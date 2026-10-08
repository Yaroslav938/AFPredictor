#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFPredictor Pipeline Core
Извлечение признаков из JSON antiSMASH, ML-предсказание и SHAP-объяснения.
Поддерживает как отдельные кластерные JSON (subclusters),
так и полные полногеномные файлы antiSMASH (multi-region/multi-contig).
"""

import os
import re
import json
import glob
import subprocess
import sys
import numpy as np
import pandas as pd
import joblib
import shap
from typing import Optional, Tuple, Dict, List, Any


# ============================================================================
# 1. ПАРСИНГ И ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ JSON antiSMASH
# ============================================================================

def parse_antismash_location(loc_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Парсит координаты из строки локации antiSMASH:
    например, '[197471:262863](+)', '[100:200]', '197471..262863'.
    """
    if not loc_str:
        return None, None
    m = re.search(r'\[?(\d+):(\d+)\]?', str(loc_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\.\.(\d+)', str(loc_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_single_cluster_record(rec: dict, source_name: str, input_file_name: str) -> dict:
    """
    Извлекает признаки из одной записи, представляющей изолированный BGC кластер
    (например, результат финальной аннотации субкластера).
    """
    areas = rec.get('areas', [])
    m_type = ", ".join(areas[0]['products']) if areas and 'products' in areas[0] else "Unknown"

    modules = rec.get('modules', {})

    # KnownClusterBlast
    cb_results = modules.get('antismash.modules.clusterblast', {}).get('knowncluster', {}).get('results', [])
    topscore_kcb, topscore_kcb_bcg, topscore_kcb_percentage = "No hit", "No hit", 0.0
    if cb_results and cb_results[0].get('ranking'):
        try:
            topscore_kcb = cb_results[0]['ranking'][0][0].get('description', 'No hit')
            topscore_kcb_bcg = cb_results[0]['ranking'][0][0].get('accession', '').split('.')[0]
            topscore_kcb_percentage = cb_results[0]['ranking'][0][1].get('similarity', 0.0)
        except (KeyError, IndexError):
            pass

    # Мономеры
    nrps_pks = modules.get('antismash.modules.nrps_pks', {})
    monomers = list(nrps_pks.get('consensus', {}).values())

    # SMCOGs и модификации
    smcogs, smcogs_desc, modifications = [], [], []
    genefuncs = modules.get('antismash.detection.genefunctions', {}).get('tools', {}).get('smcogs', {})
    for v in genefuncs.get('best_hits', {}).values():
        smcogs.append(v.get('reference_id', ''))
        smcogs_desc.append(v.get('description', ''))
    funcs = []
    for v in genefuncs.get('subfunction_mapping', {}).values():
        funcs.extend(v)
    modifications = list(set(funcs))

    # Домены
    domains, domains_motiff = [], []
    for v1 in modules.get('antismash.detection.nrps_pks_domains', {}).get('cds_results', {}).values():
        for v2 in v1.get('domain_hmms', []):
            domains.append(v2.get('hit_id', ''))
        for v2 in v1.get('motif_hmms', []):
            domains_motiff.append(v2.get('hit_id', ''))

    # Pfam
    pfam_id = []
    for v in modules.get('antismash.detection.cluster_hmmer', {}).get('hits', []):
        ident = v.get('identifier', '').split('.')[0]
        if ident:
            pfam_id.append(ident)

    # Классы аминокислот
    nrp_domains = []
    for v in nrps_pks.get('domain_predictions', {}).values():
        if 'nrpys' in v:
            for cluster in ['physiochemical_class', 'large_cluster', 'small_cluster']:
                v3 = v['nrpys'].get(cluster, {}).get('name')
                if v3 and v3 not in ('N/A', ''):
                    nrp_domains.append(v3)

    return {
        'filename': input_file_name if input_file_name and input_file_name != 'Unknown' else source_name,
        'product_type': m_type,
        'Cluster_Blast_Hit': topscore_kcb,
        'CBH_percent_identity': topscore_kcb_percentage,
        'CBH_BCG': topscore_kcb_bcg,
        'predicted_monomers': " ".join(monomers),
        'domains': " ".join(domains),
        'domains_motiff': " ".join(domains_motiff),
        'modification_enzymes': " ".join(modifications),
        'pfam_id': " ".join(pfam_id),
        'smcogs_id': " ".join(smcogs),
        'smcogs_description': ", ".join(smcogs_desc),
        'aa_classes': ", ".join(nrp_domains),
    }


def extract_from_antismash_data(data: dict, source_name: str = "Unknown") -> List[dict]:
    """
    Универсальный парсер antiSMASH JSON данных:
    Корректно обрабатывает как одиночные кластеры, так и полногеномные файлы
    с множеством контигов и областей (areas).
    """
    records = data.get('records', [])
    if not records:
        return []

    total_areas = sum(len(r.get('areas', [])) for r in records)
    input_file_name = data.get('input_file', source_name)

    # Сценарий 1: JSON содержит изолированный кластер (1 запись и <=1 область)
    if len(records) == 1 and total_areas <= 1:
        return [_extract_single_cluster_record(records[0], source_name, input_file_name)]

    # Сценарий 2: Полногеномный файл с множеством BGC областей
    results = []
    base_name = os.path.splitext(source_name)[0].replace('.fasta', '').replace('.fna', '').replace('.gbk', '')

    for r_idx, rec in enumerate(records):
        contig_id = rec.get('id', f'contig_{r_idx+1}')
        areas = rec.get('areas', [])
        modules = rec.get('modules', {})
        features = rec.get('features', [])

        # Индексируем CDS по координатам и locus_tag
        cds_loci = {}
        for f in features:
            if f.get('type') == 'CDS':
                loc = f.get('location', '')
                s, e = parse_antismash_location(loc)
                lt = f.get('qualifiers', {}).get('locus_tag', [''])[0]
                if lt and s is not None and e is not None:
                    cds_loci[lt] = (s, e)

        # Модули аннотации
        nrps_mod = modules.get('antismash.modules.nrps_pks', {})
        consensus_dict = nrps_mod.get('consensus', {})
        domain_preds = nrps_mod.get('domain_predictions', {})
        reg_preds = nrps_mod.get('region_predictions', {})

        nrps_domains_mod = modules.get('antismash.detection.nrps_pks_domains', {})
        cds_results = nrps_domains_mod.get('cds_results', {})

        genefunc_mod = modules.get('antismash.detection.genefunctions', {})
        smcogs_hits = genefunc_mod.get('tools', {}).get('smcogs', {}).get('best_hits', {})
        smcogs_funcs = genefunc_mod.get('tools', {}).get('smcogs', {}).get('subfunction_mapping', {})

        cluster_hmmer_mod = modules.get('antismash.detection.cluster_hmmer', {})
        pfam_hits = cluster_hmmer_mod.get('hits', [])

        cb_mod = modules.get('antismash.modules.clusterblast', {}).get('knowncluster', {})
        cb_results = cb_mod.get('results', [])

        if areas:
            for a_idx, area in enumerate(areas):
                a_start = area.get('start', 0)
                a_end = area.get('end', 0)
                products = area.get('products', ['Unknown'])
                m_type = ", ".join(products) if products else "Unknown"

                # Находим CDS, входящие в эту область
                area_cdss = set()
                for lt, (cs, ce) in cds_loci.items():
                    if cs >= a_start and ce <= a_end:
                        area_cdss.add(lt)

                # 1. Мономеры в данной области
                area_monomers = []
                for dom_key, mon in consensus_dict.items():
                    matched = False
                    for lt in area_cdss:
                        if lt in dom_key:
                            matched = True
                            break
                    if matched:
                        area_monomers.append(mon)

                # Если по locus_tag не сопоставилось, проверяем region_predictions
                if not area_monomers:
                    reg_key = str(a_idx + 1)
                    if reg_key in reg_preds:
                        for pred_item in reg_preds[reg_key]:
                            poly = pred_item.get('polymer', '')
                            # Извлекаем названия мономеров из полимера e.g. (Glu - Leu - D-Leu)
                            found_mons = re.findall(r'\b[A-Z][a-zA-Z0-9_-]+\b', poly)
                            for fm in found_mons:
                                if fm not in ['D', 'L']:
                                    area_monomers.append(fm)

                # 2. Домены
                area_domains, area_motifs = [], []
                for lt, v1 in cds_results.items():
                    if lt in area_cdss:
                        for v2 in v1.get('domain_hmms', []):
                            area_domains.append(v2.get('hit_id', ''))
                        for v2 in v1.get('motif_hmms', []):
                            area_motifs.append(v2.get('hit_id', ''))

                # 3. SMCOGs и модификации
                area_smcogs, area_smcogs_desc, area_funcs = [], [], []
                for lt, v in smcogs_hits.items():
                    if lt in area_cdss:
                        area_smcogs.append(v.get('reference_id', ''))
                        area_smcogs_desc.append(v.get('description', ''))
                for lt, flist in smcogs_funcs.items():
                    if lt in area_cdss:
                        area_funcs.extend(flist)
                area_modifications = list(set(area_funcs))

                # 4. Pfam
                area_pfam = []
                for hit in pfam_hits:
                    lt = hit.get('locus_tag', '')
                    hs = hit.get('start', 0)
                    he = hit.get('end', 0)
                    if lt in area_cdss or (hs >= a_start and he <= a_end):
                        ident = hit.get('identifier', '').split('.')[0]
                        if ident:
                            area_pfam.append(ident)

                # 5. NRP физико-химические классы
                area_aa_classes = []
                for dom_key, v in domain_preds.items():
                    matched = any(lt in dom_key for lt in area_cdss)
                    if matched and 'nrpys' in v:
                        for cluster in ['physiochemical_class', 'large_cluster', 'small_cluster']:
                            v3 = v['nrpys'].get(cluster, {}).get('name')
                            if v3 and v3 not in ('N/A', ''):
                                area_aa_classes.append(v3)

                # 6. KnownClusterBlast
                topscore_kcb, topscore_kcb_bcg, topscore_kcb_percentage = "No hit", "No hit", 0.0
                if cb_results:
                    cb_item = None
                    if a_idx < len(cb_results):
                        cb_item = cb_results[a_idx]
                    elif len(cb_results) == 1:
                        cb_item = cb_results[0]

                    if cb_item and cb_item.get('ranking'):
                        try:
                            topscore_kcb = cb_item['ranking'][0][0].get('description', 'No hit')
                            topscore_kcb_bcg = cb_item['ranking'][0][0].get('accession', '').split('.')[0]
                            topscore_kcb_percentage = cb_item['ranking'][0][1].get('similarity', 0.0)
                        except (KeyError, IndexError):
                            pass

                cluster_name = f"{base_name}_{contig_id}_BGC_{a_idx+1}_subcluster_1"

                results.append({
                    'filename': cluster_name,
                    'product_type': m_type,
                    'Cluster_Blast_Hit': topscore_kcb,
                    'CBH_percent_identity': topscore_kcb_percentage,
                    'CBH_BCG': topscore_kcb_bcg,
                    'predicted_monomers': " ".join(area_monomers),
                    'domains': " ".join(area_domains),
                    'domains_motiff': " ".join(area_motifs),
                    'modification_enzymes': " ".join(area_modifications),
                    'pfam_id': " ".join(area_pfam),
                    'smcogs_id': " ".join(area_smcogs),
                    'smcogs_description': ", ".join(area_smcogs_desc),
                    'aa_classes': ", ".join(area_aa_classes),
                })

    return results


def extract_features_from_files(file_paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Извлекает признаки из списка путей к JSON файлам.
    """
    rows = []
    errors = []
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            extracted = extract_from_antismash_data(data, os.path.basename(fp))
            if extracted:
                rows.extend(extracted)
            else:
                errors.append(f"{os.path.basename(fp)}: Не удалось найти BGC области в файле")
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, errors


def extract_features_from_directory(input_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Извлекает признаки из всех JSON файлов в директории (рекурсивно).
    """
    file_list = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    if not file_list:
        return pd.DataFrame(), [f"JSON файлы не найдены в {input_dir}"]
    return extract_features_from_files(file_list)


def extract_features_from_uploaded(uploaded_files) -> Tuple[pd.DataFrame, List[str]]:
    """
    Извлекает признаки из загруженных файлов (Streamlit UploadedFile).
    """
    rows = []
    errors = []
    for uf in uploaded_files:
        try:
            raw_bytes = uf.getvalue() if hasattr(uf, 'getvalue') else uf.read()
            text_content = raw_bytes.decode('utf-8', errors='replace')
            data = json.loads(text_content)
            extracted = extract_from_antismash_data(data, uf.name)
            if extracted:
                rows.extend(extracted)
            else:
                errors.append(f"{uf.name}: В файле не обнаружено BGC записей")
        except Exception as e:
            errors.append(f"{uf.name}: Ошибка парсинга JSON ({e})")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, errors


# ============================================================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ ML
# ============================================================================

def prepare_features(table_merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготовка признаков для ML-предсказания.
    
    Возвращает:
        annotation_df: таблица со ВСЕМИ кластерами (полная аннотация)
        ml_df: отфильтрованная таблица для ML (только с predicted_monomers)
    """
    if table_merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    table_merged = table_merged.copy()
    table_merged['filename'] = (
        table_merged['filename']
        .astype(str)
        .str.replace('.fasta', '', regex=False)
        .str.replace('.gbk', '', regex=False)
        .str.replace('.json', '', regex=False)
    )

    annotation_df = table_merged.copy()

    # Фильтрация для ML: только кластеры с предсказанными мономерами
    ml_df = table_merged[
        table_merged['predicted_monomers'].notna() &
        (table_merged['predicted_monomers'].astype(str).str.strip() != '')
    ].copy()

    if not ml_df.empty:
        ml_df['aa_classes'] = (
            ml_df['aa_classes']
            .fillna('')
            .astype(str)
            .str.replace(' ', '_', regex=False)
            .str.replace(',_', ' ', regex=False)
        )
        ml_df['all'] = (
            ml_df[['pfam_id', 'smcogs_id', 'domains', 'aa_classes']]
            .fillna('')
            .astype(str)
            .agg(' '.join, axis=1)
        )

    return annotation_df, ml_df


# ============================================================================
# 3. ЗАГРУЗКА ML-МОДЕЛЕЙ
# ============================================================================

def load_models(models_dir: str) -> Dict[str, Any]:
    """
    Загружает 4 ML-модели и словарь оптимальных порогов отсечки.
    """
    with open(os.path.join(models_dir, "optimal_thresholds.json"), 'r', encoding='utf-8') as f:
        thresholds = json.load(f)

    models = {
        'pu': joblib.load(os.path.join(models_dir, "Balanced_Bagging_PU_pipeline.joblib")),
        'rf': joblib.load(os.path.join(models_dir, "Single_RF_pipeline.joblib")),
        'brf': joblib.load(os.path.join(models_dir, "Balanced_RF_pipeline.joblib")),
        'ensemble': joblib.load(os.path.join(models_dir, "Ensemble_Voting_pipeline.joblib")),
    }

    return {'models': models, 'thresholds': thresholds}


# ============================================================================
# 4. ML-ПРЕДСКАЗАНИЕ
# ============================================================================

def predict(df: pd.DataFrame, model_pack: dict) -> pd.DataFrame:
    """
    Запускает расчет вероятностей всеми 4 моделями и определяет согласие моделей.
    """
    if df.empty:
        return df

    df = df.copy()
    models = model_pack['models']
    thresholds = model_pack['thresholds']

    # PU Learning
    df['PU_Probability'] = models['pu'].predict_proba(df)[:, 1]
    df['PU_Prediction'] = (df['PU_Probability'] >= thresholds['Balanced_Bagging_PU']).astype(int)

    # Random Forest
    df['RF_Probability'] = models['rf'].predict_proba(df)[:, 1]
    df['RF_Prediction'] = (df['RF_Probability'] >= thresholds['Single_RF']).astype(int)

    # Balanced Random Forest
    df['BRF_Probability'] = models['brf'].predict_proba(df)[:, 1]
    df['BRF_Prediction'] = (df['BRF_Probability'] >= thresholds['Balanced_RF']).astype(int)

    # Ensemble
    df['Ensemble_Probability'] = models['ensemble'].predict_proba(df)[:, 1]
    df['Ensemble_Prediction'] = (df['Ensemble_Probability'] >= thresholds['Ensemble_Voting']).astype(int)

    # Согласие моделей
    conditions = [
        (df['PU_Prediction'] == 1) & (df['RF_Prediction'] == 1),
        (df['PU_Prediction'] == 0) & (df['RF_Prediction'] == 0),
    ]
    df['Models_Agreement'] = np.select(conditions, [1, 0], default=-1)

    return df


# ============================================================================
# 5. SHAP-ОБЪЯСНЕНИЯ
# ============================================================================

def _get_top_reasons(shap_row, feat_names, top_n=10, positive=True):
    """Извлекает топ-N признаков по SHAP-вкладу."""
    sorted_indices = np.argsort(shap_row)[::-1] if positive else np.argsort(shap_row)
    reasons = []
    for idx in sorted_indices[:top_n]:
        val = shap_row[idx]
        if (positive and val > 0.005) or (not positive and val < -0.005):
            clean_name = feat_names[idx].replace('all__', '')
            reasons.append(f"{clean_name} ({val:+.3f})")
    return " | ".join(reasons) if reasons else "Нет сильных триггеров"


def compute_shap(df: pd.DataFrame, brf_model) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Вычисляет SHAP-значения с использованием Balanced RF модели.
    """
    if df.empty:
        return df, None, None

    df = df.copy()

    try:
        transformer = brf_model[:-1]
        model_for_shap = brf_model.named_steps['model']

        X_transformed = transformer.transform(df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        feature_names = np.array(transformer.get_feature_names_out())
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_pos = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_pos = shap_values[:, :, 1]
        else:
            shap_pos = shap_values

        df['Top_Positive_Features'] = [
            _get_top_reasons(row, feature_names, top_n=10, positive=True)
            for row in shap_pos
        ]
        df['Top_Negative_Features'] = [
            _get_top_reasons(row, feature_names, top_n=10, positive=False)
            for row in shap_pos
        ]

        return df, shap_pos, feature_names

    except Exception:
        df['Top_Positive_Features'] = "Не рассчитано"
        df['Top_Negative_Features'] = "Не рассчитано"
        return df, None, None


# ============================================================================
# 6. ФОРМАТИРОВАНИЕ И РАЗДЕЛЕНИЕ НА ШТАММ / КОНТИГ / BGC
# ============================================================================

def _split_strain_contig(val: str) -> Tuple[str, str]:
    """Разделяет префикс на штамм и контиг."""
    if not val or pd.isna(val):
        return 'unknown', 'unknown'
    val = str(val)
    if '_contig_' in val:
        parts = val.rsplit('_contig_', 1)
        return parts[0], 'contig_' + parts[1]
    elif '_ctg' in val:
        parts = val.rsplit('_ctg', 1)
        return parts[0], 'ctg' + parts[1]
    else:
        parts = val.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], 'contig_' + parts[1]
        return val, 'unknown'


def format_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Форматирует DataFrame: надежно извлекает strain, contig, BGC, subcluster
    из поля filename для любого типа файлов.
    """
    df = df.copy()
    if df.empty:
        for c in ['filename', 'strain', 'contig', 'BGC', 'subcluster']:
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)
        return df

    extracted = df['filename'].astype(str).str.extract(r'^(.*)_(BGC_\d+)_(subcluster_\d+)$')

    bgc_list = []
    sub_list = []
    strain_list = []
    contig_list = []

    for idx in range(len(df)):
        matched = False
        if not extracted.empty and idx in extracted.index:
            row_ext = extracted.iloc[idx]
            if pd.notna(row_ext[0]) and pd.notna(row_ext[1]) and pd.notna(row_ext[2]):
                s_val, c_val = _split_strain_contig(row_ext[0])
                strain_list.append(s_val)
                contig_list.append(c_val)
                bgc_list.append(row_ext[1])
                sub_list.append(row_ext[2])
                matched = True

        if not matched:
            fname = str(df.iloc[idx].get('filename', 'unknown'))
            clean_name = fname.replace('.json', '').replace('.fasta', '').replace('.gbk', '')
            s_val, c_val = _split_strain_contig(clean_name)
            if s_val == 'unknown' or not s_val:
                s_val = clean_name
                c_val = 'contig_1'
            strain_list.append(s_val)
            contig_list.append(c_val)
            bgc_list.append('BGC_1')
            sub_list.append('subcluster_1')

    df['strain'] = strain_list
    df['contig'] = contig_list
    df['BGC'] = bgc_list
    df['subcluster'] = sub_list

    return df


def format_ml_results(df: pd.DataFrame) -> pd.DataFrame:
    """Форматирует ML-результаты с унифицированным порядком колонок."""
    cols_order = [
        'filename', 'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers', 'PU_Probability', 'PU_Prediction',
        'RF_Probability', 'RF_Prediction', 'BRF_Probability', 'BRF_Prediction',
        'Ensemble_Probability', 'Ensemble_Prediction', 'Models_Agreement',
        'Top_Positive_Features', 'Top_Negative_Features',
        'pfam_id', 'smcogs_id', 'domains', 'aa_classes',
    ]
    if df.empty:
        return pd.DataFrame(columns=[c for c in cols_order if c in ['filename', 'strain', 'contig', 'BGC', 'subcluster', 'product_type', 'predicted_monomers', 'Ensemble_Prediction', 'Ensemble_Probability']])

    df = format_results(df)
    cols = [c for c in cols_order if c in df.columns]
    return df[cols]


def format_annotation_results(df: pd.DataFrame) -> pd.DataFrame:
    """Форматирует общую таблицу аннотации для всех обнаруженных BGC."""
    cols_order = [
        'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers',
    ]
    if df.empty:
        return pd.DataFrame(columns=cols_order)

    df = format_results(df)
    cols = [c for c in cols_order if c in df.columns]
    return df[cols]


# ============================================================================
# 7. ГЕНЕРАЦИЯ ВЫХОДНЫХ ТАБЛИЦ
# ============================================================================

def generate_output_tables(ml_df: pd.DataFrame, annotation_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Генерирует все итоговые таблицы отчета:
    - annotation_table: все BGC кластеры
    - ml_result_merged: кластеры, прошедшие ML-классификацию
    - positive_results: предсказанные антифунгальные кластеры
    - new_antifungal: новые кандидаты (без близких гомологов в MIBiG)
    - new_antifungal_minimal: краткая сводка новых кандидатов
    """
    annotation_formatted = format_annotation_results(annotation_df)
    ml_formatted = format_ml_results(ml_df)

    # Положительные кластеры (Ensemble = 1)
    if not ml_formatted.empty and 'Ensemble_Prediction' in ml_formatted.columns:
        positive = ml_formatted[ml_formatted['Ensemble_Prediction'] == 1].copy()
    else:
        positive = pd.DataFrame(columns=ml_formatted.columns)

    # Новые антигрибковые кандидаты
    if not ml_formatted.empty and 'Cluster_Blast_Hit' in ml_formatted.columns and 'Ensemble_Prediction' in ml_formatted.columns:
        new_af = ml_formatted[
            (ml_formatted['Cluster_Blast_Hit'].isna() | (ml_formatted['Cluster_Blast_Hit'] == 'No hit')) &
            (ml_formatted['Ensemble_Prediction'] == 1)
        ].copy()
    else:
        new_af = pd.DataFrame(columns=ml_formatted.columns)

    mini_cols = [
        'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'predicted_monomers', 'Ensemble_Probability', 'Ensemble_Prediction',
        'Models_Agreement',
    ]
    mini_cols_exist = [c for c in mini_cols if c in new_af.columns]
    new_af_minimal = new_af[mini_cols_exist].copy() if not new_af.empty else pd.DataFrame(columns=mini_cols)

    return {
        'annotation_table': annotation_formatted,
        'ml_result_merged': ml_formatted,
        'positive_results': positive,
        'new_antifungal': new_af,
        'new_antifungal_minimal': new_af_minimal,
    }


# ============================================================================
# 8. ПОЛНЫЙ ПАЙПЛАЙН (SUBPROCESS)
# ============================================================================

def check_antismash_available() -> bool:
    """Проверяет доступность команды antismash в окружении."""
    try:
        result = subprocess.run(
            ["antismash", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def run_full_pipeline(
    input_dir: str,
    ext: str,
    output_dir: str,
    threads: int,
    models_dir: str,
    out_predict: str,
) -> subprocess.Popen:
    """
    Запускает полный консольный пайплайн AFPredictor_full.py.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_script = os.path.join(script_dir, "AFPredictor_full.py")

    cmd = [
        sys.executable, full_script,
        "-i", input_dir,
        "-e", ext,
        "-o", output_dir,
        "-t", str(threads),
        "-m", models_dir,
        "--out_predict", out_predict,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


# ============================================================================
# 9. ТОЧКА ВХОДА ML-ПРЕДСКАЗАНИЯ
# ============================================================================

def run_ml_prediction(
    raw_df: pd.DataFrame,
    models_dir: str,
) -> Dict[str, Any]:
    """
    Запуск полного цикла ML: подготовка признаков -> классификация -> SHAP -> генерация таблиц.
    """
    annotation_df, ml_df = prepare_features(raw_df)

    if ml_df.empty:
        tables = generate_output_tables(ml_df, annotation_df)
        return {
            'tables': tables,
            'shap_values': None,
            'feature_names': None,
            'n_total': len(annotation_df),
            'n_ml': 0,
            'n_positive': 0,
            'n_new': 0,
        }

    # Загрузка моделей
    model_pack = load_models(models_dir)

    # Классификация
    ml_df = predict(ml_df, model_pack)

    # SHAP анализ
    ml_df, shap_vals, feat_names = compute_shap(ml_df, model_pack['models']['brf'])

    # Формирование итоговых таблиц
    tables = generate_output_tables(ml_df, annotation_df)

    n_positive = int((ml_df['Ensemble_Prediction'] == 1).sum()) if 'Ensemble_Prediction' in ml_df.columns else 0
    n_new_af = len(tables['new_antifungal'])

    return {
        'tables': tables,
        'shap_values': shap_vals,
        'feature_names': feat_names,
        'n_total': len(annotation_df),
        'n_ml': len(ml_df),
        'n_positive': n_positive,
        'n_new': n_new_af,
    }
