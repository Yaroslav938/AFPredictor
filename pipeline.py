#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFPredictor Pipeline Core
Извлечение признаков из JSON antiSMASH, ML-предсказание и SHAP-объяснения.
Модуль используется веб-интерфейсом (app.py) и может быть импортирован отдельно.
"""

import os
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
# 1. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ JSON
# ============================================================================

def _extract_single_json(data: dict, source_name: str = "Unknown") -> Optional[dict]:
    """
    Извлекает признаки из одного распарсенного JSON файла antiSMASH.
    Возвращает словарь с признаками или None при ошибке.
    """
    try:
        pfam_id, smcogs, smcogs_desc = [], [], []
        domains, domains_motiff, funcs = [], [], []
        nrp_domains, modifications, monomers = [], [], []

        name = data.get('input_file', source_name)

        # Тип продукта
        areas = data.get('records', [{}])[0].get('areas', [])
        m_type = ", ".join(areas[0]['products']) if areas and 'products' in areas[0] else "Unknown"

        # ClusterBlast хиты
        topscore_kcb, topscore_kcb_bcg, topscore_kcb_percentage = "No hit", "No hit", 0.0
        try:
            cb_results = data['records'][0]['modules']['antismash.modules.clusterblast']['knowncluster'].get('results', [])
            if cb_results and cb_results[0].get('ranking'):
                topscore_kcb = cb_results[0]['ranking'][0][0].get('description', 'No hit')
                topscore_kcb_bcg = cb_results[0]['ranking'][0][0].get('accession', '').split('.')[0]
                topscore_kcb_percentage = cb_results[0]['ranking'][0][1].get('similarity', 0.0)
        except (KeyError, IndexError):
            pass

        # Мономеры
        try:
            consensus = data['records'][0]['modules']['antismash.modules.nrps_pks'].get('consensus', {})
            monomers = list(consensus.values())
        except KeyError:
            pass

        # smcogs и модификации
        try:
            for v in data['records'][0]['modules']['antismash.detection.genefunctions']['tools']['smcogs']['best_hits'].values():
                smcogs.append(v['reference_id'])
                smcogs_desc.append(v['description'])
            for v in data['records'][0]['modules']['antismash.detection.genefunctions']['tools']['smcogs']['subfunction_mapping'].values():
                funcs.extend(v)
            modifications = list(set(funcs))
        except KeyError:
            pass

        # Домены
        try:
            for v1 in data['records'][0]['modules']['antismash.detection.nrps_pks_domains']['cds_results'].values():
                for v2 in v1.get('domain_hmms', []):
                    domains.append(v2['hit_id'])
                for v2 in v1.get('motif_hmms', []):
                    domains_motiff.append(v2['hit_id'])
        except KeyError:
            pass

        # pfam_id
        try:
            for v in data['records'][0]['modules']['antismash.detection.cluster_hmmer']['hits']:
                pfam_id.append(v['identifier'].split('.')[0])
        except KeyError:
            pass

        # Классы аминокислот (NRP)
        try:
            for v in data['records'][0]['modules']['antismash.modules.nrps_pks']['domain_predictions'].values():
                if 'nrpys' in v:
                    for cluster in ['physiochemical_class', 'large_cluster', 'small_cluster']:
                        v3 = v['nrpys'].get(cluster, {}).get('name')
                        if v3 and v3 not in ('N/A', ''):
                            nrp_domains.append(v3)
        except KeyError:
            pass

        return {
            'filename': name,
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

    except Exception:
        return None


def extract_features_from_files(file_paths: List[str]) -> pd.DataFrame:
    """
    Извлекает признаки из списка путей к JSON файлам.
    """
    rows = []
    errors = []
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            result = _extract_single_json(data, os.path.basename(fp))
            if result:
                rows.append(result)
            else:
                errors.append(os.path.basename(fp))
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
            content = uf.read()
            data = json.loads(content)
            result = _extract_single_json(data, uf.name)
            if result:
                rows.append(result)
            else:
                errors.append(uf.name)
        except Exception as e:
            errors.append(f"{uf.name}: {e}")
    
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, errors


# ============================================================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================================

def prepare_features(table_merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготовка признаков для ML-предсказания.
    
    Возвращает:
        annotation_df: таблица со ВСЕМИ кластерами (без ML)
        ml_df: отфильтрованная таблица для ML (только с predicted_monomers)
    """
    # Очистка имён файлов
    table_merged = table_merged.copy()
    table_merged['filename'] = (
        table_merged['filename']
        .str.replace('.fasta', '', regex=False)
        .str.replace('.gbk', '', regex=False)
    )
    
    # Копия для аннотационной таблицы (все кластеры)
    annotation_df = table_merged.copy()
    
    # Фильтрация: только кластеры с мономерами (для ML)
    ml_df = table_merged[
        table_merged['predicted_monomers'].notna() & 
        (table_merged['predicted_monomers'] != '')
    ].copy()
    
    if not ml_df.empty:
        ml_df['aa_classes'] = (
            ml_df['aa_classes']
            .fillna('')
            .str.replace(' ', '_', regex=False)
            .str.replace(',_', ' ', regex=False)
        )
        ml_df['all'] = (
            ml_df[['pfam_id', 'smcogs_id', 'domains', 'aa_classes']]
            .fillna('')
            .agg(' '.join, axis=1)
        )
    
    return annotation_df, ml_df


# ============================================================================
# 3. ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================================

def load_models(models_dir: str) -> Dict[str, Any]:
    """
    Загружает все 4 модели и пороги.
    """
    with open(os.path.join(models_dir, "optimal_thresholds.json"), 'r') as f:
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
    Запускает предсказание всеми 4 моделями.
    """
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
    
    Возвращает:
        df: DataFrame с добавленными колонками Top_Positive/Negative_Features
        shap_values_matrix: матрица SHAP-значений (n_samples x n_features)
        feature_names: массив имён признаков
    """
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
    
    except Exception as e:
        df['Top_Positive_Features'] = "Ошибка SHAP"
        df['Top_Negative_Features'] = "Ошибка SHAP"
        return df, None, None


# ============================================================================
# 6. ФОРМАТИРОВАНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def _split_strain_contig(val):
    """Разделяет строку на штамм и контиг."""
    if pd.isna(val):
        return pd.Series(['unknown', 'unknown'])
    if '_contig_' in val:
        parts = val.rsplit('_contig_', 1)
        return pd.Series([parts[0], 'contig_' + parts[1]])
    else:
        parts = val.rsplit('_', 1)
        if len(parts) == 2:
            return pd.Series([parts[0], parts[1]])
        return pd.Series([val, 'unknown'])


def format_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Форматирует результаты: извлекает strain, contig, BGC, subcluster из filename.
    """
    df = df.copy()
    
    extracted = df['filename'].str.extract(r'^(.*)_(BGC_\d+)_(subcluster_\d+)$')
    df['BGC'] = extracted[1].fillna('unknown')
    df['subcluster'] = extracted[2].fillna('unknown')
    df[['strain', 'contig']] = extracted[0].apply(_split_strain_contig)
    
    return df


def format_ml_results(df: pd.DataFrame) -> pd.DataFrame:
    """Форматирует ML-результаты с нужным порядком колонок."""
    df = format_results(df)
    
    cols_order = [
        'filename', 'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers', 'PU_Probability', 'PU_Prediction',
        'RF_Probability', 'RF_Prediction', 'BRF_Probability', 'BRF_Prediction',
        'Ensemble_Probability', 'Ensemble_Prediction', 'Models_Agreement',
        'Top_Positive_Features', 'Top_Negative_Features',
        'pfam_id', 'smcogs_id', 'domains', 'aa_classes',
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    return df[cols_order]


def format_annotation_results(df: pd.DataFrame) -> pd.DataFrame:
    """Форматирует аннотационную таблицу."""
    df = format_results(df)
    
    cols_order = [
        'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers',
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    return df[cols_order]


# ============================================================================
# 7. ГЕНЕРАЦИЯ ВЫХОДНЫХ ТАБЛИЦ
# ============================================================================

def generate_output_tables(ml_df: pd.DataFrame, annotation_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Генерирует все выходные таблицы:
    - annotation_table: все кластеры (без ML)
    - ml_result_merged: все ML-предсказания
    - positive_results: только положительные предсказания
    - new_antifungal: положительные БЕЗ известных аналогов
    - new_antifungal_minimal: минимальная таблица новых кандидатов
    """
    annotation_formatted = format_annotation_results(annotation_df)
    ml_formatted = format_ml_results(ml_df)
    
    # Положительные (Ensemble = 1)
    positive = ml_formatted[ml_formatted['Ensemble_Prediction'] == 1].copy()
    
    # Новые антигрибковые (нет известного аналога + положительное предсказание)
    new_af = ml_formatted[
        (ml_formatted['Cluster_Blast_Hit'].isna() | (ml_formatted['Cluster_Blast_Hit'] == 'No hit')) &
        (ml_formatted['Ensemble_Prediction'] == 1)
    ].copy()
    
    mini_cols = [
        'strain', 'contig', 'BGC', 'subcluster', 'product_type',
        'predicted_monomers', 'Ensemble_Probability', 'Ensemble_Prediction',
        'Models_Agreement',
    ]
    mini_cols = [c for c in mini_cols if c in new_af.columns]
    new_af_minimal = new_af[mini_cols].copy()
    
    return {
        'annotation_table': annotation_formatted,
        'ml_result_merged': ml_formatted,
        'positive_results': positive,
        'new_antifungal': new_af,
        'new_antifungal_minimal': new_af_minimal,
    }


# ============================================================================
# 8. ПОЛНЫЙ ПАЙПЛАЙН (ЧЕРЕЗ SUBPROCESS)
# ============================================================================

def check_antismash_available() -> bool:
    """Проверяет, доступен ли antiSMASH."""
    try:
        result = subprocess.run(
            ["antismash", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
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
    Запускает полный пайплайн через AFPredictor_full.py в subprocess.
    Возвращает Popen-объект для чтения вывода.
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
# 9. ЗАПУСК ТОЛЬКО ML-ПРЕДСКАЗАНИЯ (ОСНОВНОЙ МЕТОД)
# ============================================================================

def run_ml_prediction(
    raw_df: pd.DataFrame,
    models_dir: str,
) -> Dict[str, Any]:
    """
    Запускает полный ML-пайплайн: подготовка → предсказание → SHAP → форматирование.
    
    Возвращает словарь с результатами:
        - tables: dict с выходными таблицами
        - shap_values: матрица SHAP
        - feature_names: имена признаков
        - n_total: всего кластеров
        - n_ml: кластеров с ML
        - n_positive: положительных
        - n_new: новых кандидатов
    """
    # 1. Подготовка
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
    
    # 2. Загрузка моделей
    model_pack = load_models(models_dir)
    
    # 3. Предсказание
    ml_df = predict(ml_df, model_pack)
    
    # 4. SHAP
    ml_df, shap_vals, feat_names = compute_shap(ml_df, model_pack['models']['brf'])
    
    # 5. Формирование таблиц
    tables = generate_output_tables(ml_df, annotation_df)
    
    n_positive = int((ml_df['Ensemble_Prediction'] == 1).sum())
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
