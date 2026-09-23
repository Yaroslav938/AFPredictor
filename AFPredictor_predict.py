#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import shap

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from antiSMASH JSONs and predict antifungal activity.")
    parser.add_argument("-i", "--input", required=True, help="Путь к директории с JSON файлами (поиск рекурсивный).")
    parser.add_argument("-m", "--models", required=True, help="Путь к директории с моделями и optimal_thresholds.json.")
    parser.add_argument("-o", "--output", required=True, help="Директория для сохранения итоговых таблиц.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Поиск файлов в: {args.input}")
    filelist = glob.glob(os.path.join(args.input, "**", "*.json"), recursive=True)
    print(f"Найдено JSON файлов: {len(filelist)}")
    
    if not filelist:
        print("Файлы не найдены. Завершение работы.")
        return

    # ==========================================
    # 1. ИЗВЛЕЧЕНИЕ ДАННЫХ
    # ==========================================
    table_merged = pd.DataFrame()
    print("Извлечение данных из JSON...")

    for file in filelist:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                pfam_id, smcogs, smcogs_desc = [], [], []
                domains, domains_motiff, funcs = [], [], []
                nrp_domains, modifications, monomers = [], [], []

                name = data.get('input_file', 'Unknown')
                
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

                # smcogs & модификации
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
                
                # Классы аминокислот
                try:
                    for v in data['records'][0]['modules']['antismash.modules.nrps_pks']['domain_predictions'].values():
                        if 'nrpys' in v:
                            for cluster in ['physiochemical_class', 'large_cluster', 'small_cluster']:
                                v3 = v['nrpys'].get(cluster, {}).get('name')
                                if v3 and v3 not in ('N/A', ''):
                                    nrp_domains.append(v3)
                except KeyError:
                    pass

                # Сборка строки
                table = pd.DataFrame({
                    'filename': [name],
                    'product_type': [m_type],
                    'Cluster_Blast_Hit': [topscore_kcb],
                    'CBH_percent_identity': [topscore_kcb_percentage],
                    'CBH_BCG': [topscore_kcb_bcg],
                    'predicted_monomers': [" ".join(monomers)],
                    'domains': [" ".join(domains)],
                    'domains_motiff': [" ".join(domains_motiff)],
                    'modification_enzymes': [" ".join(modifications)],
                    'pfam_id': [" ".join(pfam_id)],
                    'smcogs_id': [" ".join(smcogs)],
                    'smcogs_description': [", ".join(smcogs_desc)],
                    'aa_classes': [", ".join(nrp_domains)]
                })
                table_merged = pd.concat([table_merged, table], ignore_index=True)
            
            except Exception as e:
                print(f"  Пропуск файла {os.path.basename(file)} (ошибка: {e})")
                continue

    if table_merged.empty:
        print("Данные не извлечены. Завершение работы.")
        return

    # ==========================================
    # 2. ПОДГОТОВКА ФИЧЕЙ
    # ==========================================
    table_merged['filename'] = table_merged['filename'].str.replace('.fasta', '', regex=False).str.replace('.gbk', '', regex=False)
    first_merged = table_merged.copy()
    table_merged = table_merged[table_merged['predicted_monomers'].notna() & (table_merged['predicted_monomers'] != '')]
    table_merged['aa_classes'] = table_merged['aa_classes'].fillna('').str.replace(' ', '_', regex=False).str.replace(',_', ' ', regex=False)
    table_merged['all'] = table_merged[['pfam_id',  'smcogs_id', 'domains', 'aa_classes']].fillna('').agg(' '.join, axis=1)
    # ==========================================
    # 3. ML ПРЕДСКАЗАНИЯ
    # ==========================================
    print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Загрузка моделей из: {args.models}")
    try:
        with open(os.path.join(args.models, "optimal_thresholds.json"), 'r') as f:
            thresholds = json.load(f)
            
        pu_model = joblib.load(os.path.join(args.models, "Balanced_Bagging_PU_pipeline.joblib"))
        rf_model = joblib.load(os.path.join(args.models, "Single_RF_pipeline.joblib"))
        brf_model = joblib.load(os.path.join(args.models, "Balanced_RF_pipeline.joblib"))
        ens_model = joblib.load(os.path.join(args.models, "Ensemble_Voting_pipeline.joblib"))
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        return

    print("Расчет вероятностей...")
    table_merged['PU_Probability'] = pu_model.predict_proba(table_merged)[:, 1]
    table_merged['PU_Prediction'] = (table_merged['PU_Probability'] >= thresholds['Balanced_Bagging_PU']).astype(int)

    table_merged['RF_Probability'] = rf_model.predict_proba(table_merged)[:, 1]
    table_merged['RF_Prediction'] = (table_merged['RF_Probability'] >= thresholds['Single_RF']).astype(int)

    table_merged['BRF_Probability'] = brf_model.predict_proba(table_merged)[:, 1]
    table_merged['BRF_Prediction'] = (table_merged['BRF_Probability'] >= thresholds['Balanced_RF']).astype(int)

    table_merged['Ensemble_Probability'] = ens_model.predict_proba(table_merged)[:, 1]
    table_merged['Ensemble_Prediction'] = (table_merged['Ensemble_Probability'] >= thresholds['Ensemble_Voting']).astype(int)

    conditions = [
        (table_merged['PU_Prediction'] == 1) & (table_merged['RF_Prediction'] == 1),
        (table_merged['PU_Prediction'] == 0) & (table_merged['RF_Prediction'] == 0) 
    ]
    table_merged['Models_Agreement'] = np.select(conditions, [1, 0], default=-1)

    # ==========================================
    # 4. SHAP ОБЪЯСНЕНИЯ
    # ==========================================
    print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Расчет SHAP значений...")
    explain_pipe = brf_model
    transformer = explain_pipe[:-1]
    model_for_shap = explain_pipe.named_steps['model']

    X_transformed = transformer.transform(table_merged)
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

    def get_top_reasons(shap_row, feat_names, top_n=3, positive=True):
        sorted_indices = np.argsort(shap_row)[::-1] if positive else np.argsort(shap_row)
        reasons = []
        for idx in sorted_indices[:top_n]:
            val = shap_row[idx]
            if (positive and val > 0.005) or (not positive and val < -0.005):
                clean_name = feat_names[idx].replace('all__', '')
                reasons.append(f"{clean_name} ({val:+.3f})")
        return " | ".join(reasons) if reasons else "Нет сильных триггеров"

    table_merged['Top_Positive_Features'] = [get_top_reasons(row, feature_names, top_n=10, positive=True) for row in shap_pos]
    table_merged['Top_Negative_Features'] = [get_top_reasons(row, feature_names, top_n=10, positive=False) for row in shap_pos]

    # ==========================================
    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================
    print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Сборка финальных таблиц...")
    
    out_merged = table_merged.copy()

    # Разделение имени: Strain_contig_X_BGC_Y_subcluster_Z
    extracted = out_merged['filename'].str.extract(r'^(.*)_(BGC_\d+)_(subcluster_\d+)$')
    
    out_merged['BGC'] = extracted[1].fillna('unknown')
    out_merged['subcluster'] = extracted[2].fillna('unknown')
    
    # Функция для отделения штамма от контига
    def split_strain_contig(val):
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

    out_merged[['strain', 'contig']] = extracted[0].apply(split_strain_contig)

    cols_order = [
        'filename', 'strain', 'contig', 'BGC', 'subcluster', 'product_type', 
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers', 'PU_Probability', 'PU_Prediction', 'RF_Probability', 'RF_Prediction', 
        'BRF_Probability', 'BRF_Prediction', 'Ensemble_Probability', 'Ensemble_Prediction',
        'Models_Agreement', 'Top_Positive_Features', 'Top_Negative_Features',
        'pfam_id', 'smcogs_id', 'domains', 'aa_classes'
    ]
    # Оставляем только те колонки, которые реально есть в датафрейме
    cols_order = [c for c in cols_order if c in out_merged.columns]
    out_merged = out_merged[cols_order]

# Сохранение файлов
    # Только аннотация без предсказания ML
    merged_path = os.path.join(args.output, 'annotation_table.tsv')
    
    # Извлекаем базовые части из filename
    extracted = first_merged['filename'].str.extract(r'^(.*)_(BGC_\d+)_(subcluster_\d+)$')
    first_merged['BGC'] = extracted[1].fillna('unknown')
    first_merged['subcluster'] = extracted[2].fillna('unknown')
    
    # Функция для отделения штамма от контига (если 'contig_' есть в названии)
    def split_strain_contig_annot(val):
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

    # Применяем функцию к первой группе (всё что до _BGC_)
    first_merged[['strain', 'contig']] = extracted[0].apply(split_strain_contig_annot)

    # Устанавливаем нужный вам порядок колонок
    annot_cols_order = [
        'strain', 'contig', 'BGC', 'subcluster', 'product_type', 
        'Cluster_Blast_Hit', 'CBH_percent_identity', 'CBH_BCG',
        'predicted_monomers'
    ]
    
    # Защита от ошибок: оставляем только те колонки, которые реально существуют
    annot_cols_order = [c for c in annot_cols_order if c in first_merged.columns]
    
    # Сохраняем отфильтрованную таблицу с нужным порядком столбцов
    first_merged[annot_cols_order].to_csv(merged_path, index=False, sep='\t')
    
    merged_path = os.path.join(args.output, 'my_ML_result_merged.tsv')
    out_merged.to_csv(merged_path, index=False, sep='\t')

    out_merged_positive = out_merged[out_merged['Ensemble_Prediction'] == 1]
    pos_path = os.path.join(args.output, 'positive_results.tsv')
    out_merged_positive.to_csv(pos_path, index=False, sep='\t')

    new_antifungal = out_merged[out_merged['Cluster_Blast_Hit'].isna() | (out_merged['Cluster_Blast_Hit'] == 'No hit')]
    new_antifungal = new_antifungal[new_antifungal['Ensemble_Prediction'] == 1]
    
    new_path = os.path.join(args.output, 'new_antifungal.tsv')
    new_antifungal.to_csv(new_path, index=False, sep='\t')

    new_mini_cols = ['strain', 'contig', 'BGC', 'subcluster', 'product_type', 'predicted_monomers', 'Ensemble_Probability', 'Ensemble_Prediction', 'Models_Agreement']
    new_mini_cols = [c for c in new_mini_cols if c in new_antifungal.columns]
    
    new_mini_path = os.path.join(args.output, 'new_antifungal_minimal.tsv')
    new_antifungal[new_mini_cols].to_csv(new_mini_path, index=False, sep='\t')

    print(f"Готово! Найдено новых кандидатов: {len(new_antifungal)}")
    print(f"Все таблицы сохранены в: {args.output}")

if __name__ == "__main__":
    main()