#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFPredictor — Веб-интерфейс
Интерактивное приложение для предсказания антигрибковой активности
биосинтетических генных кластеров (BGC).

Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import tempfile
import shutil
import time
from io import BytesIO

# Добавляем директорию скрипта в PATH для импорта pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (
    extract_features_from_uploaded,
    extract_features_from_directory,
    run_ml_prediction,
    run_full_pipeline,
    check_antismash_available,
    load_models,
    format_annotation_results,
)

# ============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================================

st.set_page_config(
    page_title="AFPredictor — Предсказание антигрибковой активности",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# ПОЛЬЗОВАТЕЛЬСКИЕ СТИЛИ (CSS)
# ============================================================================

st.markdown("""
<style>
    /* Основные стили */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Заголовок */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #E0E0E0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: #2E86AB;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        letter-spacing: 1px;
    }
    .app-header p {
        color: #666;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Метрики (карточки) */
    .metric-card {
        background: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E86AB;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #777;
        margin-top: 0.3rem;
    }
    .metric-value.green { color: #28A745; }
    .metric-value.orange { color: #E07C24; }
    .metric-value.red { color: #DC3545; }
    
    /* Информационные блоки */
    .info-box {
        background: #F0F7FA;
        border-left: 4px solid #2E86AB;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #333;
    }
    
    /* Таблицы — аккуратнее */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }
    
    /* Кнопки */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    
    /* Скрыть footer Streamlit */
    footer { visibility: hidden; }
    
    /* Разделитель */
    .section-divider {
        border-top: 1px solid #E0E0E0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_base_dir():
    """Путь к директории приложения."""
    return os.path.dirname(os.path.abspath(__file__))


def get_models_dir():
    """Путь к папке с моделями."""
    return os.path.join(get_base_dir(), "Models")


def get_test_data_dir():
    """Путь к тестовым данным."""
    return os.path.join(get_base_dir(), "test_data")


def render_metric_card(value, label, color=""):
    """Рендер карточки метрики."""
    color_class = f" {color}" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value{color_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в Excel (bytes)."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Результаты')
    return output.getvalue()


def df_to_tsv_bytes(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в TSV (bytes)."""
    return df.to_csv(index=False, sep='\t').encode('utf-8')


@st.cache_resource(show_spinner=False)
def cached_load_models(models_dir):
    """Кэшированная загрузка моделей."""
    return load_models(models_dir)


# ============================================================================
# ВИЗУАЛИЗАЦИИ
# ============================================================================

SCIENTIFIC_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
    '#44AF69', '#6B4C9A', '#E15759', '#76B7B2', '#59A14F',
]


def plot_probability_distribution(df):
    """Гистограмма распределения вероятностей Ensemble модели."""
    fig = px.histogram(
        df, x='Ensemble_Probability',
        nbins=25,
        color_discrete_sequence=['#2E86AB'],
        labels={'Ensemble_Probability': 'Вероятность (Ensemble)', 'count': 'Кол-во кластеров'},
    )
    
    # Линия порога
    fig.add_vline(x=0.5, line_dash="dash", line_color="#DC3545", line_width=2,
                  annotation_text="Порог (0.5)", annotation_position="top right",
                  annotation_font_color="#DC3545")
    
    fig.update_layout(
        title="Распределение вероятностей антигрибковой активности",
        xaxis_title="Вероятность (Ensemble модель)",
        yaxis_title="Количество кластеров",
        template="plotly_white",
        height=400,
        bargap=0.05,
        font=dict(family="Arial, sans-serif"),
    )
    return fig


def plot_model_comparison(df):
    """Scatter-plot: PU vs RF вероятности."""
    fig = px.scatter(
        df,
        x='PU_Probability',
        y='RF_Probability',
        color='Ensemble_Prediction',
        color_discrete_map={1: '#28A745', 0: '#ADB5BD'},
        hover_data=['filename', 'product_type', 'Ensemble_Probability'],
        labels={
            'PU_Probability': 'PU Learning',
            'RF_Probability': 'Random Forest',
            'Ensemble_Prediction': 'Предсказание',
        },
        category_orders={'Ensemble_Prediction': [0, 1]},
    )
    
    fig.add_hline(y=0.44, line_dash="dot", line_color="#E0E0E0", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#E0E0E0", line_width=1)
    
    fig.update_layout(
        title="Сравнение моделей: PU Learning vs Random Forest",
        template="plotly_white",
        height=450,
        font=dict(family="Arial, sans-serif"),
        legend_title_text="Ensemble",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')))
    return fig


def plot_product_types(df):
    """Pie chart: распределение типов продуктов."""
    type_counts = df['product_type'].value_counts().head(12)
    
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        color_discrete_sequence=SCIENTIFIC_COLORS,
        hole=0.35,
    )
    fig.update_layout(
        title="Типы биосинтетических продуктов",
        template="plotly_white",
        height=420,
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_traces(textposition='inside', textinfo='percent+label',
                      textfont_size=11)
    return fig


def plot_strain_predictions(df):
    """Bar chart: предсказания по штаммам."""
    if 'strain' not in df.columns:
        return None
    
    strain_data = df.groupby('strain').agg(
        total=('Ensemble_Prediction', 'count'),
        positive=('Ensemble_Prediction', 'sum'),
    ).reset_index()
    strain_data['negative'] = strain_data['total'] - strain_data['positive']
    strain_data = strain_data.sort_values('total', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=strain_data['strain'], x=strain_data['positive'],
        name='Антигрибковые', marker_color='#28A745',
        orientation='h',
    ))
    fig.add_trace(go.Bar(
        y=strain_data['strain'], x=strain_data['negative'],
        name='Не антигрибковые', marker_color='#ADB5BD',
        orientation='h',
    ))
    
    fig.update_layout(
        title="Предсказания по штаммам",
        barmode='stack',
        template="plotly_white",
        height=max(300, len(strain_data) * 40 + 100),
        font=dict(family="Arial, sans-serif"),
        xaxis_title="Количество кластеров",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_shap_waterfall(shap_values_row, feature_names, cluster_name, top_n=15):
    """SHAP waterfall plot для одного кластера."""
    indices = np.argsort(np.abs(shap_values_row))[::-1][:top_n]
    
    values = shap_values_row[indices]
    names = [fn.replace('all__', '') for fn in feature_names[indices]]
    
    colors = ['#28A745' if v > 0 else '#DC3545' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
        textfont=dict(size=10),
    ))
    
    fig.update_layout(
        title=f"SHAP: вклад признаков — {cluster_name}",
        xaxis_title="SHAP значение (вклад в предсказание)",
        template="plotly_white",
        height=max(350, top_n * 28 + 100),
        font=dict(family="Arial, sans-serif", size=11),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180),
    )
    return fig


def plot_shap_global(shap_values_matrix, feature_names, top_n=20):
    """Глобальная важность признаков по SHAP."""
    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    names = [fn.replace('all__', '') for fn in feature_names[top_indices]]
    values = mean_abs_shap[top_indices]
    
    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=names[::-1],
        orientation='h',
        marker_color='#2E86AB',
    ))
    
    fig.update_layout(
        title=f"Глобальная важность признаков (топ-{top_n})",
        xaxis_title="Средний |SHAP|",
        template="plotly_white",
        height=max(400, top_n * 25 + 100),
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=200),
    )
    return fig


def plot_all_models_box(df):
    """Box plot вероятностей для всех 4 моделей."""
    prob_cols = {
        'PU_Probability': 'PU Learning',
        'RF_Probability': 'Random Forest',
        'BRF_Probability': 'Balanced RF',
        'Ensemble_Probability': 'Ensemble',
    }
    
    plot_data = []
    for col, name in prob_cols.items():
        if col in df.columns:
            for val in df[col]:
                plot_data.append({'Модель': name, 'Вероятность': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.box(
        plot_df, x='Модель', y='Вероятность',
        color='Модель',
        color_discrete_sequence=SCIENTIFIC_COLORS[:4],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#DC3545", line_width=1,
                  annotation_text="Порог", annotation_position="top right")
    
    fig.update_layout(
        title="Распределение вероятностей по моделям",
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Arial, sans-serif"),
    )
    return fig


# ============================================================================
# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def display_results(results: dict):
    """Основной блок отображения результатов."""
    tables = results['tables']
    shap_vals = results.get('shap_values')
    feat_names = results.get('feature_names')
    
    ml_df = tables['ml_result_merged']
    annotation_df = tables['annotation_table']
    positive_df = tables['positive_results']
    new_af_df = tables['new_antifungal']
    
    # ── Сводка ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📋 Сводка результатов")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card(results['n_total'], "Всего кластеров"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card(results['n_ml'], "Проанализировано ML"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card(results['n_positive'], "Антигрибковые", "green"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card(results['n_new'], "Новые кандидаты", "orange"), unsafe_allow_html=True)
    
    if results['n_ml'] > 0 and results['n_total'] > 0:
        st.markdown(
            f'<div class="info-box">'
            f'Из <b>{results["n_total"]}</b> обнаруженных кластеров ML-анализу подлежат '
            f'<b>{results["n_ml"]}</b> (содержащие предсказанные мономеры NRPS/PKS). '
            f'Из них <b>{results["n_positive"]}</b> предсказаны как потенциально антигрибковые. '
            f'<b>{results["n_new"]}</b> не имеют известных аналогов в MIBiG — '
            f'это кандидаты на открытие новых антигрибковых соединений.'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif results['n_total'] > 0 and results['n_ml'] == 0:
        st.markdown(
            f'<div class="info-box">'
            f'Обнаружено <b>{results["n_total"]}</b> BGC кластеров. '
            f'Среди них не выявлено NRPS/PKS кластеров с аминокислотными мономерами (например, все кластеры относятся к терпенам, риппам или беталактонам). '
            f'ML-модели обучены на нерибосомных пептидах и поликетидах. '
            f'Полная таблица аннотации доступна на вкладке <b>«Все кластеры (аннотация)»</b> ниже.'
            f'</div>',
            unsafe_allow_html=True,
        )
    
    # ── Вкладки ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    tab_table, tab_viz, tab_shap, tab_download = st.tabs([
        "📊 Таблица результатов",
        "📈 Визуализации",
        "🔬 SHAP-анализ",
        "💾 Скачать",
    ])
    
    # ── Таблица ──
    with tab_table:
        default_index = 0 if not ml_df.empty else 1
        table_choice = st.radio(
            "Выберите таблицу:",
            ["ML-предсказания", "Все кластеры (аннотация)", "Только антигрибковые", "Новые кандидаты"],
            horizontal=True,
            index=default_index,
            key="table_choice",
        )
        
        if table_choice == "ML-предсказания":
            display_df = ml_df
        elif table_choice == "Все кластеры (аннотация)":
            display_df = annotation_df
        elif table_choice == "Только антигрибковые":
            display_df = positive_df
        else:
            display_df = new_af_df
        
        if display_df.empty:
            st.info("Нет данных для отображения.")
        else:
            # Фильтры
            with st.expander("🔍 Фильтры", expanded=False):
                filter_cols = st.columns(3)
                
                filtered_df = display_df.copy()
                
                if 'strain' in filtered_df.columns:
                    strains = ['Все'] + sorted(filtered_df['strain'].unique().tolist())
                    with filter_cols[0]:
                        sel_strain = st.selectbox("Штамм", strains, key="filter_strain")
                    if sel_strain != 'Все':
                        filtered_df = filtered_df[filtered_df['strain'] == sel_strain]
                
                if 'product_type' in filtered_df.columns:
                    types = ['Все'] + sorted(filtered_df['product_type'].unique().tolist())
                    with filter_cols[1]:
                        sel_type = st.selectbox("Тип продукта", types, key="filter_type")
                    if sel_type != 'Все':
                        filtered_df = filtered_df[filtered_df['product_type'] == sel_type]
                
                if 'Ensemble_Prediction' in filtered_df.columns:
                    with filter_cols[2]:
                        sel_pred = st.selectbox("Предсказание", ['Все', 'Антигрибковые (1)', 'Не антигрибковые (0)'], key="filter_pred")
                    if sel_pred == 'Антигрибковые (1)':
                        filtered_df = filtered_df[filtered_df['Ensemble_Prediction'] == 1]
                    elif sel_pred == 'Не антигрибковые (0)':
                        filtered_df = filtered_df[filtered_df['Ensemble_Prediction'] == 0]
            
            st.caption(f"Показано записей: {len(filtered_df)}")
            
            # Настройка отображения колонок
            column_config = {}
            if 'Ensemble_Probability' in filtered_df.columns:
                column_config['Ensemble_Probability'] = st.column_config.ProgressColumn(
                    "Вероятность (Ensemble)",
                    min_value=0, max_value=1,
                    format="%.3f",
                )
            if 'CBH_percent_identity' in filtered_df.columns:
                column_config['CBH_percent_identity'] = st.column_config.NumberColumn(
                    "Сходство с MIBiG (%)",
                    format="%.1f%%",
                )
            if 'Ensemble_Prediction' in filtered_df.columns:
                column_config['Ensemble_Prediction'] = st.column_config.NumberColumn(
                    "Предсказание",
                    help="1 = антигрибковый, 0 = не антигрибковый",
                )
            
            # Скрываем длинные технические колонки
            hide_cols = ['pfam_id', 'smcogs_id', 'domains', 'aa_classes',
                         'domains_motiff', 'modification_enzymes', 'smcogs_description',
                         'Top_Positive_Features', 'Top_Negative_Features']
            display_columns = [c for c in filtered_df.columns if c not in hide_cols]
            
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                height=450,
                column_config=column_config,
            )
            
            # Детали кластера
            if 'filename' in filtered_df.columns and not filtered_df.empty:
                with st.expander("🔎 Подробности по кластеру"):
                    sel_cluster = st.selectbox(
                        "Выберите кластер:",
                        filtered_df['filename'].tolist(),
                        key="detail_cluster",
                    )
                    cluster_row = filtered_df[filtered_df['filename'] == sel_cluster].iloc[0]
                    
                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        st.markdown("**Основная информация**")
                        for key in ['strain', 'contig', 'BGC', 'subcluster', 'product_type']:
                            if key in cluster_row.index:
                                st.write(f"• **{key}:** {cluster_row[key]}")
                        if 'Cluster_Blast_Hit' in cluster_row.index:
                            st.write(f"• **Ближайший аналог:** {cluster_row['Cluster_Blast_Hit']}")
                        if 'CBH_percent_identity' in cluster_row.index:
                            st.write(f"• **Сходство:** {cluster_row['CBH_percent_identity']}%")
                        if 'predicted_monomers' in cluster_row.index:
                            st.write(f"• **Мономеры:** {cluster_row['predicted_monomers']}")
                    
                    with detail_cols[1]:
                        if 'Ensemble_Probability' in cluster_row.index:
                            st.markdown("**ML-предсказание**")
                            for model, prob_col, pred_col in [
                                ('PU Learning', 'PU_Probability', 'PU_Prediction'),
                                ('Random Forest', 'RF_Probability', 'RF_Prediction'),
                                ('Balanced RF', 'BRF_Probability', 'BRF_Prediction'),
                                ('Ensemble', 'Ensemble_Probability', 'Ensemble_Prediction'),
                            ]:
                                if prob_col in cluster_row.index:
                                    prob = cluster_row[prob_col]
                                    pred = cluster_row.get(pred_col, '?')
                                    emoji = "✅" if pred == 1 else "❌"
                                    st.write(f"• **{model}:** {prob:.4f} {emoji}")
                    
                    if 'Top_Positive_Features' in cluster_row.index:
                        st.markdown("---")
                        st.markdown("**Факторы ЗА антигрибковую активность:**")
                        st.code(cluster_row['Top_Positive_Features'])
                        st.markdown("**Факторы ПРОТИВ:**")
                        st.code(cluster_row['Top_Negative_Features'])
    
    # ── Визуализации ──
    with tab_viz:
        if ml_df.empty:
            st.info("Нет ML-данных для визуализации.")
        else:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.plotly_chart(plot_probability_distribution(ml_df), use_container_width=True)
            with viz_col2:
                st.plotly_chart(plot_model_comparison(ml_df), use_container_width=True)
            
            viz_col3, viz_col4 = st.columns(2)
            
            with viz_col3:
                st.plotly_chart(plot_product_types(ml_df), use_container_width=True)
            with viz_col4:
                st.plotly_chart(plot_all_models_box(ml_df), use_container_width=True)
            
            strain_fig = plot_strain_predictions(ml_df)
            if strain_fig:
                st.plotly_chart(strain_fig, use_container_width=True)
    
    # ── SHAP ──
    with tab_shap:
        if shap_vals is not None and feat_names is not None:
            st.plotly_chart(
                plot_shap_global(shap_vals, feat_names),
                use_container_width=True,
            )
            
            st.markdown("---")
            st.subheader("SHAP для отдельного кластера")
            
            if 'filename' in ml_df.columns and not ml_df.empty:
                selected = st.selectbox(
                    "Выберите кластер для детального SHAP-анализа:",
                    ml_df['filename'].tolist(),
                    key="shap_cluster",
                )
                idx = ml_df[ml_df['filename'] == selected].index
                if len(idx) > 0:
                    # Находим позиционный индекс в shap_vals
                    pos_idx = ml_df.index.get_loc(idx[0])
                    if isinstance(pos_idx, int) or isinstance(pos_idx, np.integer):
                        st.plotly_chart(
                            plot_shap_waterfall(shap_vals[pos_idx], feat_names, selected),
                            use_container_width=True,
                        )
                    else:
                        st.warning("Не удалось определить индекс кластера.")
        else:
            st.info(
                "SHAP-анализ недоступен для демо-данных. "
                "Загрузите свои JSON файлы и запустите предсказание для получения SHAP-объяснений."
            )
    
    # ── Скачивание ──
    with tab_download:
        st.subheader("Скачать результаты")
        
        download_items = [
            ("annotation_table", "Таблица аннотации", "Все обнаруженные кластеры"),
            ("ml_result_merged", "ML-предсказания (полная)", "Все кластеры с ML-результатами"),
            ("positive_results", "Антигрибковые кластеры", "Только положительные предсказания"),
            ("new_antifungal", "Новые кандидаты (полная)", "Положительные без аналогов в MIBiG"),
            ("new_antifungal_minimal", "Новые кандидаты (краткая)", "Минимальная таблица новых кандидатов"),
        ]
        
        for key, title, description in download_items:
            if key in tables and not tables[key].empty:
                dl_col1, dl_col2, dl_col3 = st.columns([3, 1, 1])
                with dl_col1:
                    st.markdown(f"**{title}** — {description} ({len(tables[key])} записей)")
                with dl_col2:
                    st.download_button(
                        label="📥 TSV",
                        data=df_to_tsv_bytes(tables[key]),
                        file_name=f"{key}.tsv",
                        mime="text/tab-separated-values",
                        key=f"dl_tsv_{key}",
                    )
                with dl_col3:
                    try:
                        st.download_button(
                            label="📥 Excel",
                            data=df_to_excel_bytes(tables[key]),
                            file_name=f"{key}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_xlsx_{key}",
                        )
                    except Exception:
                        st.caption("Excel недоступен")
                
                st.markdown("---")


# ============================================================================
# ЗАГРУЗКА ДЕМО-ДАННЫХ
# ============================================================================

def load_demo_data() -> dict:
    """Загружает демо-данные из reference results."""
    pred_dir = os.path.join(get_test_data_dir(), "AFPredictor_reference_results", "prediction")
    
    tables = {}
    file_map = {
        'annotation_table': 'annotation_table.tsv',
        'ml_result_merged': 'my_ML_result_merged.tsv',
        'positive_results': 'positive_results.tsv',
        'new_antifungal': 'new_antifungal.tsv',
        'new_antifungal_minimal': 'new_antifungal_minimal.tsv',
    }
    
    for key, filename in file_map.items():
        path = os.path.join(pred_dir, filename)
        if os.path.exists(path):
            tables[key] = pd.read_csv(path, sep='\t')
        else:
            tables[key] = pd.DataFrame()
    
    ml_df = tables['ml_result_merged']
    n_positive = int((ml_df['Ensemble_Prediction'] == 1).sum()) if not ml_df.empty and 'Ensemble_Prediction' in ml_df.columns else 0
    
    return {
        'tables': tables,
        'shap_values': None,
        'feature_names': None,
        'n_total': len(tables['annotation_table']),
        'n_ml': len(ml_df),
        'n_positive': n_positive,
        'n_new': len(tables['new_antifungal']),
    }


# ============================================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ============================================================================

def main():
    # ── Заголовок ──
    st.markdown("""
    <div class="app-header">
        <h1>🧬 AFPredictor</h1>
        <p>Предсказание антигрибковой активности биосинтетических генных кластеров</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Боковая панель ──
    with st.sidebar:
        st.title("⚙️ Параметры")
        st.markdown("---")
        
        mode = st.radio(
            "Режим работы",
            ["🔬 Полный пайплайн", "⚡ Только ML-предсказание"],
            index=0,
            help="Полный пайплайн: FASTA → antiSMASH → ML.\n"
                 "Только ML: загрузка готовых JSON из antiSMASH.",
        )
        
        st.markdown("---")
        
        # Демо-данные
        st.markdown("### 📦 Демо-режим")
        demo_btn = st.button(
            "Загрузить демо-данные",
            help="Загрузить предрассчитанные результаты для ознакомления с интерфейсом.",
            use_container_width=True,
        )
        
        if demo_btn:
            with st.spinner("Загрузка демо-данных..."):
                st.session_state['results'] = load_demo_data()
                st.session_state['mode_used'] = 'demo'
            st.rerun()
        
        # Информация
        st.markdown("---")
        st.markdown(
            "### ℹ️ О программе\n"
            "**AFPredictor** использует ансамбль из 4 ML-моделей "
            "для предсказания антигрибковой активности BGC, "
            "обнаруженных antiSMASH.\n\n"
            "**Модели:**\n"
            "- PU Learning (Bagging)\n"
            "- Random Forest\n"
            "- Balanced Random Forest\n"
            "- Ensemble Voting\n\n"
            "Объяснения через **SHAP**."
        )
    
    # ── Основной контент ──
    
    if mode == "🔬 Полный пайплайн":
        render_full_pipeline_mode()
    else:
        render_ml_only_mode()
    
    # ── Отображение результатов ──
    if 'results' in st.session_state:
        display_results(st.session_state['results'])


def render_full_pipeline_mode():
    """UI для полного пайплайна."""
    st.subheader("🔬 Полный пайплайн: FASTA → antiSMASH → ML")
    
    st.markdown(
        '<div class="info-box">'
        '<b>Как это работает:</b> Загрузите файлы геномов в формате FASTA. '
        'Программа запустит antiSMASH для аннотации биосинтетических кластеров, '
        'извлечёт субкластеры, выполнит финальную аннотацию, '
        'а затем ML-модели предскажут антигрибковую активность каждого кластера.'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Проверка antiSMASH
    antismash_ok = check_antismash_available()
    if not antismash_ok:
        st.error(
            "⚠️ **antiSMASH не найден!** "
            "Для работы полного пайплайна необходимо установить antiSMASH. "
            "Убедитесь, что conda-окружение `afpredictor` активно.\n\n"
            "Если antiSMASH недоступен, используйте режим **«Только ML-предсказание»** "
            "— загрузите готовые JSON файлы из antiSMASH."
        )
    
    # Загрузка FASTA файлов
    uploaded_files = st.file_uploader(
        "Загрузите геномы (FASTA)",
        type=['fasta', 'fna', 'fa'],
        accept_multiple_files=True,
        help="Один или несколько файлов с геномными последовательностями.",
    )
    
    # Параметры
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        ext = st.selectbox("Расширение файлов", ['fasta', 'fna'], index=0)
    with param_col2:
        threads = st.number_input("Потоки (CPU)", min_value=1, max_value=64, value=4, step=1)
    
    # Кнопка запуска
    run_disabled = not uploaded_files or not antismash_ok
    if st.button("🚀 Запустить пайплайн", disabled=run_disabled, type="primary", use_container_width=True):
        run_full_pipeline_ui(uploaded_files, ext, threads)


def run_full_pipeline_ui(uploaded_files, ext, threads):
    """Запускает полный пайплайн и показывает прогресс."""
    models_dir = get_models_dir()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Сохранение загруженных файлов
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "antismash_output")
        predict_dir = os.path.join(tmpdir, "predictions")
        os.makedirs(input_dir, exist_ok=True)
        
        progress_bar = st.progress(0, text="Сохранение загруженных файлов...")
        
        for uf in uploaded_files:
            file_path = os.path.join(input_dir, uf.name)
            with open(file_path, 'wb') as f:
                f.write(uf.getbuffer())
        
        progress_bar.progress(5, text="Файлы сохранены. Запуск antiSMASH...")
        
        # Запуск пайплайна
        log_container = st.expander("📋 Лог выполнения", expanded=True)
        log_text = log_container.empty()
        full_log = []
        
        try:
            process = run_full_pipeline(
                input_dir, ext, output_dir, threads, models_dir, predict_dir
            )
            
            for line in iter(process.stdout.readline, ''):
                full_log.append(line)
                # Показываем последние 30 строк
                log_text.code(''.join(full_log[-30:]), language="text")
                
                # Обновляем прогресс на основе ключевых слов
                if "ЧЕРНОВАЯ АННОТАЦИЯ" in line:
                    progress_bar.progress(10, text="Черновая аннотация...")
                elif "ИЗВЛЕЧЕНИЕ" in line.upper() or "Извлечено" in line:
                    progress_bar.progress(30, text="Извлечение субкластеров...")
                elif "ФИНАЛЬНАЯ АННОТАЦИЯ" in line:
                    progress_bar.progress(50, text="Финальная аннотация...")
                elif "Загрузка моделей" in line:
                    progress_bar.progress(70, text="Загрузка ML-моделей...")
                elif "SHAP" in line:
                    progress_bar.progress(85, text="Расчёт SHAP...")
                elif "Готово!" in line:
                    progress_bar.progress(100, text="Готово!")
            
            process.wait()
            
            if process.returncode == 0:
                # Загружаем результаты из сохранённых TSV
                tables = {}
                file_map = {
                    'annotation_table': 'annotation_table.tsv',
                    'ml_result_merged': 'my_ML_result_merged.tsv',
                    'positive_results': 'positive_results.tsv',
                    'new_antifungal': 'new_antifungal.tsv',
                    'new_antifungal_minimal': 'new_antifungal_minimal.tsv',
                }
                for key, fname in file_map.items():
                    path = os.path.join(predict_dir, fname)
                    tables[key] = pd.read_csv(path, sep='\t') if os.path.exists(path) else pd.DataFrame()
                
                ml_df = tables.get('ml_result_merged', pd.DataFrame())
                n_pos = int((ml_df['Ensemble_Prediction'] == 1).sum()) if not ml_df.empty and 'Ensemble_Prediction' in ml_df.columns else 0
                
                st.session_state['results'] = {
                    'tables': tables,
                    'shap_values': None,
                    'feature_names': None,
                    'n_total': len(tables.get('annotation_table', pd.DataFrame())),
                    'n_ml': len(ml_df),
                    'n_positive': n_pos,
                    'n_new': len(tables.get('new_antifungal', pd.DataFrame())),
                }
                st.success("✅ Пайплайн завершён успешно!")
                st.rerun()
            else:
                st.error(f"❌ Пайплайн завершился с ошибкой (код {process.returncode})")
        
        except Exception as e:
            st.error(f"❌ Ошибка выполнения: {e}")


def render_ml_only_mode():
    """UI для режима только ML-предсказания."""
    st.subheader("⚡ ML-предсказание по данным antiSMASH")
    
    st.markdown(
        '<div class="info-box">'
        '<b>Поддерживаются любые JSON-файлы antiSMASH:</b><br>'
        '• <b>Полногеномные JSON</b> (из веб-сервера <a href="https://antismash.secondarymetabolites.org" target="_blank">antiSMASH</a> '
        'или локального запуска) — приложение автоматически найдет все контиги и извлечет каждый BGC кластер отдельно.<br>'
        '• <b>Изолированные кластерные JSON</b> (файлы отдельных субкластеров).<br>'
        '<i>ML-модели оценивают профили доменов, Pfam, SMCOG, мономеры и предсказывают вероятность антифунгальной активности.</i>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Загрузка JSON файлов
    uploaded_jsons = st.file_uploader(
        "Загрузите JSON файл(ы) antiSMASH",
        type=['json'],
        accept_multiple_files=True,
        help="Полногеномный JSON или JSON отдельных кластеров из antiSMASH.",
    )
    
    # Проверка моделей
    models_dir = get_models_dir()
    models_exist = os.path.exists(os.path.join(models_dir, "optimal_thresholds.json"))
    if not models_exist:
        st.error(f"⚠️ Модели не найдены в `{models_dir}`. Убедитесь, что папка `Models` содержит .joblib файлы и optimal_thresholds.json.")
    
    # Кнопка запуска пользовательских файлов
    run_disabled = not uploaded_jsons or not models_exist
    if st.button("🚀 Запустить предсказание для загруженных файлов", disabled=run_disabled, type="primary", use_container_width=True):
        run_ml_only_ui(uploaded_jsons)

    # ── Блок быстрого тестирования на реальных геномах ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🧪 Быстрое тестирование на реальных общедоступных файлах antiSMASH")
    st.caption("Выберите готовый JSON-файл аннотации генома для проверки работы без загрузки собственных данных:")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    # Проверяем наличие скачанного из официального репозитория antiSMASH файла
    online_nc_path = os.path.join(get_test_data_dir(), "NC_003888.json")
    b_subtilis_path = os.path.join(get_test_data_dir(), "AFPredictor_reference_results", "draft_annotation", "Bacillus_subtilis_Kam3", "Bacillus_subtilis_Kam3.json")
    b_pumilis_path = os.path.join(get_test_data_dir(), "AFPredictor_reference_results", "draft_annotation", "Bacillus_pumilis_C76.2", "Bacillus_pumilis_C76.2.json")
    
    with test_col1:
        if os.path.exists(online_nc_path):
            if st.button("🧬 Streptomyces coelicolor (NC_003888)", use_container_width=True, help="Официальный пример антибиотикопродуцента с сервера antiSMASH (29 кластеров)"):
                run_sample_file_ui(online_nc_path, "Streptomyces_coelicolor_A3(2)_NC_003888.json")
    
    with test_col2:
        if os.path.exists(b_subtilis_path):
            if st.button("🧫 Bacillus subtilis (Kam3)", use_container_width=True, help="Геномный файл B. subtilis (13 кластеров: сурфактин, фенгицин, бациллаен и др.)"):
                run_sample_file_ui(b_subtilis_path, "Bacillus_subtilis_Kam3.json")
                
    with test_col3:
        if os.path.exists(b_pumilis_path):
            if st.button("🦠 Bacillus pumilus (C76.2)", use_container_width=True, help="Многоконтиговый геном B. pumilis (14 кластеров)"):
                run_sample_file_ui(b_pumilis_path, "Bacillus_pumilis_C76.2.json")


def run_sample_file_ui(file_path: str, display_name: str):
    """Запускает пайплайн на локальном тестовом/скачанном файле."""
    models_dir = get_models_dir()
    progress = st.progress(0, text=f"Загрузка {display_name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        progress.progress(25, text="Извлечение BGC областей из JSON...")
        from pipeline import extract_from_antismash_data
        extracted = extract_from_antismash_data(data, display_name)
        raw_df = pd.DataFrame(extracted)
        
        if raw_df.empty:
            st.error("Не удалось обнаружить BGC кластеры в данном файле.")
            return
            
        progress.progress(50, text=f"Обнаружено {len(raw_df)} BGC. Запуск ML-моделей...")
        results = run_ml_prediction(raw_df, models_dir)
        
        progress.progress(100, text="Готово!")
        time.sleep(0.4)
        progress.empty()
        
        st.session_state['results'] = results
        st.session_state['mode_used'] = 'ml_only'
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Ошибка при анализе файла: {e}")
        import traceback
        with st.expander("Подробности ошибки"):
            st.code(traceback.format_exc())


def run_ml_only_ui(uploaded_jsons):
    """Запускает ML-предсказание по загруженным пользователем файлам."""
    models_dir = get_models_dir()
    
    progress = st.progress(0, text="Извлечение BGC признаков из загруженных файлов...")
    
    # 1. Извлечение
    raw_df, errors = extract_features_from_uploaded(uploaded_jsons)
    
    if errors:
        with st.expander(f"⚠️ Предупреждения ({len(errors)})", expanded=False):
            for err in errors:
                st.warning(err)
    
    if raw_df.empty:
        st.error("Не удалось извлечь BGC кластеры ни из одного загруженного файла. Убедитесь, что это корректный JSON от antiSMASH.")
        return
    
    progress.progress(30, text=f"Извлечено {len(raw_df)} BGC записей. Применение ML-моделей...")
    
    # 2-5. ML-пайплайн
    try:
        results = run_ml_prediction(raw_df, models_dir)
        
        progress.progress(100, text="Готово!")
        time.sleep(0.4)
        progress.empty()
        
        st.session_state['results'] = results
        st.session_state['mode_used'] = 'ml_only'
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Ошибка обработки: {e}")
        import traceback
        with st.expander("Подробности ошибки"):
            st.code(traceback.format_exc())


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    main()
