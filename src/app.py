"""
ECO Change Detection - Streamlit 대시보드
3종 Score (Shift / Tail / Outlier Wafer) 기반 변경점 분석을 수행합니다.
실행: streamlit run src/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from eco_change_detection import (
    generate_synthetic_data, run_eco_change_detection
)

st.set_page_config(page_title="ECO Change Detection", layout="wide", page_icon="🔬")


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("⚙️ 설정")

st.sidebar.header("1. 데이터 입력")
data_mode = st.sidebar.radio("데이터 소스", ["📂 파일 업로드", "🧪 데모 데이터"], index=1)

df_ref, df_comp = None, None

if data_mode == "📂 파일 업로드":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ref 데이터")
    ref_file = st.sidebar.file_uploader("Ref (기준군) CSV/Excel", type=['csv', 'xlsx'], key='ref')
    st.sidebar.subheader("Compare 데이터")
    comp_file = st.sidebar.file_uploader("Compare (변경군) CSV/Excel", type=['csv', 'xlsx'], key='comp')
    index_col = st.sidebar.checkbox("첫 번째 열을 Index로 사용", value=True)

    if ref_file and comp_file:
        try:
            if ref_file.name.endswith('.csv'):
                df_ref = pd.read_csv(ref_file, index_col=0 if index_col else None)
            else:
                df_ref = pd.read_excel(ref_file, index_col=0 if index_col else None)
            if comp_file.name.endswith('.csv'):
                df_comp = pd.read_csv(comp_file, index_col=0 if index_col else None)
            else:
                df_comp = pd.read_excel(comp_file, index_col=0 if index_col else None)
        except Exception as e:
            st.sidebar.error(f"파일 읽기 오류: {e}")
else:
    n_ref = st.sidebar.slider("Ref wafer 수", 100, 2000, 1000, 100)
    n_comp = st.sidebar.slider("Compare wafer 수", 20, 500, 80, 10)
    n_feat = st.sidebar.slider("Feature 수", 100, 5000, 5000, 100)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    df_ref, df_comp, _ = generate_synthetic_data(n_ref, n_comp, n_feat, seed)

st.sidebar.markdown("---")
st.sidebar.header("2. 분석 파라미터")
step_id = st.sidebar.text_input("Step ID", "S310_ETCH")
change_code = st.sidebar.text_input("Change Code", "CHG_001")
top_k_ratio = st.sidebar.slider("Top-K Ratio (Shift)", 0.005, 0.05, 0.01, 0.005)
tail_percentile = st.sidebar.select_slider("Tail Percentile", [0.95, 0.97, 0.99, 0.995], value=0.99)
outlier_thresh = st.sidebar.slider("Outlier Feature Threshold", 0.01, 0.10, 0.05, 0.01)
min_sample = st.sidebar.number_input("최소 Compare 수", value=30, step=5)
top_n = st.sidebar.slider("Top-N Features", 5, 50, 20, 5)


# ============================================================
# Main
# ============================================================
st.title("🔬 ECO Change Detection Dashboard")
st.markdown("반도체 공정 변경점(ECO) 전후 품질 차이를 **3종 Score (Shift / Tail / Outlier)** 로 자동 정량화합니다.")

if df_ref is None or df_comp is None:
    st.info("👈 왼쪽에서 Ref / Compare 데이터를 업로드해주세요.")
    st.stop()


# 데이터 미리보기
with st.expander("📊 데이터 미리보기", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Ref** ({df_ref.shape[0]:,} wafers × {df_ref.shape[1]:,} features)")
        st.dataframe(df_ref.head(5), height=180)
    with c2:
        st.markdown(f"**Compare** ({df_comp.shape[0]:,} wafers × {df_comp.shape[1]:,} features)")
        st.dataframe(df_comp.head(5), height=180)


# 분석 실행
if st.button("🚀 분석 실행", type="primary", use_container_width=True):
    with st.spinner("3종 Score 분석 중..."):
        result = run_eco_change_detection(
            df_ref, df_comp, step_id=step_id, change_code=change_code,
            top_k_ratio=top_k_ratio, tail_percentile=tail_percentile,
            outlier_feature_thresh=outlier_thresh,
            min_sample=min_sample, top_n_features=top_n
        )
    st.session_state['result'] = result
    st.session_state['df_ref'] = df_ref
    st.session_state['df_comp'] = df_comp

if 'result' not in st.session_state:
    st.info("▲ '분석 실행' 버튼을 클릭하세요.")
    st.stop()

result = st.session_state['result']
dec = result['decision']
meta = result['metadata']
scaled_ref = result['scaled_ref']
scaled_comp = result['scaled_comp']

# ============================================================
# 판정 배너
# ============================================================
dc = {'SAFE': ('🟢', '#4CAF50'), 'CAUTION': ('🟡', '#FFC107'),
      'RISK': ('🟠', '#FF9800'), 'HIGH_RISK': ('🔴', '#E53935'),
      'INSUFFICIENT_DATA': ('⚪', '#9E9E9E')}
icon, color = dc.get(dec['decision'], ('⚪', '#9E9E9E'))

st.markdown(f"""
<div style="background:{color}22; border-left:6px solid {color}; padding:20px; border-radius:8px; margin:20px 0;">
    <h2 style="color:{color}; margin:0;">{icon} 판정: {dec['decision']}</h2>
    <p style="margin:5px 0 0 0;">{meta['step_id']} / {meta['change_code']} | Ref: {meta['ref_count']:,} | Comp: {meta['comp_count']} | Features: {meta['feature_count']:,}</p>
    {''.join(f'<p style="margin:2px 0; color:#555;">• {r}</p>' for r in dec['reasons'])}
</div>
""", unsafe_allow_html=True)


# Score cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shift Score", f"{dec['scores']['shift_score']:.3f}")
c2.metric("Tail Score (max)", f"{dec['scores']['tail_score_max']:.1%}")
c3.metric("Tail Feature Count", f"{dec['scores']['tail_feature_count']}")
c4.metric("Outlier Wafer Rate", f"{dec['scores']['outlier_wafer_rate']:.1%}",
          f"{result['detail']['outlier']['outlier_count']}개 wafer")


# ============================================================
# Score 상세 탭
# ============================================================
st.markdown("---")
st.subheader("📈 Score 상세 분석")

tab1, tab2, tab3 = st.tabs(["Shift Score", "Tail Score", "Outlier Wafer Score"])

with tab1:
    shift = result['detail']['shift']
    z_shift = shift['z_shift_all']

    c1, c2 = st.columns(2)
    with c1:
        # Z-shift 분포
        z_sorted = z_shift.sort_values()
        colors = ['#E53935' if v > 0.5 else '#1E88E5' if v < -0.5 else '#BDBDBD' for v in z_sorted]
        fig = go.Figure(go.Bar(y=z_sorted.values, marker_color=colors))
        fig.add_hline(y=1.0, line_dash='dash', line_color='red', annotation_text='1σ')
        fig.add_hline(y=-1.0, line_dash='dash', line_color='blue', annotation_text='-1σ')
        fig.update_layout(title=f'Feature별 Z-shift 분포 (Score = {shift["score"]:.3f})',
                          xaxis_title='Features (sorted)', yaxis_title='Z-shift', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Top-N shift features
        shift_imp = result['importance']['shift_features'].head(top_n)
        colors_bar = ['#E53935' if d == '악화' else '#1E88E5' for d in shift_imp['direction']]
        fig = go.Figure(go.Bar(
            y=shift_imp['feature'][::-1], x=shift_imp['z_shift'].values[::-1],
            orientation='h', marker_color=colors_bar[::-1]
        ))
        fig.update_layout(title=f'Shift 원인 Feature Top-{top_n}', height=max(300, top_n * 25),
                          xaxis_title='Z-shift', yaxis_title='Feature')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    tail = result['detail']['tail']

    c1, c2 = st.columns(2)
    with c1:
        # Tail rate 분포
        tail_sorted = tail['tail_rate_all'].sort_values(ascending=False)
        top_show = min(100, len(tail_sorted))
        colors_t = ['#E53935' if v > 0.03 else '#FF9800' if v > 0.01 else '#BDBDBD'
                     for v in tail_sorted.values[:top_show]]
        fig = go.Figure(go.Bar(y=tail_sorted.values[:top_show] * 100, marker_color=colors_t))
        fig.add_hline(y=1, line_dash='dash', line_color='gray', annotation_text='기대치 (1%)')
        fig.add_hline(y=3, line_dash='dash', line_color='orange', annotation_text='경고 (3%)')
        fig.add_hline(y=10, line_dash='dash', line_color='red', annotation_text='위험 (10%)')
        fig.update_layout(title=f'Feature별 Tail Rate (Max={tail["score_max"]:.1%})',
                          xaxis_title='Features (sorted)', yaxis_title='Tail Rate (%)', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Elevated features
        tail_imp = result['importance']['tail_features'].head(top_n)
        if len(tail_imp) > 0:
            fig = go.Figure(go.Bar(
                y=tail_imp['feature'][::-1], x=tail_imp['tail_rate_pct'].values[::-1],
                orientation='h', marker_color='#FF9800'
            ))
            fig.add_vline(x=1, line_dash='dash', line_color='gray')
            fig.add_vline(x=3, line_dash='dash', line_color='orange')
            fig.update_layout(title=f'Tail 원인 Feature Top-{top_n} ({tail["score_count"]}개 elevated)',
                              height=max(300, top_n * 25),
                              xaxis_title='Tail Rate (%)', yaxis_title='Feature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Elevated Feature 없음")

with tab3:
    outlier = result['detail']['outlier']

    c1, c2 = st.columns(2)
    with c1:
        # Wafer별 초과 비율
        exceed = outlier['exceed_ratio_per_wafer'].sort_values(ascending=False)
        colors_o = ['#E53935' if v > outlier_thresh else '#4CAF50' for v in exceed.values]
        fig = go.Figure(go.Bar(y=exceed.values * 100, marker_color=colors_o))
        fig.add_hline(y=outlier_thresh * 100, line_dash='dash', line_color='red',
                      annotation_text=f'Outlier 기준 ({outlier_thresh:.0%})')
        fig.update_layout(title=f'Wafer별 Feature 초과 비율 (Outlier={outlier["outlier_count"]}개)',
                          xaxis_title='Compare Wafers (sorted)', yaxis_title='Feature Exceed Ratio (%)',
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Outlier 공통 Feature
        common = outlier['common_features'].head(top_n)
        if len(common) > 0:
            fig = go.Figure(go.Bar(
                y=common.index[::-1], x=common.values[::-1] * 100,
                orientation='h', marker_color='#7B1FA2'
            ))
            fig.update_layout(title=f'Outlier Wafer 공통 이상 Feature (Rate={outlier["score"]:.1%})',
                              height=max(300, len(common) * 25),
                              xaxis_title='Outlier Wafer 중 초과 비율 (%)', yaxis_title='Feature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Outlier Wafer 없음")


# ============================================================
# Feature 상세 비교
# ============================================================
st.markdown("---")
st.subheader("🔍 Feature 상세 비교")

# 모든 top features를 합쳐서 선택지 제공
all_imp_features = list(dict.fromkeys(
    result['importance']['shift_features']['feature'].head(top_n).tolist() +
    result['importance']['tail_features']['feature'].head(top_n).tolist() +
    result['importance']['outlier_features']['feature'].head(min(top_n, len(result['importance']['outlier_features']))).tolist()
))

selected_feat = st.selectbox("Feature 선택", all_imp_features)

if selected_feat and selected_feat in scaled_ref.columns:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scaled_ref[selected_feat], name='Ref', opacity=0.6,
                                    marker_color='#2196F3', histnorm='probability density'))
        fig.add_trace(go.Histogram(x=scaled_comp[selected_feat], name='Compare', opacity=0.6,
                                    marker_color='#FF5722', histnorm='probability density'))
        fig.update_layout(title=f'{selected_feat} 분포 비교', barmode='overlay', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=scaled_ref[selected_feat], name='Ref', marker_color='#2196F3'))
        fig.add_trace(go.Box(y=scaled_comp[selected_feat], name='Compare', marker_color='#FF5722'))
        fig.update_layout(title=f'{selected_feat} Box Plot', height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature 상세 정보
    z_val = result['detail']['shift']['z_shift_all'].get(selected_feat, 0)
    tail_val = result['detail']['tail']['tail_rate_all'].get(selected_feat, 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Z-shift", f"{z_val:+.4f}")
    c2.metric("Tail Rate", f"{tail_val:.2%}")
    c3.metric("방향", "악화 ↑" if z_val > 0 else "개선 ↓" if z_val < 0 else "변화 없음")


# ============================================================
# Outlier Wafer 분석
# ============================================================
st.markdown("---")
st.subheader("⚠️ Outlier Wafer 분석")

outlier_ids = result['detail']['outlier']['outlier_wafer_ids']
if outlier_ids:
    st.warning(f"**{len(outlier_ids)}개** Outlier Wafer 검출: {', '.join(outlier_ids[:20])}")

    # Shift vs Tail Feature scatter
    z_all = result['detail']['shift']['z_shift_all'].abs()
    t_all = result['detail']['tail']['tail_rate_all']
    merged = pd.DataFrame({'z_shift': z_all, 'tail_rate': t_all * 100}).dropna()

    fig = go.Figure()
    # 분류별 색상
    for _, row in merged.iterrows():
        pass
    fig.add_trace(go.Scatter(
        x=merged['z_shift'], y=merged['tail_rate'],
        mode='markers', marker=dict(size=5, opacity=0.5,
            color=['#E53935' if z > 0.5 and t > 3 else '#FF9800' if z > 0.5 else '#7B1FA2' if t > 3 else '#BDBDBD'
                   for z, t in zip(merged['z_shift'], merged['tail_rate'])]),
        text=merged.index, hoverinfo='text+x+y', name='Features'
    ))
    fig.add_hline(y=3, line_dash='dash', line_color='orange')
    fig.add_vline(x=0.5, line_dash='dash', line_color='orange')
    fig.update_layout(title='Shift vs Tail: Feature 분류', xaxis_title='|Z-shift|',
                      yaxis_title='Tail Rate (%)', height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("Outlier Wafer 없음")


# ============================================================
# 다운로드
# ============================================================
st.markdown("---")
st.subheader("📥 결과 다운로드")
c1, c2, c3 = st.columns(3)

with c1:
    shift_imp = result['importance']['shift_features']
    tail_imp = result['importance']['tail_features']
    outlier_imp = result['importance']['outlier_features']
    imp_df = pd.concat([
        shift_imp.assign(score_type='shift'),
        tail_imp.assign(score_type='tail'),
        outlier_imp.assign(score_type='outlier')
    ], ignore_index=True)
    st.download_button("Feature Importance (CSV)", imp_df.to_csv(index=False),
                       "feature_importance.csv", "text/csv")

with c2:
    exceed_ratio = result['detail']['outlier']['exceed_ratio_per_wafer']
    outlier_df = pd.DataFrame({
        'Wafer': scaled_comp.index,
        'Exceed_Ratio': exceed_ratio.values,
        'Is_Outlier': exceed_ratio.values > outlier_thresh,
    })
    st.download_button("Outlier Wafer (CSV)", outlier_df.to_csv(index=False),
                       "outlier_wafers.csv", "text/csv")

with c3:
    report = f"""ECO Change Detection Report
{'='*45}
Step: {meta['step_id']} | Code: {meta['change_code']}
Ref: {meta['ref_count']:,} | Comp: {meta['comp_count']} | Features: {meta['feature_count']:,}
{'='*45}
Decision: {dec['decision']}

Shift Score: {dec['scores']['shift_score']:.3f}
Tail Score (max): {dec['scores']['tail_score_max']:.4f}
Tail Feature Count: {dec['scores']['tail_feature_count']}
Outlier Wafer Rate: {dec['scores']['outlier_wafer_rate']:.4f}
{'='*45}
Reasons:
""" + '\n'.join(f'  - {r}' for r in dec['reasons']) + f"""
{'='*45}
Shift Top Features:
""" + '\n'.join(f'  #{i+1} {row["feature"]}: z={row["z_shift"]:+.3f} ({row["direction"]})'
                for i, (_, row) in enumerate(result['importance']['shift_features'].head(10).iterrows())) + f"""
{'='*45}
Outlier Wafers ({len(outlier_ids)}): {', '.join(outlier_ids[:20])}
"""
    st.download_button("Summary Report (TXT)", report, "report.txt", "text/plain")
