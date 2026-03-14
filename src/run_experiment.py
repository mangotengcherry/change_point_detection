"""
ECO Change Detection PoC - Full Experiment Runner
가상 데이터를 생성하고 파이프라인을 실행하며, 과정별 시각화를 생성합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats

from eco_change_detection import (
    generate_synthetic_data, filter_features, robust_scale, winsorize,
    calc_shift_score, calc_tail_score, calc_outlier_wafer_score,
    get_feature_importance, make_decision, run_eco_change_detection
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save figure to both results and docs/images"""
    fig.savefig(os.path.join(RESULTS_DIR, name), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(DOCS_DIR, name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")


# ============================================================
# 1. 합성 데이터 생성 및 시각화
# ============================================================
def visualize_synthetic_data(df_ref, df_comp):
    """합성 데이터 구조 시각화"""
    print("\n[1/8] Visualizing synthetic data structure...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Step 1: 합성 데이터 구조 분석', fontsize=16, fontweight='bold', y=0.98)

    # 1-1: Ref vs Comp 데이터 크기 비교
    ax = axes[0, 0]
    bars = ax.bar(['Ref\n(기준군)', 'Compare\n(변경군)'],
                  [len(df_ref), len(df_comp)],
                  color=['#2196F3', '#FF5722'], width=0.5, edgecolor='white', linewidth=2)
    ax.set_ylabel('Wafer 수')
    ax.set_title('데이터 크기 비교 (비대칭)', fontweight='bold')
    for bar, val in zip(bars, [len(df_ref), len(df_comp)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}', ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(len(df_ref), len(df_comp)) * 1.15)

    # 1-2: Feature 분포 (정상 feature)
    ax = axes[0, 1]
    normal_feat = 'EDS_0050'  # 변화 없는 feature
    ax.hist(df_ref[normal_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[normal_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.set_title(f'정상 Feature 분포 ({normal_feat})', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    # 1-3: Feature 분포 (Shift Pattern A)
    ax = axes[0, 2]
    shift_feat = 'EDS_0110'
    ax.hist(df_ref[shift_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[shift_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.axvline(df_ref[shift_feat].mean(), color='#1565C0', linestyle='--', linewidth=2, label='Ref mean')
    ax.axvline(df_comp[shift_feat].mean(), color='#D32F2F', linestyle='--', linewidth=2, label='Comp mean')
    ax.set_title(f'Pattern A: Systematic Shift ({shift_feat})', fontweight='bold')
    ax.set_xlabel('Value')
    ax.legend(fontsize=8)

    # 1-4: Feature 분포 (Spike Pattern B)
    ax = axes[1, 0]
    spike_feat = 'EDS_0505'
    ax.hist(df_ref[spike_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[spike_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.set_title(f'Pattern B: Intermittent Spike ({spike_feat})', fontweight='bold')
    ax.set_xlabel('Value')
    ax.legend()

    # 1-5: Outlier Wafer Pattern C heatmap
    ax = axes[1, 1]
    outlier_feats = [f'EDS_{i:04d}' for i in range(200, 260)]
    comp_subset = df_comp.iloc[65:80][outlier_feats[:30]]
    ref_mean = df_ref[outlier_feats[:30]].mean()
    diff_map = comp_subset - ref_mean
    im = ax.imshow(diff_map.values, aspect='auto', cmap='RdYlBu_r', vmin=-2, vmax=4)
    ax.set_title('Pattern C: Outlier Wafer Heatmap', fontweight='bold')
    ax.set_ylabel('Wafer Index (65~79)')
    ax.set_xlabel('Feature Index (200~229)')
    ax.axhline(y=4.5, color='red', linewidth=2, linestyle='--')
    ax.text(15, 2, 'Outlier\nWafers\n(70~74)', ha='center', va='center',
            fontsize=10, color='red', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Deviation from Ref Mean')

    # 1-6: 3가지 패턴 요약 다이어그램
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('삽입된 불량 패턴 요약', fontweight='bold')

    patterns = [
        ("Pattern A", "Systematic Shift", "Feature 100~120\n전체 wafer +1.5σ 이동", '#E53935', 7.5),
        ("Pattern B", "Intermittent Spike", "Feature 500~510\n10% wafer에서 극단값", '#FF9800', 5.0),
        ("Pattern C", "Multi-Feature Outlier", "Feature 200~259\nWafer 70~74 동시 이상", '#7B1FA2', 2.5),
    ]
    for name, desc, detail, color, y in patterns:
        box = FancyBboxPatch((0.5, y-0.8), 9, 1.5, boxstyle="round,pad=0.2",
                             facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(1.5, y+0.15, f'{name}: {desc}', fontsize=11, fontweight='bold', color=color)
        ax.text(1.5, y-0.35, detail, fontsize=9, color='#333333')

    plt.tight_layout()
    save_fig(fig, '01_synthetic_data_structure.png')


# ============================================================
# 2. 전처리 과정 시각화
# ============================================================
def visualize_preprocessing(df_ref, df_comp, features, scaled_ref, scaled_comp):
    """전처리 단계별 효과 시각화"""
    print("[2/8] Visualizing preprocessing steps...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Step 2: 전처리 과정 (Robust Scaling + Winsorizing)', fontsize=16, fontweight='bold', y=0.98)

    # 2-1: Feature 필터링 결과
    ax = axes[0, 0]
    total_feats = df_ref.shape[1]
    filtered_feats = len(features)
    removed = total_feats - filtered_feats
    ax.pie([filtered_feats, removed], labels=[f'유효\n({filtered_feats})', f'제거\n({removed})'],
           colors=['#4CAF50', '#E0E0E0'], autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'Feature 필터링 결과\n(총 {total_feats} → {filtered_feats})', fontweight='bold')

    # 2-2: Scaling 전 원본 분포 (여러 Feature)
    ax = axes[0, 1]
    sample_feats = ['EDS_0050', 'EDS_0110', 'EDS_0505', 'EDS_0230']
    for i, f in enumerate(sample_feats):
        if f in df_ref.columns:
            ax.hist(df_ref[f], bins=30, alpha=0.4, label=f, density=True)
    ax.set_title('Scaling 전: 원본 Feature 분포', fontweight='bold')
    ax.set_xlabel('Raw Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    # 2-3: Scaling 후 분포
    ax = axes[0, 2]
    for i, f in enumerate(sample_feats):
        if f in scaled_ref.columns:
            ax.hist(scaled_ref[f], bins=30, alpha=0.4, label=f, density=True)
    ax.set_title('Scaling 후: Robust-Scaled 분포', fontweight='bold')
    ax.set_xlabel('Scaled Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    # 2-4: Scaling 효과 - ref 기준 정규화 확인
    ax = axes[1, 0]
    ref_medians = scaled_ref.median()
    ref_iqrs = scaled_ref.quantile(0.75) - scaled_ref.quantile(0.25)
    ax.scatter(ref_medians.values[:200], ref_iqrs.values[:200], alpha=0.5, s=10, color='#2196F3')
    ax.axhline(y=1.0, color='red', linestyle='--', label='IQR=1 (기대값)')
    ax.axvline(x=0.0, color='orange', linestyle='--', label='Median=0 (기대값)')
    ax.set_title('Scaling 검증: Ref의 Median/IQR', fontweight='bold')
    ax.set_xlabel('Median')
    ax.set_ylabel('IQR')
    ax.legend()

    # 2-5: Ref vs Comp scaled 비교 (shift feature)
    ax = axes[1, 1]
    shift_feat = 'EDS_0110'
    if shift_feat in scaled_ref.columns and shift_feat in scaled_comp.columns:
        ax.hist(scaled_ref[shift_feat], bins=40, alpha=0.6, label='Ref (scaled)',
                color='#2196F3', density=True)
        ax.hist(scaled_comp[shift_feat], bins=20, alpha=0.6, label='Compare (scaled)',
                color='#FF5722', density=True)
        ax.set_title(f'Scaled 비교: {shift_feat} (Shift 패턴)', fontweight='bold')
        ax.set_xlabel('Scaled Value')
        ax.legend()

    # 2-6: Winsorizing 효과
    ax = axes[1, 2]
    feat = 'EDS_0505'
    if feat in scaled_ref.columns and feat in scaled_comp.columns:
        # Before winsorizing
        raw_vals = (df_comp[feat] - df_ref[feat].median()) / max(
            df_ref[feat].quantile(0.75) - df_ref[feat].quantile(0.25), 1e-10)
        ax.hist(raw_vals, bins=20, alpha=0.5, label='Before Winsorize', color='#FF9800', density=True)
        ax.hist(scaled_comp[feat], bins=20, alpha=0.5, label='After Winsorize', color='#4CAF50', density=True)
        ax.set_title(f'Winsorizing 효과: {feat}', fontweight='bold')
        ax.set_xlabel('Scaled Value')
        ax.legend()

    plt.tight_layout()
    save_fig(fig, '02_preprocessing_steps.png')


# ============================================================
# 3. Score 산출 과정 시각화
# ============================================================
def visualize_score_calculation(scaled_ref, scaled_comp, shift_result, tail_result, outlier_result):
    """3종 Score 산출 과정 상세 시각화"""
    print("[3/8] Visualizing score calculation process...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Step 3: Score 산출 과정 (3종 병렬)', fontsize=16, fontweight='bold', y=0.98)

    # --- Shift Score ---
    # 3-1: Feature별 z-shift 전체 분포
    ax1 = fig.add_subplot(gs[0, 0])
    z_vals = shift_result["z_shift_all"].sort_values()
    colors = ['#E53935' if v > 0.5 else '#1E88E5' if v < -0.5 else '#BDBDBD' for v in z_vals]
    ax1.bar(range(len(z_vals)), z_vals.values, width=1.0, color=colors)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=1.0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.0, color='blue', linewidth=1, linestyle='--', alpha=0.5)
    ax1.set_title('Shift Score: Feature별 Z-shift 분포', fontweight='bold')
    ax1.set_xlabel('Features (sorted)')
    ax1.set_ylabel('Z-shift (σ)')

    # 3-2: Top-K shift features
    ax2 = fig.add_subplot(gs[0, 1])
    top_shift = shift_result["z_shift_all"].abs().sort_values(ascending=True).tail(15)
    colors = ['#E53935' if shift_result["z_shift_all"][f] > 0 else '#1E88E5'
              for f in top_shift.index]
    ax2.barh(range(len(top_shift)), top_shift.values, color=colors)
    ax2.set_yticks(range(len(top_shift)))
    ax2.set_yticklabels(top_shift.index, fontsize=8)
    ax2.set_title(f'Shift Score Top-15 Features\n(Score = {shift_result["score"]:.3f})',
                  fontweight='bold')
    ax2.set_xlabel('|Z-shift|')

    # 3-3: Shift Score 의미 설명 다이어그램
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-4, 6)
    ax3.set_ylim(0, 0.5)
    x_range = np.linspace(-4, 6, 300)
    ax3.plot(x_range, stats.norm.pdf(x_range, 0, 1), color='#2196F3', linewidth=2, label='Ref 분포')
    ax3.plot(x_range, stats.norm.pdf(x_range, 1.5, 1), color='#FF5722', linewidth=2, label='Compare 분포')
    ax3.fill_between(x_range, stats.norm.pdf(x_range, 0, 1), alpha=0.2, color='#2196F3')
    ax3.fill_between(x_range, stats.norm.pdf(x_range, 1.5, 1), alpha=0.2, color='#FF5722')
    ax3.annotate('', xy=(1.5, 0.42), xytext=(0, 0.42),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax3.text(0.75, 0.44, 'Z-shift = 1.5σ', ha='center', fontweight='bold', color='red')
    ax3.set_title('Shift Score 개념도', fontweight='bold')
    ax3.legend()

    # --- Tail Score ---
    # 3-4: Feature별 tail rate 분포
    ax4 = fig.add_subplot(gs[1, 0])
    tail_vals = tail_result["tail_rate_all"].sort_values(ascending=False)
    top_n_show = min(80, len(tail_vals))
    colors = ['#E53935' if v > 0.03 else '#FF9800' if v > 0.01 else '#BDBDBD'
              for v in tail_vals.values[:top_n_show]]
    ax4.bar(range(top_n_show), tail_vals.values[:top_n_show], color=colors, width=1.0)
    ax4.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='기대치 (1%)')
    ax4.axhline(y=0.03, color='orange', linestyle='--', linewidth=1, label='경고 (3%)')
    ax4.axhline(y=0.10, color='red', linestyle='--', linewidth=1, label='위험 (10%)')
    ax4.set_title('Tail Score: Feature별 Tail Rate', fontweight='bold')
    ax4.set_ylabel('Tail Rate')
    ax4.set_xlabel('Features (Top 80)')
    ax4.legend(fontsize=8)

    # 3-5: Top tail features
    ax5 = fig.add_subplot(gs[1, 1])
    elevated = tail_result["elevated_features"].head(15)
    if len(elevated) > 0:
        colors = ['#E53935' if v > 0.10 else '#FF9800' if v > 0.05 else '#FFC107'
                  for v in elevated.values]
        ax5.barh(range(len(elevated)), elevated.values * 100, color=colors)
        ax5.set_yticks(range(len(elevated)))
        ax5.set_yticklabels(elevated.index, fontsize=8)
        ax5.axvline(x=1, color='gray', linestyle='--', label='기대치 (1%)')
        ax5.axvline(x=3, color='orange', linestyle='--', label='경고 (3%)')
    ax5.set_title(f'Tail Score: Elevated Features\n(Max = {tail_result["score_max"]:.1%}, Count = {tail_result["score_count"]})',
                  fontweight='bold')
    ax5.set_xlabel('Tail Rate (%)')
    ax5.legend(fontsize=8)

    # 3-6: Tail 개념도
    ax6 = fig.add_subplot(gs[1, 2])
    x_range = np.linspace(-4, 6, 300)
    ax6.plot(x_range, stats.norm.pdf(x_range, 0, 1), color='#2196F3', linewidth=2, label='Ref 분포')
    threshold_x = stats.norm.ppf(0.99) # 99th percentile
    ax6.axvline(x=threshold_x, color='red', linewidth=2, linestyle='--', label='99th percentile')
    ax6.fill_between(x_range[x_range > threshold_x],
                     stats.norm.pdf(x_range[x_range > threshold_x], 0, 1),
                     alpha=0.4, color='red', label='Ref tail (~1%)')
    # Compare with heavier tail
    comp_dist = 0.9 * stats.norm.pdf(x_range, 0, 1) + 0.1 * stats.norm.pdf(x_range, 3, 0.8)
    ax6.plot(x_range, comp_dist, color='#FF5722', linewidth=2, label='Compare (heavy tail)')
    ax6.fill_between(x_range[x_range > threshold_x],
                     comp_dist[x_range > threshold_x],
                     alpha=0.3, color='#FF5722')
    ax6.set_title('Tail Score 개념도', fontweight='bold')
    ax6.legend(fontsize=8)

    # --- Outlier Wafer Score ---
    # 3-7: Wafer별 초과 feature 비율
    ax7 = fig.add_subplot(gs[2, 0])
    exceed = outlier_result["exceed_ratio_per_wafer"].sort_values(ascending=False)
    colors = ['#E53935' if v > 0.05 else '#FF9800' if v > 0.03 else '#4CAF50'
              for v in exceed.values]
    ax7.bar(range(len(exceed)), exceed.values * 100, color=colors, width=1.0)
    ax7.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Outlier 기준 (5%)')
    ax7.set_title('Outlier Wafer Score: Wafer별 초과 비율', fontweight='bold')
    ax7.set_xlabel('Compare Wafers (sorted)')
    ax7.set_ylabel('Feature Exceed Ratio (%)')
    ax7.legend()

    # 3-8: Outlier wafer의 공통 Feature
    ax8 = fig.add_subplot(gs[2, 1])
    common = outlier_result["common_features"].head(15)
    if len(common) > 0:
        ax8.barh(range(len(common)), common.values * 100, color='#7B1FA2', alpha=0.7)
        ax8.set_yticks(range(len(common)))
        ax8.set_yticklabels(common.index, fontsize=8)
    ax8.set_title(f'Outlier Wafer 공통 이상 Feature\n(Outlier Rate = {outlier_result["score"]:.1%})',
                  fontweight='bold')
    ax8.set_xlabel('Outlier Wafer 중 초과 비율 (%)')

    # 3-9: Score 요약 게이지
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_xlim(0, 10)
    ax9.set_ylim(0, 10)
    ax9.axis('off')
    ax9.set_title('3종 Score 요약', fontweight='bold')

    scores_info = [
        ('Shift Score', shift_result["score"], [0.5, 1.0, 2.0], 7.5),
        ('Tail Score (max)', tail_result["score_max"], [0.03, 0.05, 0.10], 5.0),
        ('Outlier Rate', outlier_result["score"], [0.03, 0.05, 0.10], 2.5),
    ]
    for name, val, thresholds, y in scores_info:
        if val > thresholds[2]:
            color = '#E53935'
            level = 'HIGH_RISK'
        elif val > thresholds[1]:
            color = '#FF9800'
            level = 'RISK'
        elif val > thresholds[0]:
            color = '#FFC107'
            level = 'CAUTION'
        else:
            color = '#4CAF50'
            level = 'SAFE'

        box = FancyBboxPatch((0.5, y-0.6), 9, 1.2, boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax9.add_patch(box)
        ax9.text(1.0, y+0.1, f'{name}: {val:.4f}', fontsize=12, fontweight='bold')
        ax9.text(8, y+0.1, level, fontsize=11, fontweight='bold', color=color, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '03_score_calculation.png')


# ============================================================
# 4. Feature Importance 시각화
# ============================================================
def visualize_feature_importance(importance_result, shift_result, tail_result):
    """Feature Importance 상세 시각화"""
    print("[4/8] Visualizing feature importance...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Step 4: Feature Importance 분석', fontsize=16, fontweight='bold', y=0.98)

    # 4-1: Shift 원인 Feature Top-20
    ax = axes[0, 0]
    shift_imp = importance_result["shift_features"].head(20)
    colors = ['#E53935' if d == '악화' else '#1E88E5' for d in shift_imp['direction']]
    y_pos = range(len(shift_imp) - 1, -1, -1)
    ax.barh(y_pos, shift_imp['abs_z_shift'].values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(shift_imp['feature'].values, fontsize=8)
    ax.set_title('Shift 원인 Feature Top-20', fontweight='bold')
    ax.set_xlabel('|Z-shift|')
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#E53935', label='악화 ▲'),
                       Patch(color='#1E88E5', label='개선 ▼')],
              loc='lower right')

    # 4-2: Tail 원인 Feature
    ax = axes[0, 1]
    tail_imp = importance_result["tail_features"].head(20)
    if len(tail_imp) > 0:
        y_pos = range(len(tail_imp) - 1, -1, -1)
        ax.barh(y_pos, tail_imp['tail_rate_pct'].values, color='#FF9800')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(tail_imp['feature'].values, fontsize=8)
        ax.axvline(x=1, color='gray', linestyle='--', label='기대치 (1%)')
        ax.axvline(x=3, color='orange', linestyle='--', label='경고 (3%)')
        ax.legend()
    ax.set_title('Tail 원인 Feature (Tail Rate %)', fontweight='bold')
    ax.set_xlabel('Tail Rate (%)')

    # 4-3: Shift vs Tail 원인 비교 (벤다이어그램 대체 - scatter)
    ax = axes[1, 0]
    z_all = shift_result["z_shift_all"].abs()
    t_all = tail_result["tail_rate_all"]
    merged = pd.DataFrame({'z_shift': z_all, 'tail_rate': t_all}).dropna()
    colors_scatter = []
    for _, row in merged.iterrows():
        if row['z_shift'] > 0.5 and row['tail_rate'] > 0.03:
            colors_scatter.append('#E53935')
        elif row['z_shift'] > 0.5:
            colors_scatter.append('#FF9800')
        elif row['tail_rate'] > 0.03:
            colors_scatter.append('#7B1FA2')
        else:
            colors_scatter.append('#BDBDBD')
    ax.scatter(merged['z_shift'], merged['tail_rate'] * 100,
               c=colors_scatter, alpha=0.5, s=15, edgecolors='none')
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('|Z-shift|')
    ax.set_ylabel('Tail Rate (%)')
    ax.set_title('Shift vs Tail: Feature 분류', fontweight='bold')
    ax.legend(handles=[
        Patch(color='#FF9800', label='Shift만 (중심 이동)'),
        Patch(color='#7B1FA2', label='Tail만 (간헐 극단)'),
        Patch(color='#E53935', label='Shift + Tail'),
        Patch(color='#BDBDBD', label='정상'),
    ], fontsize=8)

    # 4-4: Feature Importance Heatmap (Top features by all 3 scores)
    ax = axes[1, 1]
    top_shift_feats = importance_result["shift_features"]["feature"].head(10).tolist()
    top_tail_feats = importance_result["tail_features"]["feature"].head(10).tolist()
    top_outlier_feats = importance_result["outlier_features"]["feature"].head(10).tolist()

    all_top = list(dict.fromkeys(top_shift_feats + top_tail_feats + top_outlier_feats))[:20]

    heatmap_data = pd.DataFrame(index=all_top)
    heatmap_data['Shift (|Z|)'] = [z_all.get(f, 0) for f in all_top]
    heatmap_data['Tail Rate (%)'] = [t_all.get(f, 0) * 100 for f in all_top]

    outlier_feats_dict = importance_result["outlier_features"].set_index("feature")["outlier_exceed_rate"]
    heatmap_data['Outlier (%)'] = [outlier_feats_dict.get(f, 0) * 100 for f in all_top]

    # Normalize for heatmap
    heatmap_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))
    sns.heatmap(heatmap_norm, ax=ax, cmap='YlOrRd', annot=heatmap_data.round(2).values,
                fmt='', linewidths=0.5, cbar_kws={'label': 'Normalized Score'})
    ax.set_title('Multi-Score Feature Importance Heatmap', fontweight='bold')
    ax.set_ylabel('')

    plt.tight_layout()
    save_fig(fig, '04_feature_importance.png')


# ============================================================
# 5. 최종 결과 리포트 시각화
# ============================================================
def visualize_final_report(result):
    """최종 판정 결과 대시보드"""
    print("[5/8] Visualizing final report dashboard...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)
    fig.suptitle('ECO Change Detection Report — 최종 결과', fontsize=18, fontweight='bold', y=0.99)

    decision = result['decision']
    meta = result['metadata']

    # Color mapping
    decision_colors = {
        'SAFE': '#4CAF50', 'CAUTION': '#FFC107',
        'RISK': '#FF9800', 'HIGH_RISK': '#E53935',
        'INSUFFICIENT_DATA': '#9E9E9E'
    }
    dec_color = decision_colors.get(decision['decision'], '#9E9E9E')

    # 5-1: 메타정보 + 판정 결과 (큰 영역)
    ax_header = fig.add_subplot(gs[0, :2])
    ax_header.set_xlim(0, 10)
    ax_header.set_ylim(0, 10)
    ax_header.axis('off')

    box = FancyBboxPatch((0.2, 0.2), 9.5, 9.5, boxstyle="round,pad=0.3",
                         facecolor=dec_color, alpha=0.1, edgecolor=dec_color, linewidth=3)
    ax_header.add_patch(box)

    ax_header.text(5, 8.5, f'판정: {decision["decision"]}',
                   ha='center', fontsize=28, fontweight='bold', color=dec_color)
    ax_header.text(5, 6.5, f'Step: {meta["step_id"]}  |  Code: {meta["change_code"]}',
                   ha='center', fontsize=14)
    ax_header.text(5, 5, f'Ref: {meta["ref_count"]:,} wafers  |  Compare: {meta["comp_count"]} wafers  |  Features: {meta["feature_count"]:,}',
                   ha='center', fontsize=12, color='#555')

    for i, reason in enumerate(decision['reasons'][:3]):
        ax_header.text(1, 3.5 - i * 1.2, f'• {reason}', fontsize=10, color='#333')

    # 5-2: Score 게이지 차트
    ax_gauge = fig.add_subplot(gs[0, 2:])
    ax_gauge.set_xlim(0, 10)
    ax_gauge.set_ylim(0, 10)
    ax_gauge.axis('off')
    ax_gauge.set_title('Score Summary', fontweight='bold', fontsize=14)

    scores_data = [
        ('Shift Score', result['scores']['shift_score'], 2.0, 8),
        ('Tail Score', result['scores']['tail_score_max'], 0.10, 5.5),
        ('Outlier Rate', result['scores']['outlier_wafer_rate'], 0.10, 3),
    ]
    for name, val, max_val, y in scores_data:
        ratio = min(val / max_val, 1.0)
        bar_color = '#E53935' if ratio > 0.8 else '#FF9800' if ratio > 0.5 else '#FFC107' if ratio > 0.3 else '#4CAF50'

        ax_gauge.text(0.5, y + 0.3, f'{name}: {val:.4f}', fontsize=12, fontweight='bold')
        # Background bar
        ax_gauge.add_patch(FancyBboxPatch((0.5, y - 0.5), 9, 0.5,
                                          boxstyle="round,pad=0.05", facecolor='#E0E0E0'))
        # Value bar
        bar_width = max(0.1, ratio * 9)
        ax_gauge.add_patch(FancyBboxPatch((0.5, y - 0.5), bar_width, 0.5,
                                          boxstyle="round,pad=0.05", facecolor=bar_color, alpha=0.8))

    # 5-3: Shift Feature Top-10 (bar)
    ax_shift = fig.add_subplot(gs[1, :2])
    shift_feats = result['importance']['shift_features'].head(10)
    colors = ['#E53935' if d == '악화' else '#1E88E5' for d in shift_feats['direction']]
    y_pos = range(len(shift_feats) - 1, -1, -1)
    ax_shift.barh(y_pos, shift_feats['z_shift'].values, color=colors, edgecolor='white')
    ax_shift.set_yticks(list(y_pos))
    ax_shift.set_yticklabels([f"#{i+1} {f}" for i, f in enumerate(shift_feats['feature'])], fontsize=9)
    ax_shift.axvline(x=0, color='black', linewidth=0.5)
    ax_shift.set_title('Shift 원인 Feature Top-10 (Z-shift 값)', fontweight='bold')
    ax_shift.set_xlabel('Z-shift (σ) — 양수: 악화, 음수: 개선')

    # 5-4: Tail Feature Top-10
    ax_tail = fig.add_subplot(gs[1, 2:])
    tail_feats = result['importance']['tail_features'].head(10)
    if len(tail_feats) > 0:
        y_pos = range(len(tail_feats) - 1, -1, -1)
        ax_tail.barh(y_pos, tail_feats['tail_rate_pct'].values, color='#FF9800',
                     edgecolor='white')
        ax_tail.set_yticks(list(y_pos))
        ax_tail.set_yticklabels([f"#{i+1} {f}" for i, f in enumerate(tail_feats['feature'])], fontsize=9)
        ax_tail.axvline(x=1, color='gray', linestyle='--', label='기대치 (1%)')
        ax_tail.legend()
    ax_tail.set_title('Tail 원인 Feature Top-10 (Tail Rate %)', fontweight='bold')
    ax_tail.set_xlabel('Tail Rate (%)')

    # 5-5: Outlier wafer 상세
    ax_outlier = fig.add_subplot(gs[2, :2])
    exceed = result['detail']['outlier']['exceed_ratio_per_wafer'].sort_values(ascending=False)
    colors = ['#E53935' if v > 0.05 else '#4CAF50' for v in exceed.values]
    ax_outlier.bar(range(len(exceed)), exceed.values * 100, color=colors, width=1.0)
    ax_outlier.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Outlier 기준 (5%)')
    ax_outlier.set_title(f'Outlier Wafer 분포 (총 {result["detail"]["outlier"]["outlier_count"]}개)',
                         fontweight='bold')
    ax_outlier.set_xlabel('Compare Wafers')
    ax_outlier.set_ylabel('Feature Exceed Ratio (%)')
    ax_outlier.legend()

    # 5-6: 판정 근거 + 액션 제안
    ax_action = fig.add_subplot(gs[2, 2:])
    ax_action.set_xlim(0, 10)
    ax_action.set_ylim(0, 10)
    ax_action.axis('off')
    ax_action.set_title('권장 액션', fontweight='bold', fontsize=14)

    actions = {
        'SAFE': ['변경점 적용 승인 가능', '정기 모니터링 유지', 'Score 기록 보관'],
        'CAUTION': ['추가 wafer 확보 후 재평가 권장', '원인 Feature 엔지니어 확인 필요', 'Tail 패턴 주시'],
        'RISK': ['변경점 적용 보류 권장', '원인 Feature 긴급 분석 필요', 'Outlier wafer 상세 조사'],
        'HIGH_RISK': ['변경점 적용 중단 권장', '즉시 엔지니어링 검토', '공정 조건 원복 고려'],
    }
    action_list = actions.get(decision['decision'], ['평가 불가'])
    for i, action in enumerate(action_list):
        ax_action.text(0.5, 8 - i * 2, f'▶ {action}', fontsize=12, color='#333')

    ax_action.text(0.5, 1.5, f'Outlier Wafer IDs:', fontsize=10, fontweight='bold', color='#555')
    outlier_ids = result['detail']['outlier']['outlier_wafer_ids'][:10]
    ax_action.text(0.5, 0.5, ', '.join(outlier_ids) if outlier_ids else '(없음)',
                   fontsize=9, color='#777')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, '05_final_report.png')


# ============================================================
# 6. 검증 실험: Sensitivity Analysis
# ============================================================
def visualize_sensitivity_analysis(df_ref, df_comp):
    """Score 민감도 검증 시각화"""
    print("[6/8] Running sensitivity analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 5: Score 민감도 검증 (Sensitivity Analysis)', fontsize=16, fontweight='bold', y=0.98)

    # 6-1: False Alarm Test - ref를 반으로 나누어 비교
    ax = axes[0, 0]
    np.random.seed(123)
    n_ref = len(df_ref)
    idx = np.random.permutation(n_ref)
    ref_a = df_ref.iloc[idx[:n_ref//2]]
    ref_b = df_ref.iloc[idx[n_ref//2:]]

    false_result = run_eco_change_detection(ref_a, ref_b, step_id="FALSE_ALARM_TEST")
    real_result = run_eco_change_detection(df_ref, df_comp, step_id="REAL_TEST")

    labels = ['Shift Score', 'Tail Score\n(max)', 'Outlier Rate']
    false_vals = [false_result['scores']['shift_score'],
                  false_result['scores']['tail_score_max'],
                  false_result['scores']['outlier_wafer_rate']]
    real_vals = [real_result['scores']['shift_score'],
                 real_result['scores']['tail_score_max'],
                 real_result['scores']['outlier_wafer_rate']]

    x = np.arange(len(labels))
    ax.bar(x - 0.2, false_vals, 0.35, label='False Alarm (Ref vs Ref)', color='#4CAF50', alpha=0.8)
    ax.bar(x + 0.2, real_vals, 0.35, label='Real (Ref vs Compare)', color='#E53935', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score Value')
    ax.set_title('False Alarm Test\n(Ref를 반으로 나누어 비교)', fontweight='bold')
    ax.legend()
    for i, (fv, rv) in enumerate(zip(false_vals, real_vals)):
        ax.text(i - 0.2, fv + 0.01, f'{fv:.3f}', ha='center', fontsize=8)
        ax.text(i + 0.2, rv + 0.01, f'{rv:.3f}', ha='center', fontsize=8)

    # 6-2: Shift 크기 vs Score (단조 증가 검증)
    ax = axes[0, 1]
    shift_sizes = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    shift_scores = []
    for s in shift_sizes:
        np.random.seed(42)
        n_f = 5000
        ref_d = np.random.randn(1000, n_f) * 0.5 + 3.0
        comp_d = np.random.randn(80, n_f) * 0.5 + 3.0
        comp_d[:, 100:121] += s
        df_r = pd.DataFrame(ref_d, columns=[f"F_{i}" for i in range(n_f)])
        df_c = pd.DataFrame(comp_d, columns=[f"F_{i}" for i in range(n_f)])
        r = run_eco_change_detection(df_r, df_c)
        shift_scores.append(r['scores']['shift_score'])

    ax.plot(shift_sizes, shift_scores, 'o-', color='#E53935', linewidth=2, markersize=8)
    ax.set_xlabel('Shift 크기 (σ 단위)')
    ax.set_ylabel('Shift Score')
    ax.set_title('Shift 크기 vs Score\n(단조 증가 검증)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6-3: Sample Size vs Score 변동성
    ax = axes[1, 0]
    sample_sizes = [20, 30, 50, 80, 100, 150, 200, 300]
    results_per_size = {s: [] for s in sample_sizes}
    for trial in range(10):
        for ss in sample_sizes:
            np.random.seed(42 + trial)
            n_f = 5000
            ref_d = np.random.randn(1000, n_f) * 0.5 + 3.0
            comp_d = np.random.randn(ss, n_f) * 0.5 + 3.0
            comp_d[:, 100:121] += 0.75
            spike_w = np.random.choice(ss, size=max(1, int(ss * 0.10)), replace=False)
            comp_d[np.ix_(spike_w, list(range(500, 511)))] += 3.0
            df_r = pd.DataFrame(ref_d, columns=[f"F_{i}" for i in range(n_f)])
            df_c = pd.DataFrame(comp_d, columns=[f"F_{i}" for i in range(n_f)])
            r = run_eco_change_detection(df_r, df_c)
            results_per_size[ss].append(r['scores']['shift_score'])

    means = [np.mean(results_per_size[s]) for s in sample_sizes]
    stds = [np.std(results_per_size[s]) for s in sample_sizes]
    ax.errorbar(sample_sizes, means, yerr=stds, fmt='o-', color='#1E88E5',
                capsize=5, linewidth=2, markersize=8)
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='최소 기준 (30)')
    ax.set_xlabel('Compare Wafer 수')
    ax.set_ylabel('Shift Score (mean ± std)')
    ax.set_title('Sample Size vs Score 변동성\n(n=10 trials)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6-4: 검증 체크리스트 결과
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('검증 체크리스트 결과', fontweight='bold', fontsize=14)

    # Check results
    shift_top = real_result['importance']['shift_features']['feature'].head(20).tolist()
    shift_check = any('EDS_01' in f for f in shift_top)

    tail_top = real_result['importance']['tail_features']['feature'].head(20).tolist()
    tail_check = any('EDS_05' in f for f in tail_top)

    outlier_ids = real_result['detail']['outlier']['outlier_wafer_ids']
    outlier_check = any('comp_w007' in w for w in outlier_ids)

    false_alarm_check = (false_result['scores']['shift_score'] < 0.3 and
                         false_result['scores']['tail_score_max'] < 0.03)

    monotone_check = all(shift_scores[i] <= shift_scores[i+1] for i in range(len(shift_scores)-1))

    checks = [
        (shift_check, 'Pattern A: Shift 원인에 EDS_0100~0120 포함'),
        (tail_check, 'Pattern B: Tail 원인에 EDS_0500~0510 포함'),
        (outlier_check, 'Pattern C: Outlier에 comp_w0070~0074 포함'),
        (false_alarm_check, 'False Alarm: Ref vs Ref Score ≈ 0'),
        (monotone_check, 'Shift 크기 증가 → Score 단조 증가'),
    ]

    for i, (passed, desc) in enumerate(checks):
        icon = '✓' if passed else '✗'
        color = '#4CAF50' if passed else '#E53935'
        ax.text(0.5, 8.5 - i * 1.5, f'{icon}  {desc}', fontsize=11,
                color=color, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, '06_sensitivity_analysis.png')


# ============================================================
# 7. 추가 시각화: Box plot + Violin + Correlation
# ============================================================
def visualize_additional_insights(scaled_ref, scaled_comp, shift_result, tail_result):
    """추가 인사이트 시각화"""
    print("[7/8] Generating additional insight visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('추가 인사이트: 다양한 시각화 기법', fontsize=16, fontweight='bold', y=0.98)

    # 7-1: Violin plot - Top shift features
    ax = axes[0, 0]
    top_feats = shift_result["z_shift_all"].abs().sort_values(ascending=False).head(5).index.tolist()
    plot_data = []
    for f in top_feats:
        for val in scaled_ref[f].values[:200]:  # sample for speed
            plot_data.append({'Feature': f, 'Value': val, 'Group': 'Ref'})
        for val in scaled_comp[f].values:
            plot_data.append({'Feature': f, 'Value': val, 'Group': 'Compare'})
    plot_df = pd.DataFrame(plot_data)
    sns.violinplot(data=plot_df, x='Feature', y='Value', hue='Group',
                   split=True, ax=ax, palette={'Ref': '#2196F3', 'Compare': '#FF5722'},
                   inner='quartile', density_norm='width')
    ax.set_title('Top-5 Shift Feature: Violin Plot', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    # 7-2: 2D density plot (PCA-like) using top 2 features
    ax = axes[0, 1]
    if len(top_feats) >= 2:
        f1, f2 = top_feats[0], top_feats[1]
        ax.scatter(scaled_ref[f1].values[:300], scaled_ref[f2].values[:300],
                   alpha=0.3, s=10, c='#2196F3', label='Ref')
        ax.scatter(scaled_comp[f1].values, scaled_comp[f2].values,
                   alpha=0.7, s=30, c='#FF5722', label='Compare', edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f'2D Scatter: Top Shift Features', fontweight='bold')
        ax.legend()

    # 7-3: Correlation heatmap of top features
    ax = axes[1, 0]
    top_all = list(dict.fromkeys(
        shift_result["z_shift_all"].abs().sort_values(ascending=False).head(8).index.tolist() +
        tail_result["elevated_features"].head(5).index.tolist()
    ))[:12]
    if len(top_all) > 2:
        corr_ref = scaled_ref[top_all].corr()
        mask = np.triu(np.ones_like(corr_ref, dtype=bool))
        sns.heatmap(corr_ref, mask=mask, ax=ax, cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.5,
                    annot_kws={'fontsize': 7})
        ax.set_title('Top Feature 상관관계 (Ref 기준)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    # 7-4: Cumulative distribution comparison
    ax = axes[1, 1]
    compare_feat = top_feats[0] if top_feats else scaled_ref.columns[0]
    ref_sorted = np.sort(scaled_ref[compare_feat].values)
    comp_sorted = np.sort(scaled_comp[compare_feat].values)
    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    comp_cdf = np.arange(1, len(comp_sorted) + 1) / len(comp_sorted)

    ax.plot(ref_sorted, ref_cdf, color='#2196F3', linewidth=2, label='Ref CDF')
    ax.plot(comp_sorted, comp_cdf, color='#FF5722', linewidth=2, label='Compare CDF')

    # KS statistic
    ks_stat, ks_p = stats.ks_2samp(scaled_ref[compare_feat], scaled_comp[compare_feat])
    ax.set_title(f'CDF 비교: {compare_feat}\n(KS stat = {ks_stat:.4f}, p = {ks_p:.2e})',
                 fontweight='bold')
    ax.set_xlabel('Scaled Value')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '07_additional_insights.png')


# ============================================================
# 8. Pipeline 흐름도
# ============================================================
def visualize_pipeline_flow():
    """파이프라인 전체 흐름도"""
    print("[8/8] Creating pipeline flow diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('ECO Change Detection Pipeline 흐름도', fontsize=18, fontweight='bold')

    # Steps
    steps = [
        (2, 8.5, 3, 1.2, '입력 데이터', 'Ref wafers + Compare wafers\n(Feature matrix)', '#E3F2FD', '#1565C0'),
        (2, 6.5, 3, 1.2, 'Step 1: 전처리', '결측/상수 제거\nRobust Scaling\nWinsorizing', '#E8F5E9', '#2E7D32'),
        (2, 4.0, 3, 1.2, 'Step 2: Score 산출', '3종 병렬 계산\nShift / Tail / Outlier', '#FFF3E0', '#E65100'),
        (2, 1.5, 3, 1.2, 'Step 3: Feature\nImportance', 'Score별 원인 Feature\nTop-N 추적', '#F3E5F5', '#6A1B9A'),
        (11, 4.0, 3, 1.2, 'Step 4: 판정', 'OR 조건 판정\nSAFE/CAUTION/\nRISK/HIGH_RISK', '#FFEBEE', '#C62828'),
        (11, 1.5, 3, 1.2, '출력: 리포트', 'Score + 원인 Feature\n+ 시각화 + 판정', '#E0F7FA', '#00695C'),
    ]

    for x, y, w, h, title, desc, fcolor, ecolor in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=fcolor, edgecolor=ecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='top',
                fontsize=11, fontweight='bold', color=ecolor)
        ax.text(x + w/2, y + 0.35, desc, ha='center', va='bottom',
                fontsize=8, color='#333')

    # Score detail boxes (middle column)
    score_boxes = [
        (6.5, 6.0, 'Shift Score', '중심치/산포 이동\ntop_k_ratio=1%', '#FFCDD2'),
        (6.5, 4.5, 'Tail Score', '간헐적 극단값\n99th percentile', '#FFE0B2'),
        (6.5, 3.0, 'Outlier Score', 'Wafer 단위 이상\nfeature_exceed>5%', '#E1BEE7'),
    ]
    for x, y, title, desc, color in score_boxes:
        box = FancyBboxPatch((x, y), 2.8, 1.2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#666', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1.4, y + 0.9, title, ha='center', fontsize=9, fontweight='bold')
        ax.text(x + 1.4, y + 0.25, desc, ha='center', fontsize=7, color='#555')

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#555', lw=2)
    ax.annotate('', xy=(3.5, 7.7), xytext=(3.5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3.5, 5.2), xytext=(3.5, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3.5, 2.7), xytext=(3.5, 4.0), arrowprops=arrow_props)

    # Step 2 to score boxes
    ax.annotate('', xy=(6.5, 6.6), xytext=(5.0, 5.2), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 5.1), xytext=(5.0, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 3.6), xytext=(5.0, 4.2), arrowprops=arrow_props)

    # Score boxes to Step 4
    ax.annotate('', xy=(11, 5.0), xytext=(9.3, 6.6), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 4.8), xytext=(9.3, 5.1), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 4.5), xytext=(9.3, 3.6), arrowprops=arrow_props)

    # Step 4 to output
    ax.annotate('', xy=(12.5, 2.7), xytext=(12.5, 4.0), arrowprops=arrow_props)

    # Step 3 to output
    ax.annotate('', xy=(11, 2.1), xytext=(5.0, 2.1), arrowprops=arrow_props)

    save_fig(fig, '08_pipeline_flow.png')


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  ECO Change Detection PoC — Full Experiment")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[DATA] Generating synthetic data...")
    df_ref, df_comp = generate_synthetic_data()
    print(f"  Ref: {df_ref.shape}, Compare: {df_comp.shape}")

    # 2. Run full pipeline
    print("\n[PIPELINE] Running ECO Change Detection pipeline...")
    result = run_eco_change_detection(
        df_ref, df_comp,
        step_id="S310_ETCH",
        change_code="CHG_TEST_001"
    )

    # Print text report
    print("\n" + "=" * 55)
    print("  ECO Change Detection Report")
    print("=" * 55)
    meta = result['metadata']
    print(f"\n  Step: {meta['step_id']}")
    print(f"  Change Code: {meta['change_code']}")
    print(f"  Ref Wafers: {meta['ref_count']:,}  |  Compare Wafers: {meta['comp_count']}")
    print(f"  Features: {meta['feature_count']:,} (원본: {meta['feature_count_original']:,})")
    print("-" * 55)
    print("  SCORES")
    print("-" * 55)
    s = result['scores']
    print(f"  Shift Score:          {s['shift_score']:.3f}")
    print(f"  Tail Score (max):     {s['tail_score_max']:.4f}")
    print(f"  Tail Feature Count:   {s['tail_feature_count']}")
    print(f"  Outlier Wafer Rate:   {s['outlier_wafer_rate']:.4f}")
    print("-" * 55)
    d = result['decision']
    print(f"  DECISION: {d['decision']}")
    print("-" * 55)
    for r in d['reasons']:
        print(f"  • {r}")
    print("-" * 55)
    print("  SHIFT 원인 Feature:")
    shift_f = result['importance']['shift_features'].head(10)
    for _, row in shift_f.iterrows():
        arrow = "▲" if row['direction'] == '악화' else "▼"
        print(f"    {row['feature']}: z = {row['z_shift']:+.3f} {row['direction']} {arrow}")
    print("-" * 55)
    print("  TAIL 원인 Feature:")
    tail_f = result['importance']['tail_features'].head(10)
    for _, row in tail_f.iterrows():
        print(f"    {row['feature']}: tail rate = {row['tail_rate_pct']}%")
    print("-" * 55)
    outlier_ids = result['detail']['outlier']['outlier_wafer_ids']
    print(f"  Outlier Wafers ({len(outlier_ids)}): {', '.join(outlier_ids[:10])}")
    print("=" * 55)

    # 3. Generate all visualizations
    print("\n[VISUALIZATION] Generating figures...")

    # Get preprocessed data from result
    scaled_ref = result['scaled_ref']
    scaled_comp = result['scaled_comp']

    visualize_synthetic_data(df_ref, df_comp)
    visualize_preprocessing(df_ref, df_comp,
                            list(scaled_ref.columns), scaled_ref, scaled_comp)
    visualize_score_calculation(scaled_ref, scaled_comp,
                                result['detail']['shift'],
                                result['detail']['tail'],
                                result['detail']['outlier'])
    visualize_feature_importance(result['importance'],
                                 result['detail']['shift'],
                                 result['detail']['tail'])
    visualize_final_report(result)
    visualize_sensitivity_analysis(df_ref, df_comp)
    visualize_additional_insights(scaled_ref, scaled_comp,
                                  result['detail']['shift'],
                                  result['detail']['tail'])
    visualize_pipeline_flow()

    print("\n[DONE] All visualizations saved to results/ and docs/images/")
    return result


if __name__ == '__main__':
    result = main()
