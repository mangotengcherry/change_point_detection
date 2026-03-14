"""
ECO Change Detection PoC - Full Experiment Runner (Enhanced)
가상 데이터를 생성하고 파이프라인을 실행하며, 과정별 시각화를 생성합니다.

v2 시각화: 5가지 패턴, FancyBboxPatch, gridspec 멀티패널, Violin/CDF 추가
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
from matplotlib.patches import FancyBboxPatch, Patch
import seaborn as sns
from scipy import stats

from eco_change_detection import (
    generate_synthetic_data, filter_features, robust_scale, winsorize,
    calc_shift_score, calc_tail_score, calc_outlier_wafer_score,
    get_feature_importance, make_decision, run_eco_change_detection
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(RESULTS_DIR, name), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(DOCS_DIR, name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")


# ============================================================
# 0. Pipeline 흐름도
# ============================================================
def visualize_pipeline_flow():
    print("  Creating pipeline flow diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis('off')
    ax.set_title('ECO Change Detection Pipeline 흐름도', fontsize=18, fontweight='bold')

    steps = [
        (6, 10.2, 4, 1.0, '입력 데이터', 'Ref + Compare wafers\n(Feature matrix)', '#E3F2FD', '#1565C0'),
        (6, 8.5, 4, 1.0, 'Step 1: 전처리', '결측/상수 제거 + Robust Scaling\n+ Winsorizing', '#E8F5E9', '#2E7D32'),
        (1, 5.8, 4, 1.5, 'Shift Score', '중심치/산포 이동\nTop-K z-shift 평균', '#FFF3E0', '#E65100'),
        (6, 5.8, 4, 1.5, 'Tail Score', '간헐적 극단값\n99th percentile 초과', '#E8EAF6', '#283593'),
        (11, 5.8, 4, 1.5, 'Outlier Wafer', 'Wafer별 다변량\nFeature 동시 초과', '#F3E5F5', '#6A1B9A'),
        (3.5, 3.5, 4, 1.0, 'Feature Importance', 'Score별 원인\nFeature Top-N 추적', '#FFF3E0', '#E65100'),
        (8.5, 3.5, 4, 1.0, '판정 (Decision)', 'SAFE / CAUTION\nRISK / HIGH_RISK', '#FFEBEE', '#C62828'),
        (6, 1.5, 4, 1.0, '출력: 리포트', 'Score + Feature +\n시각화 + 판정 요약', '#E0F7FA', '#00695C'),
    ]
    for x, y, w, h, title, desc, fcolor, ecolor in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=fcolor, edgecolor=ecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.15, title, ha='center', va='top', fontsize=10, fontweight='bold', color=ecolor)
        ax.text(x + w/2, y + 0.2, desc, ha='center', va='bottom', fontsize=8, color='#333')

    arrow_props = dict(arrowstyle='->', color='#555', lw=1.5)
    ax.annotate('', xy=(8, 9.5), xytext=(8, 10.2), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 7.3), xytext=(8, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 7.3), xytext=(8, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 7.3), xytext=(8, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 4.5), xytext=(3, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 4.5), xytext=(8, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(10.5, 4.5), xytext=(8, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(10.5, 4.5), xytext=(13, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 2.5), xytext=(5.5, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 2.5), xytext=(10.5, 3.5), arrowprops=arrow_props)

    ax.text(0.5, 6.5, 'Step 2\n3종 병렬', fontsize=11, fontweight='bold', color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#999'))

    save_fig(fig, '00_pipeline_flow.png')


# ============================================================
# 1. 합성 데이터 구조 시각화
# ============================================================
def visualize_synthetic_data(df_ref, df_comp, ground_truth):
    print("\n[1/9] Visualizing synthetic data structure...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Step 1: 합성 데이터 구조 분석 (5가지 패턴)', fontsize=16, fontweight='bold', y=0.98)

    # 1-1: Ref vs Comp 크기 비교
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

    # 1-2: 정상 feature 분포
    ax = axes[0, 1]
    normal_feat = 'EDS_0050'
    ax.hist(df_ref[normal_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[normal_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.set_title(f'정상 Feature 분포 ({normal_feat})', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    # 1-3: Pattern A - Systematic Shift
    ax = axes[0, 2]
    shift_feat = 'EDS_0110'
    ax.hist(df_ref[shift_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[shift_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.axvline(df_ref[shift_feat].mean(), color='#1565C0', linestyle='--', linewidth=2, label='Ref mean')
    ax.axvline(df_comp[shift_feat].mean(), color='#D32F2F', linestyle='--', linewidth=2, label='Comp mean')
    ax.set_title(f'Pattern A: Systematic Shift ({shift_feat})', fontweight='bold')
    ax.legend(fontsize=8)

    # 1-4: Pattern B - Intermittent Spike
    ax = axes[1, 0]
    spike_feat = 'EDS_0505'
    ax.hist(df_ref[spike_feat], bins=40, alpha=0.6, label='Ref', color='#2196F3', density=True)
    ax.hist(df_comp[spike_feat], bins=20, alpha=0.6, label='Compare', color='#FF5722', density=True)
    ax.set_title(f'Pattern B: Intermittent Spike ({spike_feat})', fontweight='bold')
    ax.legend()

    # 1-5: Pattern D - Gradual Trend
    ax = axes[1, 1]
    trend_feat = 'EDS_0605'
    ax.scatter(range(len(df_comp)), df_comp[trend_feat], s=15, alpha=0.7, c='#FF5722', label='Compare')
    ref_sample = df_ref[trend_feat].sample(80, random_state=42).values
    ax.scatter(range(len(ref_sample)), ref_sample, s=10, alpha=0.3, c='#2196F3', label='Ref (sample)')
    z = np.polyfit(range(len(df_comp)), df_comp[trend_feat].values, 1)
    p = np.poly1d(z)
    ax.plot(range(len(df_comp)), p(range(len(df_comp))), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    ax.set_title(f'Pattern D: Gradual Trend ({trend_feat})', fontweight='bold')
    ax.set_xlabel('Wafer Sequence')
    ax.legend(fontsize=8)

    # 1-6: 5가지 패턴 요약
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('삽입된 불량 패턴 요약', fontweight='bold')

    patterns = [
        ("A", "Systematic Shift", "F100~120, +1.5s, Easy", '#E53935', 8.5),
        ("B", "Intermittent Spike", "F500~510, 10% wafer, Easy", '#FF9800', 6.8),
        ("C", "Multi-Feature Outlier", "F200~499, W70~74, Easy", '#7B1FA2', 5.1),
        ("D", "Gradual Trend", "F600~610, 점진적, Medium", '#1565C0', 3.4),
        ("E", "Subtle Shift", "F700~720, +0.5s, Hard", '#4CAF50', 1.7),
    ]
    for name, desc, detail, color, y in patterns:
        box = FancyBboxPatch((0.3, y-0.55), 9.2, 1.0, boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.12, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(1.0, y+0.05, f'Pattern {name}: {desc}', fontsize=10, fontweight='bold', color=color)
        ax.text(1.0, y-0.35, detail, fontsize=8, color='#333')

    plt.tight_layout()
    save_fig(fig, '01_synthetic_data_structure.png')


# ============================================================
# 2. 전처리 과정 시각화
# ============================================================
def visualize_preprocessing(df_ref, df_comp, features, scaled_ref, scaled_comp):
    print("[2/9] Visualizing preprocessing steps...")

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
    ax.set_title(f'Feature 필터링 결과\n(총 {total_feats} -> {filtered_feats})', fontweight='bold')

    # 2-2: Scaling 전 분포
    ax = axes[0, 1]
    sample_feats = ['EDS_0050', 'EDS_0110', 'EDS_0505', 'EDS_0605']
    for f in sample_feats:
        if f in df_ref.columns:
            ax.hist(df_ref[f], bins=30, alpha=0.4, label=f, density=True)
    ax.set_title('Scaling 전: 원본 Feature 분포', fontweight='bold')
    ax.set_xlabel('Raw Value')
    ax.legend(fontsize=8)

    # 2-3: Scaling 후 분포
    ax = axes[0, 2]
    for f in sample_feats:
        if f in scaled_ref.columns:
            ax.hist(scaled_ref[f], bins=30, alpha=0.4, label=f, density=True)
    ax.set_title('Scaling 후: Robust-Scaled 분포', fontweight='bold')
    ax.set_xlabel('Scaled Value')
    ax.legend(fontsize=8)

    # 2-4: Scaling 검증
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

    # 2-5: Shift feature scaled 비교
    ax = axes[1, 1]
    shift_feat = 'EDS_0110'
    if shift_feat in scaled_ref.columns:
        ax.hist(scaled_ref[shift_feat], bins=40, alpha=0.6, label='Ref (scaled)', color='#2196F3', density=True)
        ax.hist(scaled_comp[shift_feat], bins=20, alpha=0.6, label='Compare (scaled)', color='#FF5722', density=True)
        ax.set_title(f'Scaled 비교: {shift_feat} (Shift 패턴)', fontweight='bold')
        ax.legend()

    # 2-6: Winsorizing 효과
    ax = axes[1, 2]
    feat = 'EDS_0505'
    if feat in scaled_ref.columns:
        raw_vals = (df_comp[feat] - df_ref[feat].median()) / max(
            df_ref[feat].quantile(0.75) - df_ref[feat].quantile(0.25), 1e-10)
        ax.hist(raw_vals, bins=20, alpha=0.5, label='Before Winsorize', color='#FF9800', density=True)
        ax.hist(scaled_comp[feat], bins=20, alpha=0.5, label='After Winsorize', color='#4CAF50', density=True)
        ax.set_title(f'Winsorizing 효과: {feat}', fontweight='bold')
        ax.legend()

    plt.tight_layout()
    save_fig(fig, '02_preprocessing_steps.png')


# ============================================================
# 3. Score 산출 과정 시각화
# ============================================================
def visualize_score_calculation(scaled_ref, scaled_comp, shift_result, tail_result, outlier_result):
    print("[3/9] Visualizing score calculation process...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Step 3: Score 산출 과정 (3종 병렬)', fontsize=16, fontweight='bold', y=0.98)

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
    ax1.set_ylabel('Z-shift (s)')

    # 3-2: Top-K shift features
    ax2 = fig.add_subplot(gs[0, 1])
    top_shift = shift_result["z_shift_all"].abs().sort_values(ascending=True).tail(15)
    colors = ['#E53935' if shift_result["z_shift_all"][f] > 0 else '#1E88E5' for f in top_shift.index]
    ax2.barh(range(len(top_shift)), top_shift.values, color=colors)
    ax2.set_yticks(range(len(top_shift)))
    ax2.set_yticklabels(top_shift.index, fontsize=8)
    ax2.set_title(f'Shift Score Top-15 Features\n(Score = {shift_result["score"]:.3f})', fontweight='bold')
    ax2.set_xlabel('|Z-shift|')

    # 3-3: Shift Score 개념도
    ax3 = fig.add_subplot(gs[0, 2])
    x_range = np.linspace(-4, 6, 300)
    ax3.plot(x_range, stats.norm.pdf(x_range, 0, 1), color='#2196F3', linewidth=2, label='Ref 분포')
    ax3.plot(x_range, stats.norm.pdf(x_range, 1.5, 1), color='#FF5722', linewidth=2, label='Compare 분포')
    ax3.fill_between(x_range, stats.norm.pdf(x_range, 0, 1), alpha=0.2, color='#2196F3')
    ax3.fill_between(x_range, stats.norm.pdf(x_range, 1.5, 1), alpha=0.2, color='#FF5722')
    ax3.annotate('', xy=(1.5, 0.42), xytext=(0, 0.42), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax3.text(0.75, 0.44, 'Z-shift = 1.5s', ha='center', fontweight='bold', color='red')
    ax3.set_title('Shift Score 개념도', fontweight='bold')
    ax3.legend()

    # 3-4: Tail rate 분포
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
    ax4.legend(fontsize=8)

    # 3-5: Top tail features
    ax5 = fig.add_subplot(gs[1, 1])
    elevated = tail_result["elevated_features"].head(15)
    if len(elevated) > 0:
        colors_e = ['#E53935' if v > 0.10 else '#FF9800' if v > 0.05 else '#FFC107' for v in elevated.values]
        ax5.barh(range(len(elevated)), elevated.values * 100, color=colors_e)
        ax5.set_yticks(range(len(elevated)))
        ax5.set_yticklabels(elevated.index, fontsize=8)
        ax5.axvline(x=1, color='gray', linestyle='--', label='기대치 (1%)')
        ax5.axvline(x=3, color='orange', linestyle='--', label='경고 (3%)')
    ax5.set_title(f'Tail Score: Elevated Features\n(Max={tail_result["score_max"]:.1%}, Count={tail_result["score_count"]})', fontweight='bold')
    ax5.set_xlabel('Tail Rate (%)')
    ax5.legend(fontsize=8)

    # 3-6: Tail 개념도
    ax6 = fig.add_subplot(gs[1, 2])
    x_range = np.linspace(-4, 6, 300)
    ax6.plot(x_range, stats.norm.pdf(x_range, 0, 1), color='#2196F3', linewidth=2, label='Ref 분포')
    threshold_x = stats.norm.ppf(0.99)
    ax6.axvline(x=threshold_x, color='red', linewidth=2, linestyle='--', label='99th percentile')
    ax6.fill_between(x_range[x_range > threshold_x],
                     stats.norm.pdf(x_range[x_range > threshold_x], 0, 1), alpha=0.4, color='red', label='Ref tail (~1%)')
    comp_dist = 0.9 * stats.norm.pdf(x_range, 0, 1) + 0.1 * stats.norm.pdf(x_range, 3, 0.8)
    ax6.plot(x_range, comp_dist, color='#FF5722', linewidth=2, label='Compare (heavy tail)')
    ax6.set_title('Tail Score 개념도', fontweight='bold')
    ax6.legend(fontsize=8)

    # 3-7: Wafer별 초과 비율
    ax7 = fig.add_subplot(gs[2, 0])
    exceed = outlier_result["exceed_ratio_per_wafer"].sort_values(ascending=False)
    colors = ['#E53935' if v > 0.05 else '#FF9800' if v > 0.03 else '#4CAF50' for v in exceed.values]
    ax7.bar(range(len(exceed)), exceed.values * 100, color=colors, width=1.0)
    ax7.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Outlier 기준 (5%)')
    ax7.set_title('Outlier Wafer Score: Wafer별 초과 비율', fontweight='bold')
    ax7.set_xlabel('Compare Wafers (sorted)')
    ax7.set_ylabel('Feature Exceed Ratio (%)')
    ax7.legend()

    # 3-8: Outlier 공통 Feature
    ax8 = fig.add_subplot(gs[2, 1])
    common = outlier_result["common_features"].head(15)
    if len(common) > 0:
        ax8.barh(range(len(common)), common.values * 100, color='#7B1FA2', alpha=0.7)
        ax8.set_yticks(range(len(common)))
        ax8.set_yticklabels(common.index, fontsize=8)
    ax8.set_title(f'Outlier Wafer 공통 이상 Feature\n(Outlier Rate={outlier_result["score"]:.1%})', fontweight='bold')
    ax8.set_xlabel('Outlier Wafer 중 초과 비율 (%)')

    # 3-9: Score 요약
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
        if val > thresholds[2]: color, level = '#E53935', 'HIGH_RISK'
        elif val > thresholds[1]: color, level = '#FF9800', 'RISK'
        elif val > thresholds[0]: color, level = '#FFC107', 'CAUTION'
        else: color, level = '#4CAF50', 'SAFE'
        box = FancyBboxPatch((0.5, y-0.6), 9, 1.2, boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax9.add_patch(box)
        ax9.text(1.0, y+0.1, f'{name}: {val:.4f}', fontsize=12, fontweight='bold')
        ax9.text(8, y+0.1, level, fontsize=11, fontweight='bold', color=color, ha='right')

    save_fig(fig, '03_score_calculation.png')


# ============================================================
# 4. Feature Importance 시각화
# ============================================================
def visualize_feature_importance(importance_result, shift_result, tail_result):
    print("[4/9] Visualizing feature importance...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Step 4: Feature Importance 분석', fontsize=16, fontweight='bold', y=0.98)

    # 4-1: Shift 원인 Top-20
    ax = axes[0, 0]
    shift_imp = importance_result["shift_features"].head(20)
    colors = ['#E53935' if d == '악화' else '#1E88E5' for d in shift_imp['direction']]
    y_pos = range(len(shift_imp) - 1, -1, -1)
    ax.barh(y_pos, shift_imp['abs_z_shift'].values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(shift_imp['feature'].values, fontsize=8)
    ax.set_title('Shift 원인 Feature Top-20', fontweight='bold')
    ax.set_xlabel('|Z-shift|')
    ax.legend(handles=[Patch(color='#E53935', label='악화'), Patch(color='#1E88E5', label='개선')], loc='lower right')

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

    # 4-3: Shift vs Tail scatter
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
    ax.scatter(merged['z_shift'], merged['tail_rate'] * 100, c=colors_scatter, alpha=0.5, s=15)
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('|Z-shift|')
    ax.set_ylabel('Tail Rate (%)')
    ax.set_title('Shift vs Tail: Feature 분류', fontweight='bold')
    ax.legend(handles=[
        Patch(color='#FF9800', label='Shift만'), Patch(color='#7B1FA2', label='Tail만'),
        Patch(color='#E53935', label='Shift + Tail'), Patch(color='#BDBDBD', label='정상')], fontsize=8)

    # 4-4: Multi-Score Heatmap
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

    heatmap_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))
    sns.heatmap(heatmap_norm, ax=ax, cmap='YlOrRd', annot=heatmap_data.round(2).values,
                fmt='', linewidths=0.5, cbar_kws={'label': 'Normalized Score'})
    ax.set_title('Multi-Score Feature Importance Heatmap', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, '04_feature_importance.png')


# ============================================================
# 5. 최종 리포트 대시보드
# ============================================================
def visualize_final_report(result):
    print("[5/9] Visualizing final report dashboard...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)
    fig.suptitle('ECO Change Detection Report -- 최종 결과', fontsize=18, fontweight='bold', y=0.99)

    decision = result['decision']
    meta = result['metadata']
    decision_colors = {'SAFE': '#4CAF50', 'CAUTION': '#FFC107', 'RISK': '#FF9800',
                       'HIGH_RISK': '#E53935', 'INSUFFICIENT_DATA': '#9E9E9E'}
    dec_color = decision_colors.get(decision['decision'], '#9E9E9E')

    # 5-1: 판정 헤더
    ax_header = fig.add_subplot(gs[0, :2])
    ax_header.set_xlim(0, 10); ax_header.set_ylim(0, 10); ax_header.axis('off')
    box = FancyBboxPatch((0.2, 0.2), 9.5, 9.5, boxstyle="round,pad=0.3",
                         facecolor=dec_color, alpha=0.1, edgecolor=dec_color, linewidth=3)
    ax_header.add_patch(box)
    ax_header.text(5, 8.5, f'판정: {decision["decision"]}', ha='center', fontsize=28, fontweight='bold', color=dec_color)
    ax_header.text(5, 6.5, f'Step: {meta["step_id"]}  |  Code: {meta["change_code"]}', ha='center', fontsize=14)
    ax_header.text(5, 5, f'Ref: {meta["ref_count"]:,} wafers  |  Compare: {meta["comp_count"]} wafers  |  Features: {meta["feature_count"]:,}', ha='center', fontsize=12, color='#555')
    for i, reason in enumerate(decision['reasons'][:3]):
        ax_header.text(1, 3.5 - i * 1.2, f'  {reason}', fontsize=10, color='#333')

    # 5-2: Score 게이지
    ax_gauge = fig.add_subplot(gs[0, 2:])
    ax_gauge.set_xlim(0, 10); ax_gauge.set_ylim(0, 10); ax_gauge.axis('off')
    ax_gauge.set_title('3종 Score Summary', fontweight='bold', fontsize=14)

    info_lines = [
        (f'Shift Score: {result["scores"]["shift_score"]:.3f}', 7.5),
        (f'Tail Score (max): {result["scores"]["tail_score_max"]:.4f}', 5.5),
        (f'Tail Feature Count: {result["scores"]["tail_feature_count"]}', 3.5),
        (f'Outlier Wafer Rate: {result["scores"]["outlier_wafer_rate"]:.4f}', 1.5),
    ]
    for text, y in info_lines:
        ax_gauge.text(0.5, y, text, fontsize=13, fontweight='bold')

    # 5-3: Shift Top-10
    ax_shift = fig.add_subplot(gs[1, :2])
    shift_feats = result['importance']['shift_features'].head(10)
    colors = ['#E53935' if d == '악화' else '#1E88E5' for d in shift_feats['direction']]
    y_pos = range(len(shift_feats) - 1, -1, -1)
    ax_shift.barh(y_pos, shift_feats['z_shift'].values, color=colors, edgecolor='white')
    ax_shift.set_yticks(list(y_pos))
    ax_shift.set_yticklabels([f"#{i+1} {f}" for i, f in enumerate(shift_feats['feature'])], fontsize=9)
    ax_shift.axvline(x=0, color='black', linewidth=0.5)
    ax_shift.set_title('Shift 원인 Feature Top-10', fontweight='bold')
    ax_shift.set_xlabel('Z-shift (s)')

    # 5-4: Tail Top-10
    ax_tail = fig.add_subplot(gs[1, 2:])
    tail_feats = result['importance']['tail_features'].head(10)
    if len(tail_feats) > 0:
        y_pos = range(len(tail_feats) - 1, -1, -1)
        ax_tail.barh(y_pos, tail_feats['tail_rate_pct'].values, color='#FF9800', edgecolor='white')
        ax_tail.set_yticks(list(y_pos))
        ax_tail.set_yticklabels([f"#{i+1} {f}" for i, f in enumerate(tail_feats['feature'])], fontsize=9)
        ax_tail.axvline(x=1, color='gray', linestyle='--', label='기대치 (1%)')
        ax_tail.legend()
    ax_tail.set_title('Tail 원인 Feature Top-10', fontweight='bold')

    # 5-5: Outlier wafer 분포
    ax_outlier = fig.add_subplot(gs[2, :2])
    exceed = result['detail']['outlier']['exceed_ratio_per_wafer'].sort_values(ascending=False)
    colors_o = ['#E53935' if v > 0.05 else '#4CAF50' for v in exceed.values]
    ax_outlier.bar(range(len(exceed)), exceed.values * 100, color=colors_o, width=1.0)
    ax_outlier.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Outlier 기준 (5%)')
    ax_outlier.set_title(f'Outlier Wafer 분포 (총 {result["detail"]["outlier"]["outlier_count"]}개)', fontweight='bold')
    ax_outlier.set_xlabel('Compare Wafers')
    ax_outlier.set_ylabel('Feature Exceed Ratio (%)')
    ax_outlier.legend()

    # 5-6: 권장 액션
    ax_action = fig.add_subplot(gs[2, 2:])
    ax_action.set_xlim(0, 10); ax_action.set_ylim(0, 10); ax_action.axis('off')
    ax_action.set_title('권장 액션', fontweight='bold', fontsize=14)
    actions = {
        'SAFE': ['변경점 적용 승인 가능', '정기 모니터링 유지'],
        'CAUTION': ['추가 wafer 확보 후 재평가 권장', '원인 Feature 엔지니어 확인 필요'],
        'RISK': ['변경점 적용 보류 권장', '원인 Feature 긴급 분석 필요'],
        'HIGH_RISK': ['변경점 적용 중단 권장', '즉시 엔지니어링 검토', '공정 조건 원복 고려'],
    }
    action_list = actions.get(decision['decision'], ['평가 불가'])
    for i, action in enumerate(action_list):
        ax_action.text(0.5, 8.5 - i * 2, f'> {action}', fontsize=12, color='#333')
    outlier_ids = result['detail']['outlier']['outlier_wafer_ids'][:10]
    ax_action.text(0.5, 2.5, f'Outlier Wafer:', fontsize=10, fontweight='bold')
    ax_action.text(0.5, 1.3, ', '.join(outlier_ids) if outlier_ids else '(없음)', fontsize=9, color='#777')

    save_fig(fig, '05_final_report.png')


# ============================================================
# 6. 민감도 분석
# ============================================================
def visualize_sensitivity_analysis(df_ref, df_comp):
    print("[6/9] Running sensitivity analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 6: Score 민감도 검증 (Sensitivity Analysis)', fontsize=16, fontweight='bold', y=0.98)

    # 6-1: False Alarm Test
    ax = axes[0, 0]
    np.random.seed(123)
    n_ref = len(df_ref)
    idx = np.random.permutation(n_ref)
    ref_a = df_ref.iloc[idx[:n_ref//2]]
    ref_b = df_ref.iloc[idx[n_ref//2:]]
    false_result = run_eco_change_detection(ref_a, ref_b, step_id="FALSE_ALARM")
    real_result = run_eco_change_detection(df_ref, df_comp, step_id="REAL")

    labels = ['Shift Score', 'Tail Score\n(max)', 'Outlier Rate']
    false_vals = [false_result['scores']['shift_score'], false_result['scores']['tail_score_max'], false_result['scores']['outlier_wafer_rate']]
    real_vals = [real_result['scores']['shift_score'], real_result['scores']['tail_score_max'], real_result['scores']['outlier_wafer_rate']]

    x = np.arange(len(labels))
    ax.bar(x - 0.2, false_vals, 0.35, label='False Alarm (Ref vs Ref)', color='#4CAF50', alpha=0.8)
    ax.bar(x + 0.2, real_vals, 0.35, label='Real (Ref vs Compare)', color='#E53935', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('False Alarm Test', fontweight='bold')
    ax.legend()
    for i, (fv, rv) in enumerate(zip(false_vals, real_vals)):
        ax.text(i - 0.2, fv + 0.01, f'{fv:.3f}', ha='center', fontsize=8)
        ax.text(i + 0.2, rv + 0.01, f'{rv:.3f}', ha='center', fontsize=8)

    # 6-2: Shift 크기 vs Score
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
    ax.set_xlabel('Shift 크기 (s 단위)')
    ax.set_ylabel('Shift Score')
    ax.set_title('Shift 크기 vs Score (단조 증가 검증)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6-3: Sample Size vs Score
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
    ax.errorbar(sample_sizes, means, yerr=stds, fmt='o-', color='#1E88E5', capsize=5, linewidth=2, markersize=8)
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='최소 기준 (30)')
    ax.set_xlabel('Compare Wafer 수')
    ax.set_ylabel('Shift Score (mean +/- std)')
    ax.set_title('Sample Size vs Score 변동성', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6-4: 검증 체크리스트
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title('검증 체크리스트 결과', fontweight='bold', fontsize=14)

    shift_top = real_result['importance']['shift_features']['feature'].head(20).tolist()
    shift_check = any('EDS_01' in f for f in shift_top)
    tail_top = real_result['importance']['tail_features']['feature'].head(20).tolist()
    tail_check = any('EDS_05' in f for f in tail_top)
    outlier_ids = real_result['detail']['outlier']['outlier_wafer_ids']
    outlier_check = any('comp_w007' in w for w in outlier_ids)
    false_alarm_check = (false_result['scores']['shift_score'] < 0.3 and false_result['scores']['tail_score_max'] < 0.03)
    monotone_check = all(shift_scores[i] <= shift_scores[i+1] for i in range(len(shift_scores)-1))

    checks = [
        (shift_check, 'Pattern A: Shift Top-20에 EDS_0100~0120 포함'),
        (tail_check, 'Pattern B: Tail에 EDS_0500~0510 포함'),
        (outlier_check, 'Pattern C: Outlier에 comp_w0070~0074 포함'),
        (false_alarm_check, 'False Alarm: Ref vs Ref Score 약 0'),
        (monotone_check, 'Shift 크기 -> Score 단조 증가'),
    ]
    for i, (passed, desc) in enumerate(checks):
        icon = 'PASS' if passed else 'FAIL'
        color = '#4CAF50' if passed else '#E53935'
        ax.text(0.5, 8.5 - i * 1.5, f'[{icon}]  {desc}', fontsize=11, color=color, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, '06_sensitivity_analysis.png')


# ============================================================
# 7. 추가 인사이트 시각화 (Violin, Scatter, Correlation, CDF)
# ============================================================
def visualize_additional_insights(scaled_ref, scaled_comp, shift_result, tail_result):
    print("[7/9] Generating additional insight visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('추가 인사이트: 다양한 시각화 기법', fontsize=16, fontweight='bold', y=0.98)

    top_feats = shift_result["z_shift_all"].abs().sort_values(ascending=False).head(5).index.tolist()

    # 7-1: Violin plot
    ax = axes[0, 0]
    plot_data = []
    for f in top_feats:
        for val in scaled_ref[f].values[:200]:
            plot_data.append({'Feature': f, 'Value': val, 'Group': 'Ref'})
        for val in scaled_comp[f].values:
            plot_data.append({'Feature': f, 'Value': val, 'Group': 'Compare'})
    plot_df = pd.DataFrame(plot_data)
    sns.violinplot(data=plot_df, x='Feature', y='Value', hue='Group', split=True, ax=ax,
                   palette={'Ref': '#2196F3', 'Compare': '#FF5722'}, inner='quartile', density_norm='width')
    ax.set_title('Top-5 Shift Feature: Violin Plot', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    # 7-2: 2D scatter
    ax = axes[0, 1]
    if len(top_feats) >= 2:
        f1, f2 = top_feats[0], top_feats[1]
        ax.scatter(scaled_ref[f1].values[:300], scaled_ref[f2].values[:300], alpha=0.3, s=10, c='#2196F3', label='Ref')
        ax.scatter(scaled_comp[f1].values, scaled_comp[f2].values, alpha=0.7, s=30, c='#FF5722', label='Compare', edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f1); ax.set_ylabel(f2)
        ax.set_title('2D Scatter: Top Shift Features', fontweight='bold')
        ax.legend()

    # 7-3: Correlation heatmap
    ax = axes[1, 0]
    top_all = list(dict.fromkeys(
        shift_result["z_shift_all"].abs().sort_values(ascending=False).head(8).index.tolist() +
        tail_result["elevated_features"].head(5).index.tolist()))[:12]
    if len(top_all) > 2:
        corr_ref = scaled_ref[top_all].corr()
        mask = np.triu(np.ones_like(corr_ref, dtype=bool))
        sns.heatmap(corr_ref, mask=mask, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    annot=True, fmt='.2f', linewidths=0.5, annot_kws={'fontsize': 7})
        ax.set_title('Top Feature 상관관계 (Ref 기준)', fontweight='bold')

    # 7-4: CDF 비교
    ax = axes[1, 1]
    compare_feat = top_feats[0]
    ref_sorted = np.sort(scaled_ref[compare_feat].values)
    comp_sorted = np.sort(scaled_comp[compare_feat].values)
    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    comp_cdf = np.arange(1, len(comp_sorted) + 1) / len(comp_sorted)
    ax.plot(ref_sorted, ref_cdf, color='#2196F3', linewidth=2, label='Ref CDF')
    ax.plot(comp_sorted, comp_cdf, color='#FF5722', linewidth=2, label='Compare CDF')
    ks_stat, ks_p = stats.ks_2samp(scaled_ref[compare_feat], scaled_comp[compare_feat])
    ax.set_title(f'CDF 비교: {compare_feat}\n(KS stat={ks_stat:.4f}, p={ks_p:.2e})', fontweight='bold')
    ax.set_xlabel('Scaled Value'); ax.set_ylabel('Cumulative Probability')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '07_additional_insights.png')


# ============================================================
# 8. Ground Truth 검증
# ============================================================
def visualize_ground_truth_validation(result, ground_truth):
    print("[8/9] Validating against ground truth...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ground Truth 검증: 패턴별 검출 결과', fontsize=16, fontweight='bold', y=0.98)

    all_anomaly = set(f'EDS_{i:04d}' for i in ground_truth['all_anomaly_features'])
    shift_top = set(result['importance']['shift_features']['feature'].head(30).tolist())
    tail_top = set(result['importance']['tail_features']['feature'].head(30).tolist())
    outlier_top = set(result['importance']['outlier_features']['feature'].head(30).tolist())
    detected = shift_top | tail_top | outlier_top

    # 8-1: 패턴별 검출 현황
    ax = axes[0, 0]
    pattern_names = []
    pattern_recalls = []
    pattern_colors_bar = []
    for pname, pinfo in ground_truth.items():
        if not pname.startswith('pattern_'):
            continue
        feat_set = set(f'EDS_{i:04d}' for i in pinfo['features'])
        tp = len(feat_set & detected)
        recall = tp / len(feat_set) if len(feat_set) > 0 else 0
        pattern_names.append(f"{pname[-1].upper()}\n{pinfo['type']}\n({pinfo['difficulty']})")
        pattern_recalls.append(recall)
        dc = {'Easy': '#4CAF50', 'Medium': '#FF9800', 'Hard': '#E53935'}
        pattern_colors_bar.append(dc.get(pinfo['difficulty'], '#999'))

    bars = ax.bar(pattern_names, pattern_recalls, color=pattern_colors_bar, width=0.6, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, pattern_recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('패턴별 Recall (Top-30 Feature 기준)', fontweight='bold')
    ax.set_ylabel('Recall')

    # 8-2: Precision / Recall 종합
    ax = axes[0, 1]
    tp = len(detected & all_anomaly)
    fp = len(detected - all_anomaly)
    fn = len(all_anomaly - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    colors_m = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(metrics, values, color=colors_m, width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_title(f'종합 검출 성능 (TP={tp}, FP={fp}, FN={fn})', fontweight='bold')

    # 8-3: Score별 검출 Feature 수
    ax = axes[0, 2]
    methods = ['Shift\nTop-30', 'Tail\nElevated', 'Outlier\nCommon', '통합\n(합집합)']
    counts = [len(shift_top & all_anomaly), len(tail_top & all_anomaly),
              len(outlier_top & all_anomaly), tp]
    colors_bar2 = ['#E65100', '#283593', '#6A1B9A', '#C62828']
    bars = ax.bar(methods, counts, color=colors_bar2, width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', fontweight='bold', fontsize=12)
    ax.set_title('Score별 True Positive 수', fontweight='bold')
    ax.set_ylabel('TP Feature 수')

    # 8-4 ~ 8-6: 패턴별 Feature 분포 비교
    pattern_demos = [
        ('pattern_A', 'EDS_0110', 'Pattern A: Shift'),
        ('pattern_D', 'EDS_0605', 'Pattern D: Trend'),
        ('pattern_E', 'EDS_0710', 'Pattern E: Subtle'),
    ]
    scaled_ref = result['scaled_ref']
    scaled_comp = result['scaled_comp']

    for idx, (pkey, feat, title) in enumerate(pattern_demos):
        ax = axes[1, idx]
        if feat in scaled_ref.columns:
            ax.hist(scaled_ref[feat], bins=40, alpha=0.5, label='Ref', color='#2196F3', density=True)
            ax.hist(scaled_comp[feat], bins=20, alpha=0.5, label='Compare', color='#FF5722', density=True)
            ax.axvline(scaled_ref[feat].mean(), color='#1565C0', linestyle='--', linewidth=1.5)
            ax.axvline(scaled_comp[feat].mean(), color='#D32F2F', linestyle='--', linewidth=1.5)
            z_val = result['detail']['shift']['z_shift_all'].get(feat, 0)
            ax.set_title(f'{title}\n({feat}, z-shift={z_val:.2f})', fontweight='bold')
            ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, '08_ground_truth_validation.png')


# ============================================================
# 9. Wafer 단위 심층 분석
# ============================================================
def visualize_wafer_analysis(result):
    print("[9/9] Visualizing wafer-level analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Wafer 단위 심층 분석', fontsize=16, fontweight='bold', y=1.02)

    outlier_result = result['detail']['outlier']
    scaled_comp = result['scaled_comp']

    # 9-1: Wafer별 이상 Feature 수 히스토그램
    ax = axes[0]
    exceed = outlier_result['exceed_ratio_per_wafer'] * 100
    ax.hist(exceed, bins=30, color='#FF5722', alpha=0.7, edgecolor='white')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Outlier 기준 (5%)')
    ax.set_title('Compare Wafer별 이상 비율 분포', fontweight='bold')
    ax.set_xlabel('Feature Exceed Ratio (%)')
    ax.set_ylabel('Wafer 수')
    ax.legend()

    # 9-2: Top Outlier Wafer 상세
    ax = axes[1]
    outlier_ids = outlier_result['outlier_wafer_ids'][:5]
    if outlier_ids:
        for wid in outlier_ids:
            if wid in scaled_comp.index:
                wafer_vals = scaled_comp.loc[wid].sort_values(ascending=False).head(50)
                ax.plot(range(len(wafer_vals)), wafer_vals.values, alpha=0.7, linewidth=1.5, label=wid)
        ax.set_title('Top-5 Outlier Wafer: Feature Profile', fontweight='bold')
        ax.set_xlabel('Feature Rank (sorted)')
        ax.set_ylabel('Scaled Value')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No Outlier Wafers', ha='center', va='center', fontsize=14, transform=ax.transAxes)

    # 9-3: Normal vs Outlier 평균 비교
    ax = axes[2]
    if outlier_ids:
        normal_mask = ~scaled_comp.index.isin(outlier_ids)
        normal_mean = scaled_comp.loc[normal_mask].mean().sort_values(ascending=False).head(30)
        outlier_mean = scaled_comp.loc[outlier_ids].mean().reindex(normal_mean.index)
        x_pos = np.arange(len(normal_mean))
        ax.bar(x_pos - 0.15, normal_mean.values, 0.3, label='Normal', color='#4CAF50', alpha=0.7)
        ax.bar(x_pos + 0.15, outlier_mean.values, 0.3, label='Outlier', color='#E53935', alpha=0.7)
        ax.set_xticks(x_pos[::3])
        ax.set_xticklabels(normal_mean.index[::3], rotation=45, fontsize=7)
        ax.set_title('Normal vs Outlier Wafer: 상위 Feature 평균', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Outlier Wafers', ha='center', va='center', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    save_fig(fig, '09_wafer_analysis.png')


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  ECO Change Detection PoC -- Full Experiment")
    print("=" * 60)

    # 1. Generate synthetic data (5 patterns)
    print("\n[DATA] Generating synthetic data (5 patterns, 3 difficulties)...")
    df_ref, df_comp, ground_truth = generate_synthetic_data()
    print(f"  Ref: {df_ref.shape}, Compare: {df_comp.shape}")
    print(f"  Anomaly features: {len(ground_truth['all_anomaly_features'])} / {df_ref.shape[1]}")
    for name, info in ground_truth.items():
        if name.startswith('pattern_'):
            print(f"    {name}: {info['type']} ({info['difficulty']}), {len(info['features'])} features")

    # 2. Run full pipeline
    print("\n[PIPELINE] Running ECO Change Detection pipeline...")
    result = run_eco_change_detection(df_ref, df_comp, step_id="S310_ETCH", change_code="CHG_TEST_001")

    # Print text report
    print("\n" + "=" * 55)
    print("  ECO Change Detection Report")
    print("=" * 55)
    meta = result['metadata']
    print(f"  Step: {meta['step_id']}")
    print(f"  Ref: {meta['ref_count']:,}  |  Compare: {meta['comp_count']}")
    print(f"  Features: {meta['feature_count']:,}")
    print("-" * 55)
    s = result['scores']
    print(f"  Shift Score:     {s['shift_score']:.3f}")
    print(f"  Tail Score max:  {s['tail_score_max']:.4f}")
    print(f"  Tail Count:      {s['tail_feature_count']}")
    print(f"  Outlier Rate:    {s['outlier_wafer_rate']:.4f}")
    print("-" * 55)
    d = result['decision']
    print(f"  DECISION: {d['decision']}")
    for r in d['reasons']:
        print(f"    - {r}")
    print("-" * 55)
    print("  Shift Top-10:")
    for _, row in result['importance']['shift_features'].head(10).iterrows():
        arrow = ">" if row['direction'] == '악화' else "<"
        print(f"    {row['feature']}: z={row['z_shift']:+.3f} {row['direction']} {arrow}")
    print("-" * 55)
    outlier_ids = result['detail']['outlier']['outlier_wafer_ids']
    print(f"  Outlier Wafers ({len(outlier_ids)}): {', '.join(outlier_ids[:10])}")
    print("=" * 55)

    # 3. Visualizations
    print("\n[VISUALIZATION] Generating figures...")
    scaled_ref = result['scaled_ref']
    scaled_comp = result['scaled_comp']

    visualize_pipeline_flow()
    visualize_synthetic_data(df_ref, df_comp, ground_truth)
    visualize_preprocessing(df_ref, df_comp, list(scaled_ref.columns), scaled_ref, scaled_comp)
    visualize_score_calculation(scaled_ref, scaled_comp,
                                result['detail']['shift'], result['detail']['tail'], result['detail']['outlier'])
    visualize_feature_importance(result['importance'], result['detail']['shift'], result['detail']['tail'])
    visualize_final_report(result)
    visualize_sensitivity_analysis(df_ref, df_comp)
    visualize_additional_insights(scaled_ref, scaled_comp, result['detail']['shift'], result['detail']['tail'])
    visualize_ground_truth_validation(result, ground_truth)
    visualize_wafer_analysis(result)

    print("\n[DONE] All visualizations saved to results/ and docs/images/")
    return result


if __name__ == '__main__':
    result = main()
