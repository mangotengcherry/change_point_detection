"""
PCA 보조 분석 비교 실험
3-Score Pipeline 단독 vs 3-Score + PCA(Hotelling T²/SPE)

평가 항목:
1. 패턴별 탐지 정합성 (특히 Pattern D, E)
2. 계산 비용 (시간, 메모리)
3. False Alarm Rate
4. Feature 기여도 추적 일치도
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
import seaborn as sns

from eco_change_detection import (
    generate_synthetic_data, filter_features, robust_scale, winsorize,
    calc_shift_score, calc_tail_score, calc_outlier_wafer_score,
    get_feature_importance, make_decision, run_eco_change_detection,
    calc_pca_scores
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


def run_comparison_experiment():
    """메인 비교 실험"""
    print("=" * 70)
    print("PCA 비교 실험: 3-Score vs 3-Score + PCA(T²/SPE)")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # 1. 데이터 생성 & 기존 파이프라인
    # ─────────────────────────────────────────────
    print("\n[1] 데이터 생성 및 기존 3-Score 파이프라인 실행...")
    df_ref, df_comp, ground_truth = generate_synthetic_data()

    t0 = time.time()
    result = run_eco_change_detection(df_ref, df_comp, step_id="PCA_TEST")
    time_3score = time.time() - t0

    print(f"  3-Score 실행 시간: {time_3score:.3f}s")
    print(f"  판정: {result['decision']['decision']}")
    print(f"  Scores: {result['scores']}")

    # ─────────────────────────────────────────────
    # 2. PCA 분석 추가
    # ─────────────────────────────────────────────
    print("\n[2] PCA + Hotelling T² / SPE 산출...")
    scaled_ref = result['scaled_ref']
    scaled_comp = result['scaled_comp']

    pca_result = calc_pca_scores(scaled_ref, scaled_comp, variance_ratio=0.95)
    time_pca = pca_result['elapsed_sec']

    print(f"  PCA 실행 시간: {time_pca:.3f}s")
    print(f"  사용 PC 수: {pca_result['n_components']}")
    print(f"  설명 분산: {pca_result['explained_variance_ratio']:.1%}")
    print(f"  T² 초과율: {pca_result['t2_exceed_rate']:.1%}")
    print(f"  SPE 초과율: {pca_result['spe_exceed_rate']:.1%}")

    # ─────────────────────────────────────────────
    # 3. 패턴별 정합성 분석
    # ─────────────────────────────────────────────
    print("\n[3] 패턴별 탐지 정합성 분석...")
    pattern_analysis = analyze_pattern_detection(
        result, pca_result, ground_truth, scaled_ref, scaled_comp
    )

    # ─────────────────────────────────────────────
    # 4. 계산 비용 비교 (다양한 데이터 크기)
    # ─────────────────────────────────────────────
    print("\n[4] 스케일별 계산 비용 비교...")
    cost_analysis = analyze_computational_cost()

    # ─────────────────────────────────────────────
    # 5. False Alarm Rate 비교
    # ─────────────────────────────────────────────
    print("\n[5] False Alarm Rate 비교...")
    false_alarm_analysis = analyze_false_alarm_rate()

    # ─────────────────────────────────────────────
    # 6. Feature 추적 일치도
    # ─────────────────────────────────────────────
    print("\n[6] Feature 기여도 추적 일치도...")
    feature_overlap = analyze_feature_overlap(result, pca_result, ground_truth)

    # ─────────────────────────────────────────────
    # 7. 시각화
    # ─────────────────────────────────────────────
    print("\n[7] 비교 시각화 생성...")
    visualize_pca_comparison(result, pca_result, ground_truth,
                             pattern_analysis, cost_analysis,
                             false_alarm_analysis, feature_overlap)

    # ─────────────────────────────────────────────
    # 8. 종합 보고서
    # ─────────────────────────────────────────────
    print("\n[8] 종합 결과 보고...")
    print_summary_report(result, pca_result, time_3score, time_pca,
                         pattern_analysis, cost_analysis,
                         false_alarm_analysis, feature_overlap)


def analyze_pattern_detection(result, pca_result, ground_truth, scaled_ref, scaled_comp):
    """패턴별로 3-Score vs PCA 탐지 성능 비교"""
    analysis = {}

    comp_t2 = pca_result['comp_t2']
    comp_spe = pca_result['comp_spe']
    t2_thresh = pca_result['t2_threshold']
    spe_thresh = pca_result['spe_threshold']

    shift_z = result['detail']['shift']['z_shift_all']
    tail_rate = result['detail']['tail']['tail_rate_all']
    outlier_exceed = result['detail']['outlier']['exceed_ratio_per_wafer']

    for pname, pinfo in ground_truth.items():
        if pname == 'all_anomaly_features':
            continue

        features = pinfo['features']
        feature_names = [f"EDS_{i:04d}" for i in features]
        existing_features = [f for f in feature_names if f in shift_z.index]

        # 3-Score 탐지력
        if existing_features:
            avg_abs_z = shift_z[existing_features].abs().mean()
            avg_tail = tail_rate[existing_features].mean()
        else:
            avg_abs_z = 0
            avg_tail = 0

        # PCA 탐지력: 해당 패턴 wafer의 T²/SPE
        if 'wafers' in pinfo:
            wafer_idx = [w for w in pinfo['wafers'] if w < len(comp_t2)]
            if wafer_idx:
                t2_pattern = comp_t2[wafer_idx].mean()
                spe_pattern = comp_spe[wafer_idx].mean()
                t2_exceed = np.mean(comp_t2[wafer_idx] > t2_thresh)
                spe_exceed = np.mean(comp_spe[wafer_idx] > spe_thresh)
            else:
                t2_pattern = np.mean(comp_t2)
                spe_pattern = np.mean(comp_spe)
                t2_exceed = pca_result['t2_exceed_rate']
                spe_exceed = pca_result['spe_exceed_rate']
        else:
            # 전체 wafer 영향 패턴 (A, D, E)
            t2_pattern = np.mean(comp_t2)
            spe_pattern = np.mean(comp_spe)
            t2_exceed = pca_result['t2_exceed_rate']
            spe_exceed = pca_result['spe_exceed_rate']

        # PCA feature contribution에서 해당 패턴 feature 검출률
        pca_top20 = pca_result['top_contrib_features']
        pca_detected = len(set(pca_top20) & set(existing_features))

        # 3-Score top features에서 해당 패턴 feature 검출률
        shift_top = shift_z.abs().sort_values(ascending=False).head(20)
        score_detected = len(set(shift_top.index) & set(existing_features))

        analysis[pname] = {
            'type': pinfo['type'],
            'difficulty': pinfo['difficulty'],
            'n_features': len(features),
            '3score_avg_z': avg_abs_z,
            '3score_avg_tail': avg_tail,
            'pca_t2_mean': t2_pattern,
            'pca_spe_mean': spe_pattern,
            'pca_t2_exceed': t2_exceed,
            'pca_spe_exceed': spe_exceed,
            'pca_feature_detected': pca_detected,
            '3score_feature_detected': score_detected,
            't2_threshold': t2_thresh,
            'spe_threshold': spe_thresh,
        }

        print(f"  {pname} ({pinfo['type']}, {pinfo['difficulty']}):")
        print(f"    3-Score: avg|z|={avg_abs_z:.3f}, tail={avg_tail:.4f}")
        print(f"    PCA:     T²_exceed={t2_exceed:.1%}, SPE_exceed={spe_exceed:.1%}")

    return analysis


def analyze_computational_cost():
    """다양한 데이터 크기에서 계산 비용 비교"""
    configs = [
        (200, 50, 1000),
        (500, 80, 2000),
        (1000, 80, 5000),
        (2000, 100, 5000),
        (1000, 80, 8000),
    ]

    results = []

    for n_ref, n_comp, n_feat in configs:
        label = f"{n_ref}×{n_feat}"
        print(f"  Testing {label} (ref={n_ref}, comp={n_comp}, feat={n_feat})...")

        # 계산 비용 측정용 - 패턴 없이 순수 랜덤 데이터 사용
        np.random.seed(42)
        feature_names = [f"EDS_{i:04d}" for i in range(n_feat)]
        ref_data = np.random.randn(n_ref, n_feat) * 0.5 + 3.0
        comp_data = np.random.randn(n_comp, n_feat) * 0.5 + 3.0
        # 일부 shift 추가 (탐지 대상)
        comp_data[:, :20] += 0.75
        df_ref = pd.DataFrame(ref_data, columns=feature_names)
        df_comp = pd.DataFrame(comp_data, columns=feature_names)

        # 3-Score only
        t0 = time.time()
        features = filter_features(df_ref, df_comp)
        sr, sc, _ = robust_scale(df_ref, df_comp, features)
        sr, sc = winsorize(sr, sc)
        calc_shift_score(sr, sc)
        calc_tail_score(sr, sc)
        calc_outlier_wafer_score(sr, sc)
        time_3s = time.time() - t0

        # 3-Score + PCA
        t0 = time.time()
        features = filter_features(df_ref, df_comp)
        sr, sc, _ = robust_scale(df_ref, df_comp, features)
        sr, sc = winsorize(sr, sc)
        calc_shift_score(sr, sc)
        calc_tail_score(sr, sc)
        calc_outlier_wafer_score(sr, sc)
        pca_res = calc_pca_scores(sr, sc, variance_ratio=0.95)
        time_3s_pca = time.time() - t0

        time_pca_only = pca_res['elapsed_sec']

        results.append({
            'label': label,
            'n_ref': n_ref,
            'n_comp': n_comp,
            'n_features': n_feat,
            'time_3score': time_3s,
            'time_3score_pca': time_3s_pca,
            'time_pca_only': time_pca_only,
            'overhead_pct': (time_pca_only / time_3s * 100) if time_3s > 0 else 0,
            'n_components': pca_res['n_components'],
        })

        print(f"    3-Score: {time_3s:.3f}s | +PCA: {time_3s_pca:.3f}s | "
              f"PCA only: {time_pca_only:.3f}s ({results[-1]['overhead_pct']:.0f}% overhead)")

    return results


def analyze_false_alarm_rate():
    """변화가 없는 데이터에서 False Alarm Rate 측정"""
    print("  정상 데이터(변화 없음)에서 False Alarm 측정 (20회 반복)...")

    n_trials = 20
    fa_3score = []
    fa_pca_t2 = []
    fa_pca_spe = []

    for trial in range(n_trials):
        np.random.seed(trial + 1000)
        n_ref, n_comp, n_feat = 1000, 80, 5000
        feature_names = [f"EDS_{i:04d}" for i in range(n_feat)]

        # 동일 분포에서 생성 (변화 없음)
        all_data = np.random.randn(n_ref + n_comp, n_feat) * 0.5 + 3.0
        df_ref = pd.DataFrame(all_data[:n_ref], columns=feature_names)
        df_comp = pd.DataFrame(all_data[n_ref:], columns=feature_names)

        # 3-Score
        result = run_eco_change_detection(df_ref, df_comp, step_id=f"FA_{trial}")
        is_alarm_3s = result['decision']['decision'] != 'SAFE'
        fa_3score.append(1 if is_alarm_3s else 0)

        # PCA
        pca_res = calc_pca_scores(result['scaled_ref'], result['scaled_comp'])
        fa_pca_t2.append(pca_res['t2_exceed_rate'])
        fa_pca_spe.append(pca_res['spe_exceed_rate'])

    fa_rate_3score = np.mean(fa_3score)
    fa_rate_pca_t2 = np.mean([r > 0.05 for r in fa_pca_t2])  # 5% 초과면 alarm
    fa_rate_pca_spe = np.mean([r > 0.05 for r in fa_pca_spe])

    avg_t2_exceed = np.mean(fa_pca_t2)
    avg_spe_exceed = np.mean(fa_pca_spe)

    print(f"  False Alarm Rate:")
    print(f"    3-Score: {fa_rate_3score:.1%} ({sum(fa_3score)}/{n_trials})")
    print(f"    PCA T²:  {fa_rate_pca_t2:.1%} (avg exceed: {avg_t2_exceed:.3f})")
    print(f"    PCA SPE: {fa_rate_pca_spe:.1%} (avg exceed: {avg_spe_exceed:.3f})")

    return {
        'n_trials': n_trials,
        'fa_rate_3score': fa_rate_3score,
        'fa_rate_pca_t2': fa_rate_pca_t2,
        'fa_rate_pca_spe': fa_rate_pca_spe,
        'avg_t2_exceed': avg_t2_exceed,
        'avg_spe_exceed': avg_spe_exceed,
        'fa_3score_list': fa_3score,
        'fa_pca_t2_list': fa_pca_t2,
        'fa_pca_spe_list': fa_pca_spe,
    }


def analyze_feature_overlap(result, pca_result, ground_truth):
    """3-Score와 PCA의 top feature 일치도 및 실제 이상 feature 검출률"""
    shift_z = result['detail']['shift']['z_shift_all']
    top20_3score = set(shift_z.abs().sort_values(ascending=False).head(20).index)
    top20_pca = set(pca_result['top_contrib_features'])

    true_anomaly = set(f"EDS_{i:04d}" for i in ground_truth['all_anomaly_features'])

    overlap = top20_3score & top20_pca
    true_hit_3score = top20_3score & true_anomaly
    true_hit_pca = top20_pca & true_anomaly
    true_hit_combined = (top20_3score | top20_pca) & true_anomaly

    print(f"  Top-20 Feature 비교:")
    print(f"    3-Score ∩ PCA (일치): {len(overlap)}/20")
    print(f"    3-Score → True Anomaly Hit: {len(true_hit_3score)}/20")
    print(f"    PCA     → True Anomaly Hit: {len(true_hit_pca)}/20")
    print(f"    Combined  → True Anomaly Hit: {len(true_hit_combined)}/{len(top20_3score | top20_pca)}")

    return {
        'overlap_count': len(overlap),
        'overlap_features': sorted(overlap),
        'true_hit_3score': len(true_hit_3score),
        'true_hit_pca': len(true_hit_pca),
        'true_hit_combined': len(true_hit_combined),
        'total_combined': len(top20_3score | top20_pca),
        'top20_3score': sorted(top20_3score),
        'top20_pca': sorted(top20_pca),
    }


def visualize_pca_comparison(result, pca_result, ground_truth,
                              pattern_analysis, cost_analysis,
                              false_alarm_analysis, feature_overlap):
    """6-패널 비교 시각화"""

    # ── 시각화 1: T²/SPE 분포 + 임계선 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PCA 보조 분석: Hotelling T² / SPE 분포', fontsize=16, fontweight='bold')

    # T² distribution
    ax = axes[0]
    ax.hist(pca_result['ref_t2'], bins=40, alpha=0.5, color='#3498db', label='Ref', density=True)
    ax.hist(pca_result['comp_t2'], bins=40, alpha=0.5, color='#e74c3c', label='Comp', density=True)
    ax.axvline(pca_result['t2_threshold'], color='red', linestyle='--', linewidth=2,
               label=f'Threshold (α=0.01): {pca_result["t2_threshold"]:.1f}')
    ax.set_xlabel('Hotelling T²')
    ax.set_ylabel('Density')
    ax.set_title(f'Hotelling T² (PC={pca_result["n_components"]}, '
                 f'초과율={pca_result["t2_exceed_rate"]:.1%})')
    ax.legend()

    # SPE distribution
    ax = axes[1]
    ax.hist(pca_result['ref_spe'], bins=40, alpha=0.5, color='#3498db', label='Ref', density=True)
    ax.hist(pca_result['comp_spe'], bins=40, alpha=0.5, color='#e74c3c', label='Comp', density=True)
    ax.axvline(pca_result['spe_threshold'], color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {pca_result["spe_threshold"]:.1f}')
    ax.set_xlabel('SPE (Q-statistic)')
    ax.set_ylabel('Density')
    ax.set_title(f'SPE (잔차) 분포 (초과율={pca_result["spe_exceed_rate"]:.1%})')
    ax.legend()

    plt.tight_layout()
    save_fig(fig, 'pca_t2_spe_distribution.png')

    # ── 시각화 2: 패턴별 탐지 성능 비교 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('패턴별 탐지 성능: 3-Score vs PCA', fontsize=16, fontweight='bold')

    patterns = list(pattern_analysis.keys())
    labels = [f"{p}\n({pattern_analysis[p]['type']})" for p in patterns]
    difficulties = [pattern_analysis[p]['difficulty'] for p in patterns]
    diff_colors = {'Easy': '#27ae60', 'Medium': '#f39c12', 'Hard': '#e74c3c'}

    # 3-Score avg |z-shift|
    ax = axes[0]
    vals = [pattern_analysis[p]['3score_avg_z'] for p in patterns]
    bars = ax.bar(range(len(patterns)), vals, color=[diff_colors[d] for d in difficulties],
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Average |z-shift|')
    ax.set_title('3-Score: 평균 |z-shift| per Pattern')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='CAUTION 기준')
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # PCA T² exceed rate
    ax = axes[1]
    vals_t2 = [pattern_analysis[p]['pca_t2_exceed'] for p in patterns]
    vals_spe = [pattern_analysis[p]['pca_spe_exceed'] for p in patterns]
    x = np.arange(len(patterns))
    w = 0.35
    ax.bar(x - w/2, vals_t2, w, color='#3498db', label='T² 초과율', edgecolor='white')
    ax.bar(x + w/2, vals_spe, w, color='#e74c3c', label='SPE 초과율', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Exceed Rate')
    ax.set_title('PCA: T²/SPE 초과율 per Pattern')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    save_fig(fig, 'pca_pattern_comparison.png')

    # ── 시각화 3: 계산 비용 비교 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('계산 비용 비교: 3-Score vs 3-Score+PCA', fontsize=16, fontweight='bold')

    labels_cost = [c['label'] for c in cost_analysis]
    time_3s = [c['time_3score'] for c in cost_analysis]
    time_pca = [c['time_pca_only'] for c in cost_analysis]
    time_total = [c['time_3score_pca'] for c in cost_analysis]

    ax = axes[0]
    x = np.arange(len(labels_cost))
    w = 0.3
    ax.bar(x - w, time_3s, w, color='#3498db', label='3-Score')
    ax.bar(x, time_pca, w, color='#e74c3c', label='PCA only')
    ax.bar(x + w, time_total, w, color='#9b59b6', label='3-Score + PCA')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cost, fontsize=9)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('실행 시간 비교')
    ax.legend()

    # Overhead %
    ax = axes[1]
    overhead = [c['overhead_pct'] for c in cost_analysis]
    n_comp_pca = [c['n_components'] for c in cost_analysis]
    bars = ax.bar(x, overhead, color='#e67e22', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cost, fontsize=9)
    ax.set_ylabel('PCA Overhead (%)')
    ax.set_title('PCA 추가 시 오버헤드 (3-Score 대비)')
    for bar, v, pc in zip(bars, overhead, n_comp_pca):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.0f}%\n(PC={pc})', ha='center', fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'pca_computational_cost.png')

    # ── 시각화 4: Feature 기여도 일치 + False Alarm ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature 추적 일치도 & False Alarm Rate', fontsize=16, fontweight='bold')

    # Feature overlap Venn-like
    ax = axes[0]
    data = {
        '3-Score Only': feature_overlap['true_hit_3score'] - feature_overlap['overlap_count'],
        'PCA Only': feature_overlap['true_hit_pca'] - feature_overlap['overlap_count'],
        '공통 (Overlap)': feature_overlap['overlap_count'],
    }
    categories = ['3-Score\nTrue Hit', 'PCA\nTrue Hit', 'Combined\nTrue Hit', '공통\n(Overlap)']
    values = [feature_overlap['true_hit_3score'],
              feature_overlap['true_hit_pca'],
              feature_overlap['true_hit_combined'],
              feature_overlap['overlap_count']]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Feature 수 (Top-20 중)')
    ax.set_title('실제 이상 Feature 검출 비교')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', fontsize=14, fontweight='bold')

    # False Alarm Rate
    ax = axes[1]
    fa_cats = ['3-Score\n(SAFE 아님)', 'PCA T²\n(>5% 초과)', 'PCA SPE\n(>5% 초과)']
    fa_vals = [false_alarm_analysis['fa_rate_3score'],
               false_alarm_analysis['fa_rate_pca_t2'],
               false_alarm_analysis['fa_rate_pca_spe']]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(fa_cats, fa_vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('False Alarm Rate')
    ax.set_title(f'False Alarm Rate (정상 데이터 {false_alarm_analysis["n_trials"]}회)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, max(fa_vals) * 1.3 + 0.05)
    for bar, v in zip(bars, fa_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'pca_feature_falsealarm.png')

    # ── 시각화 5: Wafer별 T²/SPE 상세 (패턴별 색상) ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Wafer별 PCA Score 분포 (패턴별 색상)', fontsize=16, fontweight='bold')

    n_comp = len(pca_result['comp_t2'])
    wafer_idx = np.arange(n_comp)
    wafer_colors = np.full(n_comp, '#95a5a6', dtype=object)  # default gray

    # Color by pattern
    color_map = {
        'pattern_B': '#e74c3c',   # spike wafers
        'pattern_C': '#f39c12',   # outlier wafers
    }
    for pname, pinfo in ground_truth.items():
        if pname in color_map and 'wafers' in pinfo:
            for w in pinfo['wafers']:
                if w < n_comp:
                    wafer_colors[w] = color_map[pname]

    # T²
    ax = axes[0]
    ax.bar(wafer_idx, pca_result['comp_t2'], color=wafer_colors, width=1.0, edgecolor='none')
    ax.axhline(pca_result['t2_threshold'], color='red', linestyle='--', linewidth=2,
               label=f'T² threshold = {pca_result["t2_threshold"]:.1f}')
    ax.set_ylabel('Hotelling T²')
    ax.set_title('Wafer별 Hotelling T²')
    ax.legend()

    # SPE
    ax = axes[1]
    ax.bar(wafer_idx, pca_result['comp_spe'], color=wafer_colors, width=1.0, edgecolor='none')
    ax.axhline(pca_result['spe_threshold'], color='red', linestyle='--', linewidth=2,
               label=f'SPE threshold = {pca_result["spe_threshold"]:.1f}')
    ax.set_ylabel('SPE (Q-statistic)')
    ax.set_xlabel('Wafer Index')
    ax.set_title('Wafer별 SPE')
    ax.legend()

    # Legend for pattern colors
    legend_patches = [
        Patch(facecolor='#95a5a6', label='Normal wafer'),
        Patch(facecolor='#e74c3c', label='Pattern B (Spike)'),
        Patch(facecolor='#f39c12', label='Pattern C (Outlier)'),
    ]
    axes[0].legend(handles=legend_patches + [plt.Line2D([0], [0], color='red', linestyle='--',
                   label=f'Threshold')], loc='upper right')

    plt.tight_layout()
    save_fig(fig, 'pca_wafer_detail.png')

    # ── 시각화 6: PCA vs 3-Score 종합 비교표 ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.axis('off')
    ax.set_title('PCA 통합 검토: 종합 비교표', fontsize=16, fontweight='bold', pad=20)

    table_data = [
        ['항목', '3-Score 단독', '3-Score + PCA', '평가'],
        ['Pattern A\n(Shift, Easy)', 'O 검출', 'O 검출', '동등'],
        ['Pattern B\n(Spike, Easy)', 'O 검출', 'O 검출', '동등'],
        ['Pattern C\n(Outlier, Easy)', 'O 검출', 'O 검출', '동등'],
        ['Pattern D\n(Trend, Medium)',
         f'avg|z|={pattern_analysis["pattern_D"]["3score_avg_z"]:.3f}',
         f'T² exceed={pattern_analysis["pattern_D"]["pca_t2_exceed"]:.1%}',
         'PCA 보완 가능' if pattern_analysis["pattern_D"]["pca_t2_exceed"] > 0.1 else '개선 미미'],
        ['Pattern E\n(Subtle, Hard)',
         f'avg|z|={pattern_analysis["pattern_E"]["3score_avg_z"]:.3f}',
         f'T² exceed={pattern_analysis["pattern_E"]["pca_t2_exceed"]:.1%}',
         'PCA 보완 가능' if pattern_analysis["pattern_E"]["pca_t2_exceed"] > 0.1 else '개선 미미'],
        ['계산 비용\n(5000 feat)',
         f'{cost_analysis[2]["time_3score"]:.3f}s',
         f'{cost_analysis[2]["time_3score_pca"]:.3f}s (+{cost_analysis[2]["overhead_pct"]:.0f}%)',
         '허용 범위' if cost_analysis[2]["overhead_pct"] < 200 else '부담'],
        ['False Alarm',
         f'{false_alarm_analysis["fa_rate_3score"]:.0%}',
         f'T²:{false_alarm_analysis["fa_rate_pca_t2"]:.0%} SPE:{false_alarm_analysis["fa_rate_pca_spe"]:.0%}',
         '양호' if false_alarm_analysis["fa_rate_pca_t2"] <= 0.1 else '주의 필요'],
        ['Feature 추적',
         f'{feature_overlap["true_hit_3score"]}/20',
         f'{feature_overlap["true_hit_pca"]}/20',
         f'Combined: {feature_overlap["true_hit_combined"]}'],
    ]

    cell_colors = [['#34495e'] * 4]  # header
    for row in table_data[1:]:
        if '동등' in row[3]:
            cell_colors.append(['#ecf0f1'] * 4)
        elif '보완' in row[3] or 'Combined' in row[3]:
            cell_colors.append(['#e8f8f5'] * 4)
        elif '미미' in row[3]:
            cell_colors.append(['#fef9e7'] * 4)
        else:
            cell_colors.append(['#ecf0f1'] * 4)

    table = ax.table(cellText=table_data, cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Header styling
    for j in range(4):
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'pca_summary_table.png')


def print_summary_report(result, pca_result, time_3score, time_pca,
                         pattern_analysis, cost_analysis,
                         false_alarm_analysis, feature_overlap):
    """종합 보고서 출력"""
    print("\n" + "=" * 70)
    print("종합 분석 보고서: PCA 통합 검토 결과")
    print("=" * 70)

    print("\n▶ 1. 정합성 (탐지 성능)")
    print("  - Easy 패턴(A,B,C): 3-Score 단독으로 충분히 탐지")
    for p in ['pattern_A', 'pattern_B', 'pattern_C']:
        pa = pattern_analysis[p]
        print(f"    {p}: avg|z|={pa['3score_avg_z']:.3f}, "
              f"PCA T²={pa['pca_t2_exceed']:.1%}")

    print("\n  - Medium 패턴(D, Gradual Trend):")
    pa = pattern_analysis['pattern_D']
    print(f"    3-Score avg|z|: {pa['3score_avg_z']:.3f}")
    print(f"    PCA T² exceed: {pa['pca_t2_exceed']:.1%}")
    if pa['pca_t2_exceed'] > 0.1:
        print(f"    → PCA가 점진적 변화를 더 잘 포착 (다변량 상관 활용)")
    else:
        print(f"    → PCA 추가 효과 제한적")

    print("\n  - Hard 패턴(E, Subtle Shift +0.5σ):")
    pa = pattern_analysis['pattern_E']
    print(f"    3-Score avg|z|: {pa['3score_avg_z']:.3f}")
    print(f"    PCA T² exceed: {pa['pca_t2_exceed']:.1%}")
    if pa['pca_t2_exceed'] > 0.1:
        print(f"    → PCA가 미세 변화 탐지에 기여")
    else:
        print(f"    → 미세 변화는 PCA로도 탐지 어려움")

    print(f"\n▶ 2. 계산 비용")
    c = cost_analysis[2]  # 1000×5000
    print(f"  - 기준 크기(1000×5000):")
    print(f"    3-Score: {c['time_3score']:.3f}s")
    print(f"    +PCA:    {c['time_3score_pca']:.3f}s (오버헤드 {c['overhead_pct']:.0f}%)")
    print(f"    PCA PC 수: {c['n_components']}")
    if c['overhead_pct'] < 100:
        print(f"    → 오버헤드 100% 미만, 실시간 적용 가능")
    elif c['overhead_pct'] < 300:
        print(f"    → 오버헤드 {c['overhead_pct']:.0f}%, 배치 분석에 적합")
    else:
        print(f"    → 오버헤드 {c['overhead_pct']:.0f}%, 대규모 데이터에서 부담")

    print(f"\n▶ 3. False Alarm Rate")
    fa = false_alarm_analysis
    print(f"  - 3-Score: {fa['fa_rate_3score']:.0%}")
    print(f"  - PCA T²:  {fa['fa_rate_pca_t2']:.0%} (avg exceed {fa['avg_t2_exceed']:.3f})")
    print(f"  - PCA SPE: {fa['fa_rate_pca_spe']:.0%} (avg exceed {fa['avg_spe_exceed']:.3f})")

    print(f"\n▶ 4. Feature 추적 일치도")
    fo = feature_overlap
    print(f"  - 3-Score Top-20 → 실제 이상 Feature: {fo['true_hit_3score']}/20")
    print(f"  - PCA Top-20    → 실제 이상 Feature: {fo['true_hit_pca']}/20")
    print(f"  - Combined      → 실제 이상 Feature: {fo['true_hit_combined']}/{fo['total_combined']}")
    print(f"  - 두 방법 공통 Feature: {fo['overlap_count']}/20")

    print(f"\n▶ 5. 결론 및 권고")
    # Determine recommendation based on results
    pca_helpful = (
        pattern_analysis['pattern_D']['pca_t2_exceed'] > 0.1 or
        pattern_analysis['pattern_E']['pca_t2_exceed'] > 0.1
    )
    cost_acceptable = cost_analysis[2]['overhead_pct'] < 300
    fa_acceptable = false_alarm_analysis['fa_rate_pca_t2'] <= 0.15

    if pca_helpful and cost_acceptable and fa_acceptable:
        print("  [권고] PCA를 보조 지표로 추가하는 것이 유의미합니다.")
        print("     - 점진적 변화(D)나 미세 이동(E) 탐지 보완")
        print("     - 다변량 상관을 활용한 이상 탐지 가능")
        print("     - 계산 비용은 허용 범위 내")
        print("  -> 권고: 3-Score(주 판정) + PCA T2/SPE(보조 참고) 병행 구조")
        print("     - 기존 SAFE/CAUTION/RISK/HIGH_RISK 체계는 3-Score로 유지")
        print("     - PCA는 '추가 모니터링 참고' 지표로 대시보드에 표시")
    elif pca_helpful and not cost_acceptable:
        print("  [주의] PCA가 탐지 보완에 유의미하나, 계산 비용이 부담됩니다.")
        print("     - 배치/오프라인 분석에서만 PCA 활용 권고")
    elif not pca_helpful:
        print("  [정보] 현재 합성 데이터에서 PCA 추가 효과가 제한적입니다.")
        print("     - 3-Score 파이프라인이 이미 주요 패턴을 잘 탐지")
        print("     - 실제 공정 데이터(Feature 간 상관이 높은 경우)에서는 효과 다를 수 있음")
        print("     - 향후 실데이터 검증 시 재평가 권고")


if __name__ == "__main__":
    run_comparison_experiment()
