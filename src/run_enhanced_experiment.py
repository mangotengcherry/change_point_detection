"""
Enhanced ECO Change Detection Experiment
=========================================
기존 3-Score 파이프라인 + 강화 분석(변경 유형별 Stage 판정, 신뢰도, Bootstrap, 다변량 편차)을
통합 실행하고, 결과를 시각화 및 텍스트 리포트로 출력한다.

실행:
    python src/run_enhanced_experiment.py
"""

import sys, os, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from eco_change_detection import (
    generate_synthetic_data,
    run_eco_change_detection,
    run_enhanced_pipeline,
    calc_confidence_score,
    calc_bootstrap_stability,
    calc_global_deviation_score,
    make_enhanced_decision,
    filter_features,
    robust_scale,
    winsorize,
    calc_shift_score,
    calc_tail_score,
    calc_outlier_wafer_score,
)

# ============================================================
# 한글 폰트 설정
# ============================================================
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 출력 디렉토리
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DOCS_DIR = os.path.join(BASE_DIR, "docs", "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def save_fig(fig, name):
    for d in [RESULTS_DIR, DOCS_DIR]:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {name} saved")


# ============================================================
# 1. 합성 데이터 생성 & 기본 파이프라인 실행
# ============================================================
def run_base_pipeline():
    print("=" * 70)
    print("Phase 1: 합성 데이터 생성 & 기본 3-Score 파이프라인")
    print("=" * 70)

    df_ref, df_comp, gt = generate_synthetic_data(
        n_ref=1000, n_comp=80, n_features=5000, seed=42)

    print(f"  Ref: {df_ref.shape}, Comp: {df_comp.shape}")
    print(f"  패턴 A(Shift): F100~120, +1.5σ")
    print(f"  패턴 B(Spike): F500~510, 10% wafer, +6σ")
    print(f"  패턴 C(Outlier): F200~499, W70~74, +4σ")
    print(f"  패턴 D(Trend): F600~610, 0→+2σ 점진")
    print(f"  패턴 E(Subtle): F700~720, +0.5σ 미세")

    t0 = time.time()
    result = run_eco_change_detection(df_ref, df_comp,
                                       step_id="ETCH_S310",
                                       change_code="CHG_0241")
    base_time = time.time() - t0

    d = result["decision"]
    print(f"\n  ▶ 판정: {d['decision']} (Level {d['risk_level']})")
    print(f"  ▶ Shift={d['scores']['shift_score']:.3f}, "
          f"Tail={d['scores']['tail_score_max']:.1%}, "
          f"Outlier={d['scores']['outlier_wafer_rate']:.1%}")
    print(f"  ▶ 소요시간: {base_time:.2f}s")
    for r in d['reasons']:
        print(f"    - {r}")

    return df_ref, df_comp, gt, result


# ============================================================
# 2. 강화 파이프라인 — 효율 향상형
# ============================================================
def run_efficiency_test(df_ref, df_comp):
    print("\n" + "=" * 70)
    print("Phase 2: 강화 파이프라인 — 효율 향상형 (Equivalence Verification)")
    print("=" * 70)

    # 정상 데이터 (변화 없음)로 효율 향상형 테스트
    np.random.seed(99)
    n_ref, n_comp, n_feat = 1000, 80, 5000
    feat_names = [f"EDS_{i:04d}" for i in range(n_feat)]
    ref_clean = pd.DataFrame(
        np.random.randn(n_ref, n_feat) * 0.5 + 3.0,
        columns=feat_names,
        index=[f"ref_w{i:04d}" for i in range(n_ref)])
    comp_clean = pd.DataFrame(
        np.random.randn(n_comp, n_feat) * 0.5 + 3.0,
        columns=feat_names,
        index=[f"comp_w{i:04d}" for i in range(n_comp)])

    t0 = time.time()
    eff_result = run_enhanced_pipeline(
        ref_clean, comp_clean,
        step_id="CVD_S210", change_code="EFF_0012",
        change_type="efficiency")
    eff_time = time.time() - t0

    enh = eff_result["enhanced"]
    ed = enh["enhanced_decision"]
    print(f"\n  ▶ Stage: {ed['stage']} — {ed['stage_name']}")
    print(f"  ▶ 설명: {ed['description']}")
    conf = enh['confidence']
    conf_level = conf.get('level', conf.get('confidence_level', 'N/A'))
    conf_score = conf.get('score', conf.get('confidence_score', 0))
    print(f"  ▶ Confidence: {conf_level} ({conf_score:.2f})")
    if 'error' not in enh['bootstrap_stability']:
        bs = enh['bootstrap_stability']
        print(f"  ▶ Bootstrap Shift: {bs['mean_score']:.4f} ± {bs['std_score']:.4f} "
              f"(95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}])")
    if 'error' not in enh['global_deviation']:
        gd = enh['global_deviation']
        print(f"  ▶ Global Deviation: mean={gd['mean_distance']:.4f}, "
              f"outlier_rate={gd['outlier_rate']:.1%}")
    print(f"  ▶ 소요시간: {eff_time:.2f}s")
    for r in ed['reasons']:
        print(f"    - {r}")

    # 위험 데이터로 효율 향상형 테스트
    print("\n  --- 위험 케이스 (큰 shift 존재) ---")
    comp_risky = comp_clean.copy()
    comp_risky.iloc[:, 100:200] += 1.5  # 강한 품질 변화 삽입

    risk_result = run_enhanced_pipeline(
        ref_clean, comp_risky,
        step_id="CVD_S210", change_code="EFF_0013",
        change_type="efficiency")
    rd = risk_result["enhanced"]["enhanced_decision"]
    print(f"  ▶ Stage: {rd['stage']} — {rd['stage_name']}")
    print(f"  ▶ 설명: {rd['description']}")
    for r in rd['reasons']:
        print(f"    - {r}")

    return eff_result, risk_result


# ============================================================
# 3. 강화 파이프라인 — 불량 개선형
# ============================================================
def run_defect_improvement_test():
    print("\n" + "=" * 70)
    print("Phase 3: 강화 파이프라인 — 불량 개선형 (Defect Improvement)")
    print("=" * 70)

    np.random.seed(77)
    n_ref, n_comp, n_feat = 1000, 80, 5000
    feat_names = [f"EDS_{i:04d}" for i in range(n_feat)]

    ref_data = np.random.randn(n_ref, n_feat) * 0.5 + 3.0
    comp_data = np.random.randn(n_comp, n_feat) * 0.5 + 3.0

    # Target defect 개선 (z_shift 음수 = 감소 방향)
    target_idx = list(range(100, 121))
    target_names = [feat_names[i] for i in target_idx]
    comp_data[:, target_idx] -= 0.8  # 개선 방향

    df_ref = pd.DataFrame(ref_data, columns=feat_names,
                          index=[f"ref_w{i:04d}" for i in range(n_ref)])
    df_comp = pd.DataFrame(comp_data, columns=feat_names,
                           index=[f"comp_w{i:04d}" for i in range(n_comp)])

    # Case A: 개선 + 부작용 없음
    print("\n  --- Case A: 개선 확인, 부작용 없음 ---")
    result_a = run_enhanced_pipeline(
        df_ref, df_comp,
        step_id="IMPL_S420", change_code="DEF_0088",
        change_type="defect_improvement",
        target_features=target_names)

    ed_a = result_a["enhanced"]["enhanced_decision"]
    print(f"  ▶ Stage: {ed_a['stage']} — {ed_a['stage_name']}")
    print(f"  ▶ 설명: {ed_a['description']}")
    if 'target_assessment' in ed_a:
        ta = ed_a['target_assessment']
        print(f"  ▶ Target 개선: {ta['target_improved']}, Side Effect: {ta['side_effect_count']}개")
    for r in ed_a['reasons']:
        print(f"    - {r}")

    # Case B: 개선 + 부작용 있음
    print("\n  --- Case B: 개선 확인, 부작용 존재 ---")
    comp_side = comp_data.copy()
    side_idx = list(range(300, 320))
    comp_side[:, side_idx] += 1.5  # 부작용 삽입

    df_comp_side = pd.DataFrame(comp_side, columns=feat_names,
                                index=[f"comp_w{i:04d}" for i in range(n_comp)])
    result_b = run_enhanced_pipeline(
        df_ref, df_comp_side,
        step_id="IMPL_S420", change_code="DEF_0089",
        change_type="defect_improvement",
        target_features=target_names)

    ed_b = result_b["enhanced"]["enhanced_decision"]
    print(f"  ▶ Stage: {ed_b['stage']} — {ed_b['stage_name']}")
    print(f"  ▶ 설명: {ed_b['description']}")
    if 'target_assessment' in ed_b:
        ta = ed_b['target_assessment']
        print(f"  ▶ Target 개선: {ta['target_improved']}, Side Effect: {ta['side_effect_count']}개")
    for r in ed_b['reasons']:
        print(f"    - {r}")

    return result_a, result_b


# ============================================================
# 4. 민감도 분석
# ============================================================
def run_sensitivity_analysis():
    print("\n" + "=" * 70)
    print("Phase 4: 민감도 분석")
    print("=" * 70)

    np.random.seed(42)
    n_ref, n_feat = 1000, 5000
    feat_names = [f"EDS_{i:04d}" for i in range(n_feat)]
    ref_data = np.random.randn(n_ref, n_feat) * 0.5 + 3.0
    df_ref = pd.DataFrame(ref_data, columns=feat_names)

    # 4-1. False Alarm Test (ref vs ref)
    print("\n  [4-1] False Alarm Test (ref vs ref)")
    ref_a = df_ref.iloc[:500]
    ref_b = df_ref.iloc[500:]
    fa_result = run_eco_change_detection(ref_a, ref_b, step_id="FA_TEST")
    print(f"  ▶ 판정: {fa_result['decision']['decision']}")
    print(f"  ▶ Shift={fa_result['scores']['shift_score']:.4f}, "
          f"Tail={fa_result['scores']['tail_score_max']:.4f}, "
          f"Outlier={fa_result['scores']['outlier_wafer_rate']:.4f}")

    # 4-2. 단조 증가 검증
    print("\n  [4-2] 단조 증가 검증 (Shift 크기 vs Score)")
    shift_magnitudes = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    shift_scores = []
    for mag in shift_magnitudes:
        comp_test = np.random.randn(80, n_feat) * 0.5 + 3.0
        comp_test[:, 100:121] += mag * 0.5
        df_comp_test = pd.DataFrame(comp_test, columns=feat_names)
        r = run_eco_change_detection(df_ref, df_comp_test, step_id=f"MONO_{mag}")
        shift_scores.append(r['scores']['shift_score'])
    print(f"  Magnitudes: {shift_magnitudes}")
    print(f"  Scores:     {[f'{s:.3f}' for s in shift_scores]}")
    is_monotonic = all(shift_scores[i] <= shift_scores[i+1]
                       for i in range(len(shift_scores)-1))
    print(f"  ▶ 단조 증가: {'✓ PASS' if is_monotonic else '✗ FAIL'}")

    # 4-3. Sample Size 영향
    print("\n  [4-3] Sample Size 영향")
    sample_sizes = [10, 20, 30, 50, 80, 100, 200]
    size_results = []
    for sz in sample_sizes:
        comp_test = np.random.randn(sz, n_feat) * 0.5 + 3.0
        comp_test[:, 100:121] += 0.75  # +1.5σ
        df_comp_test = pd.DataFrame(comp_test, columns=feat_names)
        r = run_eco_change_detection(df_ref, df_comp_test, step_id=f"SZ_{sz}")
        conf = calc_confidence_score(sz)
        conf['level'] = conf.get('level', conf.get('confidence_level', 'N/A'))
        conf['score'] = conf.get('score', conf.get('confidence_score', 0))
        size_results.append({
            'n': sz,
            'shift': r['scores']['shift_score'],
            'decision': r['decision']['decision'],
            'confidence': conf['level']
        })
    for sr in size_results:
        print(f"    n={sr['n']:>4d}: Shift={sr['shift']:.3f}, "
              f"Decision={sr['decision']:<10s}, Confidence={sr['confidence']}")

    # 4-4. Bootstrap 안정성
    print("\n  [4-4] Bootstrap 안정성 (n_comp=80, 100 iterations)")
    comp_boot = np.random.randn(80, n_feat) * 0.5 + 3.0
    comp_boot[:, 100:121] += 0.75
    df_comp_boot = pd.DataFrame(comp_boot, columns=feat_names)
    features = filter_features(df_ref, df_comp_boot)
    sr, sc, _ = robust_scale(df_ref, df_comp_boot, features)
    sr, sc = winsorize(sr, sc)
    bs = calc_bootstrap_stability(sr, sc, n_iterations=200)
    print(f"  ▶ Mean Shift: {bs['mean_score']:.4f} ± {bs['std_score']:.4f}")
    print(f"  ▶ 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
    print(f"  ▶ CV: {bs['cv']:.2%}")

    return {
        'fa_result': fa_result,
        'shift_magnitudes': shift_magnitudes,
        'shift_scores': shift_scores,
        'is_monotonic': is_monotonic,
        'size_results': size_results,
        'bootstrap': bs,
    }


# ============================================================
# 5. Ground Truth 검증
# ============================================================
def run_ground_truth_validation(result, gt):
    print("\n" + "=" * 70)
    print("Phase 5: Ground Truth 검증")
    print("=" * 70)

    importance = result["importance"]
    shift_feats = set(importance["shift_features"]["feature"].tolist())
    tail_feats = set(importance["tail_features"]["feature"].tolist())
    outlier_feats = set(importance["outlier_features"]["feature"].tolist())
    all_detected = shift_feats | tail_feats | outlier_feats

    pattern_results = {}
    for pname, pinfo in gt.items():
        if pname == "all_anomaly_features":
            continue
        feat_names = [f"EDS_{i:04d}" for i in pinfo["features"]]
        gt_set = set(feat_names)
        detected = all_detected & gt_set
        recall = len(detected) / len(gt_set) if gt_set else 0
        pattern_results[pname] = {
            "type": pinfo["type"],
            "difficulty": pinfo["difficulty"],
            "total": len(gt_set),
            "detected": len(detected),
            "recall": recall
        }
        print(f"  {pname} ({pinfo['type']}, {pinfo['difficulty']}): "
              f"{len(detected)}/{len(gt_set)} = {recall:.1%}")

    # Precision
    all_gt = set(f"EDS_{i:04d}" for i in gt["all_anomaly_features"])
    tp = len(all_detected & all_gt)
    precision = tp / len(all_detected) if all_detected else 0
    recall_all = tp / len(all_gt) if all_gt else 0
    print(f"\n  ▶ 전체 Precision: {precision:.1%}")
    print(f"  ▶ 전체 Recall: {recall_all:.1%}")
    print(f"  ▶ Detected: {len(all_detected)}, Ground Truth: {len(all_gt)}, TP: {tp}")

    return pattern_results


# ============================================================
# 6. 시각화 생성
# ============================================================
def create_enhanced_visualizations(base_result, eff_result, risk_result,
                                    defect_a, defect_b, sensitivity, gt_validation):
    print("\n" + "=" * 70)
    print("Phase 6: 시각화 생성")
    print("=" * 70)

    # --- Fig 1: Enhanced Pipeline Overview ---
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Enhanced ECO Change Detection Pipeline — 강화 변경점 검증 시스템",
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # 1-1: 3-Score 결과
    ax1 = fig.add_subplot(gs[0, 0])
    scores = base_result['scores']
    bars = ax1.bar(['Shift', 'Tail(%)', 'Outlier(%)'],
                   [scores['shift_score'],
                    scores['tail_score_max'] * 100,
                    scores['outlier_wafer_rate'] * 100],
                   color=['#E53935', '#FF9800', '#9C27B0'], alpha=0.85)
    ax1.set_title("3-Score 결과 (합성 데이터)", fontweight='bold')
    ax1.set_ylabel("Score / %")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', fontsize=9)

    # 1-2: 효율 향상형 Stage 비교
    ax2 = fig.add_subplot(gs[0, 1])
    eff_stage = eff_result['enhanced']['enhanced_decision']['stage']
    risk_stage_val = risk_result['enhanced']['enhanced_decision']['stage']
    stage_labels = ['정상 (EFF_0012)', '위험 (EFF_0013)']
    stage_vals = [eff_stage, risk_stage_val]
    colors_stage = ['#4CAF50' if v >= 2 else '#E53935' for v in stage_vals]
    bars2 = ax2.barh(stage_labels, stage_vals, color=colors_stage, alpha=0.85)
    ax2.set_xlabel("Stage Level")
    ax2.set_title("효율 향상형 Stage 판정", fontweight='bold')
    ax2.set_xlim(-2, 5)
    for i, bar in enumerate(bars2):
        name = eff_result['enhanced']['enhanced_decision']['stage_name'] if i == 0 \
            else risk_result['enhanced']['enhanced_decision']['stage_name']
        ax2.text(max(bar.get_width(), 0) + 0.1, bar.get_y() + bar.get_height()/2,
                name, va='center', fontsize=8, fontweight='bold')

    # 1-3: 불량 개선형 Stage 비교
    ax3 = fig.add_subplot(gs[0, 2])
    da_stage = defect_a['enhanced']['enhanced_decision']['stage']
    db_stage = defect_b['enhanced']['enhanced_decision']['stage']
    d_labels = ['개선+부작용없음', '개선+부작용있음']
    d_vals = [da_stage, db_stage]
    colors_d = ['#4CAF50' if v >= 3 else '#FFC107' for v in d_vals]
    bars3 = ax3.barh(d_labels, d_vals, color=colors_d, alpha=0.85)
    ax3.set_xlabel("Stage Level")
    ax3.set_title("불량 개선형 Stage 판정", fontweight='bold')
    ax3.set_xlim(0, 5)
    for i, bar in enumerate(bars3):
        name = defect_a['enhanced']['enhanced_decision']['stage_name'] if i == 0 \
            else defect_b['enhanced']['enhanced_decision']['stage_name']
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                name, va='center', fontsize=8, fontweight='bold')

    # 2-1: False Alarm Test
    ax4 = fig.add_subplot(gs[1, 0])
    fa = sensitivity['fa_result']
    fa_scores = [fa['scores']['shift_score'],
                 fa['scores']['tail_score_max'] * 100,
                 fa['scores']['outlier_wafer_rate'] * 100]
    ax4.bar(['Shift', 'Tail(%)', 'Outlier(%)'], fa_scores,
            color=['#90CAF9', '#FFE082', '#CE93D8'], alpha=0.85)
    ax4.set_title("False Alarm Test (ref vs ref)", fontweight='bold')
    ax4.set_ylabel("Score / %")
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='경고 기준')
    ax4.legend(fontsize=8)
    for i, v in enumerate(fa_scores):
        ax4.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

    # 2-2: 단조 증가 검증
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(sensitivity['shift_magnitudes'], sensitivity['shift_scores'],
             'o-', color='#E53935', linewidth=2, markersize=6)
    ax5.set_xlabel("Shift Magnitude (σ)")
    ax5.set_ylabel("Shift Score")
    ax5.set_title(f"단조 증가 검증: {'✓ PASS' if sensitivity['is_monotonic'] else '✗ FAIL'}",
                  fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 2-3: Sample Size 영향
    ax6 = fig.add_subplot(gs[1, 2])
    sizes = [sr['n'] for sr in sensitivity['size_results']]
    shifts = [sr['shift'] for sr in sensitivity['size_results']]
    confs = [sr['confidence'] for sr in sensitivity['size_results']]
    colors_conf = ['#E53935' if c == 'LOW' else '#FFC107' if c == 'MEDIUM' else '#4CAF50'
                   for c in confs]
    ax6.bar(range(len(sizes)), shifts, color=colors_conf, alpha=0.85)
    ax6.set_xticks(range(len(sizes)))
    ax6.set_xticklabels([str(s) for s in sizes])
    ax6.set_xlabel("Compare Wafer 수")
    ax6.set_ylabel("Shift Score")
    ax6.set_title("Sample Size 영향 & Confidence", fontweight='bold')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E53935', label='LOW'),
                       Patch(facecolor='#FFC107', label='MEDIUM'),
                       Patch(facecolor='#4CAF50', label='HIGH')]
    ax6.legend(handles=legend_elements, title='Confidence', fontsize=7,
               loc='upper right')

    # 3-1: Bootstrap 안정성
    ax7 = fig.add_subplot(gs[2, 0])
    bs = sensitivity['bootstrap']
    ax7.hist(bs.get('scores', [bs['mean_score']]*50), bins=20,
             color='#42A5F5', alpha=0.7, edgecolor='white')
    ax7.axvline(bs['mean_score'], color='red', linewidth=2, label=f"Mean={bs['mean_score']:.4f}")
    ax7.axvline(bs['ci_lower'], color='orange', linestyle='--', label=f"95% CI lower={bs['ci_lower']:.4f}")
    ax7.axvline(bs['ci_upper'], color='orange', linestyle='--', label=f"95% CI upper={bs['ci_upper']:.4f}")
    ax7.set_title(f"Bootstrap Shift Score 분포 (CV={bs['cv']:.2%})", fontweight='bold')
    ax7.set_xlabel("Shift Score")
    ax7.legend(fontsize=7)

    # 3-2: Ground Truth 검증
    ax8 = fig.add_subplot(gs[2, 1])
    patterns = list(gt_validation.keys())
    recalls = [gt_validation[p]['recall'] for p in patterns]
    difficulties = [gt_validation[p]['difficulty'] for p in patterns]
    diff_colors = {'Easy': '#4CAF50', 'Medium': '#FFC107', 'Hard': '#E53935'}
    bar_colors = [diff_colors[d] for d in difficulties]
    bars8 = ax8.bar(range(len(patterns)), recalls, color=bar_colors, alpha=0.85)
    ax8.set_xticks(range(len(patterns)))
    ax8.set_xticklabels([f"{p}\n({gt_validation[p]['type'][:8]})" for p in patterns],
                        fontsize=7)
    ax8.set_ylabel("Recall")
    ax8.set_title("패턴별 검출률 (Ground Truth)", fontweight='bold')
    ax8.set_ylim(0, 1.1)
    legend_elements2 = [Patch(facecolor=c, label=d) for d, c in diff_colors.items()]
    ax8.legend(handles=legend_elements2, fontsize=7)

    # 3-3: 전역 편차 점수
    ax9 = fig.add_subplot(gs[2, 2])
    if 'error' not in eff_result['enhanced']['global_deviation']:
        gd = eff_result['enhanced']['global_deviation']
        ref_dist = gd['ref_distances']
        comp_dist = gd['comp_distances']
        ax9.hist(ref_dist, bins=30, alpha=0.6, color='#42A5F5', label='Ref', density=True)
        ax9.hist(comp_dist, bins=15, alpha=0.6, color='#EF5350', label='Comp', density=True)
        ax9.set_title(f"Global Deviation (Mahalanobis)\nOutlier Rate={gd['outlier_rate']:.1%}",
                      fontweight='bold')
        ax9.set_xlabel("Mahalanobis Distance")
        ax9.legend(fontsize=8)
    else:
        ax9.text(0.5, 0.5, "Global Deviation\n계산 오류", ha='center', va='center',
                transform=ax9.transAxes, fontsize=12)

    save_fig(fig, "10_enhanced_pipeline_overview.png")

    # --- Fig 2: Stage Assessment Detail ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle("Stage-Based Assessment — 단계별 검증 판정 상세", fontsize=14, fontweight='bold')

    # 효율 향상형 Stage 기준표
    ax = axes2[0, 0]
    ax.axis('off')
    stage_table_eff = [
        ['Stage', '명칭', 'Shift', 'Tail', 'Outlier', '데이터'],
        ['0', 'INSUFFICIENT_DATA', '-', '-', '-', '<30'],
        ['1', 'DATA_COLLECTED', '<0.3', '<5%', '<2%', '-'],
        ['2', 'EQUIVALENT_PRELIMINARY', '<0.5', '<8%', '<3%', '<100'],
        ['3', 'EQUIVALENT_CONFIRMED', '<0.5', '<8%', '<3%', '≥100'],
        ['4', 'EQUIVALENT_STABLE', '<0.3', '<5%', '<2%', '≥100'],
        ['-1', 'RISK', '>1.0', '>10%', '>5%', '-'],
    ]
    table1 = ax.table(cellText=stage_table_eff, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1, 1.5)
    for i in range(len(stage_table_eff[0])):
        table1[0, i].set_facecolor('#E3F2FD')
        table1[0, i].set_text_props(fontweight='bold')
    ax.set_title("효율 향상형 — Stage 판정 기준", fontweight='bold', pad=20)

    # 불량 개선형 Stage 기준표
    ax = axes2[0, 1]
    ax.axis('off')
    stage_table_def = [
        ['Stage', '명칭', 'Target 개선', 'Side Effect', 'Tail', 'Outlier'],
        ['0', 'INSUFFICIENT_DATA', '-', '-', '-', '-'],
        ['1', 'NOT_CONFIRMED', '<50%', '-', '-', '-'],
        ['2', 'PARTIAL_IMPROVEMENT', '≥50%', '>2개', '-', '-'],
        ['3', 'IMPROVEMENT_CONFIRMED', '≥50%', '≤2개', '<5%', '<5%'],
        ['4', 'IMPROVEMENT_STABLE', '≥50%', '0개', '<3%', '<3%'],
    ]
    table2 = ax.table(cellText=stage_table_def, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 1.5)
    for i in range(len(stage_table_def[0])):
        table2[0, i].set_facecolor('#FFF3E0')
        table2[0, i].set_text_props(fontweight='bold')
    ax.set_title("불량 개선형 — Stage 판정 기준", fontweight='bold', pad=20)

    # 효율 향상형 두 케이스 비교
    ax = axes2[1, 0]
    eff_ed = eff_result['enhanced']['enhanced_decision']
    risk_ed = risk_result['enhanced']['enhanced_decision']
    metrics = ['Shift Score', 'Tail Score Max', 'Outlier Rate']
    eff_vals = [eff_ed['scores']['shift_score'],
                eff_ed['scores']['tail_score_max'],
                eff_ed['scores']['outlier_wafer_rate']]
    risk_vals = [risk_ed['scores']['shift_score'],
                 risk_ed['scores']['tail_score_max'],
                 risk_ed['scores']['outlier_wafer_rate']]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, eff_vals, w, label=f'정상 (Stage {eff_ed["stage"]})', color='#4CAF50', alpha=0.8)
    ax.bar(x + w/2, risk_vals, w, label=f'위험 (Stage {risk_ed["stage"]})', color='#E53935', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("효율 향상형: 정상 vs 위험 케이스", fontweight='bold')
    ax.legend()

    # 불량 개선형 두 케이스 비교
    ax = axes2[1, 1]
    da_ed = defect_a['enhanced']['enhanced_decision']
    db_ed = defect_b['enhanced']['enhanced_decision']
    da_vals = [da_ed['scores']['shift_score'],
               da_ed['scores']['tail_score_max'],
               da_ed['scores']['outlier_wafer_rate']]
    db_vals = [db_ed['scores']['shift_score'],
               db_ed['scores']['tail_score_max'],
               db_ed['scores']['outlier_wafer_rate']]
    ax.bar(x - w/2, da_vals, w, label=f'부작용 없음 (Stage {da_ed["stage"]})', color='#4CAF50', alpha=0.8)
    ax.bar(x + w/2, db_vals, w, label=f'부작용 있음 (Stage {db_ed["stage"]})', color='#FFC107', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("불량 개선형: 부작용 없음 vs 있음", fontweight='bold')
    ax.legend()

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig2, "11_stage_assessment_detail.png")

    # --- Fig 3: Confidence & Bootstrap ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("데이터 신뢰도 & 분석 안정성", fontsize=14, fontweight='bold')

    # Confidence Score vs Sample Size
    ax = axes3[0]
    test_sizes = list(range(10, 201, 5))
    conf_scores = [calc_confidence_score(n).get('score', calc_confidence_score(n).get('confidence_score', 0)) for n in test_sizes]
    conf_levels = [calc_confidence_score(n).get('level', calc_confidence_score(n).get('confidence_level', 'N/A')) for n in test_sizes]
    colors_line = ['#E53935' if l == 'LOW' else '#FFC107' if l == 'MEDIUM' else '#4CAF50'
                   for l in conf_levels]
    ax.scatter(test_sizes, conf_scores, c=colors_line, s=10, alpha=0.8)
    ax.axvline(30, color='red', linestyle='--', alpha=0.5, label='최소 기준(30)')
    ax.axvline(100, color='green', linestyle='--', alpha=0.5, label='충분 기준(100)')
    ax.set_xlabel("Compare Wafer 수")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Confidence Score vs Sample Size", fontweight='bold')
    ax.legend(fontsize=8)

    # Bootstrap CV vs Sample Size
    ax = axes3[1]
    boot_sizes = [20, 30, 50, 80, 100, 150, 200]
    boot_cvs = []
    for sz in boot_sizes:
        np.random.seed(42)
        comp_test = np.random.randn(sz, 2000) * 0.5 + 3.0
        comp_test[:, 50:60] += 0.75
        ref_test = np.random.randn(500, 2000) * 0.5 + 3.0
        feat_n = [f"F{i}" for i in range(2000)]
        dr = pd.DataFrame(ref_test, columns=feat_n)
        dc = pd.DataFrame(comp_test, columns=feat_n)
        feats = filter_features(dr, dc)
        sr, sc, _ = robust_scale(dr, dc, feats)
        sr, sc = winsorize(sr, sc)
        b = calc_bootstrap_stability(sr, sc, n_iterations=50)
        boot_cvs.append(b['cv'] * 100)
    ax.plot(boot_sizes, boot_cvs, 'o-', color='#7B1FA2', linewidth=2)
    ax.axhline(10, color='green', linestyle='--', alpha=0.5, label='CV=10% 기준')
    ax.set_xlabel("Compare Wafer 수")
    ax.set_ylabel("CV (%)")
    ax.set_title("Bootstrap CV vs Sample Size", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Enhanced vs Base Decision
    ax = axes3[2]
    ax.axis('off')
    comparison_text = (
        "기존 판정 vs 강화 판정 비교\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"기존: SAFE/CAUTION/RISK/HIGH_RISK (4단계)\n"
        f"강화: Stage 0~4 + 변경 유형별 판정 (10단계+)\n\n"
        f"추가된 정보:\n"
        f"  • Confidence Score (데이터 신뢰도)\n"
        f"  • Bootstrap 95% CI (결과 안정성)\n"
        f"  • Global Deviation (다변량 편차)\n"
        f"  • 변경 유형별 판정 기준\n"
        f"  • Target/Side-effect 분리 분석"
    )
    ax.text(0.1, 0.5, comparison_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.8))

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig3, "12_confidence_bootstrap.png")

    print("\n  ✓ 모든 시각화 생성 완료")


# ============================================================
# 7. 텍스트 리포트 생성
# ============================================================
def generate_text_report(base_result, eff_result, risk_result,
                         defect_a, defect_b, sensitivity, gt_validation):
    print("\n" + "=" * 70)
    print("Phase 7: 텍스트 리포트 생성")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append("Enhanced ECO Change Detection — 통합 검증 리포트")
    lines.append("=" * 70)
    lines.append("")

    # 1. 기본 파이프라인
    d = base_result['decision']
    lines.append("[1] 기본 3-Score 파이프라인 (합성 데이터)")
    lines.append(f"  판정: {d['decision']} (Level {d['risk_level']})")
    lines.append(f"  Shift Score: {d['scores']['shift_score']:.3f}")
    lines.append(f"  Tail Score Max: {d['scores']['tail_score_max']:.1%}")
    lines.append(f"  Tail Feature Count: {d['scores']['tail_feature_count']}")
    lines.append(f"  Outlier Wafer Rate: {d['scores']['outlier_wafer_rate']:.1%}")
    for r in d['reasons']:
        lines.append(f"    - {r}")
    lines.append("")

    # 2. 효율 향상형
    lines.append("[2] 효율 향상형 검증")
    for label, res in [("정상 케이스", eff_result), ("위험 케이스", risk_result)]:
        ed = res['enhanced']['enhanced_decision']
        enh = res['enhanced']
        lines.append(f"  [{label}]")
        lines.append(f"    Stage: {ed['stage']} — {ed['stage_name']}")
        lines.append(f"    설명: {ed['description']}")
        _cl = enh['confidence'].get('level', enh['confidence'].get('confidence_level', 'N/A'))
        _cs = enh['confidence'].get('score', enh['confidence'].get('confidence_score', 0))
        lines.append(f"    Confidence: {_cl} ({_cs:.2f})")
        if 'error' not in enh['bootstrap_stability']:
            bs = enh['bootstrap_stability']
            lines.append(f"    Bootstrap: mean={bs['mean_score']:.4f} ± {bs['std_score']:.4f}, "
                        f"CV={bs['cv']:.2%}")
        if 'error' not in enh['global_deviation']:
            gd = enh['global_deviation']
            lines.append(f"    Global Deviation: mean={gd['mean_distance']:.4f}, "
                        f"outlier={gd['outlier_rate']:.1%}")
        for r in ed['reasons']:
            lines.append(f"      - {r}")
    lines.append("")

    # 3. 불량 개선형
    lines.append("[3] 불량 개선형 검증")
    for label, res in [("부작용 없음", defect_a), ("부작용 있음", defect_b)]:
        ed = res['enhanced']['enhanced_decision']
        lines.append(f"  [{label}]")
        lines.append(f"    Stage: {ed['stage']} — {ed['stage_name']}")
        lines.append(f"    설명: {ed['description']}")
        if 'target_assessment' in ed:
            ta = ed['target_assessment']
            lines.append(f"    Target 개선: {ta['target_improved']}, "
                        f"Side Effect: {ta['side_effect_count']}개")
        for r in ed['reasons']:
            lines.append(f"      - {r}")
    lines.append("")

    # 4. 민감도 분석
    lines.append("[4] 민감도 분석")
    fa = sensitivity['fa_result']
    lines.append(f"  False Alarm: {fa['decision']['decision']} "
                f"(Shift={fa['scores']['shift_score']:.4f})")
    lines.append(f"  단조 증가: {'PASS' if sensitivity['is_monotonic'] else 'FAIL'}")
    bs = sensitivity['bootstrap']
    lines.append(f"  Bootstrap: mean={bs['mean_score']:.4f}, "
                f"CV={bs['cv']:.2%}, "
                f"95%CI=[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
    lines.append("")

    # 5. Ground Truth
    lines.append("[5] Ground Truth 검증")
    for pname, pres in gt_validation.items():
        lines.append(f"  {pname} ({pres['type']}, {pres['difficulty']}): "
                    f"{pres['detected']}/{pres['total']} = {pres['recall']:.1%}")
    lines.append("")

    # 6. 검증 체크리스트
    lines.append("[6] 검증 체크리스트")
    checks = [
        ("False Alarm = 0", fa['decision']['decision'] == 'SAFE'),
        ("단조 증가 검증", sensitivity['is_monotonic']),
        ("Bootstrap CV < 20%", bs['cv'] < 0.20),
        ("효율 향상형 정상→Stage≥1", eff_result['enhanced']['enhanced_decision']['stage'] >= 1),
        ("효율 향상형 위험→RISK", risk_result['enhanced']['enhanced_decision']['stage_name'] == 'RISK'),
        ("불량개선 부작용없음→Stage≥3", defect_a['enhanced']['enhanced_decision']['stage'] >= 3),
        ("불량개선 부작용있음→Stage≤2", defect_b['enhanced']['enhanced_decision']['stage'] <= 2),
    ]
    all_pass = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"  [{status}] {name}")
        if not passed:
            all_pass = False
    lines.append(f"\n  ▶ 전체: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    report = "\n".join(lines)

    # 파일 저장
    report_path = os.path.join(RESULTS_DIR, "enhanced_validation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ 리포트 저장: {report_path}")

    return report, all_pass


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n>>> Enhanced ECO Change Detection Experiment")
    print("=" * 70)
    total_start = time.time()

    # Phase 1: 기본 파이프라인
    df_ref, df_comp, gt, base_result = run_base_pipeline()

    # Phase 2: 효율 향상형
    eff_result, risk_result = run_efficiency_test(df_ref, df_comp)

    # Phase 3: 불량 개선형
    defect_a, defect_b = run_defect_improvement_test()

    # Phase 4: 민감도 분석
    sensitivity = run_sensitivity_analysis()

    # Phase 5: Ground Truth 검증
    gt_validation = run_ground_truth_validation(base_result, gt)

    # Phase 6: 시각화
    create_enhanced_visualizations(
        base_result, eff_result, risk_result,
        defect_a, defect_b, sensitivity, gt_validation)

    # Phase 7: 리포트
    report, all_pass = generate_text_report(
        base_result, eff_result, risk_result,
        defect_a, defect_b, sensitivity, gt_validation)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"✅ 전체 실험 완료 — 소요시간: {total_time:.1f}s")
    print(f"{'=' * 70}")
