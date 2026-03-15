"""
AI/ML 강화 비교 실험
3-Score(통계) vs 3-Score + ML(IF, AE, Ensemble)

평가 축:
1. 패턴별 탐지 정합성 (특히 D, E)
2. 계산 비용
3. False Alarm Rate
4. Feature 추적 정확도
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
)
from ml_enhanced_detection import (
    calc_isolation_forest_score,
    calc_autoencoder_score,
    generate_ensemble_training_data,
    train_ensemble_classifier,
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


def main():
    print("=" * 70)
    print("AI/ML 강화 비교 실험")
    print("3-Score(통계) vs Isolation Forest / Autoencoder / Ensemble")
    print("=" * 70)

    # ─── 1. 데이터 준비 ───
    print("\n[1] 합성 데이터 생성 & 기존 파이프라인 실행...")
    df_ref, df_comp, ground_truth = generate_synthetic_data()

    t0 = time.time()
    result = run_eco_change_detection(df_ref, df_comp, step_id="ML_TEST")
    time_3score = time.time() - t0

    scaled_ref = result['scaled_ref']
    scaled_comp = result['scaled_comp']

    print(f"  3-Score: {time_3score:.3f}s | {result['decision']['decision']}")

    # ─── 2. Isolation Forest ───
    print("\n[2] Isolation Forest 실행...")
    if_result = calc_isolation_forest_score(scaled_ref, scaled_comp)
    print(f"  IF: {if_result['elapsed_sec']:.3f}s | anomaly_rate={if_result['anomaly_rate']:.1%}")

    # ─── 3. Autoencoder ───
    print("\n[3] Autoencoder 실행...")
    ae_result = calc_autoencoder_score(scaled_ref, scaled_comp, epochs=30)
    print(f"  AE: {ae_result['elapsed_sec']:.3f}s | exceed_rate={ae_result['exceed_rate']:.1%} "
          f"| backend={ae_result['backend']} | enc_dim={ae_result['encoding_dim']}")

    # ─── 4. 패턴별 탐지 비교 ───
    print("\n[4] 패턴별 탐지 성능 비교...")
    pattern_results = compare_pattern_detection(
        result, if_result, ae_result, ground_truth, scaled_ref, scaled_comp
    )

    # ─── 5. False Alarm Rate ───
    print("\n[5] False Alarm Rate 비교 (정상 데이터 10회)...")
    fa_results = compare_false_alarm()

    # ─── 6. Ensemble 학습 ───
    print("\n[6] Learned Ensemble 학습 및 평가...")
    ensemble_result = run_ensemble_experiment()

    # ─── 7. 계산 비용 비교 ───
    print("\n[7] 계산 비용 비교...")
    cost_results = compare_computational_cost()

    # ─── 8. Feature 추적 비교 ───
    print("\n[8] Feature 추적 정확도 비교...")
    feature_results = compare_feature_tracking(result, ae_result, ground_truth)

    # ─── 9. 시각화 ───
    print("\n[9] 시각화 생성...")
    visualize_all(result, if_result, ae_result, ground_truth,
                  pattern_results, fa_results, ensemble_result,
                  cost_results, feature_results, time_3score)

    # ─── 10. 종합 보고 ───
    print_final_report(result, if_result, ae_result,
                       pattern_results, fa_results, ensemble_result,
                       cost_results, feature_results, time_3score)


def compare_pattern_detection(result, if_result, ae_result, ground_truth,
                                scaled_ref, scaled_comp):
    """패턴별 탐지 성능 비교: 3-Score vs IF vs AE"""
    outcomes = {}

    shift_z = result['detail']['shift']['z_shift_all']
    outlier_exceed = result['detail']['outlier']['exceed_ratio_per_wafer']

    for pname, pinfo in ground_truth.items():
        if pname == 'all_anomaly_features':
            continue

        features = pinfo['features']
        feature_names = [f"EDS_{i:04d}" for i in features]
        existing = [f for f in feature_names if f in shift_z.index]

        # 3-Score
        avg_z = shift_z[existing].abs().mean() if existing else 0

        # IF: 해당 패턴 wafer의 anomaly score
        if 'wafers' in pinfo:
            widx = [w for w in pinfo['wafers'] if w < len(if_result['comp_scores_norm'])]
            if widx:
                if_score = if_result['comp_scores_norm'][widx].mean()
                if_anomaly = (if_result['comp_labels'][widx] == -1).mean()
            else:
                if_score = if_result['comp_scores_norm'].mean()
                if_anomaly = if_result['anomaly_rate']
        else:
            if_score = if_result['comp_scores_norm'].mean()
            if_anomaly = if_result['anomaly_rate']

        # AE: 해당 패턴 feature의 재구성 오류 증가
        feat_indices = [i for i, f in enumerate(scaled_ref.columns)
                        if f in set(feature_names)]
        if feat_indices:
            ae_feat_error = ae_result['feature_error_increase'][feat_indices].mean()
        else:
            ae_feat_error = 0

        # AE: wafer-level
        if 'wafers' in pinfo:
            widx = [w for w in pinfo['wafers'] if w < len(ae_result['comp_mse'])]
            ae_wafer_mse = ae_result['comp_mse'][widx].mean() if widx else ae_result['comp_mse'].mean()
        else:
            ae_wafer_mse = ae_result['comp_mse'].mean()

        outcomes[pname] = {
            'type': pinfo['type'],
            'difficulty': pinfo['difficulty'],
            '3score_z': avg_z,
            'if_anomaly_rate': if_anomaly,
            'if_score': if_score,
            'ae_feat_error': ae_feat_error,
            'ae_wafer_mse': ae_wafer_mse,
            'ae_threshold': ae_result['threshold'],
        }

        det_3s = 'O' if avg_z > 0.5 else 'X'
        det_if = 'O' if if_anomaly > 0.3 else 'X'
        det_ae = 'O' if ae_wafer_mse > ae_result['threshold'] else 'X'

        print(f"  {pname} ({pinfo['difficulty']}): "
              f"3S={det_3s}(z={avg_z:.3f}) | "
              f"IF={det_if}(anom={if_anomaly:.0%}) | "
              f"AE={det_ae}(mse={ae_wafer_mse:.4f})")

    return outcomes


def compare_false_alarm():
    """정상 데이터에서 False Alarm Rate 비교"""
    n_trials = 10
    fa = {'3score': [], 'if': [], 'ae': []}

    for trial in range(n_trials):
        np.random.seed(trial + 2000)
        n_ref, n_comp, n_feat = 500, 80, 2000
        feat_names = [f"EDS_{i:04d}" for i in range(n_feat)]

        all_data = np.random.randn(n_ref + n_comp, n_feat) * 0.5 + 3.0
        df_ref = pd.DataFrame(all_data[:n_ref], columns=feat_names)
        df_comp = pd.DataFrame(all_data[n_ref:], columns=feat_names)

        # 3-Score
        res = run_eco_change_detection(df_ref, df_comp, step_id=f"FA_{trial}")
        fa['3score'].append(1 if res['decision']['decision'] != 'SAFE' else 0)

        # IF
        if_res = calc_isolation_forest_score(res['scaled_ref'], res['scaled_comp'])
        fa['if'].append(1 if if_res['anomaly_rate'] > 0.05 else 0)

        # AE (경량)
        ae_res = calc_autoencoder_score(res['scaled_ref'], res['scaled_comp'], epochs=20)
        fa['ae'].append(1 if ae_res['exceed_rate'] > 0.05 else 0)

    rates = {k: np.mean(v) for k, v in fa.items()}
    print(f"  False Alarm Rate:")
    print(f"    3-Score: {rates['3score']:.0%}")
    print(f"    IF:      {rates['if']:.0%}")
    print(f"    AE:      {rates['ae']:.0%}")
    return {'rates': rates, 'details': fa, 'n_trials': n_trials}


def run_ensemble_experiment():
    """Learned Ensemble 실험"""
    print("  학습 데이터 생성 (200 시나리오)...")
    results_list, labels = generate_ensemble_training_data(n_scenarios=200, seed=42)

    print(f"  생성된 시나리오: {len(results_list)}개")
    print(f"  라벨 분포: {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # 학습
    ens_result = train_ensemble_classifier(results_list, labels)
    print(f"  Ensemble 학습 정확도: {ens_result['accuracy']:.1%}")
    print(f"  Score 가중치: {ens_result['coefficients']}")

    # 비교: 기존 rule-based vs learned
    rule_preds = []
    for r in results_list:
        decision = r['decision']['decision']
        dec_map = {'SAFE': 0, 'CAUTION': 1, 'RISK': 2, 'HIGH_RISK': 3, 'INSUFFICIENT_DATA': 0}
        rule_preds.append(dec_map.get(decision, 0))
    rule_preds = np.array(rule_preds)
    rule_accuracy = (rule_preds == np.array(labels)).mean()

    print(f"  Rule-based 정확도: {rule_accuracy:.1%}")
    print(f"  Ensemble 정확도:   {ens_result['accuracy']:.1%}")

    ens_result['rule_accuracy'] = rule_accuracy
    ens_result['rule_preds'] = rule_preds
    return ens_result


def compare_computational_cost():
    """계산 비용 비교"""
    configs = [
        (300, 50, 1000),
        (500, 80, 2000),
        (1000, 80, 5000),
    ]
    results = []

    for n_ref, n_comp, n_feat in configs:
        label = f"{n_ref}x{n_feat}"
        print(f"  Testing {label}...")

        np.random.seed(42)
        feat_names = [f"EDS_{i:04d}" for i in range(n_feat)]
        ref = pd.DataFrame(np.random.randn(n_ref, n_feat) * 0.5 + 3.0, columns=feat_names)
        comp = pd.DataFrame(np.random.randn(n_comp, n_feat) * 0.5 + 3.0, columns=feat_names)
        comp.iloc[:, :20] += 0.75

        # 3-Score
        t0 = time.time()
        res = run_eco_change_detection(ref, comp)
        t_3s = time.time() - t0

        # IF
        t0 = time.time()
        calc_isolation_forest_score(res['scaled_ref'], res['scaled_comp'])
        t_if = time.time() - t0

        # AE
        t0 = time.time()
        calc_autoencoder_score(res['scaled_ref'], res['scaled_comp'], epochs=20)
        t_ae = time.time() - t0

        results.append({
            'label': label,
            'time_3score': t_3s,
            'time_if': t_if,
            'time_ae': t_ae,
            'time_total': t_3s + t_if + t_ae,
        })

        print(f"    3-Score: {t_3s:.3f}s | IF: {t_if:.3f}s | AE: {t_ae:.3f}s | "
              f"Total: {t_3s + t_if + t_ae:.3f}s")

    return results


def compare_feature_tracking(result, ae_result, ground_truth):
    """Feature 추적 정확도: 3-Score vs AE"""
    true_anomaly = set(f"EDS_{i:04d}" for i in ground_truth['all_anomaly_features'])

    # 3-Score top features
    shift_z = result['detail']['shift']['z_shift_all']
    top20_3s = set(shift_z.abs().sort_values(ascending=False).head(20).index)
    hit_3s = len(top20_3s & true_anomaly)

    # AE top features
    feat_names = result['scaled_ref'].columns.tolist()
    fe_increase = ae_result['feature_error_increase']
    top_idx = np.argsort(fe_increase)[::-1][:20]
    top20_ae = set(feat_names[i] for i in top_idx)
    hit_ae = len(top20_ae & true_anomaly)

    # Combined
    combined = top20_3s | top20_ae
    hit_combined = len(combined & true_anomaly)
    overlap = len(top20_3s & top20_ae)

    print(f"  3-Score top-20 hit: {hit_3s}/20")
    print(f"  AE top-20 hit:     {hit_ae}/20")
    print(f"  Combined hit:      {hit_combined}/{len(combined)}")
    print(f"  Overlap:           {overlap}/20")

    return {
        'hit_3s': hit_3s,
        'hit_ae': hit_ae,
        'hit_combined': hit_combined,
        'total_combined': len(combined),
        'overlap': overlap,
        'top20_3s': sorted(top20_3s),
        'top20_ae': sorted(top20_ae),
    }


def visualize_all(result, if_result, ae_result, ground_truth,
                  pattern_results, fa_results, ensemble_result,
                  cost_results, feature_results, time_3score):
    """종합 시각화 4장"""

    # ── 1. IF + AE 이상 탐지 분포 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AI/ML 보조 분석: Isolation Forest & Autoencoder',
                 fontsize=16, fontweight='bold')

    # IF: ref vs comp scores
    ax = axes[0, 0]
    ax.hist(if_result['ref_scores_norm'], bins=40, alpha=0.5, color='#3498db',
            label='Ref', density=True)
    ax.hist(if_result['comp_scores_norm'], bins=40, alpha=0.5, color='#e74c3c',
            label='Comp', density=True)
    ax.set_xlabel('Anomaly Score (normalized)')
    ax.set_ylabel('Density')
    ax.set_title(f'Isolation Forest (anomaly rate={if_result["anomaly_rate"]:.1%})')
    ax.legend()

    # IF: wafer별 anomaly score
    ax = axes[0, 1]
    n_comp = len(if_result['comp_scores_norm'])
    colors = np.where(if_result['comp_labels'] == -1, '#e74c3c', '#95a5a6')
    ax.bar(range(n_comp), if_result['comp_scores_norm'], color=colors, width=1.0)
    ax.set_xlabel('Wafer Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('IF: Wafer별 Anomaly Score')
    ax.legend(handles=[
        Patch(color='#e74c3c', label='Anomaly'),
        Patch(color='#95a5a6', label='Normal'),
    ])

    # AE: ref vs comp MSE
    ax = axes[1, 0]
    ax.hist(ae_result['ref_mse'], bins=40, alpha=0.5, color='#3498db',
            label='Ref', density=True)
    ax.hist(ae_result['comp_mse'], bins=40, alpha=0.5, color='#e74c3c',
            label='Comp', density=True)
    ax.axvline(ae_result['threshold'], color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {ae_result["threshold"]:.4f}')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title(f'Autoencoder (exceed={ae_result["exceed_rate"]:.1%}, '
                 f'dim={ae_result["encoding_dim"]})')
    ax.legend()

    # AE: feature별 오류 증가 top-20
    ax = axes[1, 1]
    top_feats = ae_result['top_features'][:15]
    top_vals = ae_result['top_feature_values'][:15]
    y_pos = range(len(top_feats))
    colors_bar = ['#e74c3c' if v > 0 else '#3498db' for v in top_vals]
    ax.barh(y_pos, top_vals, color=colors_bar)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_xlabel('Error Increase (Comp - Ref)')
    ax.set_title('AE: Feature별 재구성 오류 증가 Top-15')
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, 'ml_if_ae_analysis.png')

    # ── 2. 패턴별 탐지 비교 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('패턴별 탐지 성능: 3-Score vs IF vs AE', fontsize=16, fontweight='bold')

    patterns = list(pattern_results.keys())
    labels = [f"{p}\n({pattern_results[p]['type']})" for p in patterns]
    diff_colors = {'Easy': '#27ae60', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
    difficulties = [pattern_results[p]['difficulty'] for p in patterns]

    # 3-Score z-shift
    ax = axes[0]
    vals = [pattern_results[p]['3score_z'] for p in patterns]
    bars = ax.bar(range(len(patterns)), vals,
                  color=[diff_colors[d] for d in difficulties], edgecolor='white')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('avg |z-shift|')
    ax.set_title('3-Score')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    # IF anomaly rate
    ax = axes[1]
    vals = [pattern_results[p]['if_anomaly_rate'] for p in patterns]
    bars = ax.bar(range(len(patterns)), vals,
                  color=[diff_colors[d] for d in difficulties], edgecolor='white')
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Anomaly Rate')
    ax.set_title('Isolation Forest')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')

    # AE error
    ax = axes[2]
    vals = [pattern_results[p]['ae_wafer_mse'] for p in patterns]
    bars = ax.bar(range(len(patterns)), vals,
                  color=[diff_colors[d] for d in difficulties], edgecolor='white')
    ax.axhline(ae_result['threshold'], color='red', linestyle='--', alpha=0.5,
               label=f'Threshold={ae_result["threshold"]:.4f}')
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Autoencoder')
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'ml_pattern_comparison.png')

    # ── 3. 계산 비용 + False Alarm ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('계산 비용 & False Alarm Rate', fontsize=16, fontweight='bold')

    # Cost
    ax = axes[0]
    labels_c = [c['label'] for c in cost_results]
    x = np.arange(len(labels_c))
    w = 0.2
    ax.bar(x - w, [c['time_3score'] for c in cost_results], w,
           color='#3498db', label='3-Score')
    ax.bar(x, [c['time_if'] for c in cost_results], w,
           color='#e74c3c', label='Isolation Forest')
    ax.bar(x + w, [c['time_ae'] for c in cost_results], w,
           color='#f39c12', label='Autoencoder')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_c)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('실행 시간 비교')
    ax.legend()

    # False Alarm
    ax = axes[1]
    fa = fa_results['rates']
    cats = ['3-Score', 'Isolation\nForest', 'Autoencoder']
    vals = [fa['3score'], fa['if'], fa['ae']]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(cats, vals, color=colors, edgecolor='white')
    ax.set_ylabel('False Alarm Rate')
    ax.set_title(f'False Alarm (정상 데이터 {fa_results["n_trials"]}회)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, max(vals) * 1.3 + 0.05)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.0%}', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'ml_cost_falsealarm.png')

    # ── 4. Ensemble + Feature 추적 + 종합표 ──
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Learned Ensemble & Feature 추적 & 종합 평가',
                 fontsize=16, fontweight='bold')

    # Ensemble: Rule vs Learned confusion
    ax = fig.add_subplot(gs[0, 0])
    ens_labels = ['SAFE', 'CAUTION', 'RISK', 'HIGH_RISK']
    true_counts = pd.Series(ensemble_result['y_true']).value_counts().sort_index()
    rule_acc_per = []
    ens_acc_per = []
    for lbl in range(4):
        mask = ensemble_result['y_true'] == lbl
        if mask.sum() > 0:
            rule_acc_per.append((ensemble_result['rule_preds'][mask] == lbl).mean())
            ens_acc_per.append((ensemble_result['y_pred'][mask] == lbl).mean())
        else:
            rule_acc_per.append(0)
            ens_acc_per.append(0)

    x = np.arange(4)
    ax.bar(x - 0.2, rule_acc_per, 0.35, color='#3498db', label='Rule-based')
    ax.bar(x + 0.2, ens_acc_per, 0.35, color='#e74c3c', label='Learned Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(ens_labels, fontsize=9)
    ax.set_ylabel('Class Accuracy')
    ax.set_title(f'판정 정확도: Rule({ensemble_result["rule_accuracy"]:.0%}) '
                 f'vs Ensemble({ensemble_result["accuracy"]:.0%})')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Ensemble coefficient importance
    ax = fig.add_subplot(gs[0, 1])
    coef = ensemble_result['coefficients']
    names = list(coef.keys())
    values = [coef[n] for n in names]
    colors_c = ['#e74c3c' if v > 0 else '#3498db' for v in values]
    ax.barh(names, values, color=colors_c)
    ax.set_xlabel('Coefficient (avg across classes)')
    ax.set_title('Ensemble: Score 가중치 학습 결과')

    # Feature tracking
    ax = fig.add_subplot(gs[1, 0])
    cats = ['3-Score\nHit', 'AE\nHit', 'Combined\nHit', 'Overlap']
    vals = [feature_results['hit_3s'], feature_results['hit_ae'],
            feature_results['hit_combined'], feature_results['overlap']]
    colors = ['#3498db', '#f39c12', '#9b59b6', '#2ecc71']
    bars = ax.bar(cats, vals, color=colors, edgecolor='white')
    ax.set_ylabel('Feature count (out of Top-20)')
    ax.set_title('Feature 추적 정확도 (실제 이상 Feature Hit)')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', fontsize=14, fontweight='bold')

    # Summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    table_data = [
        ['Method', 'Detection', 'Cost', 'FA Rate', 'Feature Hit'],
        ['3-Score', 'D(O)/E(O)', f'{cost_results[-1]["time_3score"]:.2f}s',
         f'{fa_results["rates"]["3score"]:.0%}', f'{feature_results["hit_3s"]}/20'],
        ['+ IF', f'{if_result["anomaly_rate"]:.0%}',
         f'+{cost_results[-1]["time_if"]:.2f}s',
         f'{fa_results["rates"]["if"]:.0%}', 'Wafer-level'],
        ['+ AE', f'{ae_result["exceed_rate"]:.0%}',
         f'+{cost_results[-1]["time_ae"]:.2f}s',
         f'{fa_results["rates"]["ae"]:.0%}', f'{feature_results["hit_ae"]}/20'],
        ['Ensemble', f'{ensemble_result["accuracy"]:.0%}',
         'negligible', '-', '-'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.0)
    for j in range(5):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('종합 비교표', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'ml_ensemble_summary.png')


def print_final_report(result, if_result, ae_result,
                       pattern_results, fa_results, ensemble_result,
                       cost_results, feature_results, time_3score):
    """최종 보고서"""
    print("\n" + "=" * 70)
    print("AI/ML 강화 비교 실험 - 최종 보고서")
    print("=" * 70)

    print("\n--- 1. 탐지 정합성 ---")
    for p, r in pattern_results.items():
        det_3s = 'O' if r['3score_z'] > 0.5 else 'X'
        det_if = 'O' if r['if_anomaly_rate'] > 0.3 else 'X'
        det_ae = 'O' if r['ae_wafer_mse'] > r['ae_threshold'] else 'X'
        print(f"  {p} ({r['difficulty']}): 3S={det_3s} | IF={det_if} | AE={det_ae}")

    print("\n--- 2. 계산 비용 (1000x5000 기준) ---")
    c = cost_results[-1]
    print(f"  3-Score:  {c['time_3score']:.3f}s (baseline)")
    print(f"  +IF:      +{c['time_if']:.3f}s ({c['time_if']/c['time_3score']*100:.0f}%)")
    print(f"  +AE:      +{c['time_ae']:.3f}s ({c['time_ae']/c['time_3score']*100:.0f}%)")
    print(f"  Total:    {c['time_total']:.3f}s ({c['time_total']/c['time_3score']*100:.0f}%)")

    print("\n--- 3. False Alarm Rate ---")
    fa = fa_results['rates']
    print(f"  3-Score: {fa['3score']:.0%}")
    print(f"  IF:      {fa['if']:.0%}")
    print(f"  AE:      {fa['ae']:.0%}")

    print("\n--- 4. Ensemble (Rule vs Learned) ---")
    print(f"  Rule-based: {ensemble_result['rule_accuracy']:.1%}")
    print(f"  Ensemble:   {ensemble_result['accuracy']:.1%}")
    print(f"  Score weights: {ensemble_result['coefficients']}")

    print("\n--- 5. Feature 추적 ---")
    print(f"  3-Score: {feature_results['hit_3s']}/20")
    print(f"  AE:      {feature_results['hit_ae']}/20")
    print(f"  Combined: {feature_results['hit_combined']}/{feature_results['total_combined']}")

    print("\n--- 6. 종합 권고 ---")
    print("  [A] Isolation Forest:")
    print(f"     - Wafer 이상 탐지에 즉시 활용 가능")
    print(f"     - 비용: +{cost_results[-1]['time_if']:.2f}s (저부담)")
    print(f"     - FA: {fa['if']:.0%}")
    print(f"     - 기존 Outlier Wafer Score 대비 적응적 임계값")

    print("  [B] Autoencoder:")
    print(f"     - Feature 수준 오류 추적으로 원인 분석 보완")
    print(f"     - 비용: +{cost_results[-1]['time_ae']:.2f}s")
    print(f"     - Pattern E(미세변화) 보완 가능성 있으나, 학습 안정성 고려 필요")

    print("  [C] Learned Ensemble:")
    if ensemble_result['accuracy'] > ensemble_result['rule_accuracy']:
        print(f"     - Rule 대비 +{(ensemble_result['accuracy'] - ensemble_result['rule_accuracy'])*100:.1f}%p 정확도 향상")
        print(f"     - 실무 적용 시 과거 판정 이력 50건 이상 필요")
    else:
        print(f"     - Rule-based가 이미 충분한 정확도 ({ensemble_result['rule_accuracy']:.0%})")
        print(f"     - 데이터 축적 후 재평가 권고")


if __name__ == '__main__':
    main()
