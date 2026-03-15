"""
ECO Change Detection Pipeline
반도체 공정 변경점(ECO) 적용 전후의 품질 차이를 자동으로 정량화하고,
차이의 주요 원인 Feature를 식별하는 분석 파이프라인.
"""

import pandas as pd
import numpy as np


# ============================================================
# Step 1: 전처리
# ============================================================

def filter_features(df_ref, df_comp, missing_thresh=0.3, var_thresh=1e-10):
    """결측률/상수 변수 제거"""
    missing_rate = df_ref.isnull().mean()
    valid_features = missing_rate[missing_rate < missing_thresh].index

    variance = df_ref[valid_features].var()
    valid_features = variance[variance > var_thresh].index

    return valid_features.tolist()


def robust_scale(df_ref, df_comp, features):
    """ref 기준 robust scaling (median / IQR)"""
    median_ref = df_ref[features].median()
    iqr_ref = df_ref[features].quantile(0.75) - df_ref[features].quantile(0.25)
    iqr_ref = iqr_ref.replace(0, 1e-10)

    scaled_ref = (df_ref[features] - median_ref) / iqr_ref
    scaled_comp = (df_comp[features] - median_ref) / iqr_ref

    params = {"median": median_ref, "iqr": iqr_ref}
    return scaled_ref, scaled_comp, params


def winsorize(scaled_ref, scaled_comp, lower=0.005, upper=0.995):
    """극단적 measurement 오류 제거"""
    lower_bound = scaled_ref.quantile(lower)
    upper_bound = scaled_ref.quantile(upper)
    clipped_ref = scaled_ref.clip(lower_bound, upper_bound, axis=1)
    clipped_comp = scaled_comp.clip(lower_bound, upper_bound, axis=1)
    return clipped_ref, clipped_comp


# ============================================================
# Step 2: Score 산출 (3종)
# ============================================================

def calc_shift_score(scaled_ref, scaled_comp, top_k_ratio=0.01):
    """Shift Score: 중심치/산포 이동 탐지"""
    mean_ref = scaled_ref.mean()
    mean_comp = scaled_comp.mean()
    std_ref = scaled_ref.std()
    std_ref = std_ref.replace(0, 1e-10)

    z_shift = (mean_comp - mean_ref) / std_ref
    abs_z = z_shift.abs().sort_values(ascending=False)

    top_k = max(1, int(len(abs_z) * top_k_ratio))
    shift_score = abs_z.iloc[:top_k].mean()

    return {
        "score": shift_score,
        "z_shift_all": z_shift,
        "top_features": abs_z.head(top_k)
    }


def calc_tail_score(scaled_ref, scaled_comp, percentile=0.99):
    """Tail Score: 간헐적 극단값 증가 탐지"""
    threshold = scaled_ref.quantile(percentile)
    n_comp = len(scaled_comp)
    tail_rate = (scaled_comp > threshold).sum() / n_comp
    expected_rate = 1 - percentile

    elevated_features = tail_rate[tail_rate > expected_rate * 3].sort_values(ascending=False)

    tail_score_max = tail_rate.max()
    tail_score_count = len(elevated_features)

    return {
        "score_max": tail_score_max,
        "score_count": tail_score_count,
        "tail_rate_all": tail_rate,
        "elevated_features": elevated_features
    }


def calc_outlier_wafer_score(scaled_ref, scaled_comp, percentile=0.99,
                              feature_exceed_ratio=0.05):
    """Outlier Wafer Score: wafer 단위 이상 탐지"""
    threshold = scaled_ref.quantile(percentile)

    exceed_count = (scaled_comp > threshold).sum(axis=1)
    total_features = scaled_comp.shape[1]
    exceed_ratio = exceed_count / total_features

    outlier_mask = exceed_ratio > feature_exceed_ratio
    outlier_wafer_ids = scaled_comp.index[outlier_mask].tolist()
    outlier_rate = outlier_mask.mean()

    if len(outlier_wafer_ids) > 0:
        outlier_data = scaled_comp.loc[outlier_mask]
        common_features = (outlier_data > threshold).mean().sort_values(ascending=False)
    else:
        common_features = pd.Series(dtype=float)

    return {
        "score": outlier_rate,
        "outlier_count": outlier_mask.sum(),
        "outlier_wafer_ids": outlier_wafer_ids,
        "exceed_ratio_per_wafer": exceed_ratio,
        "common_features": common_features.head(20)
    }


# ============================================================
# Step 3: Feature Importance 추적
# ============================================================

def get_feature_importance(shift_result, tail_result, outlier_result, top_n=20):
    """3종 Score의 원인 Feature를 통합 정리"""
    z_shift = shift_result["z_shift_all"]
    shift_top = z_shift.abs().sort_values(ascending=False).head(top_n)
    shift_importance = pd.DataFrame({
        "feature": shift_top.index,
        "abs_z_shift": shift_top.values,
        "z_shift": z_shift[shift_top.index].values,
        "direction": ["악화" if z_shift[f] > 0 else "개선" for f in shift_top.index]
    })

    tail_elevated = tail_result["elevated_features"].head(top_n)
    tail_importance = pd.DataFrame({
        "feature": tail_elevated.index,
        "tail_rate": tail_elevated.values,
        "tail_rate_pct": (tail_elevated.values * 100).round(1)
    })

    common = outlier_result["common_features"].head(top_n)
    outlier_importance = pd.DataFrame({
        "feature": common.index,
        "outlier_exceed_rate": common.values
    })

    return {
        "shift_features": shift_importance,
        "tail_features": tail_importance,
        "outlier_features": outlier_importance
    }


# ============================================================
# Step 4: 판정
# ============================================================

def make_decision(shift_result, tail_result, outlier_result, comp_count,
                  min_sample=30):
    """3종 Score 기반 판정"""
    if comp_count < min_sample:
        return {
            "decision": "INSUFFICIENT_DATA",
            "confidence": "LOW",
            "risk_level": -1,
            "reason": f"compare wafer 수({comp_count})가 최소 기준({min_sample}) 미달",
            "reasons": [f"compare wafer 수({comp_count})가 최소 기준({min_sample}) 미달"],
            "scores": {
                "shift_score": round(shift_result["score"], 3),
                "tail_score_max": round(tail_result["score_max"], 4),
                "tail_feature_count": tail_result["score_count"],
                "outlier_wafer_rate": round(outlier_result["score"], 4)
            }
        }

    shift_score = shift_result["score"]
    tail_max = tail_result["score_max"]
    tail_count = tail_result["score_count"]
    outlier_rate = outlier_result["score"]

    reasons = []
    risk_level = 0

    if shift_score > 2.0:
        risk_level = max(risk_level, 3)
        reasons.append(f"Shift Score {shift_score:.2f}: 상위 feature에서 평균 2σ 이상 이동")
    elif shift_score > 1.0:
        risk_level = max(risk_level, 2)
        reasons.append(f"Shift Score {shift_score:.2f}: 상위 feature에서 유의미한 이동")
    elif shift_score > 0.5:
        risk_level = max(risk_level, 1)
        reasons.append(f"Shift Score {shift_score:.2f}: 소폭 이동 관찰")

    if tail_max > 0.10:
        risk_level = max(risk_level, 3)
        reasons.append(f"Tail Score {tail_max:.1%}: 심각한 tail 증가 (feature {tail_count}개)")
    elif tail_max > 0.05:
        risk_level = max(risk_level, 2)
        reasons.append(f"Tail Score {tail_max:.1%}: 간헐적 이상 발생 (feature {tail_count}개)")
    elif tail_max > 0.03:
        risk_level = max(risk_level, 1)
        reasons.append(f"Tail Score {tail_max:.1%}: 소수 feature에서 tail 증가")

    if outlier_rate > 0.10:
        risk_level = max(risk_level, 3)
        reasons.append(f"Outlier Rate {outlier_rate:.1%}: 다수 wafer 다변량 이상")
    elif outlier_rate > 0.05:
        risk_level = max(risk_level, 2)
        reasons.append(f"Outlier Rate {outlier_rate:.1%}: 일부 wafer 이상")
    elif outlier_rate > 0.03:
        risk_level = max(risk_level, 1)
        reasons.append(f"Outlier Rate {outlier_rate:.1%}: 소수 wafer 이상")

    decision_map = {0: "SAFE", 1: "CAUTION", 2: "RISK", 3: "HIGH_RISK"}
    decision = decision_map.get(risk_level, "SAFE")

    if not reasons:
        reasons.append("모든 Score가 정상 범위 내")

    return {
        "decision": decision,
        "risk_level": risk_level,
        "reasons": reasons,
        "scores": {
            "shift_score": round(shift_score, 3),
            "tail_score_max": round(tail_max, 4),
            "tail_feature_count": tail_count,
            "outlier_wafer_rate": round(outlier_rate, 4)
        }
    }


# ============================================================
# Main Pipeline
# ============================================================

def run_eco_change_detection(df_ref, df_comp,
                              step_id="UNKNOWN",
                              change_code="UNKNOWN",
                              top_k_ratio=0.01,
                              tail_percentile=0.99,
                              outlier_feature_thresh=0.05,
                              min_sample=30,
                              top_n_features=20):
    """ECO 변경점 검증 메인 파이프라인"""
    # Step 1: 전처리
    features = filter_features(df_ref, df_comp)
    scaled_ref, scaled_comp, scale_params = robust_scale(df_ref, df_comp, features)
    scaled_ref, scaled_comp = winsorize(scaled_ref, scaled_comp)

    # Step 2: Score 산출
    shift_result = calc_shift_score(scaled_ref, scaled_comp, top_k_ratio)
    tail_result = calc_tail_score(scaled_ref, scaled_comp, tail_percentile)
    outlier_result = calc_outlier_wafer_score(scaled_ref, scaled_comp,
                                              tail_percentile, outlier_feature_thresh)

    # Step 3: Feature Importance
    importance = get_feature_importance(shift_result, tail_result,
                                        outlier_result, top_n_features)

    # Step 4: 판정
    decision = make_decision(shift_result, tail_result, outlier_result,
                              len(df_comp), min_sample)

    result = {
        "metadata": {
            "step_id": step_id,
            "change_code": change_code,
            "ref_count": len(df_ref),
            "comp_count": len(df_comp),
            "feature_count": len(features),
            "feature_count_original": df_ref.shape[1]
        },
        "scores": decision["scores"],
        "decision": decision,
        "importance": importance,
        "detail": {
            "shift": shift_result,
            "tail": tail_result,
            "outlier": outlier_result
        },
        "params": {
            "scale_params": scale_params,
            "top_k_ratio": top_k_ratio,
            "tail_percentile": tail_percentile,
            "outlier_feature_thresh": outlier_feature_thresh
        },
        "scaled_ref": scaled_ref,
        "scaled_comp": scaled_comp
    }

    return result


# ============================================================
# Synthetic Data Generator (v2: 5가지 패턴, 난이도별)
# ============================================================

def generate_synthetic_data(n_ref=1000, n_comp=80, n_features=5000, seed=42):
    """5가지 불량 패턴을 시뮬레이션한 합성 데이터 생성 (난이도별 분류)

    Returns:
        df_ref, df_comp, ground_truth
        ground_truth: dict with pattern info and feature indices
    """
    np.random.seed(seed)
    feature_names = [f"EDS_{i:04d}" for i in range(n_features)]

    ref_data = np.random.randn(n_ref, n_features) * 0.5 + 3.0
    comp_data = np.random.randn(n_comp, n_features) * 0.5 + 3.0

    # Pattern A: Systematic shift (Feature 100~120) - Easy
    shift_features = list(range(100, 121))
    comp_data[:, shift_features] += 0.75  # +1.5σ shift (std=0.5)

    # Pattern B: Intermittent spike (Feature 500~510) - Easy
    spike_features = list(range(500, 511))
    spike_wafers = np.random.choice(n_comp, size=int(n_comp * 0.10), replace=False)
    comp_data[np.ix_(spike_wafers, spike_features)] += 3.0

    # Pattern C: Multi-feature outlier wafers (Wafer 70~74) - Easy
    outlier_wafers = list(range(70, min(75, n_comp)))
    outlier_features = list(range(200, 500))  # 300개 feature (6%)에 걸쳐 이상
    for w in outlier_wafers:
        if w < n_comp:
            comp_data[w, outlier_features] += 2.0

    # Pattern D: Gradual Trend (Feature 600~610) - Medium
    trend_features = list(range(600, 611))
    for i, w in enumerate(range(n_comp)):
        drift = (i / n_comp) * 1.0  # 0 -> 1.0 점진적 증가
        comp_data[w, trend_features] += drift

    # Pattern E: Subtle Shift (Feature 700~720) - Hard
    subtle_features = list(range(700, 721))
    comp_data[:, subtle_features] += 0.25  # +0.5σ shift (미세)

    df_ref = pd.DataFrame(ref_data, columns=feature_names,
                          index=[f"ref_w{i:04d}" for i in range(n_ref)])
    df_comp = pd.DataFrame(comp_data, columns=feature_names,
                           index=[f"comp_w{i:04d}" for i in range(n_comp)])

    # Ground truth 정보
    all_anomaly_features = set(shift_features + spike_features + outlier_features
                               + trend_features + subtle_features)
    ground_truth = {
        "pattern_A": {
            "type": "Systematic Shift",
            "difficulty": "Easy",
            "features": shift_features,
            "description": "F100~120, +1.5σ, 전체 wafer"
        },
        "pattern_B": {
            "type": "Intermittent Spike",
            "difficulty": "Easy",
            "features": spike_features,
            "wafers": spike_wafers.tolist(),
            "description": "F500~510, 10% wafer, +6σ spike"
        },
        "pattern_C": {
            "type": "Multi-Feature Outlier",
            "difficulty": "Easy",
            "features": outlier_features,
            "wafers": outlier_wafers,
            "description": "F200~499, W70~74, +4σ 동시 이상"
        },
        "pattern_D": {
            "type": "Gradual Trend",
            "difficulty": "Medium",
            "features": trend_features,
            "description": "F600~610, 점진적 0→+2σ drift"
        },
        "pattern_E": {
            "type": "Subtle Shift",
            "difficulty": "Hard",
            "features": subtle_features,
            "description": "F700~720, +0.5σ 미세 이동"
        },
        "all_anomaly_features": sorted(all_anomaly_features)
    }

    return df_ref, df_comp, ground_truth


# ============================================================
# Step 5: 변경 유형 분류 및 단계 평가 (Enhanced Analysis)
# ============================================================

def calc_confidence_score(comp_count, min_reliable=30, full_reliable=100):
    """비교 데이터 수량 기반 신뢰도 점수 산출

    비교(comp) 웨이퍼 수에 따라 분석 결과의 신뢰 수준을 LOW/MEDIUM/HIGH로
    판정하고, 0~1 사이의 수치 점수를 반환한다.

    Parameters:
        comp_count: 비교 웨이퍼 수
        min_reliable: 최소 신뢰 기준 (이 이상이면 MEDIUM)
        full_reliable: 완전 신뢰 기준 (이 이상이면 HIGH)

    Returns:
        dict: confidence_level (str), confidence_score (float 0-1)
    """
    if comp_count < min_reliable:
        level = "LOW"
        score = comp_count / min_reliable
    elif comp_count < full_reliable:
        level = "MEDIUM"
        score = 0.5 + 0.5 * (comp_count - min_reliable) / (full_reliable - min_reliable)
    else:
        level = "HIGH"
        score = 1.0

    return {
        "confidence_level": level,
        "confidence_score": round(score, 4)
    }


def calc_bootstrap_stability(scaled_ref, scaled_comp, n_iterations=100,
                              top_k_ratio=0.01, seed=42):
    """부트스트랩 리샘플링 기반 Shift Score 안정성 평가

    비교 데이터를 반복 리샘플링하여 Shift Score의 분포를 추정하고,
    평균·표준편차·95% 신뢰구간·변동계수(CV)를 반환한다.
    CV가 낮을수록 분석 결과가 안정적임을 의미한다.

    Parameters:
        scaled_ref: 전처리된 ref 데이터 (DataFrame)
        scaled_comp: 전처리된 comp 데이터 (DataFrame)
        n_iterations: 부트스트랩 반복 횟수
        top_k_ratio: Shift Score 산출 시 상위 feature 비율
        seed: 난수 시드

    Returns:
        dict: mean_score, std_score, ci_lower, ci_upper, cv
    """
    rng = np.random.RandomState(seed)
    n_comp = len(scaled_comp)
    scores = []

    mean_ref = scaled_ref.mean()
    std_ref = scaled_ref.std().replace(0, 1e-10)
    top_k = max(1, int(len(scaled_ref.columns) * top_k_ratio))

    for _ in range(n_iterations):
        # 비교 데이터에서 복원 추출
        sample_idx = rng.choice(n_comp, size=n_comp, replace=True)
        sample_comp = scaled_comp.iloc[sample_idx]

        mean_comp = sample_comp.mean()
        z_shift = (mean_comp - mean_ref) / std_ref
        abs_z = z_shift.abs().sort_values(ascending=False)
        shift_score = abs_z.iloc[:top_k].mean()
        scores.append(shift_score)

    scores = np.array(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    ci_lower = float(np.percentile(scores, 2.5))
    ci_upper = float(np.percentile(scores, 97.5))
    cv = float(std_score / mean_score) if mean_score > 0 else 0.0

    return {
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "cv": round(cv, 4)
    }


def calc_global_deviation_score(scaled_ref, scaled_comp, n_components=30):
    """PCA 기반 마할라노비스 거리를 이용한 전역 편차 점수 산출

    ref 데이터에 PCA를 적합하고, PC 공간에서 ref 중심으로부터의
    마할라노비스 거리를 산출하여 comp 웨이퍼의 전역적 이탈 정도를 평가한다.
    이상치 비율(outlier_rate)은 ref 99번째 백분위수를 초과하는 comp 비율이다.

    Parameters:
        scaled_ref: 전처리된 ref 데이터 (DataFrame)
        scaled_comp: 전처리된 comp 데이터 (DataFrame)
        n_components: PCA 성분 수 (자동 조정됨)

    Returns:
        dict: mean_distance, median_distance, p90_distance,
              ref_distances, comp_distances, outlier_rate, n_components_used
    """
    from sklearn.decomposition import PCA

    ref_values = scaled_ref.fillna(0).values
    comp_values = scaled_comp.fillna(0).values

    # 성분 수 자동 조정
    n_comp_actual = min(n_components, ref_values.shape[0] - 1, ref_values.shape[1])
    n_comp_actual = max(1, n_comp_actual)

    pca = PCA(n_components=n_comp_actual)
    pca.fit(ref_values)

    ref_scores = pca.transform(ref_values)
    comp_scores = pca.transform(comp_values)

    # ref PC 공간에서 평균 벡터 및 공분산 산출
    ref_mean = np.mean(ref_scores, axis=0)
    ref_cov = np.cov(ref_scores, rowvar=False)

    # 공분산 행렬 역행렬 (특이행렬 방지를 위해 정규화)
    try:
        if ref_cov.ndim == 0:
            # 성분이 1개인 경우 스칼라 처리
            ref_cov = np.array([[float(ref_cov)]])
        reg = np.eye(ref_cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(ref_cov + reg)
    except np.linalg.LinAlgError:
        # 역행렬 계산 실패 시 의사역행렬 사용
        cov_inv = np.linalg.pinv(ref_cov)

    # 마할라노비스 거리 산출
    def _mahal_distances(scores, mean_vec, cov_inv_mat):
        diff = scores - mean_vec
        left = diff @ cov_inv_mat
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    ref_distances = _mahal_distances(ref_scores, ref_mean, cov_inv)
    comp_distances = _mahal_distances(comp_scores, ref_mean, cov_inv)

    # ref 99번째 백분위수 기준 이상치 비율
    ref_p99 = float(np.percentile(ref_distances, 99))
    outlier_rate = float(np.mean(comp_distances > ref_p99))

    return {
        "mean_distance": round(float(np.mean(comp_distances)), 4),
        "median_distance": round(float(np.median(comp_distances)), 4),
        "p90_distance": round(float(np.percentile(comp_distances, 90)), 4),
        "ref_distances": ref_distances,
        "comp_distances": comp_distances,
        "outlier_rate": round(outlier_rate, 4),
        "n_components_used": n_comp_actual
    }


def make_enhanced_decision(shift_result, tail_result, outlier_result, comp_count,
                            change_type="efficiency", target_features=None,
                            min_sample=30):
    """변경 유형별 다단계 판정 (효율 동등성 검증 / 불량 개선 검증)

    기존 make_decision의 확장판으로, 변경의 목적(효율 유지 vs 불량 개선)에 따라
    단계(Stage 0~4)를 부여하고 한국어 설명과 함께 반환한다.

    Parameters:
        shift_result: calc_shift_score 결과
        tail_result: calc_tail_score 결과
        outlier_result: calc_outlier_wafer_score 결과
        comp_count: 비교 웨이퍼 수
        change_type: "efficiency" (효율 동등성) 또는 "defect_improvement" (불량 개선)
        target_features: 개선 대상 feature 이름 목록 (defect_improvement 시 필수)
        min_sample: 최소 샘플 수

    Returns:
        dict: stage, stage_name, change_type, description, reasons, scores,
              target_assessment (defect_improvement 전용)
    """
    shift_score = shift_result["score"]
    tail_max = tail_result["score_max"]
    outlier_rate = outlier_result["score"]

    scores = {
        "shift_score": round(float(shift_score), 3),
        "tail_score_max": round(float(tail_max), 4),
        "tail_feature_count": tail_result["score_count"],
        "outlier_wafer_rate": round(float(outlier_rate), 4)
    }

    # Stage 0: 데이터 부족
    if comp_count < min_sample:
        return {
            "stage": 0,
            "stage_name": "INSUFFICIENT_DATA",
            "change_type": change_type,
            "description": f"비교 웨이퍼 수({comp_count})가 최소 기준({min_sample}) 미달로 판정 불가",
            "reasons": [f"비교 웨이퍼 수({comp_count})가 최소 기준({min_sample})에 미달합니다."],
            "scores": scores
        }

    if change_type == "efficiency":
        return _decide_efficiency(shift_score, tail_max, outlier_rate,
                                   comp_count, scores)
    elif change_type == "defect_improvement":
        return _decide_defect_improvement(shift_result, tail_result, outlier_result,
                                           comp_count, target_features, scores)
    else:
        return {
            "stage": -1,
            "stage_name": "UNKNOWN_CHANGE_TYPE",
            "change_type": change_type,
            "description": f"알 수 없는 변경 유형: {change_type}",
            "reasons": [f"지원되지 않는 change_type입니다: {change_type}"],
            "scores": scores
        }


def _decide_efficiency(shift_score, tail_max, outlier_rate, comp_count, scores):
    """효율 동등성 검증을 위한 단계별 판정 (내부 함수)

    공정 변경 후 품질이 기존과 동등한지 단계적으로 확인한다.
    Stage 1~4는 점진적으로 높은 신뢰도를 나타내며,
    위험 임계치를 초과하면 RISK 단계로 판정된다.

    Note: Tail Score 임계치는 고차원 데이터(feature 수천개)에서의 다중비교를
    고려하여 10%로 설정. 5000개 feature에서 1% tail 기준일 때 우연 발생 가능
    tail_max는 약 5~8% 수준이므로, 10% 이상이면 의미있는 신호로 판단.
    """
    reasons = []

    # RISK 판정 (임계치 초과) — tail은 다중비교 보정하여 10%로 설정
    if shift_score > 1.0 or tail_max > 0.10 or outlier_rate > 0.05:
        reasons.append(f"Shift Score {shift_score:.2f}가 위험 임계치(1.0)를 초과합니다." if shift_score > 1.0 else "")
        reasons.append(f"Tail Score {tail_max:.1%}가 위험 임계치(10%)를 초과합니다." if tail_max > 0.10 else "")
        reasons.append(f"Outlier Rate {outlier_rate:.1%}가 위험 임계치(5%)를 초과합니다." if outlier_rate > 0.05 else "")
        reasons = [r for r in reasons if r]
        return {
            "stage": -1,
            "stage_name": "RISK",
            "change_type": "efficiency",
            "description": "품질 동등성 미확인 — 위험 수준의 차이가 감지되어 추가 조사 필요",
            "reasons": reasons,
            "scores": scores
        }

    # Stage 4: 안정 확인 (엄격 기준 + 충분한 데이터)
    # Note: tail 임계치는 고차원 다중비교 보정 반영 (5000 feature → 5~8% 우연 발생 가능)
    if shift_score < 0.3 and tail_max < 0.05 and outlier_rate < 0.02 and comp_count >= 100:
        reasons.append(f"Shift Score {shift_score:.2f}가 매우 낮습니다 (< 0.3).")
        reasons.append(f"Tail Score {tail_max:.1%}가 안정 범위입니다 (< 5%).")
        reasons.append(f"Outlier Rate {outlier_rate:.1%}가 안정 범위입니다 (< 2%).")
        reasons.append(f"비교 웨이퍼 {comp_count}매로 충분한 데이터가 확보되었습니다.")
        return {
            "stage": 4,
            "stage_name": "EQUIVALENT_STABLE",
            "change_type": "efficiency",
            "description": "품질 동등성 안정 확인, 전면 적용 가능",
            "reasons": reasons,
            "scores": scores
        }

    # Stage 3: 동등성 확인 (완화 기준 + 충분한 데이터)
    if shift_score < 0.5 and tail_max < 0.08 and outlier_rate < 0.03 and comp_count >= 100:
        reasons.append(f"Shift Score {shift_score:.2f}가 허용 범위입니다 (< 0.5).")
        reasons.append(f"Tail Score {tail_max:.1%}가 허용 범위입니다 (< 8%).")
        reasons.append(f"Outlier Rate {outlier_rate:.1%}가 허용 범위입니다 (< 3%).")
        reasons.append(f"비교 웨이퍼 {comp_count}매로 충분한 데이터가 확보되었습니다.")
        return {
            "stage": 3,
            "stage_name": "EQUIVALENT_CONFIRMED",
            "change_type": "efficiency",
            "description": "품질 동등성 확인, 확대 적용 가능",
            "reasons": reasons,
            "scores": scores
        }

    # Stage 2: 예비 확인 (완화 기준, 데이터 부족 가능)
    if shift_score < 0.5 and tail_max < 0.08 and outlier_rate < 0.03:
        reasons.append(f"Shift Score {shift_score:.2f}가 허용 범위입니다 (< 0.5).")
        reasons.append(f"Tail Score {tail_max:.1%}가 허용 범위입니다 (< 8%).")
        reasons.append(f"Outlier Rate {outlier_rate:.1%}가 허용 범위입니다 (< 3%).")
        reasons.append(f"비교 웨이퍼 {comp_count}매 — 100매 이상 확보 시 확정 가능합니다.")
        return {
            "stage": 2,
            "stage_name": "EQUIVALENT_PRELIMINARY",
            "change_type": "efficiency",
            "description": "품질 동등성 예비 확인, 확대 검증 권고",
            "reasons": reasons,
            "scores": scores
        }

    # Stage 1: 데이터 수집 단계 (엄격 기준 충족)
    if shift_score < 0.3 and tail_max < 0.05 and outlier_rate < 0.02:
        reasons.append(f"Shift Score {shift_score:.2f}가 매우 낮습니다 (< 0.3).")
        reasons.append(f"Tail Score {tail_max:.1%}가 매우 낮습니다 (< 5%).")
        reasons.append(f"Outlier Rate {outlier_rate:.1%}가 매우 낮습니다 (< 2%).")
        return {
            "stage": 1,
            "stage_name": "DATA_COLLECTED",
            "change_type": "efficiency",
            "description": "품질 동등성 초기 확인",
            "reasons": reasons,
            "scores": scores
        }

    # 기준 미충족 — 관찰 필요
    reasons.append(f"Shift Score {shift_score:.2f}.")
    reasons.append(f"Tail Score {tail_max:.1%}.")
    reasons.append(f"Outlier Rate {outlier_rate:.1%}.")
    reasons.append("동등성 기준을 충족하지 못하지만 위험 수준은 아닙니다. 추가 모니터링이 필요합니다.")
    return {
        "stage": 1,
        "stage_name": "DATA_COLLECTED",
        "change_type": "efficiency",
        "description": "데이터 수집 중 — 동등성 판정을 위해 추가 데이터 필요",
        "reasons": reasons,
        "scores": scores
    }


def _decide_defect_improvement(shift_result, tail_result, outlier_result,
                                comp_count, target_features, scores):
    """불량 개선 검증을 위한 단계별 판정 (내부 함수)

    특정 불량 항목의 개선 여부를 확인하고, 동시에 다른 항목에 대한
    부작용(side effect) 발생 여부를 점검한다.
    target_features의 z_shift가 음수이면 개선 방향으로 판단한다.

    핵심 설계:
    - target feature의 변화는 '의도된 개선'이므로 score 판정에서 제외
    - 비대상 feature만으로 부작용 여부를 판단
    - Tail/Outlier도 비대상 feature 기준으로 평가
    """
    z_shift_all = shift_result["z_shift_all"]
    tail_rate_all = tail_result["tail_rate_all"]
    outlier_rate = outlier_result["score"]
    reasons = []

    if target_features is None or len(target_features) == 0:
        return {
            "stage": -1,
            "stage_name": "INVALID_CONFIG",
            "change_type": "defect_improvement",
            "description": "불량 개선 검증에는 target_features가 필요합니다.",
            "reasons": ["target_features가 지정되지 않았습니다."],
            "scores": scores
        }

    target_set = set(target_features)

    # 대상 feature 분석 — z_shift < 0 이면 개선 방향
    target_z_shifts = {}
    target_improved_count = 0
    for feat in target_features:
        if feat in z_shift_all.index:
            z_val = float(z_shift_all[feat])
            target_z_shifts[feat] = round(z_val, 4)
            if z_val < -0.3:  # 의미 있는 개선 기준
                target_improved_count += 1
        else:
            target_z_shifts[feat] = None

    target_improved = target_improved_count >= len(target_features) * 0.5

    # 비대상 feature 부작용 분석
    non_target_features = [f for f in z_shift_all.index if f not in target_set]
    if len(non_target_features) > 0:
        non_target_z = z_shift_all[non_target_features]
        side_effect_mask = non_target_z.abs() > 0.8
        side_effect_count = int(side_effect_mask.sum())
        # 비대상 feature의 tail rate만으로 부작용 평가
        non_target_tail = tail_rate_all.reindex(non_target_features).dropna()
        non_target_tail_max = float(non_target_tail.max()) if len(non_target_tail) > 0 else 0.0
    else:
        side_effect_count = 0
        non_target_tail_max = 0.0

    target_assessment = {
        "target_improved": target_improved,
        "target_improved_count": target_improved_count,
        "target_total": len(target_features),
        "target_z_shifts": target_z_shifts,
        "side_effect_count": side_effect_count,
        "non_target_tail_max": round(non_target_tail_max, 4)
    }

    # Stage 4: 안정적 개선 확인
    if (target_improved and side_effect_count == 0
            and non_target_tail_max < 0.03 and outlier_rate < 0.03 and comp_count >= 100):
        reasons.append(f"대상 feature {target_improved_count}/{len(target_features)}개 개선 확인.")
        reasons.append("비대상 feature에서 부작용이 감지되지 않았습니다.")
        reasons.append(f"비교 웨이퍼 {comp_count}매로 충분한 데이터가 확보되었습니다.")
        return {
            "stage": 4,
            "stage_name": "IMPROVEMENT_STABLE",
            "change_type": "defect_improvement",
            "description": "불량 개선 안정 확인, 부작용 없음, 전면 적용 가능",
            "reasons": reasons,
            "target_assessment": target_assessment,
            "scores": scores
        }

    # Stage 3: 개선 확인 (부작용 허용 범위, 데이터 부족 허용)
    # Note: non_target_tail_max 임계치는 다중비교 보정 반영 (고차원 데이터 특성)
    if target_improved and side_effect_count <= 2 and non_target_tail_max < 0.08 and outlier_rate < 0.05:
        reasons.append(f"대상 feature {target_improved_count}/{len(target_features)}개 개선 확인.")
        if side_effect_count > 0:
            reasons.append(f"비대상 feature {side_effect_count}개에서 경미한 변화 감지 (허용 범위).")
        else:
            reasons.append("비대상 feature에서 부작용이 감지되지 않았습니다.")
        if comp_count < 100:
            reasons.append(f"비교 웨이퍼 {comp_count}매 — 100매 이상 확보 시 Stage 4 가능.")
        return {
            "stage": 3,
            "stage_name": "IMPROVEMENT_CONFIRMED",
            "change_type": "defect_improvement",
            "description": "불량 개선 확인, 부작용 허용 범위 내",
            "reasons": reasons,
            "target_assessment": target_assessment,
            "scores": scores
        }

    # Stage 2: 부분 개선 (대상 개선되었으나 부작용 존재)
    if target_improved and (side_effect_count > 2 or non_target_tail_max >= 0.08):
        reasons.append(f"대상 feature {target_improved_count}/{len(target_features)}개 개선 확인.")
        if side_effect_count > 2:
            reasons.append(f"비대상 feature {side_effect_count}개에서 부작용이 감지되었습니다.")
        if non_target_tail_max >= 0.08:
            reasons.append(f"비대상 feature tail 최대값 {non_target_tail_max:.1%}가 허용치(8%) 초과.")
        reasons.append("부작용 원인 분석 및 대응이 필요합니다.")
        return {
            "stage": 2,
            "stage_name": "PARTIAL_IMPROVEMENT",
            "change_type": "defect_improvement",
            "description": "대상 불량 개선 확인, 그러나 부작용 감지 — 추가 분석 필요",
            "reasons": reasons,
            "target_assessment": target_assessment,
            "scores": scores
        }

    # Stage 1: 개선 미확인
    reasons.append(f"대상 feature 중 {target_improved_count}/{len(target_features)}개만 개선 방향.")
    reasons.append("개선 효과가 명확하지 않아 추가 데이터 수집이 필요합니다.")
    return {
        "stage": 1,
        "stage_name": "IMPROVEMENT_NOT_CONFIRMED",
        "change_type": "defect_improvement",
        "description": "불량 개선 효과 미확인 — 추가 데이터 수집 및 분석 필요",
        "reasons": reasons,
        "target_assessment": target_assessment,
        "scores": scores
    }


def run_enhanced_pipeline(df_ref, df_comp,
                           step_id="UNKNOWN",
                           change_code="UNKNOWN",
                           change_type="efficiency",
                           target_features=None,
                           top_k_ratio=0.01,
                           tail_percentile=0.99,
                           outlier_feature_thresh=0.05,
                           min_sample=30,
                           top_n_features=20,
                           bootstrap_iterations=100,
                           global_deviation_components=30):
    """강화된 ECO 변경점 검증 파이프라인

    기존 run_eco_change_detection 파이프라인의 모든 분석을 수행한 뒤,
    추가로 신뢰도 점수, 부트스트랩 안정성, 전역 편차 점수, 단계별 판정을
    산출하여 'enhanced' 키 아래에 통합 반환한다.

    Parameters:
        df_ref: 기준(ref) 데이터 (DataFrame)
        df_comp: 비교(comp) 데이터 (DataFrame)
        step_id: 공정 단계 식별자
        change_code: ECO 변경 코드
        change_type: "efficiency" 또는 "defect_improvement"
        target_features: 개선 대상 feature 목록 (defect_improvement 시 필수)
        top_k_ratio: Shift Score 상위 feature 비율
        tail_percentile: Tail Score 임계 백분위수
        outlier_feature_thresh: Outlier Wafer 판정 feature 초과 비율
        min_sample: 최소 샘플 수
        top_n_features: Feature Importance 표시 상위 개수
        bootstrap_iterations: 부트스트랩 반복 횟수
        global_deviation_components: 전역 편차 PCA 성분 수

    Returns:
        dict: 기존 파이프라인 결과 + enhanced (confidence, bootstrap, global_deviation,
              enhanced_decision)
    """
    # 기존 파이프라인 실행
    base_result = run_eco_change_detection(
        df_ref, df_comp,
        step_id=step_id,
        change_code=change_code,
        top_k_ratio=top_k_ratio,
        tail_percentile=tail_percentile,
        outlier_feature_thresh=outlier_feature_thresh,
        min_sample=min_sample,
        top_n_features=top_n_features
    )

    scaled_ref = base_result["scaled_ref"]
    scaled_comp = base_result["scaled_comp"]
    comp_count = len(df_comp)

    # 신뢰도 점수
    confidence = calc_confidence_score(comp_count)

    # 부트스트랩 안정성
    try:
        bootstrap = calc_bootstrap_stability(
            scaled_ref, scaled_comp,
            n_iterations=bootstrap_iterations,
            top_k_ratio=top_k_ratio
        )
    except Exception as e:
        bootstrap = {"error": str(e)}

    # 전역 편차 점수
    try:
        global_deviation = calc_global_deviation_score(
            scaled_ref, scaled_comp,
            n_components=global_deviation_components
        )
    except Exception as e:
        global_deviation = {"error": str(e)}

    # 강화 판정
    shift_result = base_result["detail"]["shift"]
    tail_result = base_result["detail"]["tail"]
    outlier_result = base_result["detail"]["outlier"]

    enhanced_decision = make_enhanced_decision(
        shift_result, tail_result, outlier_result,
        comp_count,
        change_type=change_type,
        target_features=target_features,
        min_sample=min_sample
    )

    base_result["enhanced"] = {
        "confidence": confidence,
        "bootstrap_stability": bootstrap,
        "global_deviation": global_deviation,
        "enhanced_decision": enhanced_decision
    }

    return base_result


# ============================================================
# PCA + Hotelling T² / SPE (보조 분석)
# ============================================================

def calc_pca_scores(scaled_ref, scaled_comp, n_components=None, variance_ratio=0.95):
    """PCA 기반 Hotelling T² 및 SPE 산출

    Parameters:
        scaled_ref: 전처리된 ref 데이터
        scaled_comp: 전처리된 comp 데이터
        n_components: PC 수 (None이면 variance_ratio 기준 자동 결정)
        variance_ratio: 자동 결정 시 설명분산 비율 기준

    Returns:
        dict with T², SPE scores, thresholds, per-feature contributions
    """
    from sklearn.decomposition import PCA
    from scipy.stats import f as f_dist, chi2
    import time

    start_time = time.time()

    ref_values = scaled_ref.fillna(0).values
    comp_values = scaled_comp.fillna(0).values

    # PCA 학습 (ref 기준)
    if n_components is None:
        # 최대 PC 수 제한: min(50, 샘플수-1, 피처수)
        max_pc = min(50, ref_values.shape[0] - 1, ref_values.shape[1])
        pca_full = PCA(n_components=max_pc).fit(ref_values)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_ratio) + 1)
        n_components = max(2, min(n_components, max_pc))

    pca = PCA(n_components=n_components)
    pca.fit(ref_values)

    # Ref 기준 T², SPE
    ref_scores = pca.transform(ref_values)  # (n_ref, k)
    ref_reconstructed = pca.inverse_transform(ref_scores)

    eigenvalues = pca.explained_variance_  # λ_i

    # Hotelling T²: T² = Σ(t_i² / λ_i)
    ref_t2 = np.sum(ref_scores**2 / eigenvalues, axis=1)

    # SPE (Q-statistic): ||x - x̂||²
    ref_spe = np.sum((ref_values - ref_reconstructed)**2, axis=1)

    # Comp T², SPE
    comp_scores = pca.transform(comp_values)
    comp_reconstructed = pca.inverse_transform(comp_scores)

    comp_t2 = np.sum(comp_scores**2 / eigenvalues, axis=1)
    comp_spe = np.sum((comp_values - comp_reconstructed)**2, axis=1)

    # 임계값 (F-분포 기반 T², chi2 기반 SPE)
    n_ref = len(ref_values)
    k = n_components

    # T² threshold: (k(n-1)/(n-k)) * F(k, n-k, alpha)
    alpha = 0.01
    f_crit = f_dist.ppf(1 - alpha, k, n_ref - k)
    t2_threshold = (k * (n_ref - 1) / (n_ref - k)) * f_crit

    # SPE threshold: chi2 근사
    spe_mean = np.mean(ref_spe)
    spe_var = np.var(ref_spe)
    if spe_var > 0:
        g = spe_var / (2 * spe_mean)
        h = 2 * spe_mean**2 / spe_var
        spe_threshold = g * chi2.ppf(1 - alpha, h)
    else:
        spe_threshold = np.percentile(ref_spe, 99)

    # Comp 이상 비율
    t2_exceed_rate = np.mean(comp_t2 > t2_threshold)
    spe_exceed_rate = np.mean(comp_spe > spe_threshold)

    # Feature contribution (T²)
    # Contribution_j = Σ_i (t_i * p_ij)² / λ_i
    loadings = pca.components_  # (k, p)
    feature_names = scaled_ref.columns.tolist()

    # 평균 comp T² contribution per feature
    comp_contributions = np.zeros(len(feature_names))
    for i in range(k):
        comp_contributions += np.mean(
            (comp_scores[:, i:i+1] * loadings[i:i+1, :])**2 / eigenvalues[i],
            axis=0
        )

    ref_contributions = np.zeros(len(feature_names))
    for i in range(k):
        ref_contributions += np.mean(
            (ref_scores[:, i:i+1] * loadings[i:i+1, :])**2 / eigenvalues[i],
            axis=0
        )

    # Contribution 증가율
    contrib_increase = comp_contributions - ref_contributions
    top_contrib_idx = np.argsort(contrib_increase)[::-1][:20]
    top_contrib_features = [feature_names[i] for i in top_contrib_idx]
    top_contrib_values = contrib_increase[top_contrib_idx]

    elapsed = time.time() - start_time

    return {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
        "ref_t2": ref_t2,
        "comp_t2": comp_t2,
        "ref_spe": ref_spe,
        "comp_spe": comp_spe,
        "t2_threshold": t2_threshold,
        "spe_threshold": spe_threshold,
        "t2_exceed_rate": t2_exceed_rate,
        "spe_exceed_rate": spe_exceed_rate,
        "top_contrib_features": top_contrib_features,
        "top_contrib_values": top_contrib_values,
        "eigenvalues": eigenvalues,
        "loadings": loadings,
        "elapsed_sec": elapsed,
        "pca_model": pca
    }
