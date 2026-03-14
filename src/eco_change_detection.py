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
# Synthetic Data Generator
# ============================================================

def generate_synthetic_data(n_ref=1000, n_comp=80, n_features=5000, seed=42):
    """3가지 불량 패턴을 시뮬레이션한 합성 데이터 생성"""
    np.random.seed(seed)
    feature_names = [f"EDS_{i:04d}" for i in range(n_features)]

    ref_data = np.random.randn(n_ref, n_features) * 0.5 + 3.0
    comp_data = np.random.randn(n_comp, n_features) * 0.5 + 3.0

    # Pattern A: Systematic shift (Feature 100~120)
    shift_features = list(range(100, 121))
    comp_data[:, shift_features] += 0.75  # +1.5σ shift (std=0.5)

    # Pattern B: Intermittent spike (Feature 500~510)
    spike_features = list(range(500, 511))
    spike_wafers = np.random.choice(n_comp, size=int(n_comp * 0.10), replace=False)
    comp_data[np.ix_(spike_wafers, spike_features)] += 3.0

    # Pattern C: Multi-feature outlier wafers (Wafer 70~74)
    outlier_wafers = list(range(70, min(75, n_comp)))
    outlier_features = list(range(200, 500))  # 300개 feature (6%)에 걸쳐 이상
    for w in outlier_wafers:
        if w < n_comp:
            comp_data[w, outlier_features] += 2.0

    df_ref = pd.DataFrame(ref_data, columns=feature_names,
                          index=[f"ref_w{i:04d}" for i in range(n_ref)])
    df_comp = pd.DataFrame(comp_data, columns=feature_names,
                           index=[f"comp_w{i:04d}" for i in range(n_comp)])

    return df_ref, df_comp
