"""
AI/ML 강화 변경점 탐지 모듈
기존 3-Score 통계 파이프라인에 ML 기법을 보조적으로 결합

1. Isolation Forest  — Wafer 단위 비지도 이상 탐지
2. Autoencoder       — 재구성 오류 기반 미세 변화 감지
3. Learned Ensemble  — Score 통합 가중치 학습
"""

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Isolation Forest 기반 Wafer Anomaly Score
# ============================================================

def calc_isolation_forest_score(scaled_ref, scaled_comp,
                                 contamination='auto',
                                 n_estimators=100,
                                 random_state=42):
    """Isolation Forest로 wafer 단위 이상 탐지

    기존 Outlier Wafer Score: 5% 초과 Feature 수 기준 (규칙 기반)
    IF Score: 고차원 공간에서 '고립 용이성' 기반 (학습 기반)

    장점:
    - Feature 간 상관 구조를 암묵적으로 활용
    - 수동 임계값(5%) 불필요
    - 비선형 이상 패턴 포착 가능
    """
    start = time.time()

    ref_values = scaled_ref.fillna(0).values
    comp_values = scaled_comp.fillna(0).values

    # Ref 데이터로 학습 (정상 분포)
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(ref_values)

    # Comp 데이터에 대한 이상 점수
    # score_samples: 높을수록 정상, 낮을수록 이상
    comp_scores = clf.score_samples(comp_values)
    ref_scores = clf.score_samples(ref_values)

    # 이상 판정 (-1: 이상, 1: 정상)
    comp_labels = clf.predict(comp_values)
    anomaly_rate = (comp_labels == -1).mean()

    # Ref 기준 임계값 (1st percentile)
    ref_threshold = np.percentile(ref_scores, 1)

    # 정규화된 anomaly score (0~1, 높을수록 이상)
    score_min = min(ref_scores.min(), comp_scores.min())
    score_max = max(ref_scores.max(), comp_scores.max())
    if score_max - score_min > 0:
        norm_comp = 1 - (comp_scores - score_min) / (score_max - score_min)
        norm_ref = 1 - (ref_scores - score_min) / (score_max - score_min)
    else:
        norm_comp = np.zeros_like(comp_scores)
        norm_ref = np.zeros_like(ref_scores)

    elapsed = time.time() - start

    return {
        'anomaly_rate': anomaly_rate,
        'comp_scores_raw': comp_scores,
        'ref_scores_raw': ref_scores,
        'comp_scores_norm': norm_comp,
        'ref_scores_norm': norm_ref,
        'comp_labels': comp_labels,
        'ref_threshold': ref_threshold,
        'model': clf,
        'elapsed_sec': elapsed,
    }


# ============================================================
# 2. Autoencoder 기반 재구성 오류 Score
# ============================================================

def calc_autoencoder_score(scaled_ref, scaled_comp,
                            encoding_dim=None,
                            epochs=50,
                            batch_size=32,
                            random_state=42):
    """Autoencoder: Ref로 학습 → Comp 재구성 오류로 이상 탐지

    기존 방법으로 탐지 어려운 Pattern E(+0.5σ 미세 이동) 보완 기대
    - AE가 Ref의 정상 매니폴드를 학습
    - Comp에서 정상과 다른 부분은 재구성 실패 → 오류 증가

    장점:
    - 비선형 Feature 관계 학습
    - 차원 축소와 이상 탐지 동시 수행
    - PCA보다 복잡한 구조 포착 가능
    """
    start = time.time()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        use_torch = True
    except ImportError:
        use_torch = False

    ref_values = scaled_ref.fillna(0).values.astype(np.float32)
    comp_values = scaled_comp.fillna(0).values.astype(np.float32)

    n_features = ref_values.shape[1]
    if encoding_dim is None:
        encoding_dim = max(10, n_features // 50)  # 5000 → 100

    feat_names = (scaled_ref.columns.tolist()
                  if hasattr(scaled_ref, 'columns')
                  else [f"F{i}" for i in range(n_features)])

    if use_torch:
        return _autoencoder_torch(ref_values, comp_values, n_features,
                                   encoding_dim, epochs, batch_size,
                                   random_state, start, feat_names)
    else:
        return _autoencoder_sklearn(ref_values, comp_values, n_features,
                                     encoding_dim, start, feat_names)


def _autoencoder_torch(ref_values, comp_values, n_features,
                        encoding_dim, epochs, batch_size,
                        random_state, start_time, feat_names):
    """PyTorch 기반 Autoencoder"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)

    # 모델 정의
    class AutoEncoder(nn.Module):
        def __init__(self, input_dim, enc_dim):
            super().__init__()
            mid_dim = max(enc_dim * 2, input_dim // 10)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, enc_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(enc_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(n_features, encoding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    # 학습 (Ref만)
    ref_tensor = torch.FloatTensor(ref_values).to(device)
    dataset = TensorDataset(ref_tensor, ref_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, _ in loader:
            output = model(batch_x)
            loss = criterion(output, batch_x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(loader))

    # 재구성 오류 산출
    model.eval()
    with torch.no_grad():
        ref_recon = model(ref_tensor)
        ref_errors = criterion(ref_recon, ref_tensor)  # (n_ref, n_features)
        ref_mse_per_sample = ref_errors.mean(dim=1).cpu().numpy()
        ref_mse_per_feature = ref_errors.mean(dim=0).cpu().numpy()

        comp_tensor = torch.FloatTensor(comp_values).to(device)
        comp_recon = model(comp_tensor)
        comp_errors = criterion(comp_recon, comp_tensor)
        comp_mse_per_sample = comp_errors.mean(dim=1).cpu().numpy()
        comp_mse_per_feature = comp_errors.mean(dim=0).cpu().numpy()

    # Feature별 오류 증가율
    feature_error_increase = comp_mse_per_feature - ref_mse_per_feature
    top_idx = np.argsort(feature_error_increase)[::-1][:20]
    feature_names = [feat_names[i] for i in top_idx]

    # 임계값 (Ref 99th percentile)
    threshold = np.percentile(ref_mse_per_sample, 99)
    exceed_rate = (comp_mse_per_sample > threshold).mean()

    elapsed = time.time() - start_time

    return {
        'ref_mse': ref_mse_per_sample,
        'comp_mse': comp_mse_per_sample,
        'ref_mse_per_feature': ref_mse_per_feature,
        'comp_mse_per_feature': comp_mse_per_feature,
        'feature_error_increase': feature_error_increase,
        'top_features': feature_names,
        'top_feature_values': feature_error_increase[top_idx],
        'threshold': threshold,
        'exceed_rate': exceed_rate,
        'encoding_dim': encoding_dim,
        'loss_history': loss_history,
        'elapsed_sec': elapsed,
        'backend': 'torch',
    }


def _autoencoder_sklearn(ref_values, comp_values, n_features,
                          encoding_dim, start_time, feat_names):
    """sklearn MLPRegressor 기반 Autoencoder 대체"""
    from sklearn.neural_network import MLPRegressor

    # MLPRegressor를 AE 대용으로 사용
    model = MLPRegressor(
        hidden_layer_sizes=(encoding_dim * 2, encoding_dim, encoding_dim * 2),
        activation='relu',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    # Ref로 학습 (입력=출력)
    model.fit(ref_values, ref_values)

    ref_recon = model.predict(ref_values)
    comp_recon = model.predict(comp_values)

    ref_errors = (ref_values - ref_recon) ** 2
    comp_errors = (comp_values - comp_recon) ** 2

    ref_mse_per_sample = ref_errors.mean(axis=1)
    comp_mse_per_sample = comp_errors.mean(axis=1)
    ref_mse_per_feature = ref_errors.mean(axis=0)
    comp_mse_per_feature = comp_errors.mean(axis=0)

    feature_error_increase = comp_mse_per_feature - ref_mse_per_feature
    top_idx = np.argsort(feature_error_increase)[::-1][:20]

    threshold = np.percentile(ref_mse_per_sample, 99)
    exceed_rate = (comp_mse_per_sample > threshold).mean()

    elapsed = time.time() - start_time

    return {
        'ref_mse': ref_mse_per_sample,
        'comp_mse': comp_mse_per_sample,
        'ref_mse_per_feature': ref_mse_per_feature,
        'comp_mse_per_feature': comp_mse_per_feature,
        'feature_error_increase': feature_error_increase,
        'top_features': [feat_names[i] for i in top_idx],
        'top_feature_values': feature_error_increase[top_idx],
        'threshold': threshold,
        'exceed_rate': exceed_rate,
        'encoding_dim': encoding_dim,
        'loss_history': [],
        'elapsed_sec': elapsed,
        'backend': 'sklearn',
    }


# ============================================================
# 3. Learned Ensemble (Score 통합 학습)
# ============================================================

def train_ensemble_classifier(results_list, labels):
    """여러 ECO 결과의 Score 벡터로 학습 기반 판정기 구축

    Parameters:
        results_list: list of run_eco_change_detection() 결과
        labels: list of int (0=SAFE, 1=CAUTION, 2=RISK, 3=HIGH_RISK)

    Returns:
        trained model, feature_names, performance metrics
    """
    start = time.time()

    # Score 벡터 구성
    X = []
    for r in results_list:
        scores = r['scores']
        x = [
            scores['shift_score'],
            scores['tail_score_max'],
            scores['tail_feature_count'],
            scores['outlier_wafer_rate'],
        ]
        X.append(x)

    X = np.array(X)
    y = np.array(labels)

    feature_names = ['shift_score', 'tail_score_max', 'tail_feature_count', 'outlier_wafer_rate']

    # Logistic Regression (해석 가능)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
    )
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    accuracy = (y_pred == y).mean()

    elapsed = time.time() - start

    return {
        'model': clf,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'coefficients': dict(zip(feature_names, clf.coef_.mean(axis=0))),
        'y_pred': y_pred,
        'y_true': y,
        'elapsed_sec': elapsed,
    }


def generate_ensemble_training_data(n_scenarios=200, seed=42):
    """앙상블 학습용 다양한 시나리오 데이터 생성

    다양한 강도의 변화를 적용하여 학습 데이터를 자동 생성
    """
    from eco_change_detection import run_eco_change_detection

    np.random.seed(seed)
    n_ref, n_features = 500, 2000
    feature_names = [f"EDS_{i:04d}" for i in range(n_features)]

    results_list = []
    labels = []

    scenarios = [
        # (shift_magnitude, spike_rate, outlier_wafer_pct, label)
        (0.0, 0.0, 0.0, 0),     # SAFE
        (0.3, 0.01, 0.01, 0),   # SAFE (borderline)
        (0.6, 0.02, 0.02, 1),   # CAUTION
        (0.8, 0.03, 0.03, 1),   # CAUTION
        (1.2, 0.05, 0.05, 2),   # RISK
        (1.5, 0.07, 0.07, 2),   # RISK
        (2.5, 0.12, 0.10, 3),   # HIGH_RISK
        (3.0, 0.15, 0.15, 3),   # HIGH_RISK
    ]

    per_scenario = n_scenarios // len(scenarios)

    for shift_mag, spike_rate, outlier_pct, label in scenarios:
        for trial in range(per_scenario):
            trial_seed = seed + trial + label * 1000
            np.random.seed(trial_seed)

            n_comp = 80
            ref_data = np.random.randn(n_ref, n_features) * 0.5 + 3.0
            comp_data = np.random.randn(n_comp, n_features) * 0.5 + 3.0

            # Shift
            if shift_mag > 0:
                shift_feats = list(range(0, 20))
                comp_data[:, shift_feats] += shift_mag * 0.5  # scale to σ

            # Spike
            if spike_rate > 0:
                n_spike = max(1, int(n_comp * spike_rate))
                spike_idx = np.random.choice(n_comp, n_spike, replace=False)
                spike_feats = list(range(50, 60))
                comp_data[np.ix_(spike_idx, spike_feats)] += 3.0

            # Outlier wafers
            if outlier_pct > 0:
                n_outlier = max(1, int(n_comp * outlier_pct))
                outlier_idx = list(range(n_comp - n_outlier, n_comp))
                outlier_feats = list(range(100, 300))
                for w in outlier_idx:
                    comp_data[w, outlier_feats] += 2.0

            df_ref = pd.DataFrame(ref_data, columns=feature_names)
            df_comp = pd.DataFrame(comp_data, columns=feature_names)

            try:
                result = run_eco_change_detection(df_ref, df_comp, step_id=f"ENS_{label}_{trial}")
                results_list.append(result)
                labels.append(label)
            except Exception:
                pass

    return results_list, labels
