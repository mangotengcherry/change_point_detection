# ECO Change Detection PoC

반도체 공정 변경점(ECO) 적용 전후의 품질 차이를 자동으로 정량화하고, 차이의 주요 원인 Feature를 식별하는 분석 파이프라인입니다.

**[프로젝트 페이지 (GitHub Pages)](https://mangotengcherry.github.io/change_point_detection/)**

---

## 1. 과제 배경

### 1.1 목적

반도체 공정에서 ECO(Engineering Change Order)를 적용한 후, **변경 전(Ref)**과 **변경 후(Compare)**의 품질이 동일한지를 객관적으로 검증해야 합니다. 현업에서는 수천 개의 EDS/MSR/AWACS Feature를 엔지니어가 수작업으로 비교하며, 이 과정에서 미세한 변화나 간헐적 이상이 누락될 위험이 있습니다.

본 파이프라인은 이 과정을 **자동화**하여, 변경점 적용의 안전성을 정량적으로 평가합니다.

### 1.2 핵심 질문

| # | 질문 | 해결 방법 |
|---|------|-----------|
| 1 | Ref와 Compare 사이에 **얼마나 다른가?** | 3종 Score로 정량화 (Shift / Tail / Outlier) |
| 2 | **어떤 Feature**가 Score 변동에 가장 크게 기여하는가? | Score별 Feature Importance 추적 |

### 1.3 데이터 구조

| 항목 | 설명 |
|------|------|
| 단위 | Wafer |
| 그룹 | Ref (기존 조건) / Compare (변경 조건) |
| Feature | EDS / MSR / AWACS 값 (수백~수천 개, 모두 numeric) |
| 특성 | 값이 클수록 불량률이 커지는 방향 (단방향) |
| 비대칭 | Ref >> Compare (Ref 수천 장, Compare 수십~수백 장) |

### 1.4 설계 원칙

- **단일 Score로 합치지 않는다**: 불량 패턴이 2종류(Systematic Shift vs Intermittent Spike)이므로, 각각에 맞는 Score를 별도 산출
- **3개 Score를 OR 조건으로 판정**: 하나라도 기준 초과 → 경고 상향
- **Score별로 원인 Feature를 따로 추적**: "전반적 drift"와 "간헐 불량"은 다른 엔지니어링 액션으로 이어지기 때문

---

## 2. 실험 과정

### 2.1 파이프라인 구조

```
[입력] ref wafers + compare wafers (feature matrix)
  │
  ▼ Step 1: 전처리
  │  - 결측/상수 변수 제거
  │  - Ref 기준 Robust Scaling (median / IQR)
  │  - Winsorizing (0.5th ~ 99.5th)
  │
  ▼ Step 2: Score 산출 (3종 병렬)
  │  ├─ Shift Score: 중심치/산포 이동 (Top-K z-shift 평균)
  │  ├─ Tail Score: 간헐적 극단값 (99th percentile 초과 비율)
  │  └─ Outlier Wafer Score: 다변량 wafer 이상 (feature 동시 초과)
  │
  ▼ Step 3: Feature Importance 추적
  │  - Shift / Tail / Outlier 원인 Feature Top-N
  │
  ▼ Step 4: 판정 + 리포트
  │  - SAFE / CAUTION / RISK / HIGH_RISK
  │  - 원인 Feature 요약 + 시각화
  │
  ▼ [출력] 변경점 검증 리포트
```

![Pipeline Flow](docs/images/00_pipeline_flow.png)
> **[해석]** 입력 데이터가 전처리를 거친 후, 3종 Score가 **병렬**로 산출됩니다. 각 Score는 서로 다른 불량 패턴을 탐지하도록 설계되어 있으며, Feature Importance 추적과 최종 판정이 순차적으로 이루어집니다.

### 2.2 Step 1: 전처리

| 단계 | 방법 | 목적 |
|------|------|------|
| 변수 필터링 | 결측률 30% 이상 / 분산 ≈ 0 제거 | 무의미한 변수 배제 |
| Robust Scaling | `(x - median_ref) / IQR_ref` | Ref 기준으로 정규화, Compare의 이탈 정도에 의미 부여 |
| Winsorizing | 0.5th ~ 99.5th percentile clipping | 극단적 측정 오류(센서 glitch) 제거 |

**왜 Robust Scaling인가?** Mean/Std 기반 Z-score는 극단값에 취약합니다. Median/IQR은 극단값의 영향을 받지 않아, 반도체 공정 데이터처럼 heavy-tail 분포가 흔한 환경에 적합합니다.

### 2.3 Step 2: 3종 Score 산출

#### Score 1 — Shift Score (중심치/산포 이동)

```
z_j = (mean_compare_j - mean_ref_j) / std_ref_j
Shift Score = mean(|z_j|) for Top-K features
```

- **Top-K 방식을 쓰는 이유**: 전체 RMS는 feature 수가 수천 개일 때 소수 feature의 shift가 희석됨. 상위 K개만 보면 실제 shift한 feature가 Score에 반영됨.
- `score ≈ 0`: ref와 compare 거의 동일 / `score > 1.0`: 상위 feature 평균 1σ 이상 이동 / `score > 2.0`: 명확한 systematic shift

#### Score 2 — Tail Score (간헐적 극단값)

```
threshold_j = ref의 99th percentile
tail_rate_j = P(compare_j > threshold_j)
```

- **현장에서 중요한 이유**: 수율은 괜찮아 보이는데 간헐적으로 불량 lot이 나오는 상황. Mean 기반 z-shift로는 안 잡힘.
- `score_max < 0.03`: tail 증가 없음 / `0.03~0.10`: 간헐적 이상 / `> 0.10`: 심각한 tail 증가

#### Score 3 — Outlier Wafer Score (wafer 단위 이상)

```
exceed_ratio_w = (compare wafer w에서 threshold 초과 feature 수) / (전체 feature 수)
Outlier = exceed_ratio > 5%
```

- **핵심 아이디어**: 간헐적 불량은 특정 wafer에 집중. Feature 단위로 보면 각각은 미미해도, wafer 단위로 보면 수십 개 feature에서 동시에 튀는 패턴.
- `score < 0.03`: outlier 없음 / `0.03~0.10`: 소수 wafer 이상 / `> 0.10`: 다수 wafer 다변량 이상

### 2.4 Step 3: Feature Importance

Score별로 원인 Feature를 **따로** 추적합니다.

| Score | 추적 내용 | 의미 |
|-------|-----------|------|
| Shift | z-shift 크기 + 방향(악화↑/개선↓) | 전반적 drift의 원인 |
| Tail | tail_rate 높은 순 | 간헐적 극단값의 원인 |
| Outlier | outlier wafer들의 공통 초과 Feature | 특정 wafer 집중 이상의 원인 |

> 두 리스트가 다를 때가 더 가치 있는 정보입니다. "전반적 drift"와 "간헐 불량"은 다른 엔지니어링 액션으로 이어지기 때문입니다.

### 2.5 Step 4: 판정

| 레벨 | 판정 | 기준 (OR 조건) |
|------|------|----------------|
| 0 | SAFE | 모든 Score 정상 |
| 1 | CAUTION | Shift > 0.5 또는 Tail > 3% 또는 Outlier > 3% |
| 2 | RISK | Shift > 1.0 또는 Tail > 5% 또는 Outlier > 5% |
| 3 | HIGH_RISK | Shift > 2.0 또는 Tail > 10% 또는 Outlier > 10% |
| -1 | INSUFFICIENT_DATA | Compare wafer < 30장 |

### 2.6 합성 데이터 설계

실제 데이터 투입 전 파이프라인 검증을 위해 5가지 불량 패턴을 삽입한 합성 데이터를 생성했습니다.

| 패턴 | 유형 | 대상 | 변형 | 난이도 |
|------|------|------|------|--------|
| A | Systematic Shift | F100~120 (21개) | +1.5σ, 전체 wafer | Easy |
| B | Intermittent Spike | F500~510 (11개) | +6σ, 10% wafer만 | Easy |
| C | Multi-Feature Outlier | F200~499 (300개) | +4σ, W70~74만 | Easy |
| D | Gradual Trend | F600~610 (11개) | 0→+2σ 점진적 | Medium |
| E | Subtle Shift | F700~720 (21개) | +0.5σ, 전체 wafer | Hard |

- **Ref**: 1,000 wafers × 5,000 features (정상 분포 N(3.0, 0.5²))
- **Compare**: 80 wafers × 5,000 features (동일 기저 + 패턴 삽입)

![Synthetic Data](docs/images/01_synthetic_data_structure.png)
> **[해석]** 정상 Feature(EDS_0050)는 Ref와 Compare의 분포가 일치하는 반면, Pattern A(EDS_0110)는 Compare의 평균이 명확히 우측으로 이동했습니다. Pattern B(EDS_0505)는 대부분 정상이나 소수 wafer에서 극단값이 발생하며, Pattern D(EDS_0605)는 wafer 순서에 따라 점진적 상승 추세를 보입니다.

---

## 3. 결과

### 3.1 전처리 결과

![Preprocessing](docs/images/02_preprocessing_steps.png)
> **[해석]**
> - **(좌상) Feature 필터링**: 5,000개 전체 feature가 유효하여 제거된 것이 없습니다. 합성 데이터에 결측이나 상수 feature가 없기 때문입니다.
> - **(중상) Scaling 전**: 원본 Feature들이 동일한 스케일(평균 3.0, 표준편차 0.5)을 가지고 있어 분포가 겹칩니다.
> - **(우상) Scaling 후**: Robust Scaling 적용 후 Ref의 median=0, IQR≈1로 정규화되었습니다.
> - **(좌하) Scaling 검증**: Ref의 Median/IQR 분포가 기대값(0, 1)에 집중됩니다.
> - **(중하) Shift 패턴 확인**: EDS_0110에서 Scaled Compare가 Ref 대비 우측으로 명확히 이동합니다.
> - **(우하) Winsorizing 효과**: EDS_0505의 극단값이 Winsorizing 후 제한됩니다.

### 3.2 Score 산출 결과

| Score | 값 | 판정 레벨 |
|-------|-----|-----------|
| Shift Score | **0.994** | CAUTION (> 0.5) |
| Tail Score (max) | **31.25%** | HIGH_RISK (> 10%) |
| Tail Feature Count | **661개** | - |
| Outlier Wafer Rate | **6.25%** (5/80) | RISK (> 5%) |
| **최종 판정** | | **HIGH_RISK** |

![Score Calculation](docs/images/03_score_calculation.png)
> **[해석]**
> - **(1행) Shift Score**: 대부분의 feature는 z-shift ≈ 0이지만, Pattern A(F100~120)에 해당하는 feature들이 +1.4~1.6σ 범위에서 돌출합니다. Top-15 모두 EDS_01xx로, Pattern A가 정확히 검출되었습니다.
> - **(2행) Tail Score**: Feature별 tail rate 분포에서 상위 feature들이 10%를 크게 초과합니다. Pattern B(spike)와 Pattern C(outlier wafer)의 영향으로, Ref의 99th percentile을 대거 초과하는 feature가 661개에 달합니다.
> - **(3행) Outlier Wafer Score**: Wafer별 초과 비율에서 comp_w0070~0074 5개가 5% 기준을 크게 초과합니다. 이 wafer들은 Pattern C로 F200~499 300개 feature에서 동시 이상이 발생한 wafer들입니다.
> - **(우하) Score 요약**: 3종 Score가 각각 CAUTION/HIGH_RISK/RISK로 평가되어, OR 조건에 의해 최종 HIGH_RISK로 판정됩니다.

### 3.3 Feature Importance 결과

![Feature Importance](docs/images/04_feature_importance.png)
> **[해석]**
> - **(좌상) Shift Top-20**: 전부 EDS_0100~0120 범위로, Pattern A의 feature가 정확히 식별되었습니다. 모든 feature가 '악화'(빨간색) 방향이며, z-shift +1.2~1.6 범위입니다.
> - **(우상) Tail Top-20**: EDS_02xx~04xx 범위의 feature가 높은 tail rate(20~31%)를 보입니다. 이는 Pattern C(outlier wafer)의 5개 wafer에서 이 feature들이 대거 초과했기 때문입니다.
> - **(좌하) Shift vs Tail Scatter**: Feature들이 4개 영역으로 분류됩니다. 빨간점(Shift+Tail 동시)은 두 패턴의 교차 영향을 받는 feature이고, 주황(Shift만)은 Pattern A, 보라(Tail만)은 Pattern B/C의 영향입니다.
> - **(우하) Multi-Score Heatmap**: 상위 feature들의 3종 Score 기여도를 정규화하여 비교합니다. Shift/Tail/Outlier에서 각각 다른 feature가 top에 오르는 것을 확인할 수 있습니다.

### 3.4 최종 리포트

![Final Report](docs/images/05_final_report.png)
> **[해석]** HIGH_RISK 판정과 함께, 3가지 사유(Shift 유의미 이동, Tail 심각 증가, Outlier wafer 이상)가 제시됩니다. Shift Top-10은 Pattern A feature, Tail Top-10은 Pattern C feature, Outlier Wafer 5개는 Pattern C wafer(comp_w0070~0074)로, 각 Score가 설계된 패턴을 정확히 검출했습니다.

### 3.5 민감도 검증

![Sensitivity](docs/images/06_sensitivity_analysis.png)
> **[해석]**
> - **(좌상) False Alarm Test**: Ref를 반으로 나누어 비교했을 때 모든 Score ≈ 0입니다. 변화가 없는 데이터에서 Score가 발생하지 않아 **오탐이 없음**을 확인했습니다.
> - **(우상) 단조 증가 검증**: Shift 크기를 0→2.0σ로 점진적으로 늘렸을 때, Score가 단조 증가합니다. Score가 이상의 크기에 **비례적으로 반응**함을 검증했습니다.
> - **(좌하) Sample Size 검증**: Compare wafer 수가 30장 미만일 때 Score 변동성이 급증합니다. 최소 30장 이상의 Compare가 필요하다는 운영 기준의 근거입니다.
> - **(우하) 체크리스트**: 5가지 검증 항목 모두 PASS.

### 3.6 추가 인사이트 시각화

![Additional Insights](docs/images/07_additional_insights.png)
> **[해석]**
> - **(좌상) Violin Plot**: Top-5 Shift Feature에서 Ref(파랑)와 Compare(빨강)의 분포 차이가 시각적으로 명확합니다. Compare의 중앙값이 일관되게 우측으로 이동했습니다.
> - **(우상) 2D Scatter**: 상위 2개 Shift Feature의 2차원 공간에서 Ref(파랑)와 Compare(빨강) 클러스터가 분리되어 있습니다. 다변량 관점에서도 shift가 확인됩니다.
> - **(좌하) Correlation Heatmap**: 상위 feature 간 상관관계가 낮아(≈0), 각 feature의 shift가 독립적임을 보여줍니다. 상관이 높았다면 소수의 근본 원인이 여러 feature에 영향을 준 것으로 해석할 수 있습니다.
> - **(우하) CDF 비교**: KS test를 통해 Ref와 Compare의 분포 차이를 통계적으로 검증합니다. KS statistic이 높고 p-value가 극히 낮아, 두 분포가 통계적으로 유의미하게 다릅니다.

### 3.7 Ground Truth 검증

![Ground Truth](docs/images/08_ground_truth_validation.png)
> **[해석]**
> - **(좌상) 패턴별 Recall**: Easy 난이도(Pattern A, B, C)는 높은 검출률을 보이며, Medium(D: Gradual Trend)과 Hard(E: Subtle Shift)는 상대적으로 낮습니다. 이는 파이프라인이 명확한 이상은 잘 잡지만, 미세한 변화에는 민감도가 제한적임을 보여줍니다.
> - **(중상) Precision/Recall 종합**: 전체적으로 높은 정밀도를 유지하면서 적절한 재현율을 달성합니다.
> - **(우상) Score별 TP**: Shift Score, Tail Score, Outlier Score가 각각 다른 패턴의 feature를 검출하여, 3종 Score의 **상호 보완성**이 입증됩니다.
> - **(하단) 분포 비교**: Pattern A(명확한 shift), D(점진적 trend), E(미세 shift)의 실제 분포 차이를 보여줍니다. Pattern E의 z-shift가 0.5σ 수준으로, 현재 Top-K 방식으로는 검출이 어렵습니다.

### 3.8 Wafer 단위 심층 분석

![Wafer Analysis](docs/images/09_wafer_analysis.png)
> **[해석]**
> - **(좌) Wafer별 이상 비율 분포**: 대부분의 wafer는 1~2% 수준이지만, 5개 outlier wafer만 5%를 크게 초과하여 명확히 구분됩니다.
> - **(중) Outlier Wafer Feature Profile**: 5개 outlier wafer(comp_w0070~0074)의 상위 feature 값이 정상 wafer 대비 현저히 높습니다. 패턴이 유사하여 **동일 원인**에 의한 이상임을 시사합니다.
> - **(우) Normal vs Outlier 비교**: Outlier wafer들의 평균이 Normal wafer 대비 일관되게 높아, 다변량 관점에서의 이상이 확인됩니다.

---

## 4. 토론 (Discussion)

### 4.1 3종 Score 설계의 타당성

본 파이프라인은 **단일 Score가 아닌 3종 Score를 병렬로 산출**하는 설계를 채택했습니다. 실험 결과, 각 Score가 서로 다른 불량 패턴을 검출하는 것이 확인되었습니다:

| Score | 검출한 패턴 | 다른 Score로 검출 가능? |
|-------|-------------|------------------------|
| Shift Score | Pattern A (Systematic Shift) | Tail로는 부분적, Outlier로는 불가 |
| Tail Score | Pattern B (Spike) + C (Outlier wafer) | Shift로는 불가 |
| Outlier Score | Pattern C (Multi-feature outlier) | Shift/Tail로는 부분적 |

**만약 단일 Score를 사용했다면**, Pattern B(간헐적 spike)는 평균 기반 지표에 희석되어 검출이 어려웠을 것이며, Pattern C(wafer 집중 이상)는 feature 단위 분석으로는 개별 feature의 이상이 미미하여 놓칠 수 있었습니다.

### 4.2 Top-K 방식의 장단점

Shift Score에서 Top-K(상위 1%) 방식을 사용한 이유:
- **장점**: 5,000개 feature 중 소수만 shift한 경우에도 민감하게 반응
- **장점**: 전체 RMS 대비 noise에 강건
- **한계**: K의 선택에 따라 Score가 변동. K가 너무 작으면 noise에 취약, 너무 크면 signal이 희석

실험에서 K=1%(50개)로 설정했을 때, Pattern A의 21개 feature가 Top-50에 모두 포함되어 적절한 설정이었습니다.

### 4.3 난이도별 검출 한계

| 난이도 | 검출 | 한계점 |
|--------|------|--------|
| Easy (A, B, C) | 완전 검출 | - |
| Medium (D: Gradual Trend) | 부분 검출 | 점진적 drift는 wafer 순서 의존적이나, 현재 파이프라인은 순서 무관 |
| Hard (E: Subtle Shift) | 미검출 | +0.5σ 수준의 미세 변화는 Top-K 임계값 미달 |

**Pattern D(Gradual Trend)**: 시간에 따른 점진적 변화는 현재 파이프라인이 wafer 순서를 고려하지 않기 때문에, 전체 평균으로는 부분적으로만 검출됩니다. CUSUM이나 시계열 기반 방법의 도입이 필요합니다.

**Pattern E(Subtle Shift)**: +0.5σ 수준의 미세 변화는 현재 Top-K 방식의 검출 한계 아래입니다. PCA 기반 다변량 분석이나 통계 검정(Mann-Whitney U)의 보완이 유효합니다.

### 4.4 False Alarm vs Sensitivity Trade-off

민감도 분석에서 확인된 핵심 사항:
- **False Alarm = 0**: Ref vs Ref 비교 시 모든 Score ≈ 0으로, 오탐이 발생하지 않습니다.
- **단조 증가**: 이상 크기에 비례하여 Score가 증가하므로, **임계값 조정을 통해 민감도를 튜닝**할 수 있습니다.
- **최소 Sample Size = 30**: Compare 30장 미만에서 Score 변동성이 급증하여, 이 기준을 운영 정책에 반영해야 합니다.

### 4.5 Robust Scaling의 효과

Ref 기준 Robust Scaling(median/IQR)을 사용한 이유와 효과:
- **효과**: Feature별 스케일 차이를 제거하여, 서로 다른 물리적 단위(EDS mV, MSR Ω 등)의 feature를 동일 기준으로 비교 가능
- **Ref 기준**: Compare의 이탈 정도가 Ref 대비 몇 σ인지를 직접적으로 해석 가능
- **Robust**: Mean/Std 대비 극단값에 덜 민감하여, 소수의 이상 wafer가 baseline을 오염시키지 않음

---

## 5. 인사이트 (Key Insights)

### 5.1 실무 적용 시사점

1. **3종 Score 병렬 산출은 필수적**: 단일 Score로는 불량 패턴의 다양성을 포착할 수 없습니다. 특히 간헐적 spike(Pattern B)는 평균 기반 지표로는 검출 불가능합니다.

2. **Feature Importance의 이원화가 가치있는 정보**: Shift 원인과 Tail 원인이 다를 때가 더 중요합니다. 전반적 drift와 간헐 불량은 다른 공정 액션(공정 조건 조정 vs 설비 점검)으로 이어지기 때문입니다.

3. **Outlier Wafer 공통 Feature 추적**: 다수 feature에서 동시에 이상이 발생하는 wafer를 식별하면, 특정 설비/시간대/lot의 문제를 역추적할 수 있습니다.

4. **Compare 최소 30장 필요**: Sample size가 30장 미만이면 Score 변동성이 급증하여 신뢰도가 저하됩니다. 운영 정책에 이 기준을 반영해야 합니다.

### 5.2 파이프라인의 강점

| 강점 | 설명 |
|------|------|
| 해석 가능성 | 3종 Score + Feature Importance로 왜 그 판정인지 설명 가능 |
| 범용성 | EDS/MSR/AWACS 등 모든 numeric feature에 적용 가능 |
| 비대칭 대응 | Ref >> Compare 상황에서도 정상 작동 (Ref 기준 scaling) |
| 자동화 | 수작업 비교를 자동화하여 일관된 판정 기준 적용 |
| False Alarm 최소 | Ref vs Ref 검증에서 오탐률 0 확인 |

### 5.3 향후 확장 방향

| 순서 | 확장 내용 | 목적 | 기대 효과 |
|------|-----------|------|-----------|
| 1 | PCA 보조 Score | Feature 상관성에 의한 noise 감소 | Pattern E(미세 변화) 검출 개선 |
| 2 | 통계 검정 교차 검증 | Mann-Whitney U + KS Test | Score의 통계적 유의성 확인 |
| 3 | 시계열 기반 분석 | CUSUM, EWMA | Pattern D(점진적 trend) 검출 |
| 4 | Matched Comparison | 설비/시간 metadata 매칭 | Confounding 통제 |
| 5 | Stage 0~4 체계 연계 | 현업 프로세스 통합 | 운영 자동화 |
| 6 | 사례 DB + Supervised | 과거 판정 사례 학습 | 판정 정밀화 |

---

## 6. Streamlit 대시보드

실제 데이터를 업로드하여 3종 Score 분석을 대화형으로 수행할 수 있습니다.

```bash
streamlit run src/app.py
```

**주요 기능:**
- CSV/Excel 파일 업로드 (Ref / Compare 각각)
- 분석 파라미터 실시간 조정 (Top-K Ratio, Tail Percentile, Outlier Threshold 등)
- 3종 Score 상세 탭 (Shift / Tail / Outlier) — Interactive Plotly 차트
- Feature 상세 비교 (분포 히스토그램 + Box Plot)
- Outlier Wafer 시각화 (Shift vs Tail Feature 분류)
- 결과 다운로드 (Feature Importance CSV, Outlier Wafer CSV, Summary TXT)

---

## 7. 프로젝트 구조

```
change_point_detection/
├── README.md                      # 프로젝트 문서 (과제 배경 ~ 인사이트)
├── requirements.txt               # 의존성
├── src/
│   ├── eco_change_detection.py    # 핵심 파이프라인 (3종 Score + 판정)
│   ├── run_experiment.py          # 실험 실행 + 시각화 10종 생성
│   └── app.py                     # Streamlit 대시보드
├── results/                       # 생성된 시각화 이미지
└── docs/                          # GitHub Pages
    ├── index.html                 # 프로젝트 페이지
    └── images/                    # 시각화 이미지
```

## 8. 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 실험 실행 (합성 데이터 생성 + 파이프라인 + 시각화 10종)
python src/run_experiment.py

# Streamlit 대시보드 실행
streamlit run src/app.py
```

---

## 9. 참고 문헌

1. Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th Ed. Wiley.
2. Hawkins, D.M. & Olwell, D.H. (1998). *Cumulative Sum Charts and Charting for Quality Improvement*. Springer.
3. Rousseeuw, P.J. & Croux, C. (1993). "Alternatives to the Median Absolute Deviation." *JASA*, 88(424), 1273-1283.
4. Hubert, M. & Vandervieren, E. (2008). "An Adjusted Boxplot for Skewed Distributions." *CSDA*, 52(12), 5186-5201.
5. Hodge, V.J. & Austin, J. (2004). "A Survey of Outlier Detection Methodologies." *AI Review*, 22(2), 85-126.
6. Apley, D.W. & Shi, J. (2001). "A Factor-Analysis Method for Diagnosing Variability in Multivariate Manufacturing Processes." *Technometrics*, 43(1), 84-95.
7. Chen, S. & Nembhard, H.B. (2011). "High-Dimensional Process Monitoring and Diagnosis via Sparse Principal Components." *IIE Transactions*, 43(10), 685-699.
8. Qiu, P. (2013). *Introduction to Statistical Process Control*. Chapman & Hall/CRC.
9. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *JRSS-B*, 57(1), 289-300.
10. Miller, P., Swanson, R.E. & Heckler, C.E. (1998). "Contribution Plots: A Missing Link in Multivariate Quality Control." *Applied Mathematics and Computer Science*, 8(4), 775-792.
