# 대청댐 유해남조류 발생 예측 AI 모델

## 개요

대청댐(문의·추동·회남)에서 수집한 수질·기상·수문 데이터를 바탕으로  
유해남조류 4종(Microcystis, Anabaena, Oscillatoria, Aphanizomenon)의 **7일 선행 예측** 모델을 구축하고  
조류경보제 발령 시점을 사전에 탐지하는 의사결정 지원 시스템입니다.

---

## 사용 데이터

| 구분 | 출처 | 내용 |
|------|------|------|
| 조류 모니터링 | 한국수자원공사(K-water) | 대청댐 주간 조류 세포수 (4종), 수질 항목 |
| 기상 데이터 | 기상청 기상자료개방포털 | 대전·청주·보은 관측소 일별 기상자료 |
| 수문/댐운영 | 한국수자원공사(K-water) | 저수위·저수량·유입량·방류량 등 |

**기간**: 2016-01-05 ~ 2025-12-22

---

## 조류경보 발령 기준 (기후에너지환경부 운영지침)

| 단계 | 유해남조류 세포수 (cells/mL) | 적용 조건 |
|------|---------------------------|----------|
| 관심 | 1,000 이상 | **2회 연속** 기준 초과 시 발령 |
| 경계 | 10,000 이상 | **2회 연속** 기준 초과 시 발령 |
| 조류대발생 | 1,000,000 이상 | **2회 연속** 기준 초과 시 발령 |

---

## 전처리 과정 (`data_preprocessing.ipynb`)

- 주 단위 조류 데이터 → **일 단위 확장** (exponential growth 기반)
- 수질 데이터 → **선형 보간** (interpolation)
- 기상·수문 데이터 → **날짜 기준 merge**
- 2회 연속 측정 기준 발령단계 컬럼 생성 (`발령단계_2회`)
- 최종 통합 데이터셋 저장: `finaldata.csv`

---

## AI 모델 (`algae_prediction_model.ipynb`)

### 피처 엔지니어링
- **Lag 피처**: 1, 3, 7, 14일 전 값
- **Rolling 통계**: 7, 14, 30일 이동 평균·최대·표준편차
- **계절 인코딩**: sin/cos 변환 (월별 주기성 반영)
- **교호작용**: 수온 × 일사량 (조류 성장 환경)
- **누적 강수량**: 3, 7, 14일 합계

### 모델 구성
| 모델 | 역할 |
|------|------|
| LightGBM (분류) | 7일 후 발령단계 예측 (Optuna 50-trial HPO) |
| XGBoost (분류) | 비교 모델 |
| LightGBM (회귀) | total_cyano 수치 예측 → 조류대발생 threshold 후처리 |

### 데이터 분할
- **Train**: 2016-01-05 ~ 2021-12-31
- **Validation**: 2022-01-01 ~ 2023-12-31
- **Test**: 2024-01-01 ~ 2025-12-22

---

## 주요 결과 (테스트 기간 기준)

| 지표 | 값 |
|------|-----|
| Accuracy | 0.864 |
| Macro F1-Score | 0.622 |
| **경보 적중률 (Hit Rate)** | **98.1%** (611/623일) |
| 오경보율 (False Alarm) | 1.6% |
| 관심 단계 적중 | 96.5% |
| 경계 단계 적중 | 100% |

### 주요 영향 인자 (SHAP 분석)

1. `log_cyano_now` — 현재 유해남조류 세포수 (log)
2. `alert_binary_now` — 현재 경보 발령 여부
3. `수온(℃)_r30_max` — 30일 최고 수온
4. `total_cyano_lag1` — 1일 전 세포수
5. `수온(℃)_r14_max` — 14일 최고 수온

---

## 파일 구조

```
├── data_preprocessing.ipynb       # 데이터 전처리
├── algae_prediction_model.ipynb   # AI 모델 (메인)
├── finaldata.csv                  # 통합 학습 데이터셋
├── test_predictions.csv           # 테스트 예측 결과
├── algae_model_artifacts.pkl      # 저장된 모델 객체
├── requirements.txt               # 패키지 의존성
└── *.png                          # 시각화 결과물
```

---

## 가설 검증 결과

### 이미 검증된 가설 (모델·SHAP 기반)

> 확인 위치: `algae_prediction_model.ipynb` § 7. SHAP 분석 / § 6. 리드타임 분석 / § 2. EDA

| 가설 | 결과 | 근거 | 확인 파일 |
|------|------|------|----------|
| 현재 세포수가 미래 경보의 가장 강한 예측자다 | ✅ 채택 | SHAP 1위 `log_cyano_now` (r=0.659) | `shap_importance.png` |
| 수온(rolling max)이 핵심 환경 인자다 | ✅ 채택 | SHAP 3위 `수온_r30_max`, 5위 `수온_r14_max` | `shap_importance.png` |
| 예측 정확도는 리드타임이 길어질수록 낮아진다 | ✅ 채택 | 1~14일 리드타임 성능 곡선 | `lead_time_performance.png` |
| 조류 발생은 여름(6~9월)에 집중된다 | ✅ 채택 | 월별 평균 세포수 패턴 | `eda_overview.png` |

### 추가 가설 검증

> 확인 위치: `algae_prediction_model.ipynb` § 12. 추가 가설 검증

| # | 가설 | 결과 | 핵심 수치 | 확인 파일 |
|---|------|------|----------|----------|
| B | 2회 연속 기준이 단순 임계값보다 오경보를 줄인다 | ✅ 채택 | 관심 -136일, 경계 -128일 (총 264건 필터링) | `hypo_B_consecutive_vs_simple.png` |
| D | Chl-a가 세포수보다 선행 지표다 | ❌ 기각 | 세포수 r=0.659 vs Chl-a r=0.388 (70% 차이) | `hypo_D_chla_vs_cyano.png` |
| E | 강수량·방류량이 조류를 억제한다 | ❌ 기각 | 강수량 r=+0.166 (억제 아님), 방류량 r=+0.043 (무관) | `hypo_E_correlation.png` |
| F | 지점별 조류 피크 시기·강도가 다르다 | ✅ 채택 | 회남 17,959 cells/mL(8월) > 추동 8,920(9월) > 문의 9,775(8월) | `hypo_F_site_comparison.png` |

> **인사이트**: 강수량은 단기적으로 조류를 억제하지 않고, 오히려 영양염류 유입을 통해 이후 성장을 유발할 수 있음. 방류량 단독 조절보다 수온·체류시간 관리가 더 효과적인 억제 수단으로 판단됨.

---

## 실행 방법

```bash
# 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Mac/Linux

# Jupyter 실행
.venv/Scripts/jupyter notebook
```

1. `data_preprocessing.ipynb` 실행 → `finaldata.csv` 생성
2. `algae_prediction_model.ipynb` 실행 → 모델 학습 및 결과 저장
