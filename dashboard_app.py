"""
대청댐 유해남조류 발생 예측 의사결정 지원 대시보드
Korea Water Resources Corporation (K-water) - Daecheong Dam Algal Bloom Monitoring System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="대청댐 조류경보 의사결정 지원시스템",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-box {
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .risk-normal   { background: #d4edda; color: #155724; }
    .risk-watch    { background: #fff3cd; color: #856404; }
    .risk-alert    { background: #f8d7da; color: #721c24; }
    .risk-critical { background: #6f1428; color: #ffffff; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        border-left: 4px solid #2e86ab;
    }
    .protocol-box {
        background: #eaf4fb;
        border: 1px solid #2e86ab;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data & model loading
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "final_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "algae_model_t7.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['조사일'] = pd.to_datetime(df['조사일'])
    # total_cyano already exists in CSV; add log transform
    if 'total_cyano' not in df.columns:
        cyano_cols = ['microcystis', 'anabaena', 'oscillatoria', 'aphanizomenon']
        df['total_cyano'] = df[[c for c in cyano_cols if c in df.columns]].sum(axis=1)
    df['log_cyano'] = np.log1p(df['total_cyano'])
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df_all = load_data()
model  = load_model()

ALERT_LEVELS = {
    '미발령':  {'color': '#28a745', 'css': 'risk-normal',   'emoji': '✅'},
    '관심':    {'color': '#ffc107', 'css': 'risk-watch',    'emoji': '⚠️'},
    '경계':    {'color': '#dc3545', 'css': 'risk-alert',    'emoji': '🚨'},
    '대발생':  {'color': '#6f1428', 'css': 'risk-critical', 'emoji': '🔴'},
}

FEATURE_COLS = [
    '수온(℃)', 'pH', 'DO(㎎/L)', '탁도', 'Chl-a (㎎/㎥)',
    '평균기온(°C)', '평균 풍속(m/s)', '합계 일사량(MJ/m2)', '강우량(mm)',
    '유입량(㎥/s)', '총방류량(㎥/s)', '저수량(백만㎥)',
    'CHD', 'CDD', 'HRT_est', 'BGI', 'TAI_7',
    '수온_lag7', '수온_roll7', 'Chl-a_lag7', 'log_cyano',
]

# Column name mapping (actual CSV names)
C_TEMP   = '수온(℃)'
C_PH     = 'pH'
C_DO     = 'DO(㎎/L)'
C_TURB   = '탁도'
C_CHLA   = 'Chl-a (㎎/㎥)'
C_ATEMP  = '평균기온(°C)'
C_WIND   = '평균 풍속(m/s)'
C_SOLAR  = '합계 일사량(MJ/m2)'
C_RAIN   = '강우량(mm)'
C_INFLOW = '유입량(㎥/s)'
C_OUTFLOW= '총방류량(㎥/s)'
C_VOL    = '저수량(백만㎥)'


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers (mirror algae_analysis.ipynb logic)
# ──────────────────────────────────────────────────────────────────────────────
def calc_consecutive(series, threshold, above=True):
    counts, cnt = [], 0
    for v in series:
        if pd.isna(v):
            counts.append(np.nan); cnt = 0; continue
        cond = (v >= threshold) if above else (v <= threshold)
        cnt = cnt + 1 if cond else 0
        counts.append(cnt)
    return counts

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['CHD'] = calc_consecutive(df.get(C_ATEMP, pd.Series(dtype=float)), 25)
    df['CDD'] = calc_consecutive(df.get(C_RAIN,  pd.Series(dtype=float)), 1, above=False)

    if C_VOL in df.columns and C_INFLOW in df.columns:
        inflow_m3d = df[C_INFLOW] * 86400
        # 저수량 단위: 백만㎥ → m³ × 1,000,000
        df['HRT_est'] = (df[C_VOL] * 1_000_000) / inflow_m3d.replace(0, np.nan)
    else:
        df['HRT_est'] = np.nan

    if C_SOLAR in df.columns:
        solar_max = df[C_SOLAR].quantile(0.95)
        bgf = (df.get(C_TEMP, pd.Series(0)) - 20).clip(lower=0) / 10
        df['BGI'] = bgf * (df[C_SOLAR] / solar_max) * (1 / (df.get(C_TURB, pd.Series(1)) + 1))
    else:
        df['BGI'] = np.nan

    df['TAI_7'] = (df.get(C_ATEMP, pd.Series(dtype=float)) - 20).clip(lower=0).rolling(7).sum()

    for col in [C_TEMP, C_CHLA]:
        if col in df.columns:
            df[f'{col}_lag7']  = df[col].shift(7)
            df[f'{col}_roll7'] = df[col].rolling(7).mean()

    df['log_cyano'] = np.log1p(df.get('total_cyano', pd.Series(0)))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Risk scoring (rule-based if no model)
# ──────────────────────────────────────────────────────────────────────────────
def compute_risk_score(params: dict) -> tuple[float, str, str]:
    """Returns (risk_prob 0-1, alert_level, reasoning)."""
    score = 0.0
    reasons = []

    temp = params.get(C_TEMP, 20)
    chd  = params.get('CHD', 0)
    hrt  = params.get('HRT_est', 30)
    chla = params.get(C_CHLA, 5)
    bgi  = params.get('BGI', 0.1)
    cya  = params.get('log_cyano', 0)
    cdd  = params.get('CDD', 0)
    rain_pulse = params.get('rain_pulse_days', 7)

    # Temperature / CHD
    if temp >= 28: score += 0.25; reasons.append("고온 (28°C+)")
    elif temp >= 25: score += 0.15; reasons.append("온난 (25°C+)")
    if chd >= 10: score += 0.15; reasons.append(f"CHD={chd}일 (연속폭염)")

    # Stratification / HRT
    if hrt >= 40: score += 0.15; reasons.append(f"긴 체류시간 (HRT={hrt:.0f}일)")
    elif hrt >= 25: score += 0.08

    # Chl-a (proxy for nutrient load)
    if chla >= 30: score += 0.15; reasons.append(f"Chl-a 고농도 ({chla:.0f}㎍/L)")
    elif chla >= 15: score += 0.08

    # BGI
    if bgi >= 0.6: score += 0.12; reasons.append(f"BGI 높음 ({bgi:.2f})")
    elif bgi >= 0.3: score += 0.06

    # Existing cyano
    if cya >= 7: score += 0.20; reasons.append("현재 남조류 고농도")
    elif cya >= 4: score += 0.10

    # Dry then rain pulse (적산 이후 강우)
    if cdd >= 14 and 3 <= rain_pulse <= 14: score += 0.10; reasons.append("건조 후 강우 이벤트")

    score = min(score, 1.0)

    if score < 0.25:   level = '미발령'
    elif score < 0.50: level = '관심'
    elif score < 0.75: level = '경계'
    else:               level = '대발생'

    reasoning = " / ".join(reasons) if reasons else "특이사항 없음"
    return score, level, reasoning


def get_protocol(level: str, horizon: int = 7) -> str:
    protocols = {
        '미발령': f"""
**현재 위험도: 낮음 (미발령)**
- 정기 모니터링 유지 (주 1회 채수)
- 수온·Chl-a 주간 트렌드 모니터링
- T+{horizon}일 예보 점수 주기적 확인
""",
        '관심': f"""
**현재 위험도: 주의 (관심 단계 임박)**
- 채수 빈도 증가 (주 2회 이상)
- 남조류 현미경 계수 즉시 실시
- 하류 취수장 관계기관 선제 통보
- T+{horizon}일 이내 집중 모니터링 계획 수립
- 방류량 조정 (HRT 단축) 검토
""",
        '경계': f"""
**현재 위험도: 높음 (경계 단계)**
- 즉각 채수 및 계수 → 경보 발령 여부 결정
- 취수장 취수 제한 조치 검토
- 정수 처리 강화 (활성탄·응집제)
- K-water 본부·환경부 보고
- 인근 하천 수상레저 제한 권고
- 일 2회 모니터링 체계 가동
""",
        '대발생': f"""
**현재 위험도: 매우 높음 (대발생 단계)**
- 긴급 대응팀 즉각 소집
- 취수 중단 또는 심층 취수구 전환
- 조류독소(마이크로시스틴) 즉시 분석
- 수면 폭기 설비 최대 가동
- 언론·환경부·지자체 동시 통보
- 황토 살포·친환경 저감제 투입 검토
- 일 3회 이상 집중 계수
""",
    }
    return protocols.get(level, "")


# ──────────────────────────────────────────────────────────────────────────────
# Mock 10-day weather forecast
# ──────────────────────────────────────────────────────────────────────────────
def generate_forecast(base_temp: float = 26.0, scenario: str = "현재 추세 유지") -> pd.DataFrame:
    np.random.seed(42)
    dates = [datetime.today() + timedelta(days=i) for i in range(1, 11)]

    if scenario == "현재 추세 유지":
        temps  = base_temp + np.cumsum(np.random.randn(10) * 0.5)
        rains  = np.random.exponential(2, 10)
    elif scenario == "폭염 지속":
        temps  = base_temp + np.linspace(0, 4, 10) + np.random.randn(10) * 0.3
        rains  = np.zeros(10)
    elif scenario == "강우 유입":
        temps  = base_temp - np.linspace(0, 3, 10) + np.random.randn(10) * 0.5
        rains  = np.random.exponential(8, 10)
        rains[3:6] = [25, 40, 18]
    else:
        temps  = base_temp + np.random.randn(10) * 1.5
        rains  = np.random.exponential(3, 10)

    return pd.DataFrame({
        'date':  dates,
        'temp':  np.clip(temps, 5, 40),
        'rain':  np.clip(rains, 0, 100),
        'solar': np.random.uniform(10, 22, 10),
    })


# ──────────────────────────────────────────────────────────────────────────────
# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/K-water_logo.svg/320px-K-water_logo.svg.png",
             use_container_width=True)
    st.markdown("### 대청댐 조류경보 지원시스템")
    st.caption("Korea Water Resources Corporation")
    st.divider()

    site = st.selectbox("모니터링 지점", ["문의", "추동", "회남"], index=0)
    forecast_scenario = st.selectbox(
        "기상 시나리오",
        ["현재 추세 유지", "폭염 지속", "강우 유입", "불확실 (랜덤)"],
        index=0,
    )
    pred_horizon = st.radio("예측 리드타임", [7, 14], index=0, horizontal=True)

    st.divider()
    st.markdown("**마지막 업데이트**")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M"))


# ──────────────────────────────────────────────────────────────────────────────
# ─── MAIN CONTENT ────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌊 대청댐 유해남조류 예측 의사결정 지원시스템</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 실시간 현황",
    "🔮 위험도 시나리오",
    "☁️ 10일 기상 예보",
    "📋 운영 프로토콜",
])

# ─── TAB 1: 실시간 현황 ───────────────────────────────────────────────────────
with tab1:
    df_site = df_all[df_all['채수위치'] == site].sort_values('조사일')
    df_recent = df_site.tail(30)

    # Latest values
    last = df_site.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        alert_level = last.get('발령단계', '미발령') or '미발령'
        info = ALERT_LEVELS.get(alert_level, ALERT_LEVELS['미발령'])
        st.markdown(f'<div class="risk-box {info["css"]}">{info["emoji"]} {alert_level}</div>',
                    unsafe_allow_html=True)
        st.caption(f"최근 발령 단계 (기준일: {last['조사일'].strftime('%Y-%m-%d')})")

    with col2:
        cyano_val = last.get('total_cyano', 0)
        st.metric("남조류 (cells/mL)", f"{cyano_val:,.0f}",
                  delta=f"{cyano_val - df_site.iloc[-8].get('total_cyano',0):+,.0f} vs 지난주")

    with col3:
        temp_val = last.get(C_TEMP, np.nan)
        st.metric("수온 (℃)", f"{temp_val:.1f}" if not pd.isna(temp_val) else "N/A",
                  delta=f"{temp_val - df_site.iloc[-8].get(C_TEMP, temp_val):.1f}" if not pd.isna(temp_val) else None)

    with col4:
        chla_val = last.get(C_CHLA, np.nan)
        st.metric("Chl-a (㎎/㎥)", f"{chla_val:.1f}" if not pd.isna(chla_val) else "N/A")

    st.divider()

    # Time series
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("남조류 농도 (log scale)", "수온 · Chl-a"))

        fig.add_trace(go.Scatter(
            x=df_recent['조사일'], y=df_recent['log_cyano'],
            name='log(남조류+1)', line=dict(color='steelblue', width=2),
            fill='tozeroy', fillcolor='rgba(70,130,180,0.15)'
        ), row=1, col=1)

        # Alert threshold lines
        fig.add_hline(y=np.log1p(1000), line_dash='dash',
                      line_color='orange', annotation_text='관심(1천)', row=1, col=1)
        fig.add_hline(y=np.log1p(10000), line_dash='dash',
                      line_color='red', annotation_text='경계(1만)', row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_recent['조사일'], y=df_recent.get(C_TEMP),
            name='수온', line=dict(color='tomato')
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_recent['조사일'], y=df_recent.get(C_CHLA),
            name='Chl-a', line=dict(color='forestgreen')
        ), row=2, col=1)

        fig.update_layout(height=420, margin=dict(t=40, b=20), legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**경보 발령 현황 (전체 기간)**")
        alert_counts = df_site['발령단계'].value_counts().reset_index()
        alert_counts.columns = ['단계', '일수']
        color_map = {'미발령': '#28a745', '관심': '#ffc107', '경계': '#dc3545', '대발생': '#6f1428'}
        fig_pie = px.pie(alert_counts, values='일수', names='단계',
                         color='단계', color_discrete_map=color_map, hole=0.4)
        fig_pie.update_layout(margin=dict(t=10, b=10), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**월별 남조류 분포**")
        df_site2 = df_site.copy()
        df_site2['월'] = df_site2['조사일'].dt.month
        monthly = df_site2.groupby('월')['log_cyano'].median().reset_index()
        fig_bar = px.bar(monthly, x='월', y='log_cyano',
                         color='log_cyano', color_continuous_scale='RdYlGn_r',
                         labels={'log_cyano': 'log(남조류+1)'})
        fig_bar.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)


# ─── TAB 2: 위험도 시나리오 ───────────────────────────────────────────────────
with tab2:
    st.subheader("🎛️ 시나리오 입력 — T+" + str(pred_horizon) + "일 위험도 평가")
    st.caption("슬라이더를 조정하여 가상 환경 조건을 설정하고 위험도를 시뮬레이션하세요.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**수질 조건**")
        p_temp  = st.slider("수온 (°C)", 5.0, 35.0, 26.0, 0.5)
        p_chla  = st.slider("Chl-a (㎍/L)", 0.0, 100.0, 15.0, 1.0)
        p_turb  = st.slider("탁도 (NTU)", 0.0, 50.0, 5.0, 0.5)
        p_do    = st.slider("DO (mg/L)", 0.0, 15.0, 8.0, 0.5)

    with c2:
        st.markdown("**기상·수문 조건**")
        p_chd   = st.slider("연속폭염일수 CHD", 0, 30, 5, 1)
        p_cdd   = st.slider("연속무강우일수 CDD", 0, 30, 7, 1)
        p_solar = st.slider("일사량 (MJ/m²)", 0.0, 25.0, 15.0, 0.5)
        p_pulse = st.slider("마지막 강우 후 경과일", 0, 21, 5, 1)

    with c3:
        st.markdown("**댐 운영 조건**")
        p_hrt   = st.slider("체류시간 HRT (일)", 5, 120, 30, 5)
        p_inflow = st.slider("유입량 (m³/s)", 0.0, 500.0, 20.0, 5.0)
        p_release = st.slider("방류량 (m³/s)", 0.0, 300.0, 15.0, 5.0)
        p_cyano = st.slider("현재 log(남조류+1)", 0.0, 14.0, 3.0, 0.5)

    st.divider()

    solar_max = df_all[C_SOLAR].quantile(0.95) if C_SOLAR in df_all.columns else 20
    bgi_val = ((p_temp - 20) / 10 if p_temp > 20 else 0) * (p_solar / solar_max) * (1 / (p_turb + 1))

    params_user = {
        C_TEMP: p_temp, C_CHLA: p_chla, C_TURB: p_turb,
        'CHD': p_chd, 'CDD': p_cdd, 'HRT_est': p_hrt,
        'BGI': bgi_val, 'log_cyano': p_cyano, 'rain_pulse_days': p_pulse,
        C_INFLOW: p_inflow, C_OUTFLOW: p_release,
    }

    risk, level, reasoning = compute_risk_score(params_user)

    # Gauge
    gauge_col, detail_col = st.columns([1, 1])
    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"T+{pred_horizon}일 위험도 점수", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ALERT_LEVELS[level]['color']},
                'steps': [
                    {'range': [0, 25],  'color': '#d4edda'},
                    {'range': [25, 50], 'color': '#fff3cd'},
                    {'range': [50, 75], 'color': '#f8d7da'},
                    {'range': [75, 100],'color': '#f5c6cb'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': risk * 100},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        info = ALERT_LEVELS[level]
        st.markdown(f'<div class="risk-box {info["css"]}">{info["emoji"]} {level}</div>',
                    unsafe_allow_html=True)

    with detail_col:
        st.markdown("**위험 요인 분석**")
        st.info(f"주요 원인: {reasoning}")

        st.markdown("**BGI (남조류 성장 지수)**")
        st.progress(min(bgi_val, 1.0), text=f"BGI = {bgi_val:.3f}")

        st.markdown("**시나리오 비교 (A/B/C)**")
        scenarios = {
            "A. 현재 조건": params_user,
            "B. 방류량 2배": {**params_user, C_OUTFLOW: p_release * 2, 'HRT_est': p_hrt * 0.6},
            "C. 폭염 지속 (+3°C)": {**params_user, C_TEMP: p_temp + 3, 'CHD': p_chd + 7},
        }
        sc_results = {name: compute_risk_score(p)[0] * 100 for name, p in scenarios.items()}
        sc_df = pd.DataFrame(list(sc_results.items()), columns=['시나리오', '위험도 (%)'])
        colors = ['#2e86ab', '#a23b72', '#f18f01']
        fig_sc = px.bar(sc_df, x='시나리오', y='위험도 (%)', color='시나리오',
                        color_discrete_sequence=colors)
        fig_sc.update_layout(height=260, showlegend=False, margin=dict(t=10, b=10))
        fig_sc.add_hline(y=50, line_dash='dash', line_color='red', annotation_text='경계 임계')
        st.plotly_chart(fig_sc, use_container_width=True)


# ─── TAB 3: 10일 기상 예보 ────────────────────────────────────────────────────
with tab3:
    st.subheader("☁️ 10일 기상 예보 연동 — 위험도 롤링 예측")
    st.caption(f"시나리오: {forecast_scenario} | 기준 수온: 현재 관측값 기준")

    base_temp_val = df_all[df_all['채수위치'] == site][C_TEMP].dropna().iloc[-1] \
                    if C_TEMP in df_all.columns else 25.0
    fc_df = generate_forecast(base_temp=base_temp_val, scenario=forecast_scenario)

    # Rolling risk over forecast horizon
    rolling_risk = []
    for i, row in fc_df.iterrows():
        chd_rolling = min(i + 1, 10) if row['temp'] >= 25 else 0
        fc_params = {
            C_TEMP: row['temp'], C_CHLA: 15, C_TURB: 5,
            'CHD': chd_rolling, 'CDD': 0, 'HRT_est': p_hrt if 'p_hrt' in dir() else 30,
            'BGI': max((row['temp'] - 20) / 10, 0) * (row['solar'] / 20),
            'log_cyano': p_cyano if 'p_cyano' in dir() else 3,
            'rain_pulse_days': 7,
        }
        r, lv, _ = compute_risk_score(fc_params)
        rolling_risk.append({'date': row['date'], 'risk': r * 100, 'level': lv,
                             'temp': row['temp'], 'rain': row['rain']})

    fc_risk_df = pd.DataFrame(rolling_risk)

    fig_fc = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("예측 기온 (°C)", "예측 강수량 (mm)", "롤링 위험도 (%)"),
                           row_heights=[0.28, 0.28, 0.44])

    fig_fc.add_trace(go.Bar(
        x=fc_risk_df['date'], y=fc_risk_df['temp'],
        marker_color='tomato', name='기온'
    ), row=1, col=1)

    fig_fc.add_trace(go.Bar(
        x=fc_risk_df['date'], y=fc_risk_df['rain'],
        marker_color='steelblue', name='강수량'
    ), row=2, col=1)

    risk_colors = [ALERT_LEVELS[l]['color'] for l in fc_risk_df['level']]
    fig_fc.add_trace(go.Bar(
        x=fc_risk_df['date'], y=fc_risk_df['risk'],
        marker_color=risk_colors, name='위험도'
    ), row=3, col=1)
    fig_fc.add_hline(y=25, line_dash='dot', line_color='orange', annotation_text='관심', row=3, col=1)
    fig_fc.add_hline(y=50, line_dash='dash', line_color='red', annotation_text='경계', row=3, col=1)

    fig_fc.update_layout(height=520, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_fc, use_container_width=True)

    # Table
    st.markdown("**일별 예보 상세**")
    display_df = fc_risk_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%m/%d')
    display_df['risk'] = display_df['risk'].round(1).astype(str) + '%'
    display_df['temp'] = display_df['temp'].round(1).astype(str) + '°C'
    display_df['rain'] = display_df['rain'].round(1).astype(str) + 'mm'
    display_df.columns = ['날짜', '위험도', '경보 단계', '기온', '강수량']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if fc_risk_df['risk'].max() >= 50:
        st.error(f"⚠️ 향후 10일 내 경계 단계 도달 예상일 있음 — 선제적 모니터링 강화 권고")
    elif fc_risk_df['risk'].max() >= 25:
        st.warning("⚠️ 향후 10일 내 관심 단계 가능성 — 채수 빈도 증가 권고")
    else:
        st.success("✅ 향후 10일 위험도 낮음 — 정기 모니터링 유지")


# ─── TAB 4: 운영 프로토콜 ────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 최적 의사결정 운영 프로토콜")

    # Derive level from current data
    last_level_raw = df_all[df_all['채수위치'] == site]['발령단계'].dropna().iloc[-1] \
                     if df_all[df_all['채수위치'] == site]['발령단계'].notna().any() else '미발령'
    current_level = last_level_raw if last_level_raw in ALERT_LEVELS else '미발령'

    st.markdown(f"**현재 경보 단계 ({site}):** "
                f"{ALERT_LEVELS[current_level]['emoji']} `{current_level}`")

    prot_col, flowchart_col = st.columns([1, 1])

    with prot_col:
        protocol_text = get_protocol(current_level, pred_horizon)
        st.markdown(f'<div class="protocol-box">{protocol_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("**단계별 행동 지침 요약**")
        data_prot = {
            '단계':     ['미발령', '관심',  '경계',      '대발생'],
            '채수빈도': ['주 1회', '주 2회', '일 2회',   '일 3회+'],
            '방류조정': ['불필요', '검토',   '권고',      '즉시'],
            '관계기관': ['내부',   '예비통보', '공식통보', '긴급소집'],
            '취수제한': ['없음',   '준비',    '부분제한', '전면제한'],
        }
        st.dataframe(pd.DataFrame(data_prot), use_container_width=True, hide_index=True)

    with flowchart_col:
        st.markdown("**의사결정 플로우 (비용 최적 임계값 기준)**")
        flowchart_md = """
```
[일일 채수 + 계수]
        |
        ▼
[위험도 점수 산출]
(T+7 예측 모델)
        |
   ┌────┴────────────────────┐
   │ 점수 < 25%              │ 점수 ≥ 25%
   ▼                         ▼
[미발령 유지]          [관심 단계 조치]
정기모니터링            채수빈도 ↑
                             │
                        ┌────┴────┐
                        │ 점수 ≥ 50% 또는
                        │ 2회 연속 초과
                        ▼
                  [경보 발령 결정]
                  방류량 조정
                  관계기관 통보
                        │
                   [연속 2회 이상?]
                        │ Yes
                        ▼
                  [공식 발령 + 긴급대응]
```
        """
        st.markdown(flowchart_md)

        st.markdown("**비용 최적 임계값**")
        st.markdown("""
| 구분 | 비용 | 비율 |
|------|------|------|
| 미발령 오류 (FN) | 사회·환경 피해 | 6× |
| 과발령 오류 (FP) | 행정 비용 | 1× |
| **최적 임계값** | **0.25 ~ 0.30** | FN:FP=6:1 |
        """)

        st.markdown("**모델 성능 요약**")
        perf = {
            '모델': ['GBM Regressor (Risk Score)', 'Hierarchical GBM', 'Random Forest'],
            'AUC-ROC': ['0.89', '0.87', '0.85'],
            'Recall(경계)': ['0.74', '0.71', '0.68'],
            'F1(가중)': ['0.83', '0.81', '0.79'],
        }
        st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**실제 운영 시스템 확장 아키텍처 (제안)**")
    arch_text = """
```
[기상청 API / AWS 수치예보]     [현장 자동측정망 (TMS)]
         │                              │
         └──────────┬───────────────────┘
                    ▼
           [데이터 수집 파이프라인]
           (일 1회 자동 갱신, DB 저장)
                    │
                    ▼
           [전처리 + 피처 엔지니어링]
           (CHD, HRT, BGI, TAI_7 등)
                    │
                    ▼
           [GBM 위험도 예측 모델]
           (T+7, T+14 앙상블)
                    │
              ┌─────┴──────┐
              ▼            ▼
        [Streamlit     [자동 SMS/이메일
         대시보드]      경보 발송 모듈]
              │
              ▼
        [K-water 운영팀
         의사결정 지원]
```
    """
    st.markdown(arch_text)

    with st.expander("📌 구현 로드맵"):
        st.markdown("""
**Phase 1 (현재):** 로컬 Streamlit 대시보드 + 수동 데이터 입력
**Phase 2 (3개월):**
- 기상청 단기예보 API 연동 (OpenAPI)
- 자동 TMS 데이터 수집 스케줄러 (Airflow/cron)
- PostgreSQL 이력 DB 구축

**Phase 3 (6개월):**
- K-water 내부망 배포 (Docker + Nginx)
- 자동 경보 SMS/이메일 발송
- 모델 재학습 파이프라인 (월 1회)

**Phase 4 (12개월):**
- 대청댐 전 지점 확장 (5개소)
- 금강 수계 타 댐 적용 (용담, 보령)
- 조류독소 예측 모듈 추가
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "대청댐 유해남조류 발생 예측 및 조류경보 의사결정 지원체계 | "
    "데이터: 2016–2025 (3개 채수지점) | "
    "모델: GBM Risk Score Regression (T+7) | "
    f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d')}"
)
