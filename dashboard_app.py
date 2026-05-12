"""
대청댐 유해남조류 발생 예측 의사결정 지원 대시보드
Korea Water Resources Corporation (K-water) - Daecheong Dam Algal Bloom Monitoring System

AX-Competition 프로젝트용 (`final_data.csv` + 선택 `algae_model_t7.pkl`).
README 11절 파이프라인: data_eda → algae_analysis(모델 저장) → 본 대시보드.

개선 사항:
  - TAB1: 카드형 KPI·남조류 막대(log)·연도별 첫 이상일·전 지점 요약(전부 실데이터)
  - TAB2: 지점 분위 적응 구역 + 최근행 슬라이더 초깃값 + 운영 점검표(실데이터) + 접는 상세 권고
  - TAB3: 10일 기상에 기온·강수 **자료출처** 열 분리 + 운영 요약·읽는 법(expander) + 단기·중기·일주년 보정 설명, GBM T+7 타임라인
  - TAB4: 실데이터 기반 단계 카드 + B/C 지표 + 선택 JSON 로드맵
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import json
from html import escape
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
except ImportError:
    requests = None

try:
    import shap
except ImportError:
    shap = None

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="대청댐 조류경보 의사결정 지원시스템",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: bold; color: #1f4e79;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .risk-box {
        border-radius: 10px; padding: 1.2rem;
        text-align: center; font-size: 1.4rem; font-weight: bold;
    }
    .risk-normal   { background: #d4edda; color: #155724; }
    .risk-watch    { background: #fff3cd; color: #856404; }
    .risk-alert    { background: #f8d7da; color: #721c24; }
    .risk-critical { background: #6f1428; color: #ffffff; }
    .protocol-box {
        background: #eaf4fb; border: 1px solid #2e86ab;
        border-radius: 8px; padding: 1rem; margin-top: 0.5rem;
    }
    /* SHAP 해결책 카드 */
    .shap-card {
        border-radius: 8px; padding: 0.9rem 1rem;
        margin-bottom: 0.6rem; border-left: 5px solid;
    }
    .shap-card-high   { background: #fff3f3; border-color: #dc3545; }
    .shap-card-medium { background: #fffbf0; border-color: #ffc107; }
    .shap-card-low    { background: #f0f8f0; border-color: #28a745; }
    .shap-driver { font-weight: bold; font-size: 1rem; margin-bottom: 4px; }
    .shap-value  { font-size: 0.8rem; color: #666; margin-bottom: 6px; }
    .shap-solution { font-size: 0.9rem; }
    /* What-if 테이블 */
    .whatif-improve { color: #155724; font-weight: bold; }
    .whatif-worsen  { color: #721c24; font-weight: bold; }
    /* TAB2: 구역 뱃지 + 카드 변형 (tab2_scenario_solution) */
    .zone-badge {
        display:inline-block; border-radius:6px;
        padding:2px 10px; font-size:0.8rem; font-weight:bold;
    }
    .zone-safe { background:#d4edda; color:#155724; }
    .zone-watch{ background:#fff3cd; color:#856404; }
    .zone-alert{ background:#f8d7da; color:#721c24; }
    .shap-danger { background:#fff3f3; border-left:4px solid #dc3545; border-radius:0 8px 8px 0;
        padding:0.7rem 0.9rem; margin-bottom:0.5rem; }
    .shap-warn   { background:#fffbf0; border-left:4px solid #ffc107; border-radius:0 8px 8px 0;
        padding:0.7rem 0.9rem; margin-bottom:0.5rem; }
    .shap-info   { background:#f0f8ff; border-left:4px solid #2e86ab; border-radius:0 8px 8px 0;
        padding:0.7rem 0.9rem; margin-bottom:0.5rem; }
    .shap-val    { font-size:0.78rem; color:#666; margin-bottom:5px; }
    .shap-sol    { font-size:0.88rem; line-height:1.6; }
    .wi-best { background:#d4edda !important; }
    /* ── TAB4: 운영 프로토콜 (실무형 UI) ─────────────────────────────────── */
    .t4-hero {
        background: linear-gradient(165deg, #f0f5fa 0%, #ffffff 55%);
        border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1.1rem 1.35rem 1.15rem; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(31, 78, 121, 0.08);
    }
    .t4-title { font-size: 1.55rem; font-weight: 800; color: #1f4e79; letter-spacing: -0.03em; margin: 0 0 0.55rem 0; }
    .t4-badges { display: flex; flex-wrap: wrap; gap: 0.45rem 0.65rem; align-items: center; }
    .t4-badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #fff; border: 1px solid #dee2e6; border-radius: 999px;
        padding: 0.32rem 0.85rem; font-size: 0.88rem; font-weight: 600;
        color: #2c3e50; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .t4-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.35rem 0 0.9rem 0; }
    .t4-chip {
        font-size: 0.72rem; font-weight: 600; color: #5a6c7d;
        background: #eef2f6; border-radius: 6px; padding: 0.2rem 0.55rem;
        border: 1px dashed #c5d0dc;
    }
    .t4-sec {
        font-size: 1.05rem; font-weight: 800; color: #1f4e79;
        margin: 0.2rem 0 0.65rem 0; padding-bottom: 0.35rem;
        border-bottom: 2px solid #2e86ab; letter-spacing: -0.02em;
    }
    .t4-proto {
        border-radius: 14px; background: #fff; overflow: hidden; height: 100%;
        border: 1px solid #e8ecf1;
        box-shadow: 0 4px 20px rgba(31, 78, 121, 0.1);
        transition: transform 0.14s ease, box-shadow 0.14s ease;
    }
    .t4-proto:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(31, 78, 121, 0.14); }
    .t4-proto-active {
        outline: 2px solid #2e86ab; outline-offset: 1px;
        box-shadow: 0 6px 26px rgba(46, 134, 171, 0.22);
    }
    .t4-proto-head { padding: 11px 14px; font-weight: 800; font-size: 0.86rem; letter-spacing: -0.02em; border-bottom: 1px solid rgba(0,0,0,0.06); }
    .t4-proto-body {
        padding: 14px 15px 18px; font-size: 0.86rem; line-height: 1.58; color: #2c3e50;
        min-height: 172px; background: #fff;
    }
    .t4-steps { counter-reset: t4s; list-style: none; margin: 0; padding: 0; }
    .t4-steps li {
        counter-increment: t4s; margin-bottom: 11px; padding-left: 2.15rem; position: relative; min-height: 2.35em;
    }
    .t4-steps li::before {
        content: counter(t4s); position: absolute; left: 0; top: 1px;
        width: 1.45rem; height: 1.45rem; border-radius: 50%;
        background: linear-gradient(145deg, #e9ecef, #f8f9fa);
        color: #1f4e79; font-weight: 800; font-size: 0.68rem;
        display: flex; align-items: center; justify-content: center;
        border: 1px solid #dee2e6;
    }
    .t4-num { color: #1f4e79; font-weight: 800; font-size: 0.92em; }
    .t4-muted { color: #6c757d; font-size: 0.78rem; }
    .t4-bc {
        border-radius: 14px; border: 1px solid #cfe2f3;
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 45%);
        padding: 0; overflow: hidden;
        box-shadow: 0 4px 18px rgba(46, 134, 171, 0.1);
    }
    .t4-bc-head {
        padding: 12px 16px; font-weight: 800; font-size: 0.98rem; color: #1f4e79;
        border-bottom: 1px solid #dbe7f3; background: rgba(255,255,255,0.7);
    }
    .t4-bc-kpis { display: flex; flex-wrap: wrap; gap: 0; border-bottom: 1px solid #e9ecef; }
    .t4-bc-kpi {
        flex: 1 1 22%; min-width: 110px; text-align: center; padding: 12px 8px;
        border-right: 1px solid #e9ecef; background: #fff;
    }
    .t4-bc-kpi:last-child { border-right: none; }
    .t4-bc-kpi-val { font-size: 1.25rem; font-weight: 800; color: #1f4e79; line-height: 1.15; }
    .t4-bc-kpi-sub { font-size: 0.68rem; color: #6c757d; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .t4-bc-detail { padding: 12px 16px 14px; font-size: 0.84rem; line-height: 1.55; color: #343a40; }
    .t4-bc-foot {
        margin: 0 16px 14px; padding: 10px 12px; border-radius: 8px;
        background: #f1f5f9; font-size: 0.78rem; color: #495057; border: 1px solid #e2e8f0;
    }
    .t4-rm {
        border-radius: 14px; border: 1px solid #e8ecf1; background: #fff;
        padding: 0; overflow: hidden;
        box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    }
    .t4-rm-head {
        padding: 12px 16px; font-weight: 800; font-size: 0.98rem; color: #1f4e79;
        border-bottom: 1px solid #e9ecef; display: flex; align-items: center; gap: 8px;
    }
    .t4-tag { font-size: 0.65rem; font-weight: 700; color: #fff; background: #2e86ab; padding: 0.15rem 0.45rem; border-radius: 4px; }
    .t4-rm-body { padding: 8px 16px 16px 12px; }
    .t4-rm-item { position: relative; padding: 10px 0 10px 22px; margin-left: 6px; border-left: 2px solid #e9ecef; }
    .t4-rm-item:last-child { border-left-color: transparent; }
    .t4-rm-item::before {
        content: ""; position: absolute; left: -7px; top: 14px; width: 12px; height: 12px; border-radius: 50%;
        background: var(--t4-dot, #666); border: 2px solid #fff;
        box-shadow: 0 0 0 1px #dee2e6;
    }
    .t4-rm-ph { font-weight: 800; color: #2c3e50; font-size: 0.84rem; }
    .t4-rm-tx { font-size: 0.8rem; color: #5a6570; margin-top: 2px; line-height: 1.45; }
    /* ── TAB1: 카드형 현황 ─────────────────────────────────────────────────── */
    .tab1-kpi-card {
        background: #fff; border-radius: 12px; padding: 14px 16px;
        border: 1px solid #e8ecf1; box-shadow: 0 2px 10px rgba(31, 78, 121, 0.07);
        min-height: 128px;
    }
    .tab1-kpi-label { font-size: 0.76rem; color: #6a7580; font-weight: 600; margin-bottom: 8px; }
    .tab1-pill {
        display: inline-block; border-radius: 999px; padding: 5px 12px;
        font-weight: 700; font-size: 0.9rem;
    }
    .tab1-kpi-num { font-size: 1.5rem; font-weight: 800; color: #1f4e79; margin: 10px 0 6px; line-height: 1.15; }
    .tab1-kpi-sub { font-size: 0.76rem; color: #8b95a0; }
    .tab1-trend-up { color: #c0392b; font-weight: 600; font-size: 0.8rem; }
    .tab1-trend-down { color: #1e8449; font-weight: 600; font-size: 0.8rem; }
    .tab1-trend-flat { color: #8899a8; font-size: 0.8rem; }
    .tab1-side-card {
        background: #fff; border-radius: 12px; padding: 12px 14px;
        border: 1px solid #e8ecf1; margin-bottom: 12px;
        box-shadow: 0 1px 8px rgba(31, 78, 121, 0.05);
    }
    .tab1-side-title { font-size: 0.95rem; font-weight: 800; color: #1f4e79; margin-bottom: 8px; }
    .tab1-chip {
        display: inline-block; font-size: 0.68rem; font-weight: 600; color: #5a6c7d;
        background: #eef2f6; border-radius: 6px; padding: 3px 8px; margin-right: 6px; border: 1px dashed #c5d0dc;
    }
    /* ── TAB2: 섹션 정돈 (운영 화면 느낌) ─────────────────────────────────── */
    .t2-sec-title {
        font-size: 1.02rem; font-weight: 800; color: #1f4e79;
        margin: 0 0 6px 0; letter-spacing: -0.02em;
        padding-bottom: 6px; border-bottom: 2px solid #e9eef5;
    }
    .t2-sec-sub {
        font-size: 0.78rem; color: #6a7580; line-height: 1.5; margin: 0 0 12px 0;
    }
    .t2-zone-cell {
        background: #f8fafc; border: 1px solid #e3e8ef; border-radius: 10px;
        padding: 10px 12px; min-height: 88px; font-size: 0.84rem; line-height: 1.45;
    }
    .t2-zone-cell b { color: #2c3e50; }
    .t2-hr { height: 1px; background: linear-gradient(90deg, #e9eef5, transparent); margin: 14px 0; border: 0; }
    .t2-readme-kv { font-size: 0.8rem; color: #4a5568; line-height: 1.55; }
    .t2-readme-kv code { font-size: 0.74rem; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 상수 및 컬럼명
# ──────────────────────────────────────────────────────────────────────────────
# 기상청 공공데이터 API 인증키(단기·중기 동일). `KMA_SERVICE_KEY` 환경변수가 있으면 그 값을 우선합니다.
KMA_SERVICE_KEY_DEFAULT = (
    'hK3AQOnI6%2BX3aMyRsVdhqL07YYy8MC%2BmaPtV0uvMVzIJXpuVMgC9QTqcdH4tLSDAJsLfMXhImENSfI9WFEHrHg%3D%3D'
)
KMA_SHORT_NX = 67
KMA_SHORT_NY = 100
KMA_MID_REGID = '11C20401'

C_TEMP   = '수온(℃)'
C_CHLA   = 'Chl-a (㎎/㎥)'
C_TURB   = '탁도'
C_DO     = 'DO(㎎/L)'
C_ATEMP  = '평균기온(°C)'
C_SOLAR  = '합계 일사량(MJ/m2)'
C_RAIN   = '강우량(mm)'
C_INFLOW = '유입량(㎥/s)'
C_OUTFLOW= '총방류량(㎥/s)'
C_VOL    = '저수량(백만㎥)'

ALERT_LEVELS = {
    '미발령': {'color': '#28a745', 'css': 'risk-normal',   'emoji': '✅'},
    '관심':   {'color': '#ffc107', 'css': 'risk-watch',    'emoji': '⚠️'},
    '경계':   {'color': '#dc3545', 'css': 'risk-alert',    'emoji': '🚨'},
    '대발생': {'color': '#6f1428', 'css': 'risk-critical', 'emoji': '🔴'},
}

# TAB1 남조류 막대 색 — 발령단계별 (실데이터 범주)
TAB1_CYANO_BAR = {
    '미발령': '#87ceeb',
    '관심': '#ffa94d',
    '경계': '#fa5252',
    '대발생': '#6f1428',
}

# ──────────────────────────────────────────────────────────────────────────────
# SHAP 기반 해결책 룩업 테이블
# 구조: 변수명 → {조건별 단계: (해결책 텍스트, 단기 대응, 중장기 대응)}
# ──────────────────────────────────────────────────────────────────────────────
SOLUTION_MAP = {
    C_TEMP: {
        'label': '수온',
        'unit': '°C',
        'thresholds': [(25, '관심'), (28, '경계')],
        'solutions': {
            '관심': {
                'action': '심층 취수구 전환 검토',
                'short':  '표층(고온) 대신 수심 5~10m 취수구로 전환하여 저온수 확보',
                'long':   '수중 폭기장치 가동으로 성층 파괴 → 수온 균질화',
                'icon': '🌡️'
            },
            '경계': {
                'action': '방류량 증가 + 폭기 최대 가동',
                'short':  '방류량 20~40% 즉시 증가 → 체류시간 단축으로 고온수 교환',
                'long':   '수면 폭기설비 24시간 가동, 심층 순환 강화',
                'icon': '🌡️'
            }
        }
    },
    'CHD': {
        'label': '연속고온일수(CHD)',
        'unit': '일',
        'thresholds': [(7, '관심'), (15, '경계')],
        'solutions': {
            '관심': {
                'action': '채수 빈도 증가 + 방류 검토',
                'short':  '주 1회 → 주 2회 채수. CHD 14일 초과 전 방류량 조정 선제 준비',
                'long':   '기상청 폭염특보 연동 자동 알림 체계 구축',
                'icon': '☀️'
            },
            '경계': {
                'action': '즉각 방류량 30% 이상 증가',
                'short':  '연속 고온 누적 → 방류량 즉시 증가, HRT 30일 이하 목표',
                'long':   '열 축적 임계(CHD=20일) 도달 전 선제 경보 발령 검토',
                'icon': '☀️'
            }
        }
    },
    C_TURB: {
        'label': '탁도',
        'unit': 'NTU',
        'thresholds': [(10, '관심'), (20, '경계')],
        'solutions': {
            '관심': {
                'action': '정수처리 응집제 투입량 사전 증량',
                'short':  '탁도 상승 → 정수장 PAC(황산알루미늄) 투입량 10~20% 증가 준비',
                'long':   '상류 유역 강우 후 14일 탁도 모니터링 강화',
                'icon': '💧'
            },
            '경계': {
                'action': '취수 심도 조정 + 비상 여과 가동',
                'short':  '취수구 심도 조정(표층 탁도 회피), 정수장 비상 여과 설비 가동',
                'long':   '탁도-남조류 복합 위험(H2 가설) → 강우 이벤트 후 D+7~14 집중 감시',
                'icon': '💧'
            }
        }
    },
    'HRT_est': {
        'label': '체류시간(HRT)',
        'unit': '일',
        'thresholds': [(40, '관심'), (70, '경계')],
        'solutions': {
            '관심': {
                'action': '방류량 10~20% 증가로 교환율 확보',
                'short':  'HRT 단축 목표: 현재값의 70~80% 수준. 방류량 단계적 증가',
                'long':   '저수율-HRT 연동 운영 기준 수립(저수율 60% 이하 시 방류 자동 조정)',
                'icon': '🔄'
            },
            '경계': {
                'action': '방류량 최대화 + 유입 차단 검토',
                'short':  '방류량 40% 이상 즉시 증가. 지류 유입 조절 가능 구간 점검',
                'long':   'HRT 20일 이하 목표. 여름철(7~9월) HRT 기준 30일 이하 사전 관리',
                'icon': '🔄'
            }
        }
    },
    'CDD': {
        'label': '연속무강우일수(CDD)',
        'unit': '일',
        'thresholds': [(10, '관심'), (20, '경계')],
        'solutions': {
            '관심': {
                'action': '강우 이벤트 후 D+7~14 집중 채수 체계 전환',
                'short':  'H2 가설 확인: 장기 무강우 후 강우 시 영양염 펄스 → 7~14일 뒤 급증',
                'long':   '강우 예보 연동 자동 경보 발령 기준 추가(CDD≥14 + 강우 예보 시)',
                'icon': '🌧️'
            },
            '경계': {
                'action': '강우 발생 즉시 채수 + D+7, D+14 추가 채수 예약',
                'short':  'CDD 20일+ 후 강우 이벤트 → 남조류 급증 위험 최고조. 즉각 경계 대비',
                'long':   '무강우 기간 수질 변화 추이 일별 기록, 강우 후 14일 집중 모니터링',
                'icon': '🌧️'
            }
        }
    },
    C_CHLA: {
        'label': 'Chl-a',
        'unit': '㎍/L',
        'thresholds': [(15, '관심'), (30, '경계')],
        'solutions': {
            '관심': {
                'action': '식물플랑크톤 종 조성 분석 즉시 실시',
                'short':  'Chl-a 상승 = 전체 조류 증가 신호. 남조류 비율 확인을 위해 현미경 계수',
                'long':   '하류 취수장 Chl-a 기준 선제 통보(기준: 15㎍/L 초과 시)',
                'icon': '🔬'
            },
            '경계': {
                'action': '차단막 설치 + 황토 살포 검토',
                'short':  '남조류 확산 차단막 취수구 전방 설치. 황토 살포 구역 지정 및 준비',
                'long':   '조류독소(마이크로시스틴) 분석 즉시 의뢰. 취수 제한 여부 결정',
                'icon': '🔬'
            }
        }
    },
    'BGI': {
        'label': '남조류성장지수(BGI)',
        'unit': '',
        'thresholds': [(0.03, '관심'), (0.06, '경계')],
        'solutions': {
            '관심': {
                'action': '수중 폭기 강화 + 일사 차단 검토',
                'short':  'BGI = 수온×일사×탁도 복합 지수. 일사 강한 날 폭기 집중 가동',
                'long':   '수생식물 복원(부유식 식물섬) 등 장기 일사 차단 구조물 검토',
                'icon': '📊'
            },
            '경계': {
                'action': '수면 폭기 최대화 + 방류 증가 복합 대응',
                'short':  '성층 파괴(폭기) + HRT 단축(방류) 동시 시행으로 성장 조건 해체',
                'long':   '황토 살포로 광 차단 → 남조류 광합성 억제',
                'icon': '📊'
            }
        }
    },
    '회남_log_lag7': {
        'label': '상류 회남 선행신호(7일 전)',
        'unit': 'log cells/mL',
        'thresholds': [(4, '관심'), (7, '경계')],
        'solutions': {
            '관심': {
                'action': '하류 문의·추동 채수 빈도 선제 증가',
                'short':  'H3 가설: 회남(상류) 신호가 문의(하류)보다 7~14일 선행. 지금 대응해야 함',
                'long':   '상류 모니터링 주기 강화로 하류 조기경보 리드타임 확보',
                'icon': '📍'
            },
            '경계': {
                'action': '하류 전 지점 즉각 비상 채수 + 경보 발령 준비',
                'short':  '상류 이미 경계 수준 → 하류 7~14일 내 도달 거의 확실. 즉각 대비',
                'long':   '수계 연동 경보 체계 구축(회남 발령 시 문의·추동 자동 관심 격상)',
                'icon': '📍'
            }
        }
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "final_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "algae_model_t7.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['조사일'] = pd.to_datetime(df['조사일'])
    df = df.sort_values(['채수위치', '조사일']).reset_index(drop=True)
    if 'total_cyano' not in df.columns:
        cyano_cols = ['microcystis', 'anabaena', 'oscillatoria', 'aphanizomenon']
        df['total_cyano'] = df[[c for c in cyano_cols if c in df.columns]].sum(axis=1)
    df['log_cyano'] = np.log1p(df['total_cyano'])

    # ── 파생 변수 계산 ────────────────────────────────────────────────────────
    for site, grp in df.groupby('채수위치'):
        idx = grp.index
        # CHD: 수온 25°C 이상 연속 일수
        chd, cnt = [], 0
        for v in grp[C_TEMP]:
            cnt = (cnt + 1) if (not pd.isna(v) and v >= 25) else 0
            chd.append(cnt)
        df.loc[idx, 'CHD'] = chd
        # CDD: 강우 0mm 연속 일수
        cdd, cnt = [], 0
        rain_col = '일강수량(mm)' if '일강수량(mm)' in df.columns else C_RAIN
        for v in grp[rain_col].fillna(0):
            cnt = (cnt + 1) if v == 0 else 0
            cdd.append(cnt)
        df.loc[idx, 'CDD'] = cdd
        # HRT
        inflow_m3d = grp[C_INFLOW].clip(lower=0.01) * 86400
        storage_m3 = grp[C_VOL] * 1_000_000
        df.loc[idx, 'HRT_est'] = (storage_m3 / inflow_m3d).clip(upper=180)

    solar_95 = max(float(df[C_SOLAR].quantile(0.95)), 1e-6)
    df['BGI'] = (
        (df[C_TEMP] - 20).clip(lower=0) / 10
        * (df[C_SOLAR] / solar_95).clip(upper=1)
        * (1 / (df[C_TURB].fillna(df[C_TURB].median()) + 1))
    )
    df['TAI_7'] = df.groupby('채수위치')[C_TEMP].transform(
        lambda x: (x - 20).clip(lower=0).rolling(7, min_periods=1).sum()
    )
    # 공간 전파: 회남 lag7
    pivot = df.pivot_table(index='조사일', columns='채수위치', values='log_cyano')
    if '회남' in pivot.columns:
        for lag in [7, 14]:
            shifted = pivot['회남'].shift(lag)
            df[f'회남_log_lag{lag}'] = df['조사일'].map(shifted)
    return df

def _model_load_signature() -> tuple[bool, float]:
    """파일 유무·mtime으로 캐시 무효화(복사·교체 후에도 모델 재로드)."""
    if not os.path.exists(MODEL_PATH):
        return (False, 0.0)
    try:
        return (True, float(os.path.getmtime(MODEL_PATH)))
    except OSError:
        return (False, 0.0)


def patch_sklearn_simpleimputer_sklearn18(est) -> None:
    """
    sklearn 1.7.x로 저장한 Pipeline을 1.8+에서 로드할 때, SimpleImputer에 _fill_dtype가
    없어 transform에서 AttributeError가 나는 경우가 있음(내부 상태 차이).
    """
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except ImportError:
        return

    def walk(e):
        if e is None:
            return
        if isinstance(e, SimpleImputer):
            if hasattr(e, 'statistics_') and not hasattr(e, '_fill_dtype'):
                fd = getattr(e, '_fit_dtype', None)
                if fd is None:
                    try:
                        fd = np.asarray(e.statistics_).dtype
                    except Exception:
                        fd = np.dtype('float64')
                e._fill_dtype = fd
            return
        if isinstance(e, Pipeline):
            for _, step in e.steps:
                walk(step)
            return
        if isinstance(e, ColumnTransformer):
            for _, tr, _ in (getattr(e, 'transformers_', None) or []):
                walk(tr)
            rem = getattr(e, 'remainder', 'drop')
            if rem not in ('drop', 'passthrough'):
                walk(rem)
            return
        ns = getattr(e, 'named_steps', None)
        if isinstance(ns, dict):
            for v in ns.values():
                walk(v)

    walk(est)


@st.cache_resource
def load_model(_sig: tuple[bool, float]):
    if not _sig[0]:
        return None
    pipe = joblib.load(MODEL_PATH)
    patch_sklearn_simpleimputer_sklearn18(pipe)
    return pipe


df_all = load_data()
model = load_model(_model_load_signature())

# ── TAB2: 구역 경계 + 운영변수 전용 GBM(피클 없을 때) + 시나리오 해설 맵 ─────
TAB2_ZONE_THRESHOLDS = {
    C_TEMP:       (21.7, 23.5),
    C_TURB:       (3.5,  5.5),
    'TAI_7':      (10.0, 25.0),
    'CHD':        (3,    10),
    'CDD':        (7,    15),
    C_CHLA:       (8.0,  12.0),
    'BGI':        (0.02, 0.04),
    'HRT_est':    (40,   70),
}

FEATURE_IMPORTANCE_TAB2 = {
    'TAI_7': 0.409, 'CDD': 0.166, C_TURB: 0.159, C_CHLA: 0.079, '저수율(%)': 0.074,
    C_TEMP: 0.064, 'CHD': 0.021, 'HRT_est': 0.012, C_INFLOW: 0.008, 'BGI': 0.007,
}


def tab2_get_zone(var: str, val: float, thr: dict | None = None) -> str:
    """구역: `thr`가 있으면 지점 적응 경계, 없으면 전역 `TAB2_ZONE_THRESHOLDS`."""
    meta = thr if thr is not None else TAB2_ZONE_THRESHOLDS
    if var not in meta:
        return '안전'
    lo, hi = meta[var]
    if val >= hi:
        return '경계'
    if val >= lo:
        return '관심'
    return '안전'


TAB2_SCENARIO_SOLUTION = {
    C_TEMP: {
        'label': '수온', 'unit': '°C', 'icon': '🌡️',
        '관심': {'action': '심층 취수구 전환 검토', 'short': '표층(고온) 대신 수심 5~10m 취수구 전환으로 저온수 확보', 'long': '수중 폭기장치 가동 → 열적 성층 파괴 · 수온 균질화'},
        '경계': {'action': '방류 즉시 증가 + 폭기 최대 가동', 'short': '방류량 20~40% 즉시 증가 → 체류시간 단축으로 고온수 교환', 'long': '수면 폭기설비 24h 가동, BGI 지수 일별 모니터링'},
    },
    'TAI_7': {
        'label': '열 누적 지수 TAI_7', 'unit': '°C·일', 'icon': '☀️',
        '관심': {'action': '채수 빈도 증가 + 방류 준비', 'short': '7일 열 누적 임계 접근 → 주 2회 채수, 방류 단계적 증가 준비', 'long': '기상청 단기예보 연동, 폭염특보 시 자동 경보 격상 체계'},
        '경계': {'action': 'TAI 임계 초과 → 즉각 방류 30% 이상', 'short': '열 누적 고조 → 즉각 방류량 30%+ 증가', 'long': '선제 경보 발령 검토 (CHD 장기 동반 시)'},
    },
    'CDD': {
        'label': '연속무강우일수 CDD', 'unit': '일', 'icon': '🌧️',
        '관심': {'action': '강우 후 D+7~14 집중 채수 체계 전환', 'short': 'H2: 장기 무강우 후 강우 시 영양염 펄스 → 7~14일 뒤 남조류 급증', 'long': '강우 예보 연동 → CDD≥14 + 강우 예보 시 관심 기준 추가'},
        '경계': {'action': '강우 발생 즉시 D+7, D+14 추가 채수 예약', 'short': 'CDD 20일+ 후 강우 = 급증 위험 최고조', 'long': '강우 후 14일 일별 채수·계수'},
    },
    C_TURB: {
        'label': '탁도', 'unit': 'NTU', 'icon': '💧',
        '관심': {'action': '정수처리 응집제 투입량 사전 증량', 'short': 'PAC 투입량 10~20% 증가 준비', 'long': '상류 강우 후 14일 탁도 모니터링 강화'},
        '경계': {'action': '취수 심도 조정 + 비상 여과 가동', 'short': '표층 탁도 회피·비상 여과', 'long': '탁도-남조류 복합 위험: 강우 후 D+7~14 집중 감시'},
    },
    C_CHLA: {
        'label': 'Chl-a', 'unit': '㎎/㎥', 'icon': '🔬',
        '관심': {'action': '식물플랑크톤 종 조성 분석', 'short': '남조류 비율 확인을 위해 현미경 계수', 'long': '하류 취수장 선제 통보 체계'},
        '경계': {'action': '차단막 + 황토 살포 검토', 'short': '조류 확산 차단막·황토 살포 구역 지정', 'long': '조류독소 분석·취수 제한 여부 결정'},
    },
    'CHD': {
        'label': '연속고온일수 CHD', 'unit': '일', 'icon': '🔆',
        '관심': {'action': '방류량 조정 선제 준비', 'short': 'CHD 상승 시 방류 준비', 'long': '기상청 폭염특보 연동 알림'},
        '경계': {'action': '즉각 방류 30% 이상 + 채수 일 2회', 'short': '연속 고온 → 방류 즉시, HRT 30일 이하 목표', 'long': '선제 경보 발령 검토'},
    },
    'HRT_est': {
        'label': '체류시간 HRT', 'unit': '일', 'icon': '🔄',
        '관심': {'action': '방류량 10~20% 증가', 'short': 'HRT 단축 목표: 방류 단계적 증가', 'long': '저수율-HRT 연동 운영 기준'},
        '경계': {'action': '방류량 최대화 (40%+)', 'short': '방류 40%+ 즉시, HRT 20일 이하 목표', 'long': '여름철 HRT 30일 이하 사전 유지'},
    },
    'BGI': {
        'label': '남조류 성장 지수 BGI', 'unit': '', 'icon': '📊',
        '관심': {'action': '수중 폭기 강화 + 채수 빈도 증가', 'short': '일사 강한 날 폭기 집중 가동', 'long': '일사 차단 구조물 검토'},
        '경계': {'action': '폭기 최대화 + 방류 증가 복합 대응', 'short': '성층 파괴+ HRT 단축 동시 시행', 'long': '황토 살포로 광 차단'},
    },
}


def tab2_site_adaptive_thresholds(df_site: pd.DataFrame) -> dict:
    """
    선택 지점 역사 분포로 구역 하한(lo)·상한(hi) 추정.
    표본 부족 시 전역 TAB2_ZONE_THRESHOLDS 유지.
    """
    out = dict(TAB2_ZONE_THRESHOLDS)
    if df_site is None or len(df_site) < 30:
        return out
    for var, (def_lo, def_hi) in TAB2_ZONE_THRESHOLDS.items():
        if var not in df_site.columns:
            continue
        s = pd.to_numeric(df_site[var], errors='coerce').dropna()
        if len(s) < 25:
            continue
        lo = float(s.quantile(0.52))
        hi = float(s.quantile(0.82))
        if hi <= lo + 1e-6:
            hi = float(s.quantile(0.90))
        if hi <= lo + 1e-6:
            continue
        lo = 0.5 * lo + 0.5 * def_lo
        hi = 0.5 * hi + 0.5 * def_hi
        if hi <= lo + 1e-6:
            continue
        out[var] = (lo, hi)
    return out


def tab2_last_row_slider_defaults(last: pd.Series) -> dict:
    """최근 관측 행에서 슬라이더 초깃값."""

    def clipf(col, default, lo, hi):
        v = last.get(col, np.nan)
        if not pd.notna(v):
            v = default
        try:
            return float(np.clip(float(v), lo, hi))
        except (TypeError, ValueError):
            return float(default)

    return {
        'p_temp': clipf(C_TEMP, 22.0, 5.0, 35.0),
        'p_chla': clipf(C_CHLA, 10.0, 0.0, 100.0),
        'p_turb': clipf(C_TURB, 5.0, 0.0, 60.0),
        'p_do': clipf(C_DO, 8.0, 0.0, 15.0),
        'p_chd': int(np.clip(int(clipf('CHD', 0, 0, 40)), 0, 40)),
        'p_tai7': clipf('TAI_7', 15.0, 0.0, 100.0),
        'p_cdd': int(np.clip(int(clipf('CDD', 0, 0, 40)), 0, 40)),
        'p_solar': clipf(C_SOLAR, 14.0, 0.0, 25.0),
        'p_hrt': int(np.clip(int(round(clipf('HRT_est', 40, 5, 150))), 5, 150)),
        'p_inflow': clipf(C_INFLOW, 20.0, 0.0, 500.0),
        'p_release': clipf(C_OUTFLOW, 15.0, 0.0, 300.0),
        'p_storage': int(np.clip(int(round(clipf('저수율(%)', 65, 10, 100))), 10, 100)),
    }


def tab2_hist_percentile(df_site: pd.DataFrame, col: str, val: float) -> float | None:
    """지점 역사에서 val 이하 비율(%). 높을수록 상위 값에 가깝다."""
    if col not in df_site.columns:
        return None
    s = pd.to_numeric(df_site[col], errors='coerce').dropna()
    if len(s) < 8:
        return None
    return float(100.0 * (s <= val).mean())


def tab2_operational_checklist(df_site: pd.DataFrame, params: dict, thr: dict) -> pd.DataFrame:
    """슬라이더 값을 지점 분포·구역과 묶은 실무 점검 표."""
    rows = []
    zone_vars = [C_TEMP, C_TURB, 'TAI_7', 'CHD', C_CHLA, 'CDD', 'HRT_est']
    for var in zone_vars:
        if var not in params:
            continue
        val = float(params.get(var, 0) or 0)
        zone = tab2_get_zone(var, val, thr)
        pct = tab2_hist_percentile(df_site, var, val)
        label = TAB2_SCENARIO_SOLUTION.get(var, {}).get('label', var)
        sol = TAB2_SCENARIO_SOLUTION.get(var, {}).get(zone, {}) if zone != '안전' else {}
        act = sol.get('action', '—') if zone != '안전' else '정기 모니터링'
        pct_s = f"{pct:.0f}%" if pct is not None else '—'
        rows.append({
            '변수': label,
            '현재값': val,
            '지점누적≤비율': pct_s,
            '구역': zone,
            '우선조치': act,
        })
    return pd.DataFrame(rows)


def tab2_prob_level(p: float) -> str:
    if p > 0.55:
        return '경계'
    if p > 0.25:
        return '관심'
    return '미발령'


def tab2_compute_shap_proxy_operational(params: dict, feats: list, thr: dict | None = None) -> dict:
    contributions = {}
    for f in feats:
        importance = FEATURE_IMPORTANCE_TAB2.get(f, 0)
        val = params.get(f, 0)
        zone = tab2_get_zone(f, float(val) if val == val else 0.0, thr)
        zone_weight = {'안전': 0.1, '관심': 0.6, '경계': 1.0}.get(zone, 0.1)
        contributions[f] = importance * zone_weight
    return contributions


def tab2_driver_scores_pkl(pipe, merged: dict, feat_cols: list, thr: dict | None = None) -> dict:
    clf = pipe.named_steps.get('m') if pipe is not None else None
    if clf is None or not hasattr(clf, 'feature_importances_'):
        return {}
    imp = np.asarray(clf.feature_importances_, dtype=float)
    out = {}
    for i, f in enumerate(feat_cols):
        if i >= len(imp):
            break
        base_i = float(imp[i])
        val = merged.get(f, np.nan)
        try:
            v = float(val) if val == val else 0.0
        except (TypeError, ValueError):
            v = 0.0
        if f in TAB2_ZONE_THRESHOLDS:
            zone = tab2_get_zone(f, v, thr)
            zw = {'안전': 0.15, '관심': 0.55, '경계': 1.0}.get(zone, 0.35)
        else:
            zw = 0.35
        out[f] = base_i * zw
    return out


@st.cache_resource
def build_operational_model(df_all: pd.DataFrame):
    """피클 없을 때만: cyano_lag7 등 제외한 운영변수 전용 GBM (What-if 해석용)."""
    df2 = df_all[df_all['발령단계'].isin(['미발령', '관심', '경계'])].copy()
    feats = [C_TEMP, C_TURB, 'CHD', 'TAI_7', 'CDD', 'HRT_est', 'BGI', C_CHLA, C_INFLOW, '저수율(%)']
    for c in feats:
        if c not in df2.columns:
            raise ValueError(f'데이터에 컬럼 없음: {c}')
    df2 = df2.dropna(subset=feats + ['발령단계'])
    y = (df2['발령단계'] != '미발령').astype(int).values
    X = df2[feats].values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X, y)
    return gbm, imputer, feats


def tab2_predict_operational(params: dict, gbm, imp, feats: list) -> float:
    row = [params.get(f, 0) for f in feats]
    return float(gbm.predict_proba(imp.transform([row]))[0, 1])


def compute_whatif_tab2_operational(params: dict, gbm, imp, feats: list) -> pd.DataFrame:
    base_p = tab2_predict_operational(params, gbm, imp, feats)
    interventions = {
        '① 현행 유지 (기준)': {},
        '② 방류량 +20%':      {'HRT_est': params.get('HRT_est', 40) * 0.83},
        '③ 방류량 +40%':      {'HRT_est': params.get('HRT_est', 40) * 0.65},
        '④ 수중 폭기 (수온 -1.5°C)': {
            C_TEMP: params.get(C_TEMP, 24) - 1.5,
            'TAI_7': params.get('TAI_7', 30) * 0.87,
        },
        '⑤ 황토 살포 (탁도 차단)': {
            C_TURB: params.get(C_TURB, 5) * 1.5,
            'BGI': params.get('BGI', 0.03) * 0.65,
        },
        '⑥ 심층 취수 (수온 -2°C)': {C_TEMP: params.get(C_TEMP, 24) - 2.0},
        '⑦ 복합 (방류+폭기)': {
            'HRT_est': params.get('HRT_est', 40) * 0.70,
            C_TEMP: params.get(C_TEMP, 24) - 1.0,
            'TAI_7': params.get('TAI_7', 30) * 0.90,
        },
    }
    rows = []
    for name, ov in interventions.items():
        cond = {**params, **ov}
        p = tab2_predict_operational(cond, gbm, imp, feats)
        delta = p - base_p
        lv = tab2_prob_level(p)
        rec = '✅ 권고' if delta < -0.05 else ('⚡ 유효' if delta < 0 else '—')
        rows.append({
            '개입 방안': name, '위험도': f'{p:.1%}', '변화': f'{delta:+.1%}',
            '예상 단계': lv, '권고': rec, '_delta_raw': delta, '_prob_raw': p,
        })
    return pd.DataFrame(rows)


def _f(base: dict, k: str, default: float = 0.0) -> float:
    try:
        x = base.get(k, default)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def compute_whatif_pkl_tab2(pipe, base: dict, feat_cols: list) -> pd.DataFrame:
    """저장 GBM(T+7) 기준 What-if — 피처 딕셔너리 패치 후 predict_proba."""
    base_p = model_predict_prob_positive(pipe, base, feat_cols)
    t = _f(base, C_TEMP, 24.0)
    tai = _f(base, 'TAI_7', 25.0)
    hrt = _f(base, 'HRT_est', 40.0)
    turb = _f(base, C_TURB, 5.0)
    bgi = _f(base, 'BGI', 0.03)
    rel = base.get(C_OUTFLOW)

    def pred(d: dict) -> float:
        return model_predict_prob_positive(pipe, d, feat_cols)

    interventions: dict[str, dict] = {
        '① 현행 유지 (기준)': {},
        '② 방류량 +20%': {},
        '③ 방류량 +40%': {},
        '④ 수중 폭기 (수온 -1.5°C)': {
            C_TEMP: t - 1.5,
            'TAI_7': tai * 0.87,
        },
        '⑤ 황토 살포 (탁도 차단)': {
            C_TURB: turb * 1.5,
            'BGI': bgi * 0.65,
        },
        '⑥ 심층 취수 (수온 -2°C)': {C_TEMP: t - 2.0},
        '⑦ 복합 (방류+폭기)': {
            'HRT_est': hrt * 0.70,
            C_TEMP: t - 1.0,
            'TAI_7': tai * 0.90,
        },
    }
    if rel is not None and rel == rel and float(rel) > 0 and C_OUTFLOW in feat_cols:
        interventions['② 방류량 +20%'] = {C_OUTFLOW: float(rel) * 1.2}
        interventions['③ 방류량 +40%'] = {C_OUTFLOW: float(rel) * 1.4}
    else:
        interventions['② 방류량 +20%'] = {'HRT_est': hrt * 0.83}
        interventions['③ 방류량 +40%'] = {'HRT_est': hrt * 0.65}

    rows = []
    for name, patch in interventions.items():
        d = {**base, **patch}
        if C_OUTFLOW in patch:
            d = recalc_hrt_from_outflow(d)
        try:
            p = pred(d)
        except Exception:
            p = float('nan')
        delta = p - base_p if p == p else 0.0
        lv = tab2_prob_level(p) if p == p else '—'
        rec = '✅ 권고' if p == p and delta < -0.05 else ('⚡ 유효' if p == p and delta < 0 else '—')
        rows.append({
            '개입 방안': name,
            '위험도': f'{p:.1%}' if p == p else '—',
            '변화': f'{delta:+.1%}' if p == p else '—',
            '예상 단계': lv,
            '권고': rec,
            '_delta_raw': delta,
            '_prob_raw': p,
        })
    return pd.DataFrame(rows)

FEATURE_COLS_ML = [
    '수온(℃)', 'pH', 'DO(㎎/L)', '탁도', 'Chl-a (㎎/㎥)',
    '평균기온(°C)', '평균 풍속(m/s)', '합계 일사량(MJ/m2)', '강우량(mm)',
    '유입량(㎥/s)', '총방류량(㎥/s)', '저수량(백만㎥)',
    'CHD', 'CDD', 'HRT_est', 'BGI', 'TAI_7',
    '수온_lag7', '수온_roll7', 'Chl-a_lag7', 'log_cyano',
]


def get_model_feature_cols(pipe) -> list:
    if pipe is not None and hasattr(pipe, 'feature_names_in_') and pipe.feature_names_in_ is not None:
        return list(pipe.feature_names_in_)
    return FEATURE_COLS_ML


def quantile_zones_by_alert(df_site: pd.DataFrame) -> tuple:
    """발령단계별 수온·탁도·기온·CHD IQR (실데이터만 사용)."""
    levels = ['미발령', '관심', '경계']
    metrics = [c for c in [C_TEMP, C_TURB, C_ATEMP, 'CHD'] if c in df_site.columns]
    rows = []
    for lv in levels:
        sub = df_site[df_site.get('발령단계') == lv]
        if len(sub) < 5:
            continue
        rec = {'발령단계': lv, '행수': len(sub)}
        for c in metrics:
            s = sub[c].dropna()
            if len(s) < 3:
                continue
            rec[f'{c}_P25'] = float(s.quantile(0.25))
            rec[f'{c}_P50'] = float(s.median())
            rec[f'{c}_P75'] = float(s.quantile(0.75))
        rows.append(rec)
    zdf = pd.DataFrame(rows)
    shares = df_site['발령단계'].value_counts(normalize=True, dropna=True) if '발령단계' in df_site.columns else pd.Series(dtype=float)
    return zdf, shares


def merge_feature_row(last_row: pd.Series, overrides: dict, feature_cols: list) -> dict:
    out = {}
    for c in feature_cols:
        if c in overrides:
            out[c] = overrides[c]
        elif c in last_row.index and pd.notna(last_row.get(c)):
            out[c] = last_row[c]
        else:
            out[c] = np.nan
    return out


def _positive_class_shap_array(shap_vals):
    if isinstance(shap_vals, list):
        return np.asarray(shap_vals[1])
    a = np.asarray(shap_vals)
    if a.ndim == 3 and a.shape[-1] >= 2:
        return a[..., 1]
    return a


def model_predict_prob_positive(pipe, row_dict: dict, feature_cols: list) -> float:
    X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in feature_cols}])[feature_cols]
    return float(pipe.predict_proba(X)[0, 1])


def shap_positive_drivers_pkl(pipe, row_dict: dict, feature_cols: list, top_n: int = 5):
    if shap is None:
        return []
    imp = pipe.named_steps.get('imp')
    clf = pipe.named_steps.get('m')
    if imp is None or clf is None:
        return []
    X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in feature_cols}])[feature_cols]
    X_imp = imp.transform(X)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_imp)
    sv = _positive_class_shap_array(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    row = sv[0]
    pairs = [(feature_cols[i], float(row[i])) for i in range(len(feature_cols))]
    pos = [(f, v) for f, v in pairs if v > 0]
    pos.sort(key=lambda x: -x[1])
    return pos[:top_n]


def recalc_hrt_from_outflow(cond: dict) -> dict:
    v, q = cond.get(C_VOL), cond.get(C_OUTFLOW)
    if v is None or q is None or (isinstance(q, (int, float)) and q <= 0):
        return cond
    try:
        hrt = (float(v) * 1e6) / max(float(q) * 86400.0, 1.0)
    except (TypeError, ValueError):
        return cond
    return {**cond, 'HRT_est': hrt}


def what_if_pkl_table(pipe, base: dict, feature_cols: list) -> pd.DataFrame:
    base_prob = model_predict_prob_positive(pipe, base, feature_cols)
    interventions = [
        ('현행 유지', {}),
        ('방류량 +20%', {C_OUTFLOW: base.get(C_OUTFLOW, np.nan) * 1.2}),
        ('방류량 +40%', {C_OUTFLOW: base.get(C_OUTFLOW, np.nan) * 1.4}),
        ('수온 -2°C 가정', {C_TEMP: base.get(C_TEMP, np.nan) - 2.0}),
        ('폭기 가정', {C_TEMP: base.get(C_TEMP, np.nan) - 1.5, C_DO: 9.0}),
        ('탁도↑(황토)', {C_TURB: base.get(C_TURB, np.nan) * 1.5}),
    ]
    rows = []
    for name, patch in interventions:
        cond = {**base, **patch}
        if C_OUTFLOW in patch:
            cond = recalc_hrt_from_outflow(cond)
        try:
            prob = model_predict_prob_positive(pipe, cond, feature_cols)
        except Exception:
            prob = np.nan
        rows.append({
            '개입': name,
            '발령확률': round(prob, 4) if prob == prob else None,
            'Δ': round(prob - base_prob, 4) if prob == prob else None,
        })
    return pd.DataFrame(rows)


def _tail_consecutive_ge(temps, thresh: float = 25.0) -> int:
    """시계열 끝에서부터 thresh 이상 연속 일수 (CHD와 동일 로직)."""
    c = 0
    for v in reversed(np.asarray(temps).flatten()):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            break
        if not np.isnan(fv) and fv >= thresh:
            c += 1
        else:
            break
    return c


def _tail_cdd_dry(rains_mm, dry_max_mm: float = 0.5) -> int:
    """끝에서부터 강우 dry_max 이하 연속 일수."""
    c = 0
    for v in reversed(np.asarray(rains_mm).flatten()):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not np.isnan(fv) and fv <= dry_max_mm:
            c += 1
        else:
            break
    return c


def _tai7_tail(temps) -> float:
    t = np.asarray(temps, dtype=float).flatten()
    if len(t) == 0:
        return 0.0
    w = t[-7:]
    return float(np.sum(np.clip(w - 20.0, 0, None)))


def historical_site_probas(df_all: pd.DataFrame, site: str, pipe, feat_cols: list) -> np.ndarray:
    """동일 지점 과거 각 일의 모델 발령 확률 (임계값 분위용)."""
    dz = df_all[df_all['채수위치'] == site].sort_values('조사일')
    out = []
    for _, row in dz.iterrows():
        try:
            d = {c: row.get(c, np.nan) for c in feat_cols}
            out.append(model_predict_prob_positive(pipe, d, feat_cols))
        except Exception:
            continue
    arr = np.asarray(out, dtype=float)
    if len(arr) < 5:
        return np.linspace(0.1, 0.9, 20)
    return arr


def proba_thresholds_tertiles(probs: np.ndarray) -> tuple[float, float]:
    """
    과거 확률 분포에서 3구간 경계 (데이터 기반).
    과거 예측이 전부 0에 가깝거나 동일하면 분위가 0으로 붕괴해
    p=0도 '경계'로 잘못 분류되므로 이 경우 고정 하한을 쓴다.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs)]
    if len(probs) < 30:
        return 0.30, 0.60
    t_watch = float(np.quantile(probs, 2 / 3))
    t_alert = float(np.quantile(probs, 5 / 6))
    # 전부 0 등으로 분위가 0·0이 되면 0 < 0 이 거짓 → stage가 전부 '경계'가 됨
    if t_alert < 1e-5 or t_watch < 1e-5:
        return 0.30, 0.60
    if t_alert <= t_watch:
        t_alert = min(0.99, t_watch + 0.15)
    return t_watch, t_alert


def stage_from_prob(p: float, t_watch: float, t_alert: float) -> str:
    if p < t_watch:
        return '미발령'
    if p < t_alert:
        return '관심'
    return '경계'


def parse_pcp_mm(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip()
    if s in ('', '강수없음', '0', '0mm', 'None'):
        return 0.0
    # 기상청 표기: "1mm 미만", "1mm이하" 등 (이하·미만 제거 후 숫자만)
    s = s.replace('mm', '').replace('미만', '').replace('이하', '').strip()
    if '~' in s:
        s = s.split('~')[0].strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def latest_village_base(now: datetime | None = None) -> tuple[datetime.date, str]:
    """가장 최근에 발표된 단기예보 base_date / base_time (대략적)."""
    now = now or datetime.now()
    slots = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
    for days_back in range(5):
        day = (now - timedelta(days=days_back)).date()
        for bt in reversed(slots):
            hh = int(bt[:2])
            mm = int(bt[2:4])
            tpub = datetime.combine(day, datetime.min.time()) + timedelta(hours=hh, minutes=mm + 40)
            if tpub <= now:
                return day, bt
    return (now.date() - timedelta(days=1)), '1100'


def normalize_kma_service_key(service_key: str) -> str:
    """공공데이터포털 serviceKey: URL 인코딩된 문자열은 디코딩 후 전달."""
    key = (service_key or '').strip()
    if '%' in key:
        try:
            from urllib.parse import unquote
            key = unquote(key)
        except Exception:
            pass
    return key


def mid_tmfc_candidates(now: datetime | None = None) -> list[str]:
    """중기예보 발표시각(tmFc) 후보: 06·18시, 최근 발표부터."""
    now = now or datetime.now()
    pairs: list[tuple[datetime, str]] = []
    for days_back in range(6):
        d = (now.date() - timedelta(days=days_back))
        for hh, mm in ((18, 0), (6, 0)):
            tpub = datetime.combine(d, datetime.min.time()) + timedelta(hours=hh, minutes=mm + 15)
            if tpub <= now:
                pairs.append((tpub, d.strftime('%Y%m%d') + f'{hh:02d}{mm:02d}'))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in pairs]


def fetch_kma_mid_ta_daily(service_key: str, reg_id: str) -> pd.DataFrame | None:
    """
    기상청 중기기온(getMidTa) → n일 후 최저·최고 평균기온.
    예보일 = tmFc 날짜(YYYYMMDD) + n일 (기상청 중기기온 필드 taMinN/taMaxN).
    강수는 API에 수치가 없어 NaN으로 두고, 병합 시 계절통계로 채움.
    """
    if not service_key or not (reg_id or '').strip() or requests is None:
        return None
    key = normalize_kma_service_key(service_key)
    rid = reg_id.strip()
    url = 'http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa'
    for tmfc in mid_tmfc_candidates():
        params = {
            'serviceKey': key,
            'pageNo': 1,
            'numOfRows': 10,
            'dataType': 'JSON',
            'regId': rid,
            'tmFc': tmfc,
        }
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue
        resp = data.get('response') or {}
        header = resp.get('header') or {}
        if str(header.get('resultCode', '')).strip() not in ('00', '0'):
            continue
        body = resp.get('body') or {}
        items = body.get('items') or {}
        it = items.get('item')
        if not it:
            continue
        if isinstance(it, list):
            it = it[0]
        try:
            base_date = datetime.strptime(tmfc[:8], '%Y%m%d').date()
        except ValueError:
            continue
        rows: list[dict] = []
        for k, v in it.items():
            m = re.match(r'^taMin(\d+)$', str(k))
            if not m:
                continue
            n = int(m.group(1))
            tk = f'taMax{n}'
            if tk not in it:
                continue
            try:
                tn = float(v)
                tx = float(it[tk])
            except (TypeError, ValueError):
                continue
            tmean = (tn + tx) / 2.0
            fdate = pd.Timestamp(base_date + timedelta(days=n))
            rows.append({
                'date': fdate.normalize(),
                'temp': tmean,
                'rain': np.nan,
                'fc_source': '기상청 중기기온',
            })
        if rows:
            return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    return None


def fetch_kma_vilage_daily(service_key: str, nx: int, ny: int) -> pd.DataFrame | None:
    """
    기상청 단기예보(getVilageFcst) → 일별 평균기온(시간 TMP 평균), 일강수(PCP 합).
    공공데이터포털 serviceKey(인코딩·디코딩 혼용 대비).
    """
    if not service_key or requests is None:
        return None
    key = normalize_kma_service_key(service_key)
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    for day, btime in [latest_village_base(), *[
        ((datetime.now() - timedelta(days=d)).date(), bt)
        for d in (1, 2) for bt in ('1100', '0500', '0200')
    ]]:
        params = {
            'serviceKey': key,
            'pageNo': 1,
            'numOfRows': 1000,
            'dataType': 'JSON',
            'base_date': day.strftime('%Y%m%d'),
            'base_time': btime,
            'nx': int(nx),
            'ny': int(ny),
        }
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue
        resp = data.get('response') or {}
        header = resp.get('header') or {}
        if str(header.get('resultCode', '')).strip() not in ('00', '0'):
            continue
        body = resp.get('body') or {}
        items = body.get('items') or {}
        it = items.get('item')
        if not it:
            continue
        if isinstance(it, dict):
            it = [it]
        tmp_by_date: dict[str, list[float]] = defaultdict(list)
        rain_by_date: dict[str, float] = defaultdict(float)
        for row in it:
            ds = str(row.get('fcstDate', ''))
            if len(ds) != 8:
                continue
            cat = row.get('category')
            val = row.get('fcstValue')
            if cat == 'TMP':
                try:
                    tmp_by_date[ds].append(float(val))
                except (TypeError, ValueError):
                    pass
            elif cat == 'PCP':
                rain_by_date[ds] += parse_pcp_mm(val)
        if not tmp_by_date and not rain_by_date:
            continue
        all_dates = sorted(set(tmp_by_date.keys()) | set(rain_by_date.keys()))
        rows = []
        for ds in all_dates:
            dtp = pd.Timestamp(ds[:4] + '-' + ds[4:6] + '-' + ds[6:8])
            tlist = tmp_by_date.get(ds, [])
            tmean = float(np.mean(tlist)) if tlist else np.nan
            rsum = float(rain_by_date.get(ds, 0.0))
            rows.append({'date': dtp.normalize(), 'temp': tmean, 'rain': rsum, 'fc_source': '기상청 단기예보'})
        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    return None


def _circular_doy_diff(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 366 - d)


def climatology_for_calendar_day(
    df_site: pd.DataFrame,
    target: datetime.date,
    rain_col: str,
    doy_window: int = 4,
) -> tuple[float, float, float]:
    """동일 지점: 같은 일주년(요일·연도 무관) ±창에서 평균기온·강수·일사 중앙값."""
    if df_site.empty or '조사일' not in df_site.columns:
        return 15.0, 2.0, 12.0
    dz = df_site.dropna(subset=['조사일']).copy()
    dz['_doy'] = pd.to_datetime(dz['조사일']).dt.dayofyear
    tgt_doy = pd.Timestamp(target).dayofyear
    mask = dz['_doy'].apply(lambda x: _circular_doy_diff(int(x), tgt_doy) <= doy_window)
    sub = dz.loc[mask]
    if len(sub) < 4:
        sub = dz
    at = float(sub[C_ATEMP].median()) if C_ATEMP in sub.columns and sub[C_ATEMP].notna().any() else 15.0
    rc = rain_col if rain_col in sub.columns else None
    rn = float(sub[rc].fillna(0).median()) if rc else 2.0
    sol = float(sub[C_SOLAR].median()) if C_SOLAR in sub.columns and sub[C_SOLAR].notna().any() else 12.0
    return at, rn, sol


def apply_scenario_to_fc(fc_df: pd.DataFrame, scenario: str, seed: int = 42) -> None:
    """generate_forecast와 유사한 스트레스를 API·계절통계 행에 가산(제자리 수정)."""
    n = len(fc_df)
    if n == 0:
        return
    t = fc_df['temp'].astype(float).values.copy()
    r = fc_df['rain'].astype(float).values.copy()
    rng = np.random.default_rng(seed)
    if scenario == '폭염 지속':
        t = t + np.linspace(0, 5, n)
        r = r * 0.15
    elif scenario == '강우 유입':
        t = t - np.linspace(0, 3, n)
        bump = np.zeros(n)
        if n >= 6:
            bump[2:5] = [12, 22, 10]
        r = r + bump + rng.exponential(2, n)
    elif scenario == '불확실 (랜덤)':
        t = t + rng.normal(0, 2.0, n)
        r = np.clip(r + rng.exponential(2.5, n), 0, 120)
    fc_df['temp'] = np.clip(t, -8, 42)
    fc_df['rain'] = np.clip(r, 0, 150)
    if scenario != '현재 추세 유지' and 'temp_src' in fc_df.columns:
        sfx = ' +시나리오조정'
        fc_df['temp_src'] = fc_df['temp_src'].astype(str) + sfx
        fc_df['rain_src'] = fc_df['rain_src'].astype(str) + sfx


def build_fc_dataframe_10d(
    df_site: pd.DataFrame,
    scenario: str,
    service_key: str | None,
    nx: int,
    ny: int,
    use_kma: bool,
    use_kma_mid: bool = False,
    service_key_mid: str | None = None,
    mid_reg_id: str = '',
) -> tuple[pd.DataFrame, dict]:
    """
    내일~10일 후 날짜 축. 단기(getVilageFcst)·중기(getMidTa) 가능 구간 + 나머지는 동일지점 일주년 계절통계.
    동일 일자는 단기 우선. 실패 시 generate_forecast 폴백.
    """
    meta: dict = {
        'mode': 'synthetic',
        'detail': '시나리오 난수 예보(generate_forecast)',
        'tab3_rain_note': '기상청 API 미연동(키·응답 없음): 아래 기온·강수는 **시나리오·난수**이며 실제 일기예보가 아닙니다.',
    }
    rain_col = C_RAIN if C_RAIN in df_site.columns else (
        '일강수량(mm)' if '일강수량(mm)' in df_site.columns else C_RAIN
    )
    anchor_dates = [pd.Timestamp(datetime.now().date() + timedelta(days=i)) for i in range(1, 11)]

    kma_v = None
    if use_kma and service_key:
        kma_v = fetch_kma_vilage_daily(service_key, nx, ny)

    key_mid = (service_key_mid or service_key or '').strip() or None
    kma_m = None
    if use_kma_mid and key_mid and (mid_reg_id or '').strip():
        kma_m = fetch_kma_mid_ta_daily(key_mid, mid_reg_id)

    has_api = (kma_v is not None and len(kma_v) > 0) or (kma_m is not None and len(kma_m) > 0)

    rows: list[dict] = []
    if has_api:
        short_map: dict[pd.Timestamp, pd.Series] = {}
        if kma_v is not None:
            for _, r in kma_v.iterrows():
                short_map[pd.Timestamp(r['date']).normalize()] = r
        mid_map: dict[pd.Timestamp, pd.Series] = {}
        if kma_m is not None:
            for _, r in kma_m.iterrows():
                dn = pd.Timestamp(r['date']).normalize()
                if dn not in mid_map:
                    mid_map[dn] = r
        parts = []
        if kma_v is not None and len(kma_v):
            parts.append(f"단기 {len(kma_v)}일")
        if kma_m is not None and len(kma_m):
            parts.append(f"중기 {len(kma_m)}일")
        meta = {
            'mode': 'kma+clim',
            'detail': ' + '.join(parts) + ' (가능 일수) + 동일 지점 일주년± 계절통계로 10일 축 보완',
            'kma_short_rows': int(len(kma_v)) if kma_v is not None else 0,
            'kma_mid_rows': int(len(kma_m)) if kma_m is not None else 0,
            'tab3_rain_note': (
                '강수(mm): **단기**는 PCP(시간별) 합이며, 예보가 주는 구간은 보통 **3~5일**이라 그 밖 날짜는 단기에 없습니다. '
                '**중기(getMidTa)** 는 강수 필드가 없어, 단기가 덮지 않는 날은 **이 지점 CSV의 일주년(±몇 일) 강수 중앙값**으로 채웁니다. '
                '관측상 **무강수(0)가 많은 달·지점**이면 중앙이 0이라 막대가 비어 보이는 것이 정상입니다.'
            ),
        }
        for d in anchor_dates:
            dn = d.normalize()
            if dn in short_map:
                rr = short_map[dn]
                raw_t = rr['temp']
                temp = float(raw_t) if np.isfinite(raw_t) else np.nan
                t_src = '기상청 단기(TMP시간평균)'
                if not np.isfinite(temp):
                    temp, _, _ = climatology_for_calendar_day(df_site, d.date(), rain_col)
                    t_src = '지점관측·일주년±기온(단기기온누락시보정)'
                rain = float(rr['rain'])
                src = str(rr.get('fc_source', '기상청 단기예보'))
                r_src = '기상청 단기(PCP일합)' if rain > 0.05 else '기상청 단기(강수 0 또는 미만)'
            elif dn in mid_map:
                rr = mid_map[dn]
                raw_mt = rr['temp']
                temp = float(raw_mt) if np.isfinite(raw_mt) else np.nan
                t_src = '기상청 중기기온(최저·최고평균)'
                if not np.isfinite(temp):
                    temp, _, _ = climatology_for_calendar_day(df_site, d.date(), rain_col)
                    t_src = '지점관측·일주년±기온(중기기온누락시보정)'
                rn = rr.get('rain', np.nan)
                if pd.notna(rn) and np.isfinite(float(rn)):
                    rain = float(rn)
                    r_src = '기상청(강수필드)'
                else:
                    rain = climatology_for_calendar_day(df_site, d.date(), rain_col)[1]
                    r_src = '지점관측·일주년±강수중앙값(중기API에강수없음)'
                src = str(rr.get('fc_source', '기상청 중기기온'))
            else:
                temp, rain, _ = climatology_for_calendar_day(df_site, d.date(), rain_col)
                src = '계절통계(동일지점·일주년±)'
                t_src = '지점관측·일주년±기온중앙값'
                r_src = '지점관측·일주년±강수중앙값'
            _, _, solar = climatology_for_calendar_day(df_site, d.date(), rain_col)
            rows.append({
                'date': dn, 'temp': temp, 'rain': rain, 'solar': solar,
                'fc_source': src, 'temp_src': t_src, 'rain_src': r_src,
            })
        out = pd.DataFrame(rows)
    else:
        base_temp = float(df_site[C_TEMP].dropna().iloc[-1]) if C_TEMP in df_site.columns and df_site[C_TEMP].notna().any() else 20.0
        out = generate_forecast(base_temp=base_temp, scenario=scenario)
        out['fc_source'] = '시나리오(generate_forecast)'
        if use_kma or use_kma_mid:
            if not service_key and not key_mid:
                meta['detail'] = 'API 키 없음 → 시나리오 예보'
            elif use_kma_mid and not (mid_reg_id or '').strip():
                meta['detail'] = '중기 예보구역(regId) 없음 → 시나리오 예보'
            elif requests is None:
                meta['detail'] = 'requests 미설치 → pip install requests'
            else:
                meta['detail'] = '기상청 API 응답 없음(단기: 격자·키 / 중기: regId·키) → 시나리오 예보'

    if has_api:
        apply_scenario_to_fc(out, scenario, seed=43)
    return out, meta


def chd_alert_thresholds_from_data(df_site: pd.DataFrame) -> tuple[int, int]:
    """발령일 vs 비발령일 CHD 분포로 관심·경계 눈금(데이터 부족 시 7·15)."""
    if 'CHD' not in df_site.columns or '발령단계' not in df_site.columns:
        return 7, 15
    y = df_site['발령단계'].astype(str) != '미발령'
    hi = df_site.loc[y, 'CHD'].dropna().astype(float)
    lo = df_site.loc[~y, 'CHD'].dropna().astype(float)
    if len(hi) < 6 or len(lo) < 6:
        return 7, 15
    t_watch = int(round(float(np.clip(np.quantile(hi, 0.20), 4, 18))))
    t_alert = int(round(float(max(t_watch + 4, np.quantile(hi, 0.45)))))
    t_alert = int(np.clip(t_alert, t_watch + 2, 28))
    return t_watch, t_alert


def chd_issuance_odds_ratio(df_site: pd.DataFrame, chd_split: int = 7) -> float | None:
    """CHD≥chd_split일 vs 그 미만에서 (발령단계≠미발령) 비율 비 — H1 요약용."""
    if '발령단계' not in df_site.columns or 'CHD' not in df_site.columns:
        return None
    y = df_site['발령단계'].astype(str) != '미발령'
    hi = df_site['CHD'] >= chd_split
    lo = ~hi
    if hi.sum() < 8 or lo.sum() < 8:
        return None
    r_hi = float(y[hi].mean()) + 1e-9
    r_lo = float(y[lo].mean()) + 1e-9
    return r_hi / r_lo


def build_10day_forecast_ml(
    df_all: pd.DataFrame,
    site: str,
    fc_df: pd.DataFrame,
    last_row: pd.Series,
    pipe,
    feat_cols: list,
    solar_ref: float,
) -> pd.DataFrame:
    """
    예보일 d의 가정된 기상·파생 피처 X_d에 대해 **학습 타깃과 동일하게 P(발령 on d+7)** 근사.
    (노트북: target_7d = 발령여부.shift(-7).) 10일 각 칸은 동일 정의의 **시나리오 해석용**이며,
    d+7이 예보 창 밖이면 외삽에 가깝다는 점은 UI에서 별도 안내.
    수온은 예보 첫날 대비 기온 델타를 관측 수온에 가산.
    """
    dz = df_all[df_all['채수위치'] == site].sort_values('조사일')
    tail_w = dz[C_TEMP].dropna().tail(30).values.astype(float)
    rain_col = C_RAIN if C_RAIN in dz.columns else '일강수량(mm)'
    tail_r = dz[rain_col].fillna(0).tail(30).values.astype(float) if rain_col in dz.columns else np.zeros(min(30, len(dz)))

    base_w = float(last_row.get(C_TEMP, np.nan) or 25.0)
    base_fc0 = float(fc_df['temp'].iloc[0])
    n_fc = len(fc_df)
    rows = []

    for i in range(n_fc):
        fc_t = float(fc_df['temp'].iloc[i])
        fc_rn = float(fc_df['rain'].iloc[i])
        fc_sol = float(fc_df['solar'].iloc[i])
        water_t = base_w + (fc_t - base_fc0)

        comb_w = np.concatenate([tail_w, np.array([base_w + (float(fc_df['temp'].iloc[j]) - base_fc0) for j in range(i + 1)])])
        comb_r = np.concatenate([tail_r, fc_df['rain'].iloc[: i + 1].values.astype(float)])

        chd_i = _tail_consecutive_ge(comb_w, 25.0)
        cdd_i = _tail_cdd_dry(comb_r, 0.5)
        tai7_i = _tai7_tail(comb_w)
        lag7_w = float(comb_w[-7]) if len(comb_w) >= 7 else float(comb_w[0])
        roll7_w = float(np.mean(comb_w[-7:])) if len(comb_w) >= 7 else float(np.mean(comb_w))

        chla0 = float(last_row.get(C_CHLA, 5) or 5)
        chla_series = np.concatenate([dz[C_CHLA].dropna().tail(7).values.astype(float) if C_CHLA in dz.columns else np.array([chla0] * 7), np.full(i + 1, chla0)])
        lag7_chla = float(chla_series[-7]) if len(chla_series) >= 7 else chla0
        roll7_chla = float(np.mean(chla_series[-7:]))

        turb_med = float(dz[C_TURB].median()) if C_TURB in dz.columns else 5.0
        turb_i = float(last_row.get(C_TURB, turb_med) or turb_med)
        bgi_i = max(water_t - 20, 0) / 10.0 * min(fc_sol / max(solar_ref, 1e-6), 1.0) * (1.0 / (turb_i + 1.0))

        pulse = 7
        if i > 0 and float(fc_df['rain'].iloc[i - 1]) > 5:
            pulse = 1
        else:
            pulse = min(i + 1, 21)

        base = {c: last_row.get(c, np.nan) for c in feat_cols}
        patch_raw = {
            C_TEMP: water_t,
            C_ATEMP: fc_t,
            C_RAIN: fc_rn,
            C_SOLAR: fc_sol,
            'CHD': float(chd_i),
            'CDD': float(cdd_i),
            'TAI_7': tai7_i,
            '수온_lag7': lag7_w,
            '수온_roll7': roll7_w,
            'Chl-a_lag7': lag7_chla,
            C_CHLA: chla0,
            'BGI': float(bgi_i),
        }
        if 'Chl-a_roll7' in base:
            base['Chl-a_roll7'] = roll7_chla
        if 'rain_pulse_days' in base:
            base['rain_pulse_days'] = float(pulse)
        if 'log_cyano' in base:
            base['log_cyano'] = float(last_row.get('log_cyano', 0) or 0)
        for k, v in patch_raw.items():
            if k in base:
                base[k] = v

        prob = np.nan
        if pipe is not None:
            try:
                prob = model_predict_prob_positive(pipe, base, feat_cols)
            except Exception:
                prob = np.nan

        rp, rlv, _, _ = compute_risk_score({
            C_TEMP: water_t,
            C_CHLA: chla0,
            C_TURB: turb_i,
            C_DO: float(last_row.get(C_DO, 8) or 8),
            'CHD': float(chd_i),
            'CDD': float(cdd_i),
            'HRT_est': float(last_row.get('HRT_est', 30) or 30),
            'BGI': float(bgi_i),
            'log_cyano': float(last_row.get('log_cyano', 0) or 0),
            'rain_pulse_days': float(pulse),
        })

        d_fc = pd.Timestamp(fc_df['date'].iloc[i]).normalize()
        src = str(fc_df['fc_source'].iloc[i]) if 'fc_source' in fc_df.columns else ''
        rows.append({
            'date': d_fc,
            'target_date': d_fc + pd.Timedelta(days=7),
            'fc_source': src,
            'temp_fc': fc_t,
            'water_temp': water_t,
            'rain': fc_rn,
            'solar': fc_sol,
            'CHD': chd_i,
            'prob': prob,
            'rule_score': rp,
            'rule_level': rlv,
        })

    return pd.DataFrame(rows)


def tab3_weather_operator_summary(
    fc_df: pd.DataFrame,
    last_row: pd.Series,
    fc_meta: dict,
    scenario: str,
) -> str:
    """10일 기상을 ‘어쩌라고’ 없이 읽도록: 평균·합·직전 관측 대비 한 블록."""
    n = len(fc_df)
    if n == 0:
        return '_예보 행 없음_'
    m_t = float(fc_df['temp'].astype(float).mean())
    sum_r = float(fc_df['rain'].astype(float).sum())
    mx_t = float(fc_df['temp'].astype(float).max())
    mn_t = float(fc_df['temp'].astype(float).min())
    la = last_row.get(C_ATEMP, np.nan)
    lw = last_row.get(C_TEMP, np.nan)
    last_air = float(la) if la is not None and pd.notna(la) and np.isfinite(float(la)) else np.nan
    last_wt = float(lw) if lw is not None and pd.notna(lw) and np.isfinite(float(lw)) else np.nan
    lines = [
        f"**10일 요약 (예보일 = 내일부터 {n}일째까지):**",
        f"- 예보 **기온(공기)**: 평균 **{m_t:.1f}°C**, 최저 **{mn_t:.1f}°C**, 최고 **{mx_t:.1f}°C**",
        f"- 예보 **강수 합**: **{sum_r:.1f}** mm (일별은 아래 표·막대; 출처는 **강수자료** 열)",
    ]
    if np.isfinite(last_air):
        lines.append(f"- 직전 관측 **{C_ATEMP}** **{last_air:.1f}°C** → 10일 평균과 **{m_t - last_air:+.1f}°C**")
    if np.isfinite(last_wt):
        lines.append(f"- 직전 관측 **{C_TEMP}** **{last_wt:.1f}°C** (수온은 타임라인에서 예보 기온 델타로 가산)")
    if fc_meta.get('mode') == 'kma+clim' and scenario != '현재 추세 유지':
        lines.append(
            f"- 사이드바 시나리오 **「{scenario}」** 가 **API·계절통계 값에 추가 스트레스**를 가했습니다(기온·강수 **자료** 열 끝에 표시)."
        )
    elif fc_meta.get('mode') == 'synthetic':
        lines.append(
            "- **기상청 API 미연동** 구간: 아래 수치는 **시나리오·난수**이며, 위 **위험 타임라인**은 그 가정 하에서 GBM/규칙으로만 의미가 있습니다."
        )
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 위험도 점수 계산 (규칙 기반 + 모델 연동)
# ──────────────────────────────────────────────────────────────────────────────
def compute_risk_score(params: dict) -> tuple:
    """Returns (risk_prob 0–1, alert_level, reasoning, factor_scores)."""
    factors = {}

    temp  = params.get(C_TEMP, 20)
    chd   = params.get('CHD', 0)
    hrt   = params.get('HRT_est', 30)
    chla  = params.get(C_CHLA, 5)
    bgi   = params.get('BGI', 0.0)
    cya   = params.get('log_cyano', 0)
    cdd   = params.get('CDD', 0)
    turb  = params.get(C_TURB, 3)
    pulse = params.get('rain_pulse_days', 7)

    # 가중치는 실데이터 상관계수 기반 (수온 r=0.715 최고)
    if temp >= 28:   factors[C_TEMP] = 0.28
    elif temp >= 25: factors[C_TEMP] = 0.18
    elif temp >= 22: factors[C_TEMP] = 0.08
    else:            factors[C_TEMP] = 0.0

    if chd >= 20:   factors['CHD'] = 0.20
    elif chd >= 10: factors['CHD'] = 0.13
    elif chd >= 5:  factors['CHD'] = 0.06
    else:            factors['CHD'] = 0.0

    factors['HRT_est'] = min(hrt / 180 * 0.12, 0.12)

    if chla >= 30:   factors[C_CHLA] = 0.12
    elif chla >= 15: factors[C_CHLA] = 0.07
    else:             factors[C_CHLA] = 0.0

    factors['BGI'] = min(bgi / 0.1 * 0.10, 0.10)

    if cya >= 8:    factors['log_cyano'] = 0.18
    elif cya >= 5:  factors['log_cyano'] = 0.12
    elif cya >= 3:  factors['log_cyano'] = 0.05
    else:            factors['log_cyano'] = 0.0

    if cdd >= 14 and 3 <= pulse <= 14:
        factors['CDD'] = 0.08
    elif cdd >= 7:
        factors['CDD'] = 0.04
    else:
        factors['CDD'] = 0.0

    if turb >= 20:   factors[C_TURB] = 0.06
    elif turb >= 10: factors[C_TURB] = 0.03
    else:             factors[C_TURB] = 0.0

    score = min(sum(factors.values()), 1.0)

    if score < 0.25:   level = '미발령'
    elif score < 0.50: level = '관심'
    elif score < 0.75: level = '경계'
    else:               level = '대발생'

    # 주요 원인 텍스트
    top = sorted(factors.items(), key=lambda x: -x[1])
    reasons = []
    for f, v in top[:3]:
        if v > 0:
            label = SOLUTION_MAP.get(f, {}).get('label', f)
            reasons.append(label)
    reasoning = " · ".join(reasons) if reasons else "특이사항 없음"

    return score, level, reasoning, factors


def get_solution_level(var, value):
    """변수값 → 심각도 단계 반환"""
    info = SOLUTION_MAP.get(var)
    if not info:
        return None
    for threshold, stage in sorted(info['thresholds'], key=lambda x: -x[0]):
        if value >= threshold:
            return stage
    return None


# ──────────────────────────────────────────────────────────────────────────────
# What-if 개입 테이블
# ──────────────────────────────────────────────────────────────────────────────
def compute_whatif_table(base_params: dict) -> pd.DataFrame:
    """운영 개입별 위험도 변화 계산"""
    base_risk, _, _, _ = compute_risk_score(base_params)

    storage = base_params.get(C_VOL, 500)
    inflow  = base_params.get(C_INFLOW, 20)
    release = base_params.get(C_OUTFLOW, 15)

    interventions = {
        '① 현행 유지 (기준)': {},
        '② 방류량 +20%': {
            C_OUTFLOW: release * 1.2,
            'HRT_est': storage * 1e6 / max((inflow + release * 0.1) * 86400, 1) / 1e6 * 0.85
        },
        '③ 방류량 +40%': {
            C_OUTFLOW: release * 1.4,
            'HRT_est': base_params.get('HRT_est', 30) * 0.65
        },
        '④ 수중 폭기 (수온 -1.5°C)': {
            C_TEMP: base_params.get(C_TEMP, 25) - 1.5,
            C_DO: min(base_params.get(C_DO, 8) + 1.5, 14.0)
        },
        '⑤ 황토 살포 (탁도 ↑ → 광차단)': {
            C_TURB: base_params.get(C_TURB, 5) * 2.0,
            'BGI': base_params.get('BGI', 0.05) * 0.6
        },
        '⑥ 심층 취수 전환 (수온 -2°C)': {
            C_TEMP: base_params.get(C_TEMP, 25) - 2.0
        },
        '⑦ 복합 대응 (방류+폭기)': {
            C_OUTFLOW: release * 1.3,
            'HRT_est': base_params.get('HRT_est', 30) * 0.70,
            C_TEMP: base_params.get(C_TEMP, 25) - 1.0,
        },
    }

    rows = []
    for name, override in interventions.items():
        p = {**base_params, **override}
        r, lv, _, _ = compute_risk_score(p)
        delta = r - base_risk
        rows.append({
            '개입 방안': name,
            '위험도': f"{r*100:.1f}%",
            '변화': f"{delta*100:+.1f}%",
            '예상 단계': lv,
            '권고': '✅ 권고' if delta < -0.05 else ('⚡ 유효' if delta < 0 else '—'),
        })
    return pd.DataFrame(rows)


def get_protocol(level: str, horizon: int = 7) -> str:
    protocols = {
        '미발령': f"""**현재 위험도: 낮음 (미발령)**
- 정기 모니터링 유지 (주 1회 채수)
- 수온·Chl-a 주간 트렌드 모니터링
- T+{horizon}일 예보 점수 주기적 확인""",
        '관심': f"""**현재 위험도: 주의 (관심 단계)**
- 채수 빈도 증가 (주 2회 이상)
- 남조류 현미경 계수 즉시 실시
- 하류 취수장 관계기관 선제 통보
- T+{horizon}일 이내 집중 모니터링 계획 수립
- 방류량 조정 (HRT 단축) 검토""",
        '경계': f"""**현재 위험도: 높음 (경계 단계)**
- 즉각 채수 및 계수 → 경보 발령 여부 결정
- 취수장 취수 제한 조치 검토
- 정수 처리 강화 (활성탄·응집제)
- K-water 본부·환경부 보고
- 일 2회 모니터링 체계 가동""",
        '대발생': f"""**현재 위험도: 매우 높음 (대발생)**
- 긴급 대응팀 즉각 소집
- 취수 중단 또는 심층 취수구 전환
- 조류독소(마이크로시스틴) 즉시 분석
- 수면 폭기 설비 최대 가동
- 언론·환경부·지자체 동시 통보
- 황토 살포·친환경 저감제 투입 검토""",
    }
    return protocols.get(level, "")


PROTOCOL_LEVEL_ORDER = ['미발령', '관심', '경계', '대발생']


def _norm_alert_level(x) -> str:
    s = str(x).strip() if x is not None and not (isinstance(x, float) and np.isnan(x)) else '미발령'
    return s if s in PROTOCOL_LEVEL_ORDER else '미발령'


def _transition_rate_within_days(
    dz: pd.DataFrame, from_lv: str, to_lvs: list[str], days: int = 30,
) -> float | None:
    if dz.empty or '조사일' not in dz.columns or '발령단계' not in dz.columns:
        return None
    d = dz.sort_values('조사일').reset_index(drop=True).copy()
    d['_lv'] = d['발령단계'].map(_norm_alert_level)
    dates = pd.to_datetime(d['조사일']).values.astype('datetime64[D]')
    lv = d['_lv'].values
    idx = np.where(lv == from_lv)[0]
    if len(idx) == 0:
        return None
    hit = 0
    for i in idx:
        t0 = dates[i]
        t1 = t0 + np.timedelta64(int(days), 'D')
        m = (dates > t0) & (dates <= t1)
        sub = lv[m]
        if np.any(np.isin(sub, np.array(to_lvs))):
            hit += 1
    return float(hit / len(idx))


def compute_site_protocol_metrics(df_site: pd.DataFrame) -> dict:
    """TAB4 카드·B/C용 — 선택 지점 실측만 사용 (문구 내 숫자는 전부 여기서)."""
    empty = {'empty': True, 'n_rows': 0, 'span_years': 1.0, 'median_gap_days': 7.0,
             'mean_gap_days': 7.0, 'risk_days_per_year': 0.0, 'pct_by_level': {k: 0.25 for k in PROTOCOL_LEVEL_ORDER},
             'chd_med_clear': 0.0, 'chd_med_risk': 0.0, 'chla_med_clear': 0.0, 'chla_med_risk': 0.0,
             'trans_gwan_to_hi_30d': None, 'date_min': None, 'date_max': None, 'n_dae': 0}
    if df_site.empty or '조사일' not in df_site.columns:
        return empty
    dz = df_site.dropna(subset=['조사일']).sort_values('조사일')
    if dz.empty:
        return empty
    gaps = pd.to_datetime(dz['조사일']).diff().dt.days.dropna()
    med_gap = float(gaps.median()) if len(gaps) > 0 else 7.0
    mean_gap = float(gaps.mean()) if len(gaps) > 0 else med_gap
    d0, d1 = dz['조사일'].min(), dz['조사일'].max()
    span_y = max((d1 - d0).days / 365.25, 0.25)

    if '발령단계' not in dz.columns:
        pct = {k: (1.0 if k == '미발령' else 0.0) for k in PROTOCOL_LEVEL_ORDER}
        risk_mask = pd.Series(False, index=dz.index)
    else:
        dz = dz.copy()
        dz['_lv'] = dz['발령단계'].map(_norm_alert_level)
        vc = dz['_lv'].value_counts(normalize=True)
        pct = {k: float(vc.get(k, 0.0)) for k in PROTOCOL_LEVEL_ORDER}
        risk_mask = dz['_lv'] != '미발령'

    n_risk = int(risk_mask.sum())
    risk_per_y = n_risk / span_y

    chd_c = chd_r = chla_c = chla_r = 0.0
    if 'CHD' in dz.columns:
        a, b = dz.loc[~risk_mask, 'CHD'].dropna(), dz.loc[risk_mask, 'CHD'].dropna()
        chd_c = float(a.median()) if len(a) else 0.0
        chd_r = float(b.median()) if len(b) else 0.0
    if C_CHLA in dz.columns:
        a, b = dz.loc[~risk_mask, C_CHLA].dropna(), dz.loc[risk_mask, C_CHLA].dropna()
        chla_c = float(a.median()) if len(a) else 0.0
        chla_r = float(b.median()) if len(b) else 0.0

    tr = _transition_rate_within_days(dz, '관심', ['경계', '대발생'], 30)

    return {
        'empty': False,
        'n_rows': int(len(dz)),
        'span_years': float(span_y),
        'median_gap_days': med_gap,
        'mean_gap_days': float(mean_gap),
        'risk_days_per_year': float(risk_per_y),
        'pct_by_level': pct,
        'chd_med_clear': chd_c,
        'chd_med_risk': chd_r,
        'chla_med_clear': chla_c,
        'chla_med_risk': chla_r,
        'trans_gwan_to_hi_30d': tr,
        'date_min': d0,
        'date_max': d1,
        'n_dae': int((dz['_lv'] == '대발생').sum()) if '_lv' in dz.columns else 0,
    }


def _proto_header_style(level: str) -> tuple[str, str]:
    """(배경, 글자색)"""
    if level == '미발령':
        return '#28a745', '#ffffff'
    if level == '관심':
        return '#fff3cd', '#856404'
    if level == '경계':
        return '#f8d7da', '#721c24'
    return '#6f1428', '#ffffff'


def _t4n(x: str) -> str:
    """TAB4 본문 숫자 강조."""
    return f'<span class="t4-num">{x}</span>'


def build_protocol_steps(level: str, m: dict, horizon: int, site_name: str) -> list[str]:
    """단계별 3줄 — 숫자는 모두 compute_site_protocol_metrics 산출."""
    p = m.get('pct_by_level', {})
    pm, pr = m.get('chd_med_clear', 0), m.get('chd_med_risk', 0)
    cc, cr = m.get('chla_med_clear', 0), m.get('chla_med_risk', 0)
    gap = m.get('median_gap_days', 7.0)
    rpy = m.get('risk_days_per_year', 0.0)
    tr = m.get('trans_gwan_to_hi_30d')
    trs = f"{tr * 100:.0f}%" if tr is not None and tr == tr else "데이터 부족"
    suggest = max(1.0, gap * 0.55)
    span = m.get('span_years', 1.0)
    n_dae = m.get('n_dae', 0)
    n_all = m.get('n_rows', 1)
    pct_bal = p.get('미발령', 0) * 100
    pct_non = (1 - p.get('미발령', 0)) * 100

    if level == '미발령':
        rng = ""
        if m.get('date_min') is not None and m.get('date_max') is not None:
            rng = (
                f' <span class="t4-muted">({site_name} · {m["date_min"]:%Y-%m-%d} ~ {m["date_max"]:%Y-%m-%d})</span>'
            )
        return [
            f"실측 조사 중앙 간격 {_t4n(f'{gap:.0f}')}일 유지{rng}.",
            f"미발령 비중 {_t4n(f'{pct_bal:.0f}%')} — T+{horizon} 예보·수온·Chl-a 추세를 동일 주기로 점검.",
            f"CHD 중앙: 비발령 {_t4n(f'{pm:.0f}')}일 vs 위험 {_t4n(f'{pr:.0f}')}일 (고온 누적 시 선제 점검).",
        ]
    if level == '관심':
        tr_html = _t4n(trs) if trs != '데이터 부족' else '—'
        return [
            f"관심 이상 비중 {_t4n(f'{pct_non:.0f}%')} — 채수·계수를 "
            f"{_t4n(f'{suggest:.0f}')}일 이내로 단축 권고 (중앙 간격의 약 55%).",
            f"관심 시점 이후 30일 내 경계·대발생 출현 비율 {tr_html} (역사적 패턴).",
            f"Chl-a 중앙: 평시 {_t4n(f'{cc:.1f}')} vs 위험 {_t4n(f'{cr:.1f}')} ㎎/㎥ — 현미경 계수·방류 검토.",
        ]
    if level == '경계':
        tr_html = _t4n(trs) if trs != '데이터 부족' else '—'
        return [
            f"연간 위험일(≠미발령) 밀도 {_t4n(f'{rpy:.0f}')}일/년 "
            f"<span class='t4-muted'>(관측 {span:.1f}년 · n={n_all})</span> — 즉시 채수·발령 판단.",
            f"CHD {_t4n(f'{pr:.0f}')}일 수준(위험일 중앙)에서 방류·HRT·정수 강화 검토.",
            f"하류 취수장 공식 통보·취수 제한 시나리오 (관심→경계 30일 패턴 {tr_html} 참고).",
        ]
    pct_dae = 100 * n_dae / max(n_all, 1)
    return [
        f"대발생 표본 {_t4n(str(n_dae))}건 (전체 {_t4n(f'{pct_dae:.2f}')}%) — 비상 소집·취수 전환.",
        f"위험일 Chl-a 중앙 {_t4n(f'{cr:.1f}')} ㎎/㎥ — 조류독소 분석·황토·폭기 최대 가동.",
        f"연간 위험 노출 {_t4n(f'{rpy:.0f}')}일/년 수준을 상정한 비상 물량·언론 대응 체계 점검.",
    ]


def load_protocol_roadmap() -> list[dict]:
    """선택 JSON — 없으면 기본 단계만(문구는 ‘계획 예시’, 수치 카드와 분리)."""
    path = os.path.join(os.path.dirname(__file__), 'protocol_roadmap.json')
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return [
        {'phase': 'Phase 1', 'title': '현재', 'detail': '로컬 Streamlit + CSV (`final_data.csv`)', 'color': '#2e86ab'},
        {'phase': 'Phase 2', 'title': '3개월', 'detail': '기상청 API·자동 수집·DB (TAB3 API 연동 확장)', 'color': '#ffc107'},
        {'phase': 'Phase 3', 'title': '6개월', 'detail': '내부망 배포·경보 알림 자동화', 'color': '#28a745'},
        {'phase': 'Phase 4', 'title': '12개월', 'detail': '금강 수계 확장·독소 모듈', 'color': '#17a2b8'},
    ]


def format_tab4_bc_panel_html(m: dict, tw: float | None, ta: float | None, pred_horizon: int) -> str:
    p = m['pct_by_level']
    kpis = [
        (f"T+{pred_horizon}", "리드타임 참고"),
        (f"{m['median_gap_days']:.1f}", "조사 중앙 간격 (일)"),
        (f"{m['risk_days_per_year']:.0f}", "위험일·년"),
        (f"{p['미발령'] * 100:.0f}%", "미발령 비중"),
    ]
    khtml = "".join(
        f'<div class="t4-bc-kpi"><div class="t4-bc-kpi-val">{a}</div><div class="t4-bc-kpi-sub">{b}</div></div>'
        for a, b in kpis
    )
    dlines = [
        "<div style='margin-bottom:7px'><b>단계 비중</b> — 미발령 "
        f"<span class='t4-num'>{p['미발령'] * 100:.0f}%</span> / 관심 "
        f"<span class='t4-num'>{p['관심'] * 100:.0f}%</span> / 경계 "
        f"<span class='t4-num'>{p['경계'] * 100:.0f}%</span> / 대발생 "
        f"<span class='t4-num'>{p['대발생'] * 100:.1f}%</span></div>",
        "<div style='margin-bottom:7px'><b>CHD 중앙</b> (비발령 vs 위험): "
        f"<span class='t4-num'>{m['chd_med_clear']:.0f}</span>일 vs "
        f"<span class='t4-num'>{m['chd_med_risk']:.0f}</span>일</div>",
        "<div><b>Chl-a 중앙</b> (비발령 vs 위험): "
        f"<span class='t4-num'>{m['chla_med_clear']:.1f}</span> vs "
        f"<span class='t4-num'>{m['chla_med_risk']:.1f}</span> ㎎/㎥</div>",
    ]
    if m.get('trans_gwan_to_hi_30d') is not None:
        dlines.append(
            "<div style='margin-top:9px'><b>관심 후 30일 내 경계+</b>: "
            f"<span class='t4-num'>{m['trans_gwan_to_hi_30d'] * 100:.0f}%</span></div>"
        )
    if tw is not None and ta is not None:
        foot = (
            "<b>저장 GBM 과거 확률 분위</b> (동일 지점): 관심 ≥ "
            f"<span class='t4-num'>{tw:.2f}</span>, 경계 ≥ <span class='t4-num'>{ta:.2f}</span> "
            "— FN:FP=6:1 가정 시 운영 임계와 병행 검토."
        )
    else:
        foot = (
            "<b>임계</b>: 저장 모델 없음 — 분위 임계 미표시. "
            "비용비·운영 기준은 README·내부 가이드와 병행하세요."
        )
    return (
        '<div class="t4-bc"><div class="t4-bc-head">B/C 분석 — 시스템·운영 지표 '
        '<span class="t4-tag">실데이터</span></div>'
        f'<div class="t4-bc-kpis">{khtml}</div>'
        f'<div class="t4-bc-detail">{"".join(dlines)}</div>'
        f'<div class="t4-bc-foot">{foot}</div></div>'
    )


def format_tab4_roadmap_html(items: list[dict]) -> str:
    parts = []
    for it in items:
        c = it.get('color', '#666')
        parts.append(
            f'<div class="t4-rm-item" style="--t4-dot:{c};"><div class="t4-rm-ph">'
            f'{it.get("phase", "")} · {it.get("title", "")}</div>'
            f'<div class="t4-rm-tx">{it.get("detail", "")}</div></div>'
        )
    return (
        '<div class="t4-rm"><div class="t4-rm-head">구현 로드맵 <span class="t4-tag">계획</span></div>'
        f'<div class="t4-rm-body">{"".join(parts)}</div></div>'
    )


def generate_forecast(base_temp: float = 20.0, scenario: str = "현재 추세 유지") -> pd.DataFrame:
    """현실적인 계절 기반 예보 생성"""
    np.random.seed(42)
    month = datetime.today().month
    # 계절 기준 기온 (월별)
    seasonal_base = {1: 0, 2: 2, 3: 8, 4: 14, 5: 19, 6: 23,
                     7: 26, 8: 27, 9: 22, 10: 16, 11: 8, 12: 2}
    s_base = seasonal_base.get(month, 15)
    dates = [datetime.today() + timedelta(days=i) for i in range(1, 11)]

    if scenario == "현재 추세 유지":
        temps = s_base + np.random.randn(10) * 1.5
        rains = np.random.exponential(2, 10)
    elif scenario == "폭염 지속":
        temps = s_base + np.linspace(2, 6, 10) + np.random.randn(10) * 0.5
        rains = np.zeros(10)
    elif scenario == "강우 유입":
        temps = s_base - np.linspace(0, 4, 10) + np.random.randn(10) * 0.5
        rains = np.random.exponential(5, 10)
        rains[3:6] = [25, 40, 18]
    else:
        temps = s_base + np.random.randn(10) * 2.5
        rains = np.random.exponential(3, 10)

    df = pd.DataFrame({
        'date':  dates,
        'temp':  np.clip(temps, -5, 40),
        'rain':  np.clip(rains, 0, 100),
        'solar': np.random.uniform(8, 22, 10),
    })
    sc = str(scenario)
    df['temp_src'] = f'시나리오·난수({sc})'
    df['rain_src'] = f'시나리오·난수({sc})'
    return df


def tab1_trend_html(delta: float | None, *, fmt: str) -> str:
    """직전 유효 관측 대비 증감 HTML."""
    if delta is None or (isinstance(delta, float) and not np.isfinite(delta)):
        return '<div class="tab1-trend-flat">— 직전 관측 대비 비교 불가</div>'
    if abs(delta) < 1e-12:
        return '<div class="tab1-trend-flat">— 변화 없음</div>'
    arrow = '▲' if delta > 0 else '▼'
    cls = 'tab1-trend-up' if delta > 0 else 'tab1-trend-down'
    body = fmt.format(delta)
    return f'<div class="{cls}">{arrow} {body} 직전 관측 대비</div>'


def tab1_annual_first_signal(df_site: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    연도별 첫 '이상' 시점: (1) 발령단계≠미발령 첫일 (2) 없으면 해당 연도 total_cyano ≥ 전체 P90 첫일.
    """
    d = df_site.dropna(subset=['조사일']).sort_values('조사일').copy()
    if d.empty:
        return pd.DataFrame(), None
    cy = d['total_cyano'].dropna()
    p90 = float(cy.quantile(0.90)) if len(cy) else np.nan
    rows: list[dict] = []
    for y, g in d.groupby(d['조사일'].dt.year, sort=True):
        g = g.sort_values('조사일')
        g2 = g[g['발령단계'].astype(str) != '미발령']
        if len(g2):
            t0 = pd.Timestamp(g2['조사일'].iloc[0])
            typ = '발령'
        elif np.isfinite(p90):
            gg = g[g['total_cyano'].notna() & (g['total_cyano'] >= p90)]
            if len(gg):
                t0 = pd.Timestamp(gg['조사일'].iloc[0])
                typ = '남조류≥P90'
            else:
                continue
        else:
            continue
        rows.append({
            '연': int(y),
            '시작일': t0,
            '연일차': int(t0.dayofyear),
            '유형': typ,
        })
    out = pd.DataFrame(rows)
    note = None
    if len(out) >= 3:
        xs = out['연'].astype(float).values
        ys = out['연일차'].astype(float).values
        coef = np.polyfit(xs, ys, 1)
        dy = float(coef[0])
        if dy < -0.4:
            note = f"연도가 지날수록 위 시점이 평균 **약 {abs(dy):.1f}일/년** 앞당겨지는 추세입니다."
        elif dy > 0.4:
            note = f"연도가 지날수록 위 시점이 평균 **약 {dy:.1f}일/년** 늦어지는 추세입니다."
    return out, note


def tab1_all_sites_last(df_all: pd.DataFrame) -> pd.DataFrame:
    """채수위치별 마지막 유효 행(남조류·발령)."""
    sites = sorted(df_all['채수위치'].dropna().unique())
    rows = []
    for s in sites:
        dz = df_all[df_all['채수위치'] == s].sort_values('조사일')
        vc = dz['total_cyano'].dropna()
        last_c = float(vc.iloc[-1]) if len(vc) else np.nan
        la = dz['발령단계'].dropna()
        last_a = str(la.iloc[-1]) if len(la) else '미발령'
        if last_a not in ALERT_LEVELS:
            last_a = '미발령'
        rows.append({'채수위치': s, '남조류': last_c, '발령단계': last_a})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
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
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌊 대청댐 유해남조류 예측 의사결정 지원시스템</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 실시간 현황",
    "🔮 위험도 시나리오 + 해결책",
    "☁️ 10일 기상 예보",
    "📋 운영 프로토콜",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 실시간 현황 — 카드형 KPI + 남조류 시계열 + 연도별·전지점 요약 (실데이터)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_site = df_all[df_all['채수위치'] == site].sort_values('조사일')

    def last_valid(series: pd.Series):
        s = series.dropna()
        return s.iloc[-1] if len(s) > 0 else np.nan

    last_date = last_valid(df_site['조사일'].where(df_site['total_cyano'].notna()))
    last_cyano = last_valid(df_site['total_cyano'])
    last_temp = last_valid(df_site[C_TEMP])
    last_chla = last_valid(df_site[C_CHLA])
    last_alert = last_valid(df_site['발령단계'])
    if pd.isna(last_alert) or str(last_alert) not in ALERT_LEVELS:
        last_alert = '미발령'
    last_alert = str(last_alert)

    valid_cyano = df_site['total_cyano'].dropna()
    prev_cyano = float(valid_cyano.iloc[-2]) if len(valid_cyano) >= 2 else np.nan
    valid_temp = df_site[C_TEMP].dropna()
    prev_temp = float(valid_temp.iloc[-2]) if len(valid_temp) >= 2 else np.nan
    valid_chla = df_site[C_CHLA].dropna()
    prev_chla = float(valid_chla.iloc[-2]) if len(valid_chla) >= 2 else np.nan

    delta_cyano = (float(last_cyano) - prev_cyano) if (pd.notna(last_cyano) and np.isfinite(prev_cyano)) else None
    delta_temp = (float(last_temp) - prev_temp) if (pd.notna(last_temp) and np.isfinite(prev_temp)) else None
    delta_chla = (float(last_chla) - prev_chla) if (pd.notna(last_chla) and np.isfinite(prev_chla)) else None

    st.markdown("#### 실시간 현황")
    st.markdown(
        '<span class="tab1-chip">dropna·마지막 유효값</span>'
        '<span class="tab1-chip">발령단계 색상</span>'
        '<span class="tab1-chip">연도별 시점 = 데이터 규칙 산출</span>',
        unsafe_allow_html=True,
    )

    inf = ALERT_LEVELS[last_alert]
    date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'N/A'
    pill_bg = '#d4edda' if last_alert == '미발령' else ('#fff3cd' if last_alert == '관심' else ('#f8d7da' if last_alert == '경계' else '#e8d4dc'))
    pill_fg = '#155724' if last_alert == '미발령' else ('#856404' if last_alert == '관심' else ('#721c24' if last_alert == '경계' else '#4a0f18'))

    k1 = (
        f'<div class="tab1-kpi-card">'
        f'<div class="tab1-kpi-label">현재 발령</div>'
        f'<span class="tab1-pill" style="background:{pill_bg};color:{pill_fg};">'
        f'{escape(inf["emoji"] + " " + last_alert)}</span>'
        f'<div class="tab1-kpi-sub" style="margin-top:10px">{escape(date_str)} 기준</div></div>'
    )
    k2 = (
        f'<div class="tab1-kpi-card">'
        f'<div class="tab1-kpi-label">남조류 (cells/mL)</div>'
        f'<div class="tab1-kpi-num">{last_cyano:,.0f}</div>'
        f'{tab1_trend_html(delta_cyano, fmt="{:+,.0f}")}</div>'
        if pd.notna(last_cyano) else
        f'<div class="tab1-kpi-card"><div class="tab1-kpi-label">남조류 (cells/mL)</div>'
        f'<div class="tab1-kpi-num">N/A</div><div class="tab1-trend-flat">데이터 없음</div></div>'
    )
    k3 = (
        f'<div class="tab1-kpi-card">'
        f'<div class="tab1-kpi-label">수온 (℃)</div>'
        f'<div class="tab1-kpi-num">{last_temp:.1f}</div>'
        f'{tab1_trend_html(delta_temp, fmt="{:+.1f}")}</div>'
        if pd.notna(last_temp) else
        f'<div class="tab1-kpi-card"><div class="tab1-kpi-label">수온 (℃)</div>'
        f'<div class="tab1-kpi-num">N/A</div><div class="tab1-trend-flat">데이터 없음</div></div>'
    )
    k4 = (
        f'<div class="tab1-kpi-card">'
        f'<div class="tab1-kpi-label">{escape(C_CHLA)}</div>'
        f'<div class="tab1-kpi-num">{last_chla:.1f}</div>'
        f'{tab1_trend_html(delta_chla, fmt="{:+.1f}")}</div>'
        if pd.notna(last_chla) else
        f'<div class="tab1-kpi-card"><div class="tab1-kpi-label">{escape(C_CHLA)}</div>'
        f'<div class="tab1-kpi-num">N/A</div><div class="tab1-trend-flat">데이터 없음</div></div>'
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(k1, unsafe_allow_html=True)
    with c2:
        st.markdown(k2, unsafe_allow_html=True)
    with c3:
        st.markdown(k3, unsafe_allow_html=True)
    with c4:
        st.markdown(k4, unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    df_recent = df_site.sort_values('조사일').dropna(subset=['total_cyano']).tail(30)
    yr_df, yr_note = tab1_annual_first_signal(df_site)
    sites_df = tab1_all_sites_last(df_all)

    col_main, col_side = st.columns([1.65, 1.0])
    with col_main:
        st.markdown(f"##### 남조류 농도 시계열 (log) — `{escape(site)}` 최근 30관측")
        if len(df_recent) == 0:
            st.info('선택 지점에 남조류(`total_cyano`) 유효 관측이 없습니다.')
        else:
            plot_df = df_recent.copy()
            plot_df['_cy'] = np.maximum(plot_df['total_cyano'].astype(float), 1.0)
            plot_df['단계'] = plot_df['발령단계'].fillna('미발령').astype(str)
            unk = ~plot_df['단계'].isin(TAB1_CYANO_BAR.keys())
            plot_df.loc[unk, '단계'] = '미발령'
            fig_cy = px.bar(
                plot_df,
                x='조사일',
                y='_cy',
                color='단계',
                color_discrete_map=TAB1_CYANO_BAR,
                category_orders={'단계': ['미발령', '관심', '경계', '대발생']},
                labels={'_cy': '남조류 (cells/mL, log y)', '조사일': ''},
            )
            fig_cy.update_traces(hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f} cells/mL<extra></extra>')
            fig_cy.update_layout(
                yaxis_type='log',
                height=380,
                margin=dict(t=12, b=40),
                legend_title_text='발령단계',
                legend=dict(orientation='h', yanchor='top', y=-0.22, x=0),
                bargap=0.15,
            )
            st.plotly_chart(fig_cy, use_container_width=True)

    with col_side:
        st.markdown('<div class="tab1-side-title">연도별 이상 시점 (첫 일)</div>', unsafe_allow_html=True)
        if yr_df.empty:
            st.caption('발령 또는 상위 농도 기준으로 표시할 연도가 없습니다.')
        else:
            fig_y = go.Figure(
                go.Bar(
                    x=yr_df['연일차'],
                    y=yr_df['연'].astype(str),
                    orientation='h',
                    marker_color='#5c7cfa',
                    text=yr_df['시작일'].dt.strftime('%m/%d'),
                    textposition='outside',
                    customdata=np.stack([yr_df['시작일'].dt.strftime('%Y-%m-%d'), yr_df['유형']], axis=-1),
                    hovertemplate='%{customdata[0]} (%{customdata[1]})<extra></extra>',
                )
            )
            fig_y.update_layout(
                height=max(220, 42 * len(yr_df)),
                margin=dict(l=48, r=16, t=8, b=28),
                xaxis_title='연중 일수 (day-of-year)',
                yaxis_title='',
            )
            st.plotly_chart(fig_y, use_container_width=True)
            if yr_note:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#856404;background:#fffbf0;'
                    f'border-radius:8px;padding:8px 10px;border:1px solid #ffe8a3">{escape(yr_note)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tab1-side-title">채수위치 비교 (최근 유효)</div>', unsafe_allow_html=True)
        site_rows = []
        for _, sr in sites_df.iterrows():
            nm = escape(str(sr['채수위치']))
            em = ALERT_LEVELS.get(sr['발령단계'], ALERT_LEVELS['미발령'])
            cyv = f"{float(sr['남조류']):,.0f}" if np.isfinite(sr['남조류']) else '—'
            site_rows.append(
                f'<div class="tab1-site-row">'
                f'<span><b>{nm}</b></span>'
                f'<span style="color:#1f4e79">{cyv}</span>'
                f'<span style="color:{em["color"]};font-weight:700">{escape(str(sr["발령단계"]))}</span>'
                f'</div>'
            )
        st.markdown(
            '<div class="tab1-side-card">' + ''.join(site_rows) + '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 위험도 시나리오 — 3단계(구역·기여·해결책) + What-if + GBM(algae_model_t7.pkl 우선)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_z = df_all[df_all['채수위치'] == site].sort_values('조사일')
    last_ml = df_z.iloc[-1]
    fcols = get_model_feature_cols(model) if model is not None else []
    t2_thr = tab2_site_adaptive_thresholds(df_z)
    sv = tab2_last_row_slider_defaults(last_ml)

    st.subheader(f"🎛️ 시나리오 입력 — T+{pred_horizon}일 위험도 평가")
    st.caption(
        "흐름: **①구역 → ②기여 → ③권고 → ④What-if → ⑤시나리오** · "
        "저장 모델이 있으면 T+7 GBM으로 계산합니다."
    )
    ld = pd.to_datetime(last_ml['조사일']).strftime('%Y-%m-%d') if pd.notna(last_ml.get('조사일')) else '—'
    la = str(last_ml.get('발령단계', '미발령'))
    cy = last_ml.get('total_cyano', np.nan)
    cy_s = f"{float(cy):,.0f}" if pd.notna(cy) else '—'
    med_cy = float(df_z['total_cyano'].dropna().median()) if 'total_cyano' in df_z.columns and df_z['total_cyano'].notna().any() else np.nan
    cy_note = ''
    if pd.notna(cy) and pd.notna(med_cy) and med_cy > 1e-6:
        cy_note = f" · 남조류는 지점 중앙값 대비 **{float(cy) / med_cy:.2f}배**"
    st.markdown(
        f'<div class="t2-sec-sub" style="background:#f8fafc;border:1px solid #e3e8ef;border-radius:10px;'
        f'padding:10px 12px;margin-bottom:10px">'
        f'<b>최근 실측 기준</b> 조사일 <code>{escape(ld)}</code> · 발령 <b>{escape(la)}</b> · '
        f'남조류 <b>{escape(cy_s)}</b> cells/mL{cy_note} · 슬라이더 초깃값 = <b>이 날 관측</b> '
        f'(시나리오는 여기서 조정) · 구역 경계는 <b>이 지점 이력 분위</b>와 전역 기준을 혼합합니다.</div>',
        unsafe_allow_html=True,
    )

    col_gauge_area, col_slider_area = st.columns([1.05, 2.15])
    gauge_slot = col_gauge_area.empty()
    with col_slider_area:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**수질 조건**")
            p_temp = st.slider("수온 (°C)", 5.0, 35.0, float(sv['p_temp']), 0.5, key="t2_t")
            p_chla = st.slider("Chl-a (㎎/㎥)", 0.0, 100.0, float(sv['p_chla']), 1.0, key="t2_c")
            p_turb = st.slider("탁도 (NTU)", 0.0, 60.0, float(sv['p_turb']), 0.5, key="t2_tu")
            p_do = st.slider("DO (mg/L)", 0.0, 15.0, float(sv['p_do']), 0.5, key="t2_d")
        with c2:
            st.markdown("**기상·수문 조건**")
            p_chd = st.slider("연속폭염일수 CHD", 0, 40, int(sv['p_chd']), 1, key="t2_ch")
            p_tai7 = st.slider("열누적지수 TAI_7 (°C·일)", 0.0, 100.0, float(sv['p_tai7']), 1.0, key="t2_tai")
            p_cdd = st.slider("연속무강우일수 CDD", 0, 40, int(sv['p_cdd']), 1, key="t2_cd")
            p_solar = st.slider("일사량 (MJ/m²)", 0.0, 25.0, float(sv['p_solar']), 0.5, key="t2_s")
        with c3:
            st.markdown("**댐 운영 조건**")
            p_hrt = st.slider("체류시간 HRT (일)", 5, 150, int(sv['p_hrt']), 5, key="t2_h")
            p_inflow = st.slider("유입량 (m³/s)", 0.0, 500.0, float(sv['p_inflow']), 5.0, key="t2_i")
            p_release = st.slider("방류량 (m³/s)", 0.0, 300.0, float(sv['p_release']), 5.0, key="t2_r")
            p_storage = st.slider("저수율 (%)", 10, 100, int(sv['p_storage']), 1, key="t2_st")

    solar_max = float(df_all[C_SOLAR].quantile(0.95)) if C_SOLAR in df_all.columns else 20.0
    bgi_val = ((p_temp - 20) / 10 if p_temp > 20 else 0) * (p_solar / max(solar_max, 1e-6)) * (1 / (p_turb + 1))

    params_sliders: dict = {
        C_TEMP: p_temp,
        C_CHLA: p_chla,
        C_TURB: p_turb,
        C_DO: p_do,
        'CHD': float(p_chd),
        'CDD': float(p_cdd),
        C_SOLAR: p_solar,
        'TAI_7': float(p_tai7),
        'HRT_est': float(p_hrt),
        C_INFLOW: p_inflow,
        C_OUTFLOW: p_release,
        'BGI': float(bgi_val),
    }
    if '저수율(%)' in fcols:
        params_sliders['저수율(%)'] = float(p_storage)

    merged = merge_feature_row(last_ml, params_sliders, fcols) if fcols else {}

    params_operational = {
        C_TEMP: p_temp,
        C_TURB: p_turb,
        'CHD': float(p_chd),
        'TAI_7': float(p_tai7),
        'CDD': float(p_cdd),
        'HRT_est': float(p_hrt),
        'BGI': float(bgi_val),
        C_CHLA: p_chla,
        C_INFLOW: p_inflow,
        '저수율(%)': float(p_storage),
    }

    use_pkl = model is not None and bool(fcols)
    risk_prob = None
    if use_pkl:
        try:
            risk_prob = float(model_predict_prob_positive(model, merged, fcols))
        except Exception:
            risk_prob = None
    gbm_op = imp_op = feats_op = None
    if risk_prob is None:
        use_pkl = False
        try:
            gbm_op, imp_op, feats_op = build_operational_model(df_all)
            risk_prob = tab2_predict_operational(params_operational, gbm_op, imp_op, feats_op)
        except Exception as ex:
            st.error(f"운영 GBM 학습 실패: {ex}")
            gbm_op, imp_op, feats_op = None, None, []
            risk_prob = 0.0

    risk_pct = risk_prob * 100.0
    level = tab2_prob_level(risk_prob)
    info = ALERT_LEVELS.get(level, ALERT_LEVELS['미발령'])
    gauge_title = (
        f"P(발령 T+7) — 저장 GBM" if use_pkl else "P(발령) — 운영변수 GBM"
    )

    if use_pkl:
        contribs = tab2_driver_scores_pkl(model, merged, fcols, t2_thr)
        wi_df = compute_whatif_pkl_tab2(model, merged, fcols)
    elif gbm_op is not None and feats_op:
        contribs = tab2_compute_shap_proxy_operational(params_operational, feats_op, t2_thr)
        wi_df = compute_whatif_tab2_operational(params_operational, gbm_op, imp_op, feats_op)
    else:
        contribs = {}
        wi_df = pd.DataFrame(
            columns=["개입 방안", "위험도", "변화", "예상 단계", "권고", "_delta_raw", "_prob_raw"]
        )

    with gauge_slot.container():
        if use_pkl:
            st.caption("**적용:** `algae_model_t7.pkl` 저장 GBM — 아래 확률·What-if·시나리오 비교는 이 모델 기준입니다.")
        elif model is not None:
            st.caption(
                "**적용:** 운영변수 GBM — 저장 파일은 로드됐으나 `predict_proba` 실패·피처 불일치 등으로 "
                "이 탭에서는 운영 모델을 씁니다."
            )
        else:
            st.caption("**적용:** 운영변수 GBM — `algae_model_t7.pkl`이 없거나 로드되지 않았습니다.")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_pct, 1),
            number={"suffix": "%"},
            title={"text": gauge_title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": info["color"]},
                "steps": [
                    {"range": [0, 25], "color": "#d4edda"},
                    {"range": [25, 55], "color": "#fff3cd"},
                    {"range": [55, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": risk_pct},
            },
        ))
        fig_gauge.update_layout(height=270, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            f'<div class="risk-box {info["css"]}">{info["emoji"]} {level}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "단계: 확률 >25% 관심, >55% 경계 (What-if 테이블과 동일 기준)."
            + (
                " 저장 GBM은 **당일 피처로 7일 후 발령** 학습 정의를 따릅니다."
                if use_pkl
                else ""
            )
        )

    st.divider()

    with st.expander("① 변수 구역 진단 (실데이터 분위)", expanded=False):
        st.markdown(
            '<div class="t2-sec-sub">관심·경계 구간은 <b>이 지점 이력(분위)</b>과 전역 운영 기준을 절반씩 섞어 잡습니다 '
            f'(n={len(df_z)}행).</div>',
            unsafe_allow_html=True,
        )
        zone_vars = [C_TEMP, C_TURB, 'TAI_7', 'CHD', C_CHLA]
        zone_cols = st.columns(len(zone_vars))
        for i, var in enumerate(zone_vars):
            val = float(params_operational.get(var, 0) or 0)
            zone = tab2_get_zone(var, val, t2_thr)
            css = {'안전': 'zone-safe', '관심': 'zone-watch', '경계': 'zone-alert'}[zone]
            label = TAB2_SCENARIO_SOLUTION.get(var, {}).get('label', var)
            lo, hi = t2_thr.get(var, (np.nan, np.nan))
            thr_s = f"관심≥{lo:.1f} 경계≥{hi:.1f}" if pd.notna(lo) and pd.notna(hi) else ''
            with zone_cols[i]:
                st.markdown(
                    f'<div class="t2-zone-cell"><b>{escape(str(label))}</b><br>'
                    f'<span style="color:#64748b;font-size:0.72rem">{escape(thr_s)}</span><br>'
                    f'<span style="color:#64748b;font-size:0.8rem">현재</span> '
                    f'<code style="font-size:0.85rem">{val:.1f}</code><br><br>'
                    f'<span class="zone-badge {css}">{escape(zone)}</span></div>',
                    unsafe_allow_html=True,
                )

    top_vars = sorted(contribs.items(), key=lambda x: -x[1]) if contribs else []

    col_d, col_s = st.columns([1, 1.2])

    with col_d:
        st.markdown(
            '<div class="t2-sec-title">② 위험 기여 (GBM 피처 중요도)</div>'
            '<div class="t2-sec-sub">저장 GBM: `feature_importances_` · 운영 폴백: 고정 가중 프록시</div>',
            unsafe_allow_html=True,
        )
        if top_vars:
            labels = [TAB2_SCENARIO_SOLUTION.get(f, {}).get('label', f) for f, _ in top_vars]
            values = [v for _, v in top_vars]
            bar_colors = ["#dc3545" if v > 0.3 else "#ffc107" if v > 0.1 else "#28a745" for v in values]
            fig_shap = go.Figure(go.Bar(
                y=labels[::-1],
                x=values[::-1],
                orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{v:.3f}" for v in values[::-1]],
                textposition="outside",
            ))
            fig_shap.update_layout(
                height=310,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis_title="기여도 (프록시)",
                xaxis=dict(range=[0, max(values) * 1.35 if values else 0.5]),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("기여도 막대를 그릴 수 없습니다.")

        st.markdown(
            f'<div class="t2-sec-sub" style="margin-top:10px"><b>BGI</b> <code>{bgi_val:.3f}</code> · '
            f'(수온−20)/10 × 일사비 × 1/(탁도+1)</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(bgi_val / 0.1, 1.0))

    with col_s:
        st.markdown(
            '<div class="t2-sec-title">③ 운영 점검 · 권고</div>'
            '<div class="t2-sec-sub">표는 <b>현재 슬라이더 값</b>을 지점 누적분포와 비교한 뒤, 한 줄 조치만 둡니다. '
            '상세 문장은 아래에서 펼칩니다.</div>',
            unsafe_allow_html=True,
        )
        chk = tab2_operational_checklist(df_z, params_operational, t2_thr)
        if len(chk):
            chk2 = chk.copy()
            chk2['현재값'] = chk2['현재값'].astype(float).round(1)
            st.dataframe(chk2, use_container_width=True, hide_index=True)
        top3 = [f for f, v in top_vars if v > 0 and f in TAB2_SCENARIO_SOLUTION][:4]
        if not top3:
            st.success("현재 슬라이더 조합에서 **관심·경계 구역** 변수가 없습니다.")
        else:
            with st.expander("상세 권고 (단기·중장기 문장)", expanded=False):
                for var in top3:
                    info_sol = TAB2_SCENARIO_SOLUTION[var]
                    val = float(params_operational.get(var, merged.get(var, 0)) or 0)
                    zone = tab2_get_zone(var, val, t2_thr)
                    if zone == "안전":
                        continue
                    sol = info_sol.get(zone, {})
                    css = "shap-danger" if zone == "경계" else "shap-warn"
                    st.markdown(
                        f"""<div class="shap-card {css}">
  <div class="shap-driver">{info_sol.get('icon', '📌')} {info_sol.get('label', var)} → {sol.get('action', '—')}</div>
  <div class="shap-val">현재값: {val:.1f} {info_sol.get('unit', '')} | 위험 구역: {zone}</div>
  <div class="shap-sol"><b>단기:</b> {sol.get('short', '—')}<br><b>중장기:</b> {sol.get('long', '—')}</div>
</div>""",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.markdown(
        '<div class="t2-sec-title">④ What-if 개입 효과</div>'
        '<div class="t2-sec-sub">저장 GBM: 병합 피처 패치 후 확률 · 운영 GBM: 동일 개입을 주요 변수에 반영</div>',
        unsafe_allow_html=True,
    )

    display_cols = ["개입 방안", "위험도", "변화", "예상 단계", "권고"]
    display_df = wi_df[display_cols].copy() if len(wi_df) else pd.DataFrame(columns=display_cols)

    def _row_style(row):
        d = float(wi_df.loc[row.name, "_delta_raw"])
        n = len(display_cols)
        if d < -0.05:
            return ["background-color:#d4edda"] * n
        if d < 0:
            return ["background-color:#f0fff0"] * n
        if d > 0:
            return ["background-color:#fff3f3"] * n
        return [""] * n

    if len(wi_df):
        st.dataframe(
            display_df.style.apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("What-if 결과를 생성하지 못했습니다.")
    st.caption("표 색: 초록 = 위험도 5%p 이상 감소 · 연두 = 소폭 감소 · 빨강 = 증가")

    st.divider()
    st.markdown(
        '<div class="t2-sec-title">⑤ 시나리오 비교 (5종)</div>'
        '<div class="t2-sec-sub">고정 패치 대비 발령 확률(%). 점선: 관심 25% · 파선: 경계 55%</div>',
        unsafe_allow_html=True,
    )
    col_sc1, col_sc2 = st.columns(2)

    patches = {
        "A. 현재 조건": {},
        "B. 복합 대응": {"HRT_est": p_hrt * 0.70, C_TEMP: p_temp - 1.0, "TAI_7": p_tai7 * 0.90},
        "C. 폭염 +3°C": {C_TEMP: p_temp + 3, "CHD": float(p_chd + 7), "TAI_7": p_tai7 + 21},
        "D. 장마 이후": {C_TURB: p_turb * 3, "CDD": 0.0},
        "E. 방류 최대 +40%": {C_OUTFLOW: p_release * 1.4} if p_release > 0 else {"HRT_est": p_hrt * 0.65},
    }

    def _scenario_prob(patch: dict) -> float:
        if use_pkl:
            d = {**merged, **patch}
            if C_OUTFLOW in patch:
                d = recalc_hrt_from_outflow(d)
            try:
                return float(model_predict_prob_positive(model, d, fcols))
            except Exception:
                return float("nan")
        if gbm_op is None or not feats_op:
            return float("nan")
        d2 = {**params_operational, **patch}
        return tab2_predict_operational(d2, gbm_op, imp_op, feats_op)

    sc_probs = {n: _scenario_prob(p) for n, p in patches.items()}
    sc_levels = {n: tab2_prob_level(p) if p == p else "미발령" for n, p in sc_probs.items()}

    with col_sc1:
        sc_rows = [{"시나리오": n, "위험도 (%)": round(p * 100, 1)} for n, p in sc_probs.items() if p == p]
        sc_df = pd.DataFrame(sc_rows)
        sc_color_list = [
            ALERT_LEVELS.get(sc_levels[n], ALERT_LEVELS["미발령"])["color"]
            for n, p in sc_probs.items() if p == p
        ]
        if sc_df.empty:
            st.caption("시나리오 확률을 계산하지 못했습니다.")
        else:
            fig_sc = px.bar(
                sc_df,
                x="시나리오",
                y="위험도 (%)",
                color="시나리오",
                color_discrete_sequence=sc_color_list,
                text="위험도 (%)",
            )
            fig_sc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_sc.update_layout(height=320, showlegend=False, margin=dict(t=20, b=10))
            fig_sc.add_hline(y=25, line_dash="dot", line_color="orange", annotation_text="관심 25%")
            fig_sc.add_hline(y=55, line_dash="dash", line_color="red", annotation_text="경계 55%")
            st.plotly_chart(fig_sc, use_container_width=True)

    with col_sc2:
        st.markdown('<div class="t2-sec-sub" style="margin-bottom:8px"><b>해석</b> · 기준 대비 %p</div>', unsafe_allow_html=True)
        base_key = "A. 현재 조건"
        base_p_sc = sc_probs.get(base_key, 0.0)
        for name, p in sc_probs.items():
            lv = sc_levels[name]
            em = ALERT_LEVELS.get(lv, ALERT_LEVELS["미발령"])
            if name == base_key or (not (p == p)):
                delta_str = "(기준)" if name == base_key else "(오류)"
            else:
                delta_str = f"({(p - base_p_sc) * 100:+.1f}%p)"
            st.markdown(
                f"<span style='color:{em['color']};font-weight:bold'>{em['emoji']} {name}</span> "
                f"→ {p * 100:.1f}% <span style='color:gray;font-size:0.85rem'>{delta_str}</span>"
                if p == p
                else f"**{name}** — 계산 실패",
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("📎 분석 연동 — 노트북 · 경로 · SHAP", expanded=False):
        st.markdown(
            '<div class="t2-sec-title" style="border:none;padding:0;margin:0 0 6px 0">README 11절 요약</div>'
            '<div class="t2-sec-sub" style="margin-bottom:10px">'
            '① <code>data_eda.ipynb</code> → ② <code>algae_analysis.ipynb</code> → ③ <code>algae_model_t7.pkl</code> 생성'
            '</div>',
            unsafe_allow_html=True,
        )
        dp = escape(os.path.abspath(DATA_PATH))
        mp = escape(os.path.abspath(MODEL_PATH))
        st.markdown(
            f'<div class="t2-readme-kv"><b>데이터</b><br><code>{dp}</code></div>'
            f'<div class="t2-readme-kv" style="margin-top:6px"><b>모델</b><br><code>{mp}</code></div>',
            unsafe_allow_html=True,
        )
        if os.path.isfile(MODEL_PATH):
            st.success("모델 파일이 감지되었습니다.")
        else:
            st.warning("모델 파일이 없습니다. 노트북 실행 후 `dashboard_app.py`와 같은 폴더에 두세요.")

        st.markdown('<div class="t2-hr"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="t2-sec-title" style="border:none;padding:0;margin:0 0 8px 0">'
            f'발령단계별 변수 (IQR) · <code>{escape(site)}</code></div>',
            unsafe_allow_html=True,
        )
        zdf, zshare = quantile_zones_by_alert(df_z)
        if zdf.empty:
            st.caption("발령단계별 표본이 부족합니다.")
        else:
            st.dataframe(zdf.round(3), use_container_width=True, hide_index=True)
            st.caption(
                "**CHD 분위가 0으로 많이 보이는 이유:** CHD는 해당 조사일까지 **수온(℃)≥25°C가 끊기지 않고 이어진 일수**입니다. "
                "조사일에 수온이 25°C 미만이거나 연속이 끊기면 **0**이 됩니다. **미발령** 일자는 고온 누적이 덜한 날이 많아 P25·P50이 0인 경우가 흔하고, "
                "**경계** 등에서는 긴 연속이 섞여 P75가 커지는 식으로 나뉘는 것이 **실데이터 분포**에 맞는 현상입니다."
            )
        if len(zshare):
            st.caption("발령 단계 비율")
            st.bar_chart(zshare)

        st.markdown('<div class="t2-hr"></div>', unsafe_allow_html=True)
        if model is None:
            st.info("저장 GBM 없음 — 상단 게이지·What-if는 **운영변수 GBM**입니다.")
        elif shap is None:
            st.info("`pip install shap` 설치 후 아래 버튼으로 Tree SHAP을 사용할 수 있습니다.")
        else:
            st.caption("아래 버튼: **현재 슬라이더·병합 행** 기준으로 확률·SHAP·What-if 표를 펼칩니다.")
        if model is not None and shap is not None:
            if st.button("상세 분석: 확률 · SHAP · What-if", key="btn_pkl_shap"):
                try:
                    p0 = model_predict_prob_positive(model, merged, fcols)
                    st.metric("발령 확률 (저장 GBM, 동일 병합 행)", f"{p0:.4f}")
                    drv = shap_positive_drivers_pkl(model, merged, fcols, top_n=5)
                    if drv:
                        st.markdown("**위험 상승 SHAP (상위)**")
                        for fn, vv in drv:
                            st.markdown(f"- `{escape(str(fn))}` : {vv:.5f}")
                    st.markdown("**What-if (저장 GBM)**")
                    st.dataframe(
                        what_if_pkl_table(model, merged, fcols),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as ex:
                    st.error(f"모델/피처 불일치: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 10일 예보 — 예보 시계열 + GBM(또는 규칙) 기반 일별 위험
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("☁️ 10일 예보 — 위험도 타임라인 (데이터·모델 기반)")
    st.caption(
        f"지점 **{site}** | 시나리오 **{forecast_scenario}** | 운영 리드타임 참고 **T+{pred_horizon}** | "
        "GBM은 학습 시 **조사일 t의 정보로 t+7일 발령여부**를 맞추도록 되어 있습니다. "
        "아래 각 예보일 d에 대해 표시하는 값은 **그날 가정된 기상·파생 피처 X_d에 대한 P(발령 on d+7)** 의 "
        "**시나리오·근사 해석**이며, 맨끝 며칠은 d+7이 10일 창을 벗어나 **외삽 성격**이 강합니다."
    )

    df_t3 = df_all[df_all['채수위치'] == site].sort_values('조사일')
    last_row_t3 = df_t3.iloc[-1]
    solar_max_t3 = max(float(df_all[C_SOLAR].quantile(0.95)), 1e-6) if C_SOLAR in df_all.columns else 20.0

    kma_key = (os.environ.get('KMA_SERVICE_KEY') or '').strip() or KMA_SERVICE_KEY_DEFAULT
    fc_df, fc_meta = build_fc_dataframe_10d(
        df_t3, forecast_scenario,
        service_key=kma_key,
        nx=KMA_SHORT_NX,
        ny=KMA_SHORT_NY,
        use_kma=True,
        use_kma_mid=True,
        service_key_mid=None,
        mid_reg_id=KMA_MID_REGID,
    )
    st.caption(f"**예보 출처:** {fc_meta.get('detail', '')} (단기 격자 nx={KMA_SHORT_NX}, ny={KMA_SHORT_NY} | 중기 regId={KMA_MID_REGID})")
    if fc_meta.get('tab3_rain_note'):
        st.caption(fc_meta['tab3_rain_note'])

    st.markdown(tab3_weather_operator_summary(fc_df, last_row_t3, fc_meta, forecast_scenario))

    with st.expander("① 이 10일 표가 아래 **위험 타임라인·CHD** 와 어떻게 연결되나요", expanded=True):
        st.markdown(
            """
            - **각 예보일 d** 의 기온·강수·일사로, 학습과 같은 방식의 **가상 하루 피처 X_d** 를 만든 뒤 GBM(또는 규칙)이 **「d+7일 발령」** 확률을 냅니다. 표의 날짜는 **d**이고, 카드 맨 아래 **발령일**이 **d+7** 입니다.
            - **기온자료 / 강수자료** 열: 같은 날짜라도 출처가 다를 수 있습니다. **단기**만 강수(PCP)를 제공하고, **중기**는 기온만 있어 강수는 **이 지점 CSV의 일주년 강수 중앙값**으로 채운 날이 있습니다.
            - **강수 0 mm** 는 “비 없음”일 수도 있고, **역사적 중앙이 매우 작은 날**일 수도 있습니다. 의미는 **강수자료** 열을 함께 보세요.
            - 사이드바 **기상 시나리오**가 「현재 추세 유지」가 아니면, API·계절통계 값에 **스트레스**를 더한 뒤 위 타임라인을 계산합니다(자료 열에 `+시나리오조정` 표시).
            """
        )

    st.markdown("#### 10일 기상 예보 (기상청 단기·중기 + 지점 일주년 보정)")
    wtab = fc_df.copy()
    for c in ('temp_src', 'rain_src'):
        if c not in wtab.columns:
            wtab[c] = '—'
    wtab['날짜'] = pd.to_datetime(wtab['date']).dt.strftime('%Y-%m-%d')
    wtab = wtab.rename(columns={
        'temp': '기온(°C)', 'rain': '강수(mm)', 'solar': '일사(MJ/㎡)',
        'fc_source': '통합출처', 'temp_src': '기온자료', 'rain_src': '강수자료',
    })
    disp_cols = ['날짜', '기온(°C)', '강수(mm)', '일사(MJ/㎡)', '기온자료', '강수자료', '통합출처']
    show_wx = wtab[disp_cols].copy()
    for col in ('기온(°C)', '강수(mm)', '일사(MJ/㎡)'):
        show_wx[col] = show_wx[col].astype(float).round(2)
    st.dataframe(
        show_wx,
        use_container_width=True,
        hide_index=True,
    )
    ts_h = fc_df['temp_src'].astype(str).tolist() if 'temp_src' in fc_df.columns else [''] * len(fc_df)
    rs_h = fc_df['rain_src'].astype(str).tolist() if 'rain_src' in fc_df.columns else [''] * len(fc_df)
    fig_wx = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('예보 기온 (°C)', '예보 강수 (mm)'),
        row_heights=[0.48, 0.52],
        vertical_spacing=0.08,
    )
    fig_wx.add_trace(
        go.Scatter(
            x=fc_df['date'], y=fc_df['temp'], mode='lines+markers',
            name='기온', line=dict(color='#e85d04', width=2), marker=dict(size=7),
            customdata=np.asarray(ts_h, dtype=object).reshape(-1, 1),
            hovertemplate='%{x|%Y-%m-%d}<br>기온 %{y:.1f}°C<br>%{customdata[0]}<extra></extra>',
        ),
        row=1, col=1,
    )
    fig_wx.add_trace(
        go.Bar(
            x=fc_df['date'], y=fc_df['rain'], name='강수', marker_color='steelblue',
            customdata=np.asarray(rs_h, dtype=object).reshape(-1, 1),
            hovertemplate='%{x|%Y-%m-%d}<br>강수 %{y:.1f} mm<br>%{customdata[0]}<extra></extra>',
        ),
        row=2, col=1,
    )
    fig_wx.update_xaxes(title_text='예보일', row=2, col=1)
    fig_wx.update_layout(height=400, showlegend=False, margin=dict(t=36, b=8))
    st.plotly_chart(fig_wx, use_container_width=True)
    rn_vals = np.nan_to_num(fc_df['rain'].astype(float).values, nan=0.0)
    if len(rn_vals) and float(np.mean(rn_vals < 0.05)) >= 0.5:
        st.caption(
            f"**강수 막대가 거의 안 보일 때:** 10일 중 **{np.mean(rn_vals < 0.05):.0%}** 가 0 mm에 가깝습니다. "
            "단기는 **시간별 PCP**를 합친 값이라 ‘강수없음’만 있으면 **0**이고, 단기가 안 덮는 날은 **일주년 강수 중앙값**인데 "
            "댐·채수 지점 일강수는 **0이 매우 많아** 중앙이 0으로 나오는 경우가 흔합니다. (데이터·정의상 정상이며, **강수자료** 열로 출처를 확인하세요.)"
        )

    feat_cols = get_model_feature_cols(model) if model is not None else FEATURE_COLS_ML
    fc_out = build_10day_forecast_ml(
        df_all, site, fc_df, last_row_t3, model, feat_cols, solar_max_t3,
    )

    # --- 분위 임계: GBM 있으면 과거 확률 분포, 없으면 동일 10일 규칙점수 분포 ---
    if model is not None:
        hist_p = historical_site_probas(df_all, site, model, feat_cols)
        t_watch, t_alert = proba_thresholds_tertiles(hist_p)
        score_col = 'prob'
        st.info(
            f"**GBM 활성** — 과거 {len(hist_p)}일 예측확률 분위로 구간 경계: "
            f"관심 ≥ {t_watch:.2f}, 경계 ≥ {t_alert:.2f}"
        )
    else:
        hist_p = fc_out['rule_score'].values.astype(float)
        t_watch, t_alert = proba_thresholds_tertiles(hist_p)
        score_col = 'rule_score'
        st.warning(
            f"`algae_model_t7.pkl` 없음 — **규칙 위험점수**로 동일 타임라인 표시. "
            f"10일 구간 분위 경계: 관심 ≥ {t_watch:.2f}, 경계 ≥ {t_alert:.2f}"
        )

    fc_out['p_use'] = fc_out[score_col].astype(float)
    if model is not None:
        bad = ~np.isfinite(fc_out['prob'].astype(float))
        fc_out.loc[bad, 'p_use'] = fc_out.loc[bad, 'rule_score'].astype(float)
    fc_out['stage'] = [stage_from_prob(p, t_watch, t_alert) for p in fc_out['p_use']]

    st.caption(
        f"카드의 **%** = GBM 발령 확률×100. **단계(미발령·관심·경계)** 는 그 확률을 "
        f"과거 `{site}` 지점 예측 확률 분포의 분위와 비교한 것입니다(관심 ≥ {t_watch:.2f}, 경계 ≥ {t_alert:.2f}). "
        "과거 확률이 한쪽으로 몰리면 임계는 안전한 기본값(0.30 / 0.60)으로 대체됩니다."
    )

    chd_watch, chd_alert = chd_alert_thresholds_from_data(df_t3)

    # ── 10일 타임라인 카드 (5×2열 — 한 열이 너무 좁아 한글 줄바꿈 깨짐 방지) ───
    st.markdown("#### 10일 위험도 타임라인 (막줄: **발령 대상일 d+7**)")
    n_fc = len(fc_out)
    for row in range(2):
        start = row * 5
        end = min(start + 5, n_fc)
        if start >= n_fc:
            break
        tl_cols = st.columns(5)
        for j in range(start, end):
            col_j = j - start
            r = fc_out.iloc[j]
            stg = r['stage']
            em = ALERT_LEVELS.get(stg, ALERT_LEVELS['미발령'])
            pc = 100.0 * float(r['p_use']) if np.isfinite(r['p_use']) else 0.0
            td = r['target_date']
            td_s = td.strftime('%m/%d') if hasattr(td, 'strftime') else str(td)[:10]
            with tl_cols[col_j]:
                st.markdown(
                    "<div style='text-align:center;padding:8px 4px;border-radius:10px;"
                    "border:2px solid " + em['color'] + ";background:#fafafa;min-height:102px;"
                    "min-width:0;word-break:keep-all;overflow:hidden'>"
                    "<div style='font-size:0.78rem;color:#555;line-height:1.25'>예보 "
                    + r['date'].strftime('%m/%d') + "</div>"
                    "<div style='font-size:0.68rem;color:#888;line-height:1.2;"
                    "white-space:nowrap'>발령일 " + td_s + "</div>"
                    "<div style='font-size:0.88rem;margin:5px 0;line-height:1.15;"
                    "white-space:nowrap'>" + em['emoji'] + "<b>" + stg + "</b></div>"
                    "<div style='font-size:0.95rem;color:" + em['color'] + ";line-height:1.1'>"
                    "<b>" + f"{pc:.0f}%" + "</b></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    warn_days = fc_out[fc_out['stage'] == '경계']['date'].tolist()
    if warn_days:
        d0, d1 = warn_days[0].strftime('%m/%d'), warn_days[-1].strftime('%m/%d')
        st.error(f"**{d0}–{d1}** 구간에 **경계** 수준 예측 — 선제 채수·모니터링 강화 권고.")
    elif (fc_out['stage'] == '관심').any():
        st.warning("일부 일자에 **관심** 수준 — 채수 빈도 점검 권고.")
    else:
        st.success("10일 구간 **미발령** 중심 — 정기 모니터링 유지.")

    # ── CHD 누적 트래커 (예보 수온 시계열 기반) ───────────────────────────────
    st.markdown("#### CHD 누적 트래커 (예보 기반)")
    chd_now = int(fc_out['CHD'].iloc[0])
    chd_max = int(fc_out['CHD'].max())
    chd_max_day = fc_out.loc[fc_out['CHD'].idxmax(), 'date'].strftime('%m/%d')
    st.markdown(
        f"현재 CHD(예보 첫날 기준): **{chd_now}일** | 예보 구간 최대 CHD: **{chd_max}일** "
        f"(피크: {chd_max_day}) — 관심 **{chd_watch}일**, 경계 **{chd_alert}일** "
        f"(`{site}` 지점 발령·비발령 CHD 분포 기반, 부족 시 7·15일)"
    )
    st.progress(min(chd_max / float(max(chd_alert, 1)), 1.0))
    h1 = chd_issuance_odds_ratio(df_t3, chd_split=chd_watch)
    if h1 is not None:
        st.caption(f"H1 요약: CHD≥{chd_watch}일일 때 (발령≠미발령) 비율이 CHD<{chd_watch}일 대비 약 **{h1:.2f}배** (실데이터 `{site}` 지점).")

    # ── 기온–위험 복합 막대 (막대 색 = 단계) ─────────────────────────────────
    st.markdown("#### 기온·강수·위험 복합 (예보 기온 막대, 색=위험 단계)")
    bar_colors = [ALERT_LEVELS[s]['color'] for s in fc_out['stage']]
    fig_b = go.Figure(
        go.Bar(
            x=fc_out['date'].dt.strftime('%m/%d'),
            y=fc_out['temp_fc'].round(1),
            marker_color=bar_colors,
            text=[f"{t:.0f}°C" for t in fc_out['temp_fc']],
            textposition='outside',
            name='예보 기온',
        )
    )
    fig_b.update_layout(
        yaxis_title="예보 기온 (°C)",
        xaxis_title="예보일 d (GBM은 각 d에 대해 d+7 발령 확률 근사)",
        height=360,
        margin=dict(t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_b, use_container_width=True)

    fig_fc = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("예보 강수량 (mm)", "일별 P(발령 on d+7) 또는 규칙점수 (%)"),
        row_heights=[0.35, 0.65],
    )
    fig_fc.add_trace(
        go.Bar(x=fc_out['date'], y=fc_out['rain'], marker_color='steelblue', name='강수'),
        row=1, col=1,
    )
    fig_fc.add_trace(
        go.Bar(
            x=fc_out['date'],
            y=(fc_out['p_use'] * 100).round(1),
            marker_color=bar_colors,
            name='위험',
        ),
        row=2, col=1,
    )
    fig_fc.add_hline(y=t_watch * 100, line_dash='dot', line_color='orange', row=2, col=1, annotation_text='관심(분위)')
    fig_fc.add_hline(y=t_alert * 100, line_dash='dash', line_color='red', row=2, col=1, annotation_text='경계(분위)')
    fig_fc.update_layout(height=440, showlegend=False, margin=dict(t=40, b=20))
    fig_fc.update_yaxes(title_text="mm", row=1, col=1)
    fig_fc.update_yaxes(title_text="점수 (%)", row=2, col=1)
    st.plotly_chart(fig_fc, use_container_width=True)

    disp = fc_out.copy()
    disp['예보일'] = disp['date'].dt.strftime('%m/%d')
    disp['발령대상일(d+7)'] = disp['target_date'].dt.strftime('%m/%d')
    disp['예보출처'] = disp['fc_source'] if 'fc_source' in disp.columns else ''
    disp['위험%'] = (disp['p_use'] * 100).round(1)
    disp['단계'] = disp['stage']
    disp['기온'] = disp['temp_fc'].round(1)
    disp['강수'] = disp['rain'].round(1)
    disp['CHD'] = disp['CHD'].astype(int)
    st.dataframe(
        disp[['예보일', '발령대상일(d+7)', '위험%', '단계', '기온', '강수', 'CHD', '예보출처']],
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: 운영 프로토콜 (실무형 레이아웃 + 실데이터)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    df_p = df_all[df_all['채수위치'] == site].sort_values('조사일')
    m = compute_site_protocol_metrics(df_p)

    last_level_raw = df_p['발령단계'].dropna() if '발령단계' in df_p.columns else pd.Series(dtype=object)
    current_level = _norm_alert_level(last_level_raw.iloc[-1]) if len(last_level_raw) else '미발령'
    cur_em = ALERT_LEVELS[current_level]

    hero_badges = (
        f'<span class="t4-badge">📍 지점 · {site}</span>'
        f'<span class="t4-badge">{cur_em["emoji"]} 현재 단계 · {current_level}</span>'
        f'<span class="t4-badge">⏱ 리드타임 · T+{pred_horizon}</span>'
    )
    st.markdown(
        f'<div class="t4-hero"><div class="t4-title">📋 운영 프로토콜</div>'
        f'<div class="t4-badges">{hero_badges}</div></div>',
        unsafe_allow_html=True,
    )
    cap_txt = (
        f"카드·B/C 수치는 **`{site}`** 지점 실측만 사용했습니다."
        if m.get('date_min') is None
        else (
            f"카드·B/C 수치는 **`{site}`** `{m['date_min']:%Y-%m-%d}` ~ `{m['date_max']:%Y-%m-%d}` "
            f"실측 **{m['n_rows']}행**에서 계산했습니다."
        )
    )
    st.caption(cap_txt + " 로드맵은 `protocol_roadmap.json`(선택) 또는 기본 계획 문구입니다.")

    st.markdown(
        '<div class="t4-chips">'
        '<span class="t4-chip">지표 KPI 상단 요약</span>'
        '<span class="t4-chip">단계별 3개 권고</span>'
        '<span class="t4-chip">타임라인 로드맵</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="t4-sec">단계별 권고 (데이터 산출)</div>', unsafe_allow_html=True)
    col_lv = st.columns(4)
    for i, lv in enumerate(PROTOCOL_LEVEL_ORDER):
        em = ALERT_LEVELS[lv]
        bg, fg = _proto_header_style(lv)
        steps = build_protocol_steps(lv, m, pred_horizon, site)
        ol = ''.join(f'<li>{s}</li>' for s in steps)
        active = ' t4-proto-active' if lv == current_level else ''
        with col_lv[i]:
            st.markdown(
                f'<div class="t4-proto{active}">'
                f'<div class="t4-proto-head" style="background:{bg};color:{fg}">{em["emoji"]} {lv}</div>'
                f'<div class="t4-proto-body"><ol class="t4-steps">{ol}</ol></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    tw, ta = None, None
    if model is not None:
        try:
            _fc = get_model_feature_cols(model)
            _hp = historical_site_probas(df_all, site, model, _fc)
            tw, ta = proba_thresholds_tertiles(_hp)
        except Exception:
            pass

    bc_col, rm_col = st.columns(2)
    with bc_col:
        st.markdown(format_tab4_bc_panel_html(m, tw, ta, pred_horizon), unsafe_allow_html=True)
    with rm_col:
        st.markdown(
            format_tab4_roadmap_html(load_protocol_roadmap())
            + "<p class='t4-muted' style='margin:10px 16px 0;font-size:0.72rem'>"
            "프로젝트 루트에 <code>protocol_roadmap.json</code> 배열로 단계·색(<code>color</code> 필드)을 "
            "바꿀 수 있습니다.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("의사결정 플로우 (참고)", expanded=False):
        st.markdown("""
```
[일일 채수 + 계수]
        │
        ▼
[위험도 / GBM T+7 확률]
        │
   ┌────┴────────────┐
   │ 분위 미만      │ 분위 이상
   ▼                 ▼
[미발령 유지]     [관심·경계 조치]
```
실제 임계는 저장 모델이 있을 때 **과거 예측확률 분위(TAB3와 동일 체계)** 로 추정합니다.
""")

# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "대청댐 유해남조류 발생 예측 및 조류경보 의사결정 지원체계 | "
    "데이터: 2016–2025 (3개 채수지점) | "
    "모델: GBM Risk Score Regression (T+7) | "
    f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d')}"
)
