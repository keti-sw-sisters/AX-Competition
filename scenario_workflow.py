"""What-if(환경·수문 사전모델), 조류 보조모델 사후 감사, 보고서 구조."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

LEAD_TIMES = ["T+1", "T+3", "T+7", "T+10"]

# 슬라이더로 조작할 수 있는 원시·운영 변수 (학습 데이터 컬럼명과 동일)
CONTROLLABLE_SLIDERS: dict[str, dict[str, Any]] = {
    "수온(℃)": {"min": 5.0, "max": 35.0, "step": 0.5, "unit": "°C"},
    "탁도": {"min": 0.0, "max": 50.0, "step": 0.5, "unit": "NTU"},
    "평균기온(°C)": {"min": -5.0, "max": 40.0, "step": 0.5, "unit": "°C"},
    "총방류량(㎥/s)": {"min": 0.0, "max": 500.0, "step": 1.0, "unit": "㎥/s"},
    "저수율(%)": {"min": 20.0, "max": 100.0, "step": 0.5, "unit": "%"},
    "강우량(mm)": {"min": 0.0, "max": 150.0, "step": 1.0, "unit": "mm"},
    "합계 일사량(MJ/m2)": {"min": 0.0, "max": 30.0, "step": 0.5, "unit": "MJ/m²"},
    "Chl-a (㎎/㎥)": {"min": 0.0, "max": 100.0, "step": 0.5, "unit": "㎎/㎥"},
}

NATURAL_VARS = [
    "CHD",
    "평균기온(°C)",
    "합계 일사량(MJ/m2)",
    "dry_days",
    "rain_pulse_flag",
    "BGI_env",
    "BGI",
]
DAM_VARS = ["총방류량(㎥/s)", "저수율(%)", "HRT", "flow_balance", "유입량(㎥/s)", "저수량(백만㎥)"]


def _app():
    import app as app_module

    return app_module


def raw_with_overrides(
    raw_df: pd.DataFrame,
    site: str,
    target_date: pd.Timestamp,
    overrides: dict[str, float],
) -> pd.DataFrame:
    """운영 원장 복사 후 해당 지점·기준일 행에만 수치 오버라이드."""
    raw = raw_df.copy()
    raw["조사일"] = pd.to_datetime(raw["조사일"])
    td = pd.Timestamp(target_date).normalize()
    m = (raw["채수위치"] == site) & (raw["조사일"].dt.normalize() == td)
    if not m.any():
        sub = raw[(raw["채수위치"] == site) & (raw["조사일"] <= td)]
        if sub.empty:
            raise ValueError(f"지점 '{site}'에 기준일 이전 데이터가 없습니다.")
        last_date = sub["조사일"].max()
        m = (raw["채수위치"] == site) & (raw["조사일"] == last_date)
    for col, val in overrides.items():
        if col not in raw.columns:
            continue
        raw.loc[m, col] = val
    return raw


def engineer_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    return _app().engineer_features(raw_df)


def get_site_row(feature_df: pd.DataFrame, site: str, target_date: pd.Timestamp) -> pd.DataFrame:
    td = pd.Timestamp(target_date).normalize()
    sub = feature_df[(feature_df["채수위치"] == site) & (feature_df["조사일"].dt.normalize() <= td)]
    if sub.empty:
        return pd.DataFrame()
    row = sub.sort_values("조사일").tail(1)
    return row


def predict_all_leads(
    row_df: pd.DataFrame,
    bundles: dict[str, dict],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lead in LEAD_TIMES:
        b = bundles[lead]
        cols = b["feature_cols"]
        X = row_df.reindex(columns=cols, fill_value=np.nan)
        prob = float(b["pipeline"].predict_proba(X)[:, 1][0])
        out[lead] = {"prob": prob, "threshold": float(b["threshold"])}
    return out


def shap_top_for_row(bundle: dict, row_df: pd.DataFrame, n: int = 5) -> list[tuple[str, float]]:
    try:
        import shap
    except ImportError:
        return []

    pipe = bundle["pipeline"]
    cols = bundle["feature_cols"]
    X = row_df.reindex(columns=cols, fill_value=np.nan)
    if len(X) == 0:
        return []
    pre = pipe[:-1]
    model = pipe.named_steps["model"]
    Xt = pre.transform(X)
    feat_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    vals = np.asarray(sv).ravel()
    order = np.argsort(-np.abs(vals))[:n]
    return [(feat_names[i], float(vals[i])) for i in order]


def find_similar_conditions(sim_row: pd.Series, final_data: pd.DataFrame) -> dict[str, Any] | None:
    fd = final_data.copy()
    fd["조사일"] = pd.to_datetime(fd["조사일"], errors="coerce")
    need = ["CHD", "저수율(%)", "수온(℃)"]
    if not all(c in fd.columns for c in need):
        return None
    for c in need:
        if c not in sim_row.index or pd.isna(sim_row.get(c)):
            return None
    chd = float(sim_row["CHD"])
    sr = float(sim_row["저수율(%)"])
    wt = float(sim_row["수온(℃)"])
    mask = (
        (fd["CHD"] >= chd - 2)
        & (fd["CHD"] <= chd + 2)
        & (fd["저수율(%)"] <= sr + 5)
        & (fd["수온(℃)"] >= wt - 2)
    )
    similar = fd.loc[mask]
    if similar.empty:
        return None
    alert_col = "발령단계" if "발령단계" in similar.columns else None
    if alert_col:
        neg = {"", "미발령", "정상", "0", "nan", "None"}
        s = similar[alert_col].astype("string").fillna("").str.strip()
        alert_rate = float((~s.isin(neg)).mean())
    else:
        alert_rate = float("nan")
    return {
        "n": int(len(similar)),
        "alert_rate": alert_rate,
        "message": f"과거 유사 조건 {len(similar)}일 · 발령(또는 경보 단계) 비율 {alert_rate:.0%}",
    }


def map_solution(
    results: dict[str, dict[str, float]],
    top_features: list[tuple[str, float]],
    sim_row: pd.Series,
    final_data: pd.DataFrame,
) -> dict[str, Any]:
    probs = [results[lt]["prob"] for lt in LEAD_TIMES]
    max_prob = max(probs) if probs else 0.0
    top_names = [t[0] for t in top_features]
    top_controllable = [
        f
        for f in top_names
        if any(k in f for k in CONTROLLABLE_SLIDERS) or any(d in f for d in DAM_VARS)
    ]
    top_natural = [f for f in top_names if any(c in f for c in NATURAL_VARS)]
    similar_past = find_similar_conditions(sim_row, final_data)

    if max_prob < 0.3:
        return {
            "level": "안전",
            "message": "시나리오 조건에서 사전 위험 확률이 낮은 편입니다. 정기 모니터링을 유지하세요.",
            "actions": [],
            "past_reference": similar_past,
        }
    if max_prob < 0.6:
        actions: list[dict[str, Any]] = []
        if top_controllable:
            actions.append(
                {
                    "type": "댐·수문 운영",
                    "message": "방류·저수·유입 균형 등 조작 가능 변수가 SHAP 상위에 있습니다. 단계적 방류·취수 조정 시나리오를 검토하세요.",
                    "detail": top_controllable[:5],
                }
            )
        if top_natural:
            actions.append(
                {
                    "type": "모니터링 강화",
                    "message": f"기상·수문 패턴 관련 변수: {', '.join(top_natural[:5])}",
                    "lead_time": "T+3 / T+7 구간 집중 관찰",
                }
            )
        return {
            "level": "주의",
            "actions": actions,
            "past_reference": similar_past,
        }
    return {
        "level": "고위험",
        "actions": [
            {
                "type": "즉시 댐·수문 검토",
                "message": "사전 모델 기준 위험 확률이 높습니다. 방류·저수위 운영과 현장 조류 채수 일정을 긴급 조정하세요.",
            },
            {
                "type": "조류 모니터링",
                "message": "total_cyano 등 측정값 확보 후 보조 모델(모델 1)로 재판단하세요.",
            },
        ],
        "past_reference": similar_past,
    }


def run_whatif_scenario(
    raw_df: pd.DataFrame,
    site: str,
    target_date: pd.Timestamp,
    overrides: dict[str, float],
    env_bundles: dict[str, dict],
    final_data: pd.DataFrame,
    ref_lead_for_shap: str = "T+7",
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, float]], dict[str, Any]]:
    raw_base = raw_df.copy()
    raw_sim = raw_with_overrides(raw_df, site, target_date, overrides)
    feat_base = engineer_from_raw(raw_base)
    feat_sim = engineer_from_raw(raw_sim)
    base_row = get_site_row(feat_base, site, target_date)
    sim_row = get_site_row(feat_sim, site, target_date)
    if base_row.empty or sim_row.empty:
        raise ValueError("기준일에 대한 피처 행을 만들 수 없습니다.")

    results: dict[str, dict[str, Any]] = {}
    for lt in LEAD_TIMES:
        b = env_bundles[lt]
        cols = b["feature_cols"]
        pb = float(b["pipeline"].predict_proba(base_row.reindex(columns=cols, fill_value=np.nan))[:, 1][0])
        ps = float(b["pipeline"].predict_proba(sim_row.reindex(columns=cols, fill_value=np.nan))[:, 1][0])
        th = float(b["threshold"])
        results[lt] = {
            "prob": ps,
            "base_prob": pb,
            "delta": ps - pb,
            "threshold": th,
            "pred_sim": "발령 위험" if ps >= th else "미발령/관심",
            "pred_base": "발령 위험" if pb >= th else "미발령/관심",
        }

    bundle = env_bundles.get(ref_lead_for_shap) or env_bundles["T+7"]
    top_features = shap_top_for_row(bundle, sim_row, n=5)
    solution = map_solution(results, top_features, sim_row.iloc[0], final_data)
    return results, top_features, solution


def run_post_audit_cyano(
    raw_df: pd.DataFrame,
    site: str,
    target_date: pd.Timestamp,
    total_cyano_new: float,
    env_bundles: dict[str, dict],
    monitoring_bundles: dict[str, dict],
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """운영 원장에서 해당 일·지점 total_cyano만 갱신 후 피처 재계산 → 보조모델 예측. 사전모델은 갱신 전 행 기준."""
    raw_adj = raw_df.copy()
    raw_adj["조사일"] = pd.to_datetime(raw_adj["조사일"])
    td = pd.Timestamp(target_date).normalize()
    m = (raw_adj["채수위치"] == site) & (raw_adj["조사일"].dt.normalize() == td)
    if not m.any():
        sub = raw_adj[(raw_adj["채수위치"] == site) & (raw_adj["조사일"] <= td)]
        if sub.empty:
            raise ValueError(f"지점 '{site}'에 기준일 이전 데이터가 없습니다.")
        last_date = sub["조사일"].max()
        m = (raw_adj["채수위치"] == site) & (raw_adj["조사일"] == last_date)
    if "total_cyano" not in raw_adj.columns:
        raw_adj["total_cyano"] = np.nan
    raw_adj.loc[m, "total_cyano"] = float(total_cyano_new)

    feat_adj = engineer_from_raw(raw_adj)
    audit_row = get_site_row(feat_adj, site, target_date)
    feat_base = engineer_from_raw(raw_df)
    base_row = get_site_row(feat_base, site, target_date)
    if audit_row.empty or base_row.empty:
        raise ValueError("사후 감사용 피처 행을 구성할 수 없습니다.")

    m2 = predict_all_leads(base_row, env_bundles)
    model2_for_report = {lt: {"prob": m2[lt]["prob"], "threshold": m2[lt]["threshold"]} for lt in LEAD_TIMES}

    m1 = predict_all_leads(audit_row, monitoring_bundles)
    model1_results = {lt: m1[lt]["prob"] for lt in LEAD_TIMES}
    thresholds = {lt: m1[lt]["threshold"] for lt in LEAD_TIMES}

    report = generate_audit_report(model2_for_report, model1_results, thresholds, audit_row.iloc[0])
    return model1_results, report, base_row, audit_row


def generate_audit_report(
    model2_results: dict[str, dict[str, Any]],
    model1_results: dict[str, float],
    thresholds: dict[str, float],
    row: pd.Series,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "기준일": str(row.get("조사일", ""))[:10],
        "채수위치": row.get("채수위치", ""),
        "모델2_사전예측": {},
        "모델1_사후확인": {},
        "일치여부": {},
        "해석": {},
    }
    for lt in LEAD_TIMES:
        th2 = float(model2_results[lt].get("threshold", thresholds[lt]))
        m2 = float(model2_results[lt]["prob"])
        m1 = float(model1_results[lt])
        th1 = thresholds[lt]
        report["모델2_사전예측"][lt] = {
            "확률": round(m2, 4),
            "판정": "발령위험" if m2 >= th2 else "미발령",
            "threshold": th2,
        }
        report["모델1_사후확인"][lt] = {
            "확률": round(m1, 4),
            "판정": "발령위험" if m1 >= th1 else "미발령",
            "threshold": th1,
        }
        gap = m1 - m2
        if abs(gap) <= 0.15:
            verdict = "일치 — 사전 예측과 보조 모델이 유사합니다."
        elif gap > 0.15:
            verdict = "보조 모델이 더 높음 — 조류 신호 반영, 경보·채수 검토를 권장합니다."
        else:
            verdict = "사전 모델이 상대적으로 높음 — 환경·수문만으로는 과대 신호 가능, 현장 확인이 필요합니다."
        report["일치여부"][lt] = verdict
        report["해석"][lt] = f"Δ(모델1−모델2) = {gap:+.3f}"
    return report


def report_to_json_bytes(report: dict[str, Any]) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8-sig")
