"""
대청호 조류경보 What-if 시뮬레이션 엔진.

학습과 동일한 피처는 `engineer_features`(scenario_workflow.engineer_from_raw)로 재계산하고,
단일 행 보정·설명용으로 `recalculate_derived()`를 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import warnings

from scenario_workflow import (
    LEAD_TIMES,
    engineer_from_raw,
    get_site_row,
    raw_with_overrides,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL1_DIR = BASE_DIR / "outputs" / "modeling"
MODEL2_DIR = BASE_DIR / "outputs" / "modeling_env"

CONTROLLABLE_VARS = {
    "총방류량(㎥/s)",
    "저수율(%)",
    "저수량(백만㎥)",
    "유입량(㎥/s)",
    "HRT",
    "HRT_7d",
    "flow_balance",
    "flow_balance_7d",
    "취수량1",
    "취수량2",
    "취수량3",
}
NATURAL_VARS = {
    "CHD",
    "평균기온(°C)",
    "최고기온(°C)",
    "최저기온(°C)",
    "합계 일사량(MJ/m2)",
    "solar_mean_3d",
    "solar_mean_7d",
    "dry_days",
    "rain_pulse_flag",
    "rain_sum_3d",
    "rain_sum_7d",
    "rain_sum_14d",
    "수온(℃)",
    "water_temp_mean_3d",
    "water_temp_mean_7d",
    "BGI",
    "BGI_env",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "dayofyear",
}

SLIDER_CONFIG: dict[str, dict[str, Any]] = {
    "수온(℃)": {"min": 0.0, "max": 35.0, "step": 0.5, "unit": "°C", "group": "수질"},
    "탁도": {"min": 0.0, "max": 100.0, "step": 1.0, "unit": "NTU", "group": "수질"},
    "Chl-a (㎎/㎥)": {"min": 0.0, "max": 200.0, "step": 1.0, "unit": "㎎/㎥", "group": "수질"},
    "DO(㎎/L)": {"min": 0.0, "max": 20.0, "step": 0.1, "unit": "㎎/L", "group": "수질"},
    "pH": {"min": 5.0, "max": 10.0, "step": 0.1, "unit": "", "group": "수질"},
    "평균기온(°C)": {"min": -10.0, "max": 40.0, "step": 0.5, "unit": "°C", "group": "기상"},
    "최고기온(°C)": {"min": -5.0, "max": 45.0, "step": 0.5, "unit": "°C", "group": "기상"},
    "합계 일사량(MJ/m2)": {"min": 0.0, "max": 30.0, "step": 0.5, "unit": "MJ/m²", "group": "기상"},
    "강우량(mm)": {"min": 0.0, "max": 200.0, "step": 1.0, "unit": "mm", "group": "기상"},
    "총방류량(㎥/s)": {"min": 0.0, "max": 1000.0, "step": 5.0, "unit": "㎥/s", "group": "댐운영"},
    "저수율(%)": {"min": 10.0, "max": 100.0, "step": 1.0, "unit": "%", "group": "댐운영"},
    "유입량(㎥/s)": {"min": 0.0, "max": 2000.0, "step": 10.0, "unit": "㎥/s", "group": "댐운영"},
}

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "🌡️ 폭염 2주 지속": {
        "label": "🌡️ 폭염\n2주",
        "desc": "평균기온 +3°C, 최고기온 +4°C, 일사량 증가, 강우 없음",
        "overrides": {
            "평균기온(°C)": lambda v: v + 3.0,
            "최고기온(°C)": lambda v: v + 4.0,
            "합계 일사량(MJ/m2)": lambda v: min(v * 1.3, 28.0),
            "강우량(mm)": lambda _: 0.0,
        },
    },
    "🌧️ 장마 후 고온 (D+14)": {
        "label": "🌧️ 장마 후\nD+14 고온",
        "desc": "강우 이벤트(펄스) 이후 건조·고온 복합 — H2 가설(D+7~14 위험) 실데이터 검증",
        "overrides": {
            "강우량(mm)": lambda _: 0.0,
            "평균기온(°C)": lambda v: v + 2.0,
            "수온(℃)": lambda v: v + 1.5,
        },
        "force_derived": {"dry_days": 14, "rain_pulse_flag": 1},
    },
    "🏜️ 가뭄 심화 + 저수율 하락": {
        "label": "🏜️ 가뭄\n저수율↓",
        "desc": "저수율 하락, 방류 감소",
        "overrides": {
            "강우량(mm)": lambda _: 0.0,
            "저수율(%)": lambda v: max(v - 15.0, 10.0),
            "총방류량(㎥/s)": lambda v: max(v - 30.0, 0.0),
        },
        "force_derived": {"dry_days": 15},
    },
    "💧 방류량 +30% 증량": {
        "label": "💧 방류\n+30%",
        "desc": "댐 방류량 30% 증가",
        "overrides": {"총방류량(㎥/s)": lambda v: v * 1.30},
    },
    "☀️ 올해 같은 여름(고온·무강우·방류↑)": {
        "label": "☀️ 여름형\n고온·무강우\n방류↑",
        "desc": "여름형: 고온·무강우(0mm)·방류 25%↑ + 건조일 14일 — 사전점검용 시나리오",
        "overrides": {
            "평균기온(°C)": lambda v: v + 3.0,
            "최고기온(°C)": lambda v: v + 4.0,
            "강우량(mm)": lambda _: 0.0,
            "합계 일사량(MJ/m2)": lambda v: min(v * 1.25, 28.0),
            "총방류량(㎥/s)": lambda v: v * 1.25,
        },
        "force_derived": {"dry_days": 14},
    },
    "⚠️ 복합 고위험": {
        "label": "⚠️ 복합\n고위험",
        "desc": "고온 + 가뭄 + 저수율 하락",
        "overrides": {
            "평균기온(°C)": lambda v: v + 4.0,
            "최고기온(°C)": lambda v: v + 5.0,
            "수온(℃)": lambda v: v + 3.0,
            "강우량(mm)": lambda _: 0.0,
            "저수율(%)": lambda v: max(v - 20.0, 10.0),
            "합계 일사량(MJ/m2)": lambda v: min(v * 1.4, 28.0),
        },
        "force_derived": {"dry_days": 20},
    },
}


def _load_bundle(model_dir: Path, lead: str) -> dict:
    lead_key = lead.replace("+", "plus")
    models_dir = model_dir / "models"
    best_csv = model_dir / "tables" / "best_model_summary.csv"
    if best_csv.exists():
        best = pd.read_csv(best_csv, encoding="utf-8-sig")
        row = best[best["lead_time"] == lead]
        if not row.empty:
            model_file = str(row.iloc[0]["model_file"]).replace("\\", "/")
            candidates = [
                BASE_DIR / model_file,
                Path(model_file),
                model_dir / Path(model_file).name,
                models_dir / Path(model_file).name,
            ]
            for p in candidates:
                if p.exists():
                    import joblib

                    return joblib.load(p)
    if models_dir.exists():
        import joblib

        for f in sorted(models_dir.glob(f"*{lead_key}*.pkl")):
            if "candidate_" not in f.name:
                return joblib.load(f)
    raise FileNotFoundError(f"모델 없음: lead={lead}, dir={model_dir}")


def load_all_bundles(model_dir: Path) -> dict[str, dict]:
    bundles: dict[str, dict] = {}
    errors: list[str] = []
    for lt in LEAD_TIMES:
        try:
            bundles[lt] = _load_bundle(model_dir, lt)
        except Exception as e:
            errors.append(f"{lt}: {e}")
    if errors:
        raise FileNotFoundError("모델 로드 실패:\n" + "\n".join(errors))
    return bundles


def _predict_row(bundle: dict, row: pd.Series) -> float:
    fc = bundle["feature_cols"]
    x = pd.DataFrame([row]).reindex(columns=fc, fill_value=np.nan)
    # sklearn(SimpleImputer)의 "Skipping features without any observed values" 경고는
    # 학습 시점에 해당 컬럼이 전부 결측이었던 경우 발생합니다(입력값을 임의로 바꾸지 않음).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Skipping features without any observed values.*",
            category=UserWarning,
        )
        return float(bundle["pipeline"].predict_proba(x)[:, 1][0])


def recalculate_derived(
    sim_row: pd.Series,
    history: pd.DataFrame,
    force_derived: dict[str, Any] | None = None,
) -> pd.Series:
    """슬라이더 조작 후 주요 파생값만 빠르게 재계산 (전체 engineer와 불일치 가능)."""
    row = sim_row.copy()
    force = force_derived or {}
    rain_col = "강우량(mm)" if "강우량(mm)" in row.index else "일강수량(mm)"
    rain = float(row.get(rain_col, 0) or 0)
    water_temp = float(row.get("수온(℃)", np.nan) or np.nan)
    solar = float(row.get("합계 일사량(MJ/m2)", np.nan) or np.nan)
    inflow = float(row.get("유입량(㎥/s)", np.nan) or np.nan)
    outflow = float(row.get("총방류량(㎥/s)", np.nan) or np.nan)
    volume = float(row.get("저수량(백만㎥)", np.nan) or np.nan)

    if "CHD" in force:
        row["CHD"] = force["CHD"]
    elif not history.empty and "수온(℃)" in history.columns and not np.isnan(water_temp):
        temps = list(history["수온(℃)"].fillna(0).values) + [water_temp]
        cnt = 0
        for t in temps:
            cnt = cnt + 1 if t > 25 else 0
        row["CHD"] = cnt
    elif not np.isnan(water_temp):
        row["CHD"] = 1 if water_temp > 25 else 0

    if "dry_days" in force:
        row["dry_days"] = force["dry_days"]
    elif not history.empty and rain_col in history.columns:
        rains = list(history[rain_col].fillna(0).values) + [rain]
        cnt = 0
        for r in rains:
            cnt = cnt + 1 if r <= 1 else 0
        row["dry_days"] = cnt
    else:
        row["dry_days"] = 0 if rain > 1 else float(row.get("dry_days", 0) or 0) + 1

    if "rain_pulse_flag" in force:
        row["rain_pulse_flag"] = force["rain_pulse_flag"]
    else:
        row["rain_pulse_flag"] = int(float(row.get("dry_days", 0) or 0) >= 5 and rain >= 10)

    if not np.isnan(solar):
        if not history.empty and "합계 일사량(MJ/m2)" in history.columns:
            sh = history["합계 일사량(MJ/m2)"].dropna()
            row["solar_mean_3d"] = float(pd.concat([sh.tail(2), pd.Series([solar])]).mean())
            row["solar_mean_7d"] = float(pd.concat([sh.tail(6), pd.Series([solar])]).mean())
        else:
            row["solar_mean_3d"] = solar
            row["solar_mean_7d"] = solar

    if not np.isnan(inflow) and not np.isnan(volume) and inflow > 0:
        row["HRT"] = volume * 1e6 / (inflow * 86400)
    if not history.empty and "HRT" in history.columns and "HRT" in row.index:
        hh = history["HRT"].dropna()
        hrt_now = row.get("HRT", np.nan)
        row["HRT_7d"] = float(pd.concat([hh.tail(6), pd.Series([hrt_now])]).mean())

    if not np.isnan(inflow) and not np.isnan(outflow):
        row["flow_balance"] = inflow - outflow

    chd = float(row.get("CHD", 0) or 0)
    sol7 = float(row.get("solar_mean_7d", np.nan) or np.nan)
    hrt7 = float(row.get("HRT_7d", np.nan) or np.nan)
    if not np.isnan(sol7) and not np.isnan(hrt7) and hrt7 > 0:
        row["BGI"] = chd * sol7 / hrt7
        row["BGI_env"] = row["BGI"]
    return row


def run_whatif(
    base_row: pd.Series,
    overrides: dict[str, float],
    bundles2: dict[str, dict],
    history: pd.DataFrame,
    force_derived: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    기준 행(엔지니어링 완료) + 슬라이더 오버라이드 → `recalculate_derived` 후 모델2 재예측.
    """
    sim_row = base_row.copy()
    for k, v in overrides.items():
        if k in sim_row.index:
            sim_row[k] = v
    sim_row = recalculate_derived(sim_row, history, force_derived)

    results: dict[str, Any] = {"base": {}, "sim": {}, "delta": {}, "threshold": {}}
    for lt in LEAD_TIMES:
        b = bundles2[lt]
        thr = float(b["threshold"])
        pb = _predict_row(b, base_row)
        ps = _predict_row(b, sim_row)
        results["base"][lt] = pb
        results["sim"][lt] = ps
        results["delta"][lt] = ps - pb
        results["threshold"][lt] = thr
    results["sim_row"] = sim_row
    results["base_row"] = base_row
    return results


def run_whatif_engineer_raw(
    raw_df: pd.DataFrame,
    site: str,
    target_date: pd.Timestamp,
    overrides: dict[str, float],
    bundles2: dict[str, dict],
    force_derived: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """운영 원장 전체를 다시 엔지니어링할 때 (정확도 우선)."""
    raw_sim = raw_with_overrides(raw_df, site, target_date, overrides)
    feat_base = engineer_from_raw(raw_df)
    feat_sim = engineer_from_raw(raw_sim)
    base_df = get_site_row(feat_base, site, target_date)
    sim_df = get_site_row(feat_sim, site, target_date)
    if base_df.empty or sim_df.empty:
        raise ValueError("기준일에 대한 피처 행을 만들 수 없습니다.")
    base_row = base_df.iloc[0]
    sim_row = sim_df.iloc[0].copy()
    if force_derived:
        hist = feat_sim[(feat_sim["채수위치"] == site) & (feat_sim["조사일"] < sim_row["조사일"])].sort_values(
            "조사일"
        )
        sim_row = recalculate_derived(sim_row, hist, force_derived)
    results: dict[str, Any] = {"base": {}, "sim": {}, "delta": {}, "threshold": {}}
    for lt in LEAD_TIMES:
        b = bundles2[lt]
        thr = float(b["threshold"])
        pb = _predict_row(b, base_row)
        ps = _predict_row(b, sim_row)
        results["base"][lt] = pb
        results["sim"][lt] = ps
        results["delta"][lt] = ps - pb
        results["threshold"][lt] = thr
    results["sim_row"] = sim_row
    results["base_row"] = base_row
    return results


def simulate_discharge(
    base_row: pd.Series,
    bundles2: dict[str, dict],
    history: pd.DataFrame,
    discharge_ratios: list[float] | None = None,
    lead_focus: str = "T+7",
) -> list[dict[str, Any]]:
    if discharge_ratios is None:
        discharge_ratios = [0.10, 0.30, 0.50]
    current_discharge = float(base_row.get("총방류량(㎥/s)", 0) or 0)
    scenarios: list[dict[str, Any]] = []
    for ratio in discharge_ratios:
        new_d = current_discharge * (1 + ratio)
        res = run_whatif(base_row, {"총방류량(㎥/s)": new_d}, bundles2, history)
        scenarios.append(
            {
                "label": f"+{int(ratio * 100)}%",
                "방류량(㎥/s)": round(new_d, 1),
                "위험확률": round(res["sim"][lead_focus], 4),
                "기준확률": round(res["base"][lead_focus], 4),
                "확률변화(Δ)": round(res["delta"][lead_focus], 4),
            }
        )
    return scenarios


def find_min_discharge_to_reduce(
    base_row: pd.Series,
    bundles2: dict[str, dict],
    history: pd.DataFrame,
    target_prob: float = 0.50,
    lead_focus: str = "T+7",
    max_ratio: float = 2.0,
    steps: int = 20,
) -> dict[str, Any]:
    current = float(base_row.get("총방류량(㎥/s)", 0) or 0)
    for ratio in np.linspace(0, max_ratio, steps + 1):
        new_d = current * (1 + float(ratio))
        res = run_whatif(base_row, {"총방류량(㎥/s)": new_d}, bundles2, history)
        if res["sim"][lead_focus] < target_prob:
            return {
                "found": True,
                "ratio": float(ratio),
                "방류량(㎥/s)": round(new_d, 1),
                "달성확률": round(res["sim"][lead_focus], 4),
                "target_prob": target_prob,
            }
    return {
        "found": False,
        "방류량(㎥/s)": round(current * (1 + max_ratio), 1),
        "달성확률": None,
        "target_prob": target_prob,
    }


def find_similar_past(
    sim_row: pd.Series,
    final_data: pd.DataFrame,
    tol: dict[str, float] | None = None,
) -> dict[str, Any]:
    tol = tol or {
        "수온(℃)": 2.5,
        "저수율(%)": 8.0,
        "평균기온(°C)": 3.0,
        "강우량(mm)": 10.0,
    }
    fd = final_data.copy()
    fd["조사일"] = pd.to_datetime(fd["조사일"], errors="coerce")
    if "발령단계" not in fd.columns:
        return {"n": 0, "alert_rate": None, "years": [], "message": "발령단계 컬럼 없음", "similar_df": pd.DataFrame()}
    neg = {"", "미발령", "정상", "0", "nan", "none"}
    fd["alert_binary"] = (
        fd["발령단계"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x in neg else 1)
    )
    mask = pd.Series(True, index=fd.index)
    for col, tol_val in tol.items():
        if col in fd.columns and col in sim_row.index and pd.notna(sim_row[col]):
            mask &= (fd[col] - float(sim_row[col])).abs() <= tol_val
    similar = fd.loc[mask].copy()
    if similar.empty:
        return {
            "n": 0,
            "alert_rate": None,
            "years": [],
            "message": "과거 유사 조건 없음 (허용 범위를 넓혀 보세요)",
            "similar_df": pd.DataFrame(),
        }
    alert_rate = float(similar["alert_binary"].mean())
    years = sorted(similar["조사일"].dt.year.dropna().unique().tolist())
    n = len(similar)
    display_cols = [c for c in ["조사일", "채수위치", "발령단계", "alert_binary", "수온(℃)", "CHD", "저수율(%)"] if c in similar.columns]
    return {
        "n": n,
        "alert_rate": alert_rate,
        "years": years,
        "message": f"과거 유사 조건 {n}일 중 발령 비율 {alert_rate:.0%} (연도: {', '.join(map(str, years))})",
        "similar_df": similar[display_cols],
    }


def get_shap_top(bundle: dict, sim_row: pd.Series, n: int = 5) -> list[dict[str, Any]]:
    try:
        import shap
    except ImportError:
        return _shap_fallback_importance(bundle, sim_row, n)

    fc = bundle["feature_cols"]
    x = pd.DataFrame([sim_row]).reindex(columns=fc, fill_value=np.nan)
    pipe = bundle["pipeline"]
    try:
        pre = pipe[:-1]
        model = pipe.named_steps["model"]
        xt = pre.transform(x)
        fn = list(pre.get_feature_names_out())
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(xt)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        arr = np.asarray(sv).ravel()
        order = np.argsort(-np.abs(arr))[:n]
        out = []
        for i in order:
            fname = fn[i] if i < len(fn) else str(i)
            out.append(
                {
                    "feature": fname,
                    "shap_value": float(arr[i]),
                    "feature_value": float(xt[0, i]) if hasattr(xt, "shape") and len(np.shape(xt)) > 1 else np.nan,
                    "controllable": any(c in fname for c in CONTROLLABLE_VARS),
                }
            )
        return out
    except Exception:
        return _shap_fallback_importance(bundle, sim_row, n)


def _shap_fallback_importance(bundle: dict, sim_row: pd.Series, n: int) -> list[dict[str, Any]]:
    try:
        model = bundle["pipeline"].named_steps["model"]
        fi = getattr(model, "feature_importances_", None)
        if fi is None:
            return []
        fc = bundle["feature_cols"]
        pairs = sorted(zip(fc, fi), key=lambda x: x[1], reverse=True)[:n]
        return [
            {
                "feature": f,
                "shap_value": float(v),
                "feature_value": np.nan,
                "controllable": any(c in f for c in CONTROLLABLE_VARS),
            }
            for f, v in pairs
        ]
    except Exception:
        return []


def map_solution(
    whatif_result: dict[str, Any],
    shap_top: list[dict[str, Any]],
    sim_row: pd.Series,
    bundles2: dict[str, dict],
    history: pd.DataFrame,
    past_info: dict[str, Any],
) -> dict[str, Any]:
    max_lt = max(whatif_result["sim"], key=lambda k: whatif_result["sim"][k])
    max_prob = whatif_result["sim"][max_lt]
    threshold_max = whatif_result["threshold"].get(max_lt, 0.5)
    cause_vars = [s["feature"] for s in shap_top]
    controllable = [s["feature"] for s in shap_top if s.get("controllable")]
    natural = [s["feature"] for s in shap_top if not s.get("controllable")]
    past_msg = past_info.get("message", "") or past_info.get("narrative", "")

    base_row = whatif_result.get("base_row", sim_row)
    discharge_scenarios: list[dict[str, Any]] = []
    try:
        discharge_scenarios = simulate_discharge(base_row, bundles2, history, lead_focus=max_lt)
    except Exception:
        discharge_scenarios = []

    if max_prob < 0.30:
        return {
            "level": "안전",
            "color": "#2E86AB",
            "summary": "시뮬레이션 조건에서 사전 위험 확률이 낮습니다.",
            "cause_vars": cause_vars,
            "controllable": controllable,
            "natural": natural,
            "actions": [{"type": "모니터링 유지", "detail": "정기 채수·현장 관찰 지속"}],
            "discharge_scenarios": [],
            "past_reference": past_msg,
        }

    if max_prob < 0.60:
        actions: list[dict[str, Any]] = []
        if controllable:
            actions.append(
                {
                    "type": "댐 방류량 조정",
                    "detail": f"{max_lt} 기준 방류 증량 시나리오를 검토하세요.",
                }
            )
        if natural:
            actions.append(
                {
                    "type": "집중 모니터링",
                    "detail": f"자연 요인: {', '.join(natural[:3])}",
                }
            )
        return {
            "level": "주의",
            "color": "#F2A541",
            "summary": f"{max_lt} 확률 {max_prob:.0%} (threshold 약 {threshold_max:.0%}). 원인: {', '.join(cause_vars[:3])}",
            "cause_vars": cause_vars,
            "controllable": controllable,
            "natural": natural,
            "actions": actions,
            "discharge_scenarios": discharge_scenarios,
            "past_reference": past_msg,
        }

    try:
        min_d = find_min_discharge_to_reduce(base_row, bundles2, history, lead_focus=max_lt)
    except Exception:
        min_d = {"found": False}

    actions = []
    if min_d.get("found"):
        actions.append(
            {
                "type": "즉시 방류량 증량",
                "detail": f"최소 약 {min_d.get('방류량(㎥/s)', '?')} ㎥/s 검토 (달성 확률 {min_d.get('달성확률')})",
            }
        )
    else:
        actions.append({"type": "복합 대응", "detail": "방류만으로 완화가 어려울 수 있습니다."})
    actions.append(
        {
            "type": "조류 모니터링",
            "detail": "total_cyano 확보 후 보조 모델로 재판단하세요.",
        }
    )
    return {
        "level": "고위험",
        "color": "#D64545",
        "summary": f"{max_lt} 확률 {max_prob:.0%} — 즉각 대응 검토. 원인: {', '.join(cause_vars[:3])}",
        "cause_vars": cause_vars,
        "controllable": controllable,
        "natural": natural,
        "actions": actions,
        "discharge_scenarios": discharge_scenarios,
        "past_reference": past_msg,
    }


def run_model1_audit(
    base_row: pd.Series,
    log_cyano_raw: float,
    hoenam_cyano_raw: float,
    bundles1: dict[str, dict],
    history: pd.DataFrame,
) -> dict[str, float]:
    """조류 세포수(실측) 반영 후 보조 모델 확률. `log_cyano_raw`는 total_cyano(세포수)로 해석."""
    audit = build_model1_audit_row(base_row, log_cyano_raw, hoenam_cyano_raw, history)

    results: dict[str, float] = {}
    for lt in LEAD_TIMES:
        results[lt] = _predict_row(bundles1[lt], audit)
    return results


def build_model1_audit_row(
    base_row: pd.Series,
    log_cyano_raw: float,
    hoenam_cyano_raw: float,
    history: pd.DataFrame,
) -> pd.Series:
    """
    모델1(조류 포함) 예측에 넣을 단일 행을 생성.
    - log_cyano 계열: '현재 선택 지점' 측정값(=log_cyano_raw)
    - hoenam_* 계열: 회남 선행 신호(=hoenam_cyano_raw)
    """
    audit = base_row.copy()
    tc = max(float(log_cyano_raw), 0.0)
    audit["log_cyano"] = float(np.log1p(tc))
    if "total_cyano" in audit.index:
        audit["total_cyano"] = tc

    hist_lc = (
        history["log_cyano"].dropna()
        if not history.empty and "log_cyano" in history.columns
        else pd.Series(dtype=float)
    )
    for lag in [1, 3, 7, 10, 14, 30]:
        if len(hist_lc) >= lag:
            audit[f"log_cyano_lag{lag}"] = float(hist_lc.iloc[-lag])
        else:
            audit[f"log_cyano_lag{lag}"] = float(audit["log_cyano"])

    seq = pd.concat([hist_lc, pd.Series([audit["log_cyano"]])], ignore_index=True).dropna()
    for w in [7, 14, 30]:
        audit[f"log_cyano_roll{w}"] = float(seq.tail(w).mean()) if len(seq) else float(audit["log_cyano"])
    audit["log_cyano_roll7_max"] = float(seq.tail(7).max()) if len(seq) else float(audit["log_cyano"])

    chla_col = "Chl-a (㎎/㎥)"
    if not history.empty and chla_col in history.columns:
        ch = history[chla_col].dropna()
        for lag in [1, 3, 7, 10, 14, 30]:
            if len(ch) >= lag:
                audit[f"chla_lag{lag}"] = float(ch.iloc[-lag])
            else:
                v = base_row.get(chla_col, np.nan)
                audit[f"chla_lag{lag}"] = float(v) if pd.notna(v) else np.nan
        for w in [7, 14]:
            audit[f"chla_roll{w}"] = float(ch.tail(w).mean()) if len(ch) else float(base_row.get(chla_col, np.nan) or np.nan)

    h_lc = float(np.log1p(max(float(hoenam_cyano_raw), 0.0)))
    for lag in [7, 10, 14]:
        audit[f"hoenam_log_cyano_lag{lag}"] = h_lc
    lag7 = float(audit.get("log_cyano_lag7", audit["log_cyano"]))
    audit["hoenam_to_site_log_cyano_diff_lag7"] = h_lc - lag7
    return audit


def generate_audit_report(
    model2_whatif: dict[str, Any],
    model1_probs: dict[str, float],
    base_row: pd.Series,
    bundles1: dict[str, dict],
    bundles2: dict[str, dict],
    site: str,
    target_date: pd.Timestamp,
) -> dict[str, Any]:
    del base_row, bundles2
    report: dict[str, Any] = {
        "기준일": str(target_date.date()),
        "채수위치": site,
        "모델2_사전예측": {},
        "모델1_사후확인": {},
        "일치여부": {},
        "해석": {},
        "종합판정": "",
    }
    for lt in LEAD_TIMES:
        m2_prob = float(model2_whatif["sim"][lt])
        m1_prob = float(model1_probs[lt])
        thr2 = float(model2_whatif["threshold"][lt])
        thr1 = float(bundles1[lt]["threshold"])
        m2_label = "발령위험" if m2_prob >= thr2 else "미발령"
        m1_label = "발령위험" if m1_prob >= thr1 else "미발령"
        gap = m1_prob - m2_prob
        if abs(gap) <= 0.15:
            match, interp = "✅ 일치", "사전 예측 신뢰도가 높은 편입니다."
        elif gap > 0.15:
            match, interp = "⚠️ 모델1 > 모델2", "조류 신호가 환경 예측보다 위험을 키울 수 있습니다."
        else:
            match, interp = "ℹ️ 모델2 > 모델1", "환경 조건만 보면 상대적으로 보수적일 수 있습니다."
        report["모델2_사전예측"][lt] = {"확률": round(m2_prob, 4), "판정": m2_label, "threshold": thr2}
        report["모델1_사후확인"][lt] = {"확률": round(m1_prob, 4), "판정": m1_label, "threshold": thr1}
        report["일치여부"][lt] = match
        report["해석"][lt] = interp
    high_lt = [lt for lt in LEAD_TIMES if report["모델1_사후확인"][lt]["판정"] == "발령위험"]
    report["종합판정"] = (
        f"🔴 모델 1 기준 고위험 리드타임: {', '.join(high_lt)} — 즉시 현장 조치 검토"
        if high_lt
        else "🟢 모델 1 기준 전 리드타임 미발령 — 환경·현장 감시 지속"
    )
    return report


def generate_model1_report(
    model1_probs: dict[str, float],
    base_row: pd.Series,
    bundles1: dict[str, dict],
    site: str,
    target_date: pd.Timestamp,
) -> dict[str, Any]:
    """모델 1 단독 사후 감사 보고서."""
    del base_row
    report: dict[str, Any] = {
        "기준일": str(target_date.date()),
        "채수위치": site,
        "모델1_사후확인": {},
        "종합판정": "",
    }
    for lt in LEAD_TIMES:
        m1_prob = float(model1_probs[lt])
        thr1 = float(bundles1[lt]["threshold"])
        m1_label = "발령위험" if m1_prob >= thr1 else "미발령"
        report["모델1_사후확인"][lt] = {"확률": round(m1_prob, 4), "판정": m1_label, "threshold": thr1}

    high_lt = [lt for lt in LEAD_TIMES if report["모델1_사후확인"][lt]["판정"] == "발령위험"]
    report["종합판정"] = (
        f"🔴 모델 1 기준 고위험 리드타임: {', '.join(high_lt)} — 즉시 현장 조치 검토"
        if high_lt
        else "🟢 모델 1 기준 전 리드타임 미발령 — 현장 감시 지속"
    )
    return report


def export_report_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for lt in LEAD_TIMES:
        m1 = report.get("모델1_사후확인", {}).get(lt, {})
        m2 = report.get("모델2_사전예측", {}).get(lt, {})
        if "모델2_사전예측" in report:
            rows.append(
                {
                    "리드타임": lt,
                    "모델2 확률": m2.get("확률", 0),
                    "모델2 판정": m2.get("판정", ""),
                    "모델1 확률": m1.get("확률", 0),
                    "모델1 판정": m1.get("판정", ""),
                    "일치여부": report.get("일치여부", {}).get(lt, ""),
                    "해석": report.get("해석", {}).get(lt, ""),
                }
            )
        else:
            rows.append(
                {
                    "리드타임": lt,
                    "모델1 확률": m1.get("확률", 0),
                    "모델1 판정": m1.get("판정", ""),
                    "threshold": m1.get("threshold", np.nan),
                }
            )
    return pd.DataFrame(rows)


def get_eda_context(final_data: pd.DataFrame, target_date: pd.Timestamp, site: str) -> dict[str, Any]:
    fd = final_data.copy()
    fd["조사일"] = pd.to_datetime(fd["조사일"], errors="coerce")
    neg = {"", "미발령", "정상", "0", "nan", "none"}
    fd["alert_binary"] = (
        fd["발령단계"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x in neg else 1)
        if "발령단계" in fd.columns
        else 0
    )
    site_fd = fd[fd["채수위치"] == site].copy() if "채수위치" in fd.columns else fd.copy()
    site_fd["월"] = site_fd["조사일"].dt.month
    site_fd["연도"] = site_fd["조사일"].dt.year
    site_fd["doy"] = site_fd["조사일"].dt.dayofyear
    cur_month = int(target_date.month)
    season_months = {
        1: [12, 1, 2],
        2: [12, 1, 2],
        3: [3, 4, 5],
        4: [3, 4, 5],
        5: [3, 4, 5],
        6: [6, 7, 8],
        7: [6, 7, 8],
        8: [6, 7, 8],
        9: [9, 10, 11],
        10: [9, 10, 11],
        11: [9, 10, 11],
        12: [12, 1, 2],
    }
    monthly_rate = site_fd[site_fd["월"] == cur_month]["alert_binary"].mean()
    seasonal_rate = site_fd[site_fd["월"].isin(season_months[cur_month])]["alert_binary"].mean()
    cur_doy = target_date.timetuple().tm_yday
    doy_mask = (site_fd["doy"] >= cur_doy - 15) & (site_fd["doy"] <= cur_doy + 15)
    doy_rate = site_fd.loc[doy_mask, "alert_binary"].mean()
    yearly = site_fd.groupby("연도")["alert_binary"].sum().to_dict() if "연도" in site_fd.columns else {}
    fad = site_fd[site_fd["alert_binary"] == 1].groupby("연도")["doy"].min()
    if len(fad) >= 3:
        slope = float(np.polyfit(fad.index.astype(float), fad.values.astype(float), 1)[0])
        trend_msg = f"첫 발령 doy 연 추세: {slope:.2f} (음수면 시즌 앞당김)"
    else:
        trend_msg = "추세 분석 데이터 부족"
    return {
        "monthly_alert_rate": float(monthly_rate) if pd.notna(monthly_rate) else 0.0,
        "seasonal_alert_rate": float(seasonal_rate) if pd.notna(seasonal_rate) else 0.0,
        "doy_alert_window": f"현재±15일 구간 역대 발령 비율: {doy_rate:.0%}" if pd.notna(doy_rate) else "N/A",
        "recent_trend": trend_msg,
        "yearly_alert_counts": yearly,
    }


def _days_until_alert_after_rows(fd_site: pd.DataFrame, matched: pd.DataFrame, horizon: int = 45) -> list[int]:
    """matched 각 행(조건 충족일) 이후 horizon 일 안 첫 발령까지 일수."""
    if matched.empty or "alert_binary" not in fd_site.columns or "조사일" not in fd_site.columns:
        return []
    fd_site = fd_site.sort_values("조사일")
    days: list[int] = []
    for date in matched["조사일"].dropna().values:
        ts = pd.Timestamp(date)
        future = fd_site[(fd_site["조사일"] > ts) & (fd_site["조사일"] <= ts + pd.Timedelta(days=horizon))]
        alerts = future.loc[future["alert_binary"] == 1, "조사일"]
        if len(alerts):
            days.append(int((alerts.iloc[0] - ts).days))
    return days


def _days_pulse_to_first_alert(fd_site: pd.DataFrame, pulse_dates: pd.Series, max_days: int = 30) -> list[int]:
    """강우 펄스 발생일 이후 max_days 이내 첫 발령까지 일수 (히스토그램용)."""
    if pulse_dates.empty or "alert_binary" not in fd_site.columns:
        return []
    fd_site = fd_site.sort_values("조사일")
    days: list[int] = []
    for date in pulse_dates.head(150).values:
        ts = pd.Timestamp(date)
        fut = fd_site[(fd_site["조사일"] > ts) & (fd_site["조사일"] <= ts + pd.Timedelta(days=max_days))]
        al = fut.loc[fut["alert_binary"] == 1, "조사일"]
        if len(al):
            days.append(int((al.iloc[0] - ts).days))
    return days


def _event_window_alert_rates(
    fd_site: pd.DataFrame,
    pulse_dates: pd.Series,
    windows: list[tuple[int, int]],
) -> dict[str, float]:
    """
    펄스일 ts 기준 (ts+ws, ts+we] 구간에 발령일이 하나라도 있으면 1, 없으면 0 — 이벤트별 평균.
    windows: (시작일오프셋, 종료일오프셋) 일 단위, 구간은 (ts+a, ts+b].
    """
    if pulse_dates.empty or "alert_binary" not in fd_site.columns:
        return {}
    fd_site = fd_site.sort_values("조사일")
    out: dict[str, float] = {}
    for w_start, w_end in windows:
        label = f"{w_start}~{w_end}일"
        flags: list[int] = []
        for date in pulse_dates.head(150).values:
            ts = pd.Timestamp(date)
            win = fd_site[
                (fd_site["조사일"] > ts + pd.Timedelta(days=w_start))
                & (fd_site["조사일"] <= ts + pd.Timedelta(days=w_end))
            ]
            flags.append(1 if win["alert_binary"].max() == 1 else 0)
        out[label] = float(np.mean(flags)) if flags else 0.0
    return out


def _avg_discharge_on_alert_rows(matched: pd.DataFrame) -> float | None:
    if matched.empty or "총방류량(㎥/s)" not in matched.columns or "alert_binary" not in matched.columns:
        return None
    ad = matched.loc[matched["alert_binary"] == 1, "총방류량(㎥/s)"].dropna()
    return float(ad.mean()) if len(ad) else None


def scenario_past_analysis(
    preset_name: str,
    sim_row: pd.Series,
    feature_df: pd.DataFrame,
    site: str = "문의",
) -> dict[str, Any]:
    """
    프리셋별 맞춤 과거 필터 + 발령 비율·지연·연도 패턴·방류 인과 힌트.
    """
    fd_site = (
        feature_df[feature_df["채수위치"] == site].copy()
        if "채수위치" in feature_df.columns
        else feature_df.copy()
    )
    fd_site["조사일"] = pd.to_datetime(fd_site["조사일"], errors="coerce")
    fd_site = fd_site.dropna(subset=["조사일"]).sort_values("조사일").reset_index(drop=True)

    neg = {"", "미발령", "정상", "0", "nan", "none"}
    if "발령단계" in fd_site.columns:
        fd_site["alert_binary"] = (
            fd_site["발령단계"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x in neg else 1)
        )
    else:
        fd_site["alert_binary"] = 0
    fd_site["연도"] = fd_site["조사일"].dt.year

    matched = fd_site
    days_to_alert: list[int] = []
    filter_desc = ""
    rain_window_rates: dict[str, float] = {}
    yearly_pattern: dict[int, dict[str, Any]] = {}

    # ─── 올해 같은 여름: 6~9월 + 저강수 + (여름 내) 상대적 고방류 (+ 14일 기온 상위) ───
    if "올해 같은 여름" in preset_name:
        fd = fd_site.copy()
        fd["월"] = fd["조사일"].dt.month
        summ = fd[fd["월"].isin([6, 7, 8, 9])].copy()
        rain_c = "강우량(mm)" if "강우량(mm)" in summ.columns else None
        if rain_c is None and "일강수량(mm)" in summ.columns:
            rain_c = "일강수량(mm)"
        filter_desc = "6~9월"
        if rain_c is not None and len(summ) > 20:
            summ[rain_c] = pd.to_numeric(summ[rain_c], errors="coerce").fillna(0.0)
            summ = summ[summ[rain_c] <= 3.0]
            filter_desc += f", {rain_c}≤3mm"
        if "총방류량(㎥/s)" in summ.columns and len(summ) > 5:
            qd = float(summ["총방류량(㎥/s)"].astype(float).quantile(0.55))
            summ = summ[summ["총방류량(㎥/s)"].astype(float) >= qd]
            filter_desc += f", 방류≥여름내55%분위({qd:.0f}㎥/s)"
        if "air_temp_mean_14d" in summ.columns and len(summ) > 5:
            qt = float(summ["air_temp_mean_14d"].astype(float).quantile(0.50))
            summ = summ[summ["air_temp_mean_14d"].astype(float) >= qt]
            filter_desc += f", 14d평균기온≥여름내중앙({qt:.1f}°C)"
        matched = summ
        days_to_alert = _days_until_alert_after_rows(fd_site, matched) if len(matched) else []
        avg_d = _avg_discharge_on_alert_rows(matched)
        n = len(matched)
        ar = float(matched["alert_binary"].mean()) if n else 0.0
        years = sorted(matched["연도"].dropna().unique().astype(int).tolist()) if n else []
        for yr, grp in matched.groupby("연도"):
            yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
        mean_delay = float(np.mean(days_to_alert)) if days_to_alert else None
        narrative = (
            f"과거 '{filter_desc}' 조건 {n}일 중 발령 비율 {ar:.0%}"
            + (f", 이후 첫 발령까지 평균 {mean_delay:.1f}일" if mean_delay is not None else "")
            + (f", 발령 시 평균 방류 {avg_d:.0f} ㎥/s" if avg_d is not None else "")
            + (f" (연도: {', '.join(map(str, years[:8]))}{'…' if len(years) > 8 else ''})" if years else "")
            + ". 대시보드 What-if 프리셋과 동일 취지로 '올해 같은 여름' 사전점검에 활용."
        )
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": n,
            "alert_rate": ar,
            "years": years,
            "days_to_alert": days_to_alert,
            "avg_discharge_at_alert": avg_d,
            "yearly_pattern": yearly_pattern,
            "rain_window_rates": rain_window_rates,
            "narrative": narrative,
        }

    # ─── 폭염: CHD + 건조 동시 ───────────────────────────────────────────
    if "폭염" in preset_name and "CHD" in fd_site.columns and "dry_days" in fd_site.columns:
        chd_thr = max(int(float(sim_row.get("CHD", 7) or 7)), 5)
        dry_thr = max(int(float(sim_row.get("dry_days", 14) or 14)), 7)
        matched = fd_site[(fd_site["CHD"] >= chd_thr) & (fd_site["dry_days"] >= dry_thr)]
        filter_desc = f"CHD≥{chd_thr}일 + 건조일수≥{dry_thr}일 동시"
        days_to_alert = _days_until_alert_after_rows(fd_site, matched)
        avg_d = _avg_discharge_on_alert_rows(matched)
        n = len(matched)
        ar = float(matched["alert_binary"].mean()) if n else 0.0
        years = sorted(matched["연도"].dropna().unique().astype(int).tolist()) if n else []
        for yr, grp in matched.groupby("연도"):
            yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
        mean_delay = float(np.mean(days_to_alert)) if days_to_alert else None
        narrative = (
            f"과거 {filter_desc} 조건 {n}일 중 실제 발령 비율 {ar:.0%}"
            + (f", 발령까지 평균 {mean_delay:.1f}일" if mean_delay is not None else "")
            + (f", 발령 시 평균 방류량 {avg_d:.0f} ㎥/s" if avg_d is not None else "")
            + (f" ({', '.join(map(str, years))}년 등)" if years else "")
        )
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": n,
            "alert_rate": ar,
            "years": years,
            "days_to_alert": days_to_alert,
            "avg_discharge_at_alert": avg_d,
            "yearly_pattern": yearly_pattern,
            "rain_window_rates": rain_window_rates,
            "narrative": narrative,
        }

    # ─── 장마 후 고온: 펄스 후 구간별 발령 비율 + D→발령 일수 분포 ───────
    if "장마" in preset_name or "🌧️" in preset_name:
        if "rain_pulse_flag" in fd_site.columns:
            pulse_rows = fd_site[fd_site["rain_pulse_flag"] == 1]
            pulse_dates = pulse_rows["조사일"]
        else:
            pulse_rows = fd_site.iloc[0:0]
            pulse_dates = pd.Series(dtype="datetime64[ns]")

        filter_desc = "rain_pulse_flag=1 이후 구간별 발령 여부 (이벤트 기준)"
        rain_window_rates = _event_window_alert_rates(
            fd_site, pulse_dates, windows=[(0, 7), (7, 14), (14, 21)]
        )
        days_to_alert = _days_pulse_to_first_alert(fd_site, pulse_dates, max_days=30)
        matched = pulse_rows
        n_events = len(pulse_rows)
        d07 = rain_window_rates.get("0~7일", 0.0)
        d714 = rain_window_rates.get("7~14일", 0.0)
        d1421 = rain_window_rates.get("14~21일", 0.0)
        mean_delay = float(np.mean(days_to_alert)) if days_to_alert else None
        narrative = (
            f"과거 강우 펄스 이벤트 {n_events}건 분석. "
            f"D+0~7 구간 이후 발령 비율 {d07:.0%}, D+7~14: {d714:.0%}, D+14~21: {d1421:.0%}. "
            + (f"펄스 후 첫 발령까지 평균 {mean_delay:.1f}일. " if mean_delay is not None else "")
            + "H2 가설(장마 후 7~14일 위험) 실데이터 검증용."
        )
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": n_events,
            "alert_rate": d714,
            "years": sorted(pulse_rows["연도"].dropna().unique().astype(int).tolist()) if n_events else [],
            "days_to_alert": days_to_alert,
            "avg_discharge_at_alert": _avg_discharge_on_alert_rows(pulse_rows),
            "yearly_pattern": {},
            "rain_window_rates": rain_window_rates,
            "narrative": narrative,
        }

    # ─── 가뭄: 저수율 + 건조일 ───────────────────────────────────────────
    if "가뭄" in preset_name and "저수율(%)" in fd_site.columns:
        storage_thr = float(sim_row.get("저수율(%)", 45) or 45)
        dry_thr = max(int(float(sim_row.get("dry_days", 15) or 15)), 10)
        if "dry_days" in fd_site.columns:
            matched = fd_site[(fd_site["저수율(%)"] <= storage_thr) & (fd_site["dry_days"] >= dry_thr)]
            filter_desc = f"저수율≤{storage_thr:.0f}% + 건조일수≥{dry_thr}일"
        else:
            matched = fd_site[fd_site["저수율(%)"] <= storage_thr]
            filter_desc = f"저수율≤{storage_thr:.0f}% (dry_days 없음)"
        days_to_alert = _days_until_alert_after_rows(fd_site, matched)
        avg_d = _avg_discharge_on_alert_rows(matched)
        n = len(matched)
        ar = float(matched["alert_binary"].mean()) if n else 0.0
        years = sorted(matched["연도"].dropna().unique().astype(int).tolist()) if n else []
        for yr, grp in matched.groupby("연도"):
            yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
        mean_delay = float(np.mean(days_to_alert)) if days_to_alert else None
        narrative = (
            f"과거 {filter_desc} 조건 {n}일 중 발령 비율 {ar:.0%}"
            + (f", 이후 첫 발령까지 평균 {mean_delay:.1f}일" if mean_delay is not None else "")
            + (f", 발령 시 평균 방류 {avg_d:.0f} ㎥/s" if avg_d is not None else "")
        )
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": n,
            "alert_rate": ar,
            "years": years,
            "days_to_alert": days_to_alert,
            "avg_discharge_at_alert": avg_d,
            "yearly_pattern": yearly_pattern,
            "rain_window_rates": rain_window_rates,
            "narrative": narrative,
        }

    # ─── 방류량 증량: 고방류 vs 저방류 발령 비율 비교 ─────────────────────
    if ("방류" in preset_name or "💧" in preset_name) and "총방류량(㎥/s)" in fd_site.columns:
        cur_discharge = float(sim_row.get("총방류량(㎥/s)", 30) or 30)
        high_thr = cur_discharge * 1.3 if cur_discharge > 0 else fd_site["총방류량(㎥/s)"].quantile(0.75)
        matched_high = fd_site[fd_site["총방류량(㎥/s)"] >= high_thr]
        matched_low = fd_site[fd_site["총방류량(㎥/s)"] < cur_discharge] if cur_discharge > 0 else fd_site.iloc[0:0]
        ar_high = float(matched_high["alert_binary"].mean()) if len(matched_high) else 0.0
        ar_low = float(matched_low["alert_binary"].mean()) if len(matched_low) else 0.0
        filter_desc = f"방류량 ≥{high_thr:.0f} ㎥/s (시뮬 대비 +30% 근접) vs <{cur_discharge:.0f} ㎥/s"
        narrative = (
            f"과거 고방류 {len(matched_high)}일 발령 비율 {ar_high:.0%} vs "
            f"저방류 {len(matched_low)}일 {ar_low:.0%}. "
            f"방류 증량의 억제 효과는 {'데이터상 유리' if ar_high < ar_low else '연도·기상에 따라 혼재'}."
        )
        matched = matched_high
        for yr, grp in matched.groupby("연도"):
            yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": len(matched_high),
            "alert_rate": ar_high,
            "years": sorted(matched_high["연도"].dropna().unique().astype(int).tolist()) if len(matched_high) else [],
            "days_to_alert": [],
            "avg_discharge_at_alert": _avg_discharge_on_alert_rows(matched_high),
            "yearly_pattern": yearly_pattern,
            "rain_window_rates": {},
            "narrative": narrative,
            "alert_rate_low_discharge": ar_low,
            "n_low_discharge": len(matched_low),
        }

    # ─── 복합 고위험 ─────────────────────────────────────────────────────
    if "복합" in preset_name or "⚠️" in preset_name:
        wt = float(sim_row.get("수온(℃)", 26) or 26)
        stv = float(sim_row.get("저수율(%)", 40) or 40)
        if "수온(℃)" in fd_site.columns and "저수율(%)" in fd_site.columns:
            matched = fd_site[(fd_site["수온(℃)"] >= wt - 2) & (fd_site["저수율(%)"] <= stv + 5)]
            filter_desc = f"수온≥{wt - 2:.0f}°C + 저수율≤{stv + 5:.0f}%"
        else:
            matched = fd_site
            filter_desc = "수온·저수율 컬럼 부족 — 전체 기간"
        days_to_alert = _days_until_alert_after_rows(fd_site, matched)
        avg_d = _avg_discharge_on_alert_rows(matched)
        n = len(matched)
        ar = float(matched["alert_binary"].mean()) if n else 0.0
        years = sorted(matched["연도"].dropna().unique().astype(int).tolist()) if n else []
        for yr, grp in matched.groupby("연도"):
            yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
        narrative = (
            f"과거 {filter_desc} 조건 {n}일 중 발령 비율 {ar:.0%}"
            + (f", 발령 시 평균 방류 {avg_d:.0f} ㎥/s" if avg_d is not None else "")
            + (f", 유사 연도: {', '.join(map(str, years))}" if years else "")
        )
        return {
            "title": preset_name,
            "filter_desc": filter_desc,
            "n_matched": n,
            "alert_rate": ar,
            "years": years,
            "days_to_alert": days_to_alert,
            "avg_discharge_at_alert": avg_d,
            "yearly_pattern": yearly_pattern,
            "rain_window_rates": {},
            "narrative": narrative,
        }

    # ─── 기본 ────────────────────────────────────────────────────────────
    wt = float(sim_row.get("수온(℃)", 22) or 22)
    if "수온(℃)" in fd_site.columns:
        matched = fd_site[(fd_site["수온(℃)"] - wt).abs() <= 3]
        filter_desc = f"수온 {wt - 3:.0f}~{wt + 3:.0f}°C"
    else:
        matched = fd_site
        filter_desc = "수온 없음 — 전체"
    days_to_alert = _days_until_alert_after_rows(fd_site, matched)
    avg_d = _avg_discharge_on_alert_rows(matched)
    n = len(matched)
    ar = float(matched["alert_binary"].mean()) if n else 0.0
    years = sorted(matched["연도"].dropna().unique().astype(int).tolist()) if n else []
    for yr, grp in matched.groupby("연도"):
        yearly_pattern[int(yr)] = {"발생일수": len(grp), "발령비율": float(grp["alert_binary"].mean())}
    narrative = f"과거 {filter_desc} {n}일 중 발령 비율 {ar:.0%}."
    return {
        "title": preset_name,
        "filter_desc": filter_desc,
        "n_matched": n,
        "alert_rate": ar,
        "years": years,
        "days_to_alert": days_to_alert,
        "avg_discharge_at_alert": avg_d,
        "yearly_pattern": yearly_pattern,
        "rain_window_rates": {},
        "narrative": narrative,
    }


def discharge_delay_analysis(
    fd_site: pd.DataFrame,
    site: str | None = None,
    *,
    jump_ratio: float = 1.3,
    horizon_days: int = 45,
    max_events: int = 50,
) -> dict[str, Any]:
    """
    전일 대비 방류 `jump_ratio` 이상 급증일 이후, 첫 발령까지 지연(일) — 연도·일자 예시 포함.
    `fd_site`가 이미 지점 필터된 경우 `site`는 생략.
    """
    fd = fd_site.copy()
    if site is not None and "채수위치" in fd.columns:
        fd = fd[fd["채수위치"] == site].copy()
    if fd.empty or "총방류량(㎥/s)" not in fd.columns or "발령단계" not in fd.columns:
        return {"avg_delay_days": None, "examples": []}

    neg = {"", "미발령", "정상", "0", "nan", "none"}
    fd["alert_binary"] = fd["발령단계"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x in neg else 1)
    fd = fd.sort_values("조사일")
    fd["discharge_jump"] = (fd["총방류량(㎥/s)"] >= fd["총방류량(㎥/s)"].shift(1) * jump_ratio).fillna(False)

    delays: list[int] = []
    examples: list[dict[str, Any]] = []
    for jd in fd.loc[fd["discharge_jump"], "조사일"].head(max_events).values:
        jd_ts = pd.Timestamp(jd)
        post = fd[(fd["조사일"] > jd_ts) & (fd["조사일"] <= jd_ts + pd.Timedelta(days=horizon_days))]
        next_alerts = post.loc[post["alert_binary"] == 1, "조사일"]
        if len(next_alerts):
            delay = int((next_alerts.iloc[0] - jd_ts).days)
            delays.append(delay)
            row = fd.loc[fd["조사일"] == jd_ts, "총방류량(㎥/s)"]
            discharge_val = float(row.iloc[0]) if len(row) else float("nan")
            examples.append(
                {
                    "연도": jd_ts.year,
                    "방류증량일": str(jd_ts.date()),
                    "방류량(㎥/s)": round(discharge_val, 1),
                    "발령까지일수": delay,
                }
            )

    return {
        "avg_delay_days": float(np.mean(delays)) if delays else None,
        "examples": examples[:5],
    }


def apply_preset_to_overrides(base_row: pd.Series, preset_key: str) -> tuple[dict[str, float], dict[str, Any] | None]:
    """프리셋 → (원시 오버라이드 dict, force_derived)."""
    if preset_key not in PRESET_SCENARIOS:
        return {}, None
    spec = PRESET_SCENARIOS[preset_key]
    out: dict[str, float] = {}
    for col, fn in spec.get("overrides", {}).items():
        if col not in base_row.index:
            continue
        cur = base_row[col]
        if pd.isna(cur):
            continue
        out[col] = float(fn(float(cur)))
    force = spec.get("force_derived")
    return out, force
