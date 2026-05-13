"""
대청호 조류경보 대시보드 — What-if 시뮬레이션 & 사후 감사 탭.

app.py 예시::

    from whatif_tab import render_audit_tab, render_report_tab, render_whatif_tab
    ...
    with tab_whatif:
        render_whatif_tab(feature_df, target_date, site, bundles2, final_data)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from whatif_simulation import (
    LEAD_TIMES,
    PRESET_SCENARIOS,
    SLIDER_CONFIG,
    build_model1_audit_row,
    discharge_delay_analysis,
    export_report_df,
    find_similar_past,
    generate_model1_report,
    get_eda_context,
    get_shap_top,
    map_solution,
    run_model1_audit,
    run_whatif,
    scenario_past_analysis,
    simulate_discharge,
)

_SESSION_REPORT = "audit_report_history"
_SESSION_WHATIF = "whatif_last_result"


def _risk_label(prob: float, threshold: float) -> tuple[str, str]:
    """
    UI용 3단계 라벨.
    - 경계: 모델 threshold 이상
    - 관심: 30% 이상(임계 미만)
    - 미발령: 그 외
    """
    if prob >= threshold:
        return "경계", "#FDECEA"
    if prob >= 0.30:
        return "관심", "#FFF5E0"
    return "미발령", "#EAF4FB"


def _gauge_fig(prob: float, base_prob: float, threshold: float, title: str) -> go.Figure:
    label, _bg = _risk_label(prob, threshold)
    color = (
        "#D64545"
        if prob >= threshold
        else "#F2A541" if prob >= 0.60 else "#77B255" if prob >= 0.30 else "#2E86AB"
    )
    delta_txt = f"{prob - base_prob:+.1%}"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={
                "reference": base_prob * 100,
                "valueformat": ".1f",
                "suffix": "%",
                "increasing": {"color": "#D64545"},
                "decreasing": {"color": "#2E86AB"},
            },
            number={"suffix": "%", "font": {"size": 26}},
            title={
                "text": f"<b>{title}</b> <span style='font-size:11px'>[{label}]</span>"
                f"<br><span style='font-size:11px'>Δ {delta_txt} vs 기준</span>"
            },
            gauge={
                "axis": {"range": [0, 100], "tickformat": ".0f"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#EAF4FB"},
                    {"range": [30, 60], "color": "#FFF5E0"},
                    {"range": [60, 100], "color": "#FDECEA"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    # 카드형 4열에서 제목/델타 텍스트가 잘리지 않도록 높이/상단 여백을 확대
    fig.update_layout(height=240, margin=dict(t=88, b=10, l=12, r=12))
    return fig


def _bar_shap(shap_top: list[dict[str, Any]]) -> go.Figure | None:
    if not shap_top:
        return None
    df = pd.DataFrame(shap_top)
    df["color"] = df["controllable"].map({True: "#2E86AB", False: "#F2A541"})
    df["방향"] = df["shap_value"].apply(lambda v: "위험 증가" if v > 0 else "위험 감소")
    fig = px.bar(
        df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="color",
        color_discrete_map="identity",
        hover_data=["feature_value", "방향"],
        labels={"shap_value": "SHAP 값", "feature": "변수"},
        title="위험 기여 변수 Top (파랑=조작가능, 주황=자연조건)",
    )
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
        yaxis={"autorange": "reversed"},
    )
    return fig


def _discharge_bar(scenarios: list[dict[str, Any]]) -> go.Figure | None:
    if not scenarios:
        return None
    df = pd.DataFrame(scenarios)
    fig = go.Figure()
    fig.add_bar(
        x=df["label"],
        y=df["기준확률"] * 100,
        name="기준 확률",
        marker_color="#BBDEFB",
        opacity=0.8,
    )
    fig.add_bar(
        x=df["label"],
        y=df["위험확률"] * 100,
        name="시뮬레이션 확률",
        marker_color="#2E86AB",
    )
    fig.update_layout(
        barmode="group",
        title="방류량 증량 시나리오별 위험확률 변화",
        yaxis_title="위험확률 (%)",
        xaxis_title="방류량 증량",
        height=280,
        margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _yearly_bar(yearly: dict[Any, Any]) -> go.Figure | None:
    if not yearly:
        return None
    df = pd.DataFrame({"연도": list(yearly.keys()), "발령일수": list(yearly.values())}).sort_values("연도")
    fig = px.bar(
        df,
        x="연도",
        y="발령일수",
        title="연도별 발령 일수 (역대)",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.update_layout(height=220, margin=dict(t=40, b=10))
    return fig


def _summarize_primary_causes(shap_top: list[dict[str, Any]]) -> str:
    """SHAP 상위 요인을 '한 문장'으로 묶어주는 간단 요약."""
    feats = [str(s.get("feature", "")) for s in shap_top if s.get("feature")]
    if not feats:
        return "상위 기여 요인을 계산할 수 없습니다."

    def _has_any(keys: list[str]) -> bool:
        return any(any(k in f for k in keys) for f in feats)

    parts: list[str] = []
    if _has_any(["water_temp", "수온", "CHD"]):
        parts.append("고온/수온 지속")
    if _has_any(["dry_days", "rain_sum", "rain_pulse", "강우"]):
        parts.append("건조·강우 펄스")
    if _has_any(["log_cyano", "chla", "Chl-a", "hoenam"]):
        parts.append("조류 신호(세포수/Chl-a)")
    if _has_any(["총방류량", "HRT", "flow_balance", "저수율", "유입량"]):
        parts.append("댐운영·체류/유량")
    if _has_any(["doy", "month", "dayofyear", "일사", "solar"]):
        parts.append("계절·일사")

    if not parts:
        parts = ["복합 요인"]
    return " + ".join(parts) + " 영향이 동시에 작용한 것으로 보입니다."


def _resolve_history_col(feature_name: str, history: pd.DataFrame) -> str | None:
    """전처리된 feature_name을 history 컬럼으로 최대한 매칭."""
    if feature_name in history.columns:
        return feature_name
    # 흔한 변환 prefix 제거
    candidates = [
        feature_name.replace("num__", "").replace("cat__", ""),
        feature_name.split("__")[-1],
    ]
    for c in candidates:
        if c in history.columns:
            return c
    # 부분 문자열 매칭(너무 공격적이지 않게)
    for col in history.columns:
        if col in feature_name or feature_name in col:
            return col
    return None


@st.cache_data(show_spinner=False)
def _load_species_policy_insight() -> pd.DataFrame:
    """종별 회귀 모델 정책 인사이트 CSV 로드(T+7 기준)."""
    p = Path(__file__).resolve().parent / "outputs" / "species_regression" / "tables" / "policy_insight_T7.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(p)


def render_whatif_tab(
    feature_df: pd.DataFrame,
    target_date: pd.Timestamp,
    site: str,
    bundles2: dict[str, dict],
    final_data: pd.DataFrame,
) -> None:
    st.subheader("🔮 What-if 시뮬레이션 — 환경·수문 기반 사전예측 (모델 2)")
    st.caption(
        "슬라이더로 환경·수문 조건을 바꾸면 T+1·T+3·T+7·T+10 리드타임별(모델2 각각) 발령 확률이 재계산됩니다. "
        "파생변수(CHD, HRT, dry_days, BGI 등)는 `recalculate_derived`로 갱신됩니다."
    )

    site_history = feature_df[feature_df["채수위치"] == site].sort_values("조사일").reset_index(drop=True)
    if site_history.empty:
        st.error("선택한 지점의 데이터가 없습니다.")
        return

    before = site_history[site_history["조사일"] <= target_date]
    if before.empty:
        base_row = site_history.iloc[-1].copy()
        history = site_history.iloc[:-1].copy()
    else:
        base_row = before.iloc[-1].copy()
        history = before.iloc[:-1].copy()

    st.divider()
    st.markdown("**⚡ 프리셋 시나리오 선택 (또는 아래 슬라이더 직접 조작)**")
    n_presets = len(PRESET_SCENARIOS)
    preset_cols = st.columns(min(n_presets, 5))
    selected_preset: str | None = st.session_state.get("selected_preset")
    for i, (pname, _pinfo) in enumerate(PRESET_SCENARIOS.items()):
        col = preset_cols[i % len(preset_cols)]
        if col.button(pname, width="stretch", key=f"preset_{pname}_{site}"):
            st.session_state["selected_preset"] = pname

    if selected_preset and selected_preset in PRESET_SCENARIOS:
        st.info(f"**{selected_preset}**: {PRESET_SCENARIOS[selected_preset]['desc']}")

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.markdown("#### 🎚️ 변수 조작")
        overrides: dict[str, float] = {}
        force_derived: dict[str, Any] = {}

        preset_overrides: dict[str, float] = {}
        if selected_preset and selected_preset in PRESET_SCENARIOS:
            pinfo = PRESET_SCENARIOS[selected_preset]
            for col_name, fn in pinfo.get("overrides", {}).items():
                if col_name in base_row.index:
                    base_val = float(base_row.get(col_name, 0) or 0)
                    preset_overrides[col_name] = float(fn(base_val))
            force_derived = pinfo.get("force_derived", {}) or {}

        groups: dict[str, list[str]] = {}
        for var, cfg in SLIDER_CONFIG.items():
            groups.setdefault(cfg["group"], []).append(var)

        for group_name, vars_ in groups.items():
            st.markdown(f"**{group_name}**")
            for var in vars_:
                cfg = SLIDER_CONFIG[var]
                if var not in feature_df.columns:
                    continue
                base_val = float(base_row.get(var, cfg["min"]) or cfg["min"])
                base_val = max(cfg["min"], min(cfg["max"], base_val))
                init_val = preset_overrides.get(var, base_val)
                init_val = max(cfg["min"], min(cfg["max"], init_val))
                val = st.slider(
                    f"{var} ({cfg['unit']})",
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(round(init_val, 4)),
                    step=float(cfg["step"]),
                    key=f"slider_{site}_{var}",
                    help=f"기준값: {base_val:.2f} {cfg['unit']}",
                )
                if abs(val - base_val) > 1e-6:
                    overrides[var] = val

        if st.button("🔄 슬라이더 초기화", width="stretch"):
            st.session_state["selected_preset"] = None
            for var in SLIDER_CONFIG:
                key = f"slider_{site}_{var}"
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with right:
        st.markdown("#### 📈 예측 결과")
        with st.spinner("재예측 중..."):
            result = run_whatif(base_row, overrides, bundles2, history, force_derived)
            sim_row = result["sim_row"]
            st.session_state[_SESSION_WHATIF] = result

        gcols = st.columns(4)
        for col, lt in zip(gcols, LEAD_TIMES):
            with col:
                _lab, _bg = _risk_label(float(result["sim"][lt]), float(result["threshold"][lt]))
                st.plotly_chart(
                    _gauge_fig(
                        result["sim"][lt],
                        result["base"][lt],
                        result["threshold"][lt],
                        lt,
                    ),
                    width="stretch",
                )
                st.markdown(
                    f"""
<div style="background:{_bg};padding:10px 12px;border-radius:10px;text-align:center;font-weight:800;">
  {_lab}
</div>
""",
                    unsafe_allow_html=True,
                )

        shap_lead = max(result["sim"], key=lambda k: result["sim"][k])
        shap_top = get_shap_top(bundles2[shap_lead], sim_row, n=7)
        shap_fig = _bar_shap(shap_top)
        if shap_fig:
            st.plotly_chart(shap_fig, width="stretch")
        else:
            st.info("SHAP 계산 불가 — feature importance로 대체합니다.")

        st.divider()
        st.markdown("##### 📚 과거 데이터 기반 검증")

        past_info: dict[str, Any] = {}
        scenario_info: dict[str, Any] = {}

        with st.spinner("과거 조건 분석 중..."):
            if selected_preset and selected_preset in PRESET_SCENARIOS:
                scenario_info = scenario_past_analysis(selected_preset, sim_row, feature_df, site)
                palert = float(scenario_info.get("alert_rate") or 0)
                color = "red" if palert > 0.6 else "orange" if palert > 0.3 else "green"
                st.markdown(f"**🔍 검색 조건:** `{scenario_info.get('filter_desc', '')}`")
                st.markdown(f":{color}[{scenario_info.get('narrative', '')}]")

                rw = scenario_info.get("rain_window_rates") or {}
                if rw and ("장마" in selected_preset or "🌧️" in selected_preset):
                    k1, k2, k3 = st.columns(3)
                    k1.metric("펄스 후 D+0~7 발령", f"{rw.get('0~7일', 0):.0%}")
                    k2.metric("D+7~14 (H2)", f"{rw.get('7~14일', 0):.0%}")
                    k3.metric("D+14~21", f"{rw.get('14~21일', 0):.0%}")

                if ("장마" in selected_preset or "🌧️" in selected_preset) and scenario_info.get("days_to_alert"):
                    days = scenario_info["days_to_alert"]
                    fig_days = px.histogram(
                        x=days,
                        nbins=15,
                        labels={"x": "발령까지 걸린 일수"},
                        title="강우 펄스 후 발령까지 일수 분포",
                        color_discrete_sequence=["#2E86AB"],
                    )
                    fig_days.add_vline(
                        x=7, line_dash="dash", line_color="orange", annotation_text="D+7"
                    )
                    fig_days.add_vline(x=14, line_dash="dash", line_color="red", annotation_text="D+14")
                    fig_days.update_layout(height=240, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_days, width="stretch")

                elif "방류" in selected_preset or "💧" in selected_preset:
                    delay_info = discharge_delay_analysis(
                        feature_df[feature_df["채수위치"] == site].sort_values("조사일")
                    )
                    if delay_info.get("avg_delay_days"):
                        st.markdown(
                            f"**💧 방류량 증량 후 발령까지 평균 지연:** {delay_info['avg_delay_days']:.1f}일"
                        )
                    if delay_info.get("examples"):
                        st.dataframe(
                            pd.DataFrame(delay_info["examples"]),
                            width="stretch",
                            hide_index=True,
                        )

                if scenario_info.get("yearly_pattern"):
                    yp = scenario_info["yearly_pattern"]
                    yp_df = pd.DataFrame(
                        [
                            {
                                "연도": yr,
                                "발생일수": v["발생일수"],
                                "발령비율(%)": v["발령비율"] * 100,
                            }
                            for yr, v in sorted(yp.items())
                        ]
                    )
                    fig_yp = go.Figure()
                    fig_yp.add_bar(
                        x=yp_df["연도"],
                        y=yp_df["발생일수"],
                        name="조건 발생일수",
                        marker_color="#BBDEFB",
                    )
                    fig_yp.add_scatter(
                        x=yp_df["연도"],
                        y=yp_df["발령비율(%)"],
                        name="발령 비율(%)",
                        yaxis="y2",
                        mode="lines+markers",
                        line=dict(color="#D64545", width=2),
                    )
                    fig_yp.update_layout(
                        title="연도별 조건 발생일수 & 발령 비율",
                        yaxis=dict(title="발생일수"),
                        yaxis2=dict(
                            title="발령비율(%)",
                            overlaying="y",
                            side="right",
                            range=[0, 100],
                        ),
                        legend=dict(orientation="h", y=1.1),
                        height=260,
                        margin=dict(t=50, b=10),
                    )
                    st.plotly_chart(fig_yp, width="stretch")

                if scenario_info.get("avg_discharge_at_alert"):
                    cur_d = float(sim_row.get("총방류량(㎥/s)", 0) or 0)
                    avg_d = float(scenario_info["avg_discharge_at_alert"])
                    diff = avg_d - cur_d
                    st.markdown(
                        f"**📌 발령 시 역대 평균 방류량:** {avg_d:.0f} ㎥/s "
                        f"({'현재보다 ' + ('+' if diff > 0 else '') + f'{diff:.0f} ㎥/s'})"
                    )
            else:
                past_info = find_similar_past(sim_row, final_data)
                if past_info.get("n", 0) > 0:
                    palert = past_info.get("alert_rate") or 0
                    color = "red" if palert > 0.6 else "orange" if palert > 0.3 else "green"
                    st.markdown(f":{color}[{past_info.get('message', '')}]")
                    with st.expander("유사 조건 상세 데이터", expanded=False):
                        sdf = past_info.get("similar_df", pd.DataFrame())
                        if isinstance(sdf, pd.DataFrame) and not sdf.empty:
                            st.dataframe(sdf.head(20), width="stretch", hide_index=True)
                else:
                    st.info(past_info.get("message", "유사 조건 없음"))
                scenario_info = {"narrative": past_info.get("message", "")}

        st.divider()
        st.markdown("##### 🗺️ 해결책 매핑")

        _past_for_solution = (
            {"message": scenario_info.get("narrative", "")}
            if selected_preset
            else past_info
        )

        with st.spinner("해결책 분석 중..."):
            solution = map_solution(result, shap_top, sim_row, bundles2, history, _past_for_solution)

        level_color = solution["color"]
        st.markdown(
            f"<div style='background:{level_color};color:white;padding:12px 16px;"
            f"border-radius:10px;font-size:1.05rem;font-weight:600'>"
            f"{solution['summary']}</div>",
            unsafe_allow_html=True,
        )

        for action in solution["actions"]:
            st.markdown(f"**{action['type']}**  \n{action['detail']}")

        if solution["discharge_scenarios"]:
            st.markdown("##### 💧 방류량 증량 시나리오")
            disc_df = pd.DataFrame(solution["discharge_scenarios"])
            st.dataframe(disc_df, width="stretch", hide_index=True)
            disc_fig = _discharge_bar(solution["discharge_scenarios"])
            if disc_fig:
                st.plotly_chart(disc_fig, width="stretch")

        if solution.get("past_reference"):
            st.caption(f"📌 {solution['past_reference']}")


def render_audit_tab(
    feature_df: pd.DataFrame,
    target_date: pd.Timestamp,
    site: str,
    bundles1: dict[str, dict],
    bundles2: dict[str, dict],
    final_data: pd.DataFrame,
    scenario_recommendation: pd.DataFrame | None = None,
) -> None:
    del final_data, bundles2
    st.subheader("🔍 사후 감사 — 조류 모니터링 포함 보조 모델 (모델 1)")
    st.caption(
        "조류 세포수 측정값을 반영해 모델 1로 재예측하고, 사후 설명·대응 판단에 활용합니다."
    )

    site_history = feature_df[feature_df["채수위치"] == site].sort_values("조사일").reset_index(drop=True)
    if site_history.empty:
        st.error("선택한 지점의 데이터가 없습니다.")
        return

    before = site_history[site_history["조사일"] <= target_date]
    base_row = (before.iloc[-1] if not before.empty else site_history.iloc[-1]).copy()
    history = (before.iloc[:-1] if len(before) > 1 else pd.DataFrame()).copy()

    model2_result: dict[str, Any] | None = st.session_state.get(_SESSION_WHATIF)

    st.markdown("#### 조류 측정값 입력")
    col1, col2 = st.columns(2)
    with col1:
        log_cyano_raw = st.number_input(
            f"현재 총 남조류 세포수 ({site}, cells/mL)",
            min_value=0,
            max_value=5_000_000,
            value=1000,
            step=100,
            help="total_cyano 실측값 (log1p 변환은 자동)",
        )
        # 요구사항: 지점(문의/추동/회남)별로 "현재 지점" 세포수만 입력받음.
        # H3 선행 신호(hoenam_*)는 운영 데이터에서 자동 추정해 사용.
        hoenam_cyano_raw: float
        if site == "회남":
            hoenam_cyano_raw = float(log_cyano_raw)
            st.caption("회남 지점 선택 시 회남 선행 신호는 현재 측정값을 그대로 사용합니다.")
        else:
            hoenam_hist = (
                feature_df[
                    (feature_df["채수위치"] == "회남")
                    & (feature_df["조사일"] <= target_date)
                ]
                .sort_values("조사일")
            )
            auto_hoenam = np.nan
            if not hoenam_hist.empty:
                # 운영 데이터에 total_cyano가 있으면 그대로 사용, 없으면 log_cyano 역변환
                if "total_cyano" in hoenam_hist.columns and pd.notna(hoenam_hist.iloc[-1].get("total_cyano")):
                    auto_hoenam = float(hoenam_hist.iloc[-1]["total_cyano"])
                elif "log_cyano" in hoenam_hist.columns and pd.notna(hoenam_hist.iloc[-1].get("log_cyano")):
                    auto_hoenam = float(np.expm1(float(hoenam_hist.iloc[-1]["log_cyano"])))
            if pd.isna(auto_hoenam):
                auto_hoenam = 0.0
                st.warning("회남 선행 신호(세포수)를 운영 데이터에서 찾지 못해 0으로 처리합니다.")
            else:
                st.caption(f"회남 선행 신호(자동): {auto_hoenam:,.0f} cells/mL (운영 데이터 기반)")

            hoenam_cyano_raw = float(auto_hoenam)
    with col2:
        st.info(
            "**모델 1 사용 시점**\n\n"
            "- 정기 채수 후 측정값 확보 즉시\n"
            "- 고위험 경보 발령 전 최종 교차 검증\n"
            "- 위험 상승 원인(상위 기여 요인) 설명 및 대응 체크"
        )

    if st.button("🔍 모델 1 사후 감사 실행", type="primary", width="stretch"):
        with st.spinner("모델 1 예측 중..."):
            audit_row = build_model1_audit_row(
                base_row,
                float(log_cyano_raw),
                float(hoenam_cyano_raw),
                history,
            )
            # 모델 1 확률 (리드타임별 번들 사용)
            m1_probs = run_model1_audit(
                base_row,
                float(log_cyano_raw),
                float(hoenam_cyano_raw),
                bundles1,
                history,
            )

            report = generate_model1_report(
                m1_probs,
                base_row,
                bundles1,
                site,
                target_date,
            )
            report["권장대응"] = []

        if _SESSION_REPORT not in st.session_state:
            st.session_state[_SESSION_REPORT] = []
        st.session_state[_SESSION_REPORT].append(report)
        st.session_state["audit_last_display"] = {
            "report": report,
            "m1_probs": m1_probs,
            "audit_row": audit_row,
        }
        st.success("사후 감사 완료. 결과를 확인하세요.")
        st.rerun()

    disp = st.session_state.get("audit_last_display")
    if not disp:
        return

    report = disp["report"]
    m1_probs = disp["m1_probs"]
    audit_row_disp = disp.get("audit_row")

    report_df = export_report_df(report)

    def _style(row: pd.Series) -> list[str]:
        cell = str(row.get("일치여부", ""))
        if "⚠️" in cell:
            return ["background-color:#FDECEA"] * len(row)
        if "ℹ️" in cell:
            return ["background-color:#FFF5E0"] * len(row)
        return [""] * len(row)

    try:
        st.dataframe(report_df.style.apply(_style, axis=1), width="stretch", hide_index=True)
    except Exception:
        st.dataframe(report_df, width="stretch", hide_index=True)

    judg = report["종합판정"]
    color = "#D64545" if "🔴" in judg else "#2E86AB"
    st.markdown(
        f"<div style='background:{color};color:white;padding:12px 16px;border-radius:10px;"
        f"font-size:1.05rem;font-weight:700'>{judg}</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # 모델 1 기반: 상위 기여 요인 + 근거(타임라인) + 대응 자동 매핑(체크리스트)
    # -------------------------------------------------------------------
    st.divider()
    st.markdown("#### 🧠 왜 위험이 커졌나 (모델 1 설명)")

    # 가장 위험한 리드타임(기본) + 선택
    max_lt = max(m1_probs, key=lambda k: float(m1_probs[k]))
    lead_for_shap = st.selectbox(
        "설명 리드타임(모델 1)",
        options=LEAD_TIMES,
        index=LEAD_TIMES.index(max_lt),
        key=f"audit_lead_for_shap_{site}",
    )

    # audit_row 재구성(표시 구간에서도 동일 로직)
    if audit_row_disp is None:
        audit_row_disp = build_model1_audit_row(
            base_row,
            float(log_cyano_raw),
            float(hoenam_cyano_raw),
            history,
        )
    shap_top_m1 = get_shap_top(bundles1[lead_for_shap], audit_row_disp, n=7)
    if shap_top_m1:
        st.caption("상위 기여 요인(파랑=조작가능, 주황=자연조건)")
        shap_fig = _bar_shap(shap_top_m1)
        if shap_fig:
            st.plotly_chart(shap_fig, width="stretch")
        st.info(f"요약: {_summarize_primary_causes(shap_top_m1)}")
    else:
        st.info("모델 1 SHAP을 계산할 수 없어 설명을 생략합니다.")

    # (B) 근거 타임라인: 최근 30개(또는 30일) 스냅샷
    if not history.empty and shap_top_m1:
        st.markdown("#### 📈 근거(최근 추세)")
        h = history.sort_values("조사일").tail(30).copy()
        h["조사일"] = pd.to_datetime(h["조사일"], errors="coerce")
        for item in shap_top_m1[:5]:
            feat = str(item.get("feature", ""))
            col = _resolve_history_col(feat, h)
            if not col:
                continue
            s = pd.to_numeric(h[col], errors="coerce")
            if s.dropna().empty:
                continue
            cur_val = float(audit_row_disp.get(col, np.nan)) if col in audit_row_disp.index else float(s.iloc[-1])
            p90 = float(np.nanpercentile(s.values, 90))
            dfp = pd.DataFrame({"조사일": h["조사일"], col: s})
            fig_ts = px.line(dfp, x="조사일", y=col, title=f"{col} (최근 {len(dfp)}개)")
            fig_ts.add_hline(y=p90, line_dash="dot", line_color="orange", annotation_text="상위 10% 기준(p90)")
            fig_ts.add_hline(y=cur_val, line_dash="dash", line_color="red", annotation_text="오늘/입력값")
            fig_ts.update_layout(height=220, margin=dict(t=40, b=10))
            st.plotly_chart(fig_ts, width="stretch")

    st.markdown("#### ✅ 리드타임별 권장 대응")
    selected_actions: list[dict[str, Any]] = []
    if scenario_recommendation is None or not isinstance(scenario_recommendation, pd.DataFrame) or scenario_recommendation.empty:
        st.info("대응 시나리오 표를 불러오지 못했습니다. (app.py에서 scenario_recommendation을 전달해야 합니다.)")
    else:
        action_lead = st.selectbox(
            "대응 리드타임",
            options=LEAD_TIMES,
            index=LEAD_TIMES.index(lead_for_shap),
            key=f"audit_action_lead_{site}",
        )
        action_df = scenario_recommendation[scenario_recommendation["lead_time"] == action_lead].copy()
        cols = [c for c in ["risk_signal", "shap_feature_group", "recommended_action", "responsible_unit", "urgency_level"] if c in action_df.columns]

        # 모델1 SHAP 상위 변수 -> shap_feature_group 데이터 기반 매핑
        def _feature_tokens(f: str) -> list[str]:
            ft = f.lower()
            tokens = [f, ft, f.replace("num__", "").replace("cat__", ""), f.split("__")[-1]]
            if any(k in ft for k in ["water_temp", "수온", "chd"]):
                tokens += ["water_temp", "고온", "수온", "고온·수온 지속 신호"]
            if any(k in ft for k in ["rain_sum", "dry_days", "rain_pulse", "강우"]):
                tokens += ["rain", "dry", "강우", "건조", "강우·건조 펄스 신호"]
            if any(k in ft for k in ["탁도", "투명도", "ph", "do"]):
                tokens += ["탁도", "투명도", "일반 수질 신호", "water quality"]
            if any(k in ft for k in ["doy", "month", "dayofyear", "solar", "일사"]):
                tokens += ["doy", "month", "dayofyear", "일사", "계절", "계절·일사 신호"]
            if any(k in ft for k in ["총방류량", "hrt", "flow_balance", "저수율", "유입량"]):
                tokens += ["discharge", "hrt", "flow", "저수율", "댐운영", "체류시간"]
            if any(k in ft for k in ["log_cyano", "chla", "chl-a", "hoenam"]):
                tokens += ["cyano", "chla", "hoenam", "조류", "공간전파", "조류 모니터링/공간전파 신호"]
            # 중복 제거
            seen: set[str] = set()
            uniq: list[str] = []
            for t in tokens:
                tt = str(t).strip()
                if tt and tt not in seen:
                    seen.add(tt)
                    uniq.append(tt)
            return uniq

        shap_feats = [str(s.get("feature", "")) for s in shap_top_m1 if s.get("feature")]
        cand_tokens: list[str] = []
        for f in shap_feats[:7]:
            cand_tokens.extend(_feature_tokens(f))
        cand_tokens = list(dict.fromkeys(cand_tokens))

        matched_df = action_df
        if "shap_feature_group" in action_df.columns and cand_tokens:
            def _match_group(g: Any) -> bool:
                gs = str(g).lower()
                return any(tok.lower() in gs for tok in cand_tokens if len(tok) >= 2)

            matched_df = action_df[action_df["shap_feature_group"].apply(_match_group)].copy()
            if matched_df.empty:
                matched_df = action_df.copy()

        # 권장행 중복 제거(같은 액션 반복 방지)
        dedup_cols = [c for c in ["recommended_action", "responsible_unit", "urgency_level"] if c in matched_df.columns]
        if dedup_cols:
            matched_df = matched_df.drop_duplicates(subset=dedup_cols, keep="first")

        # 종별 회귀 정책 인사이트(데이터 기반) 추가 매핑
        species_df = _load_species_policy_insight()
        species_actions = pd.DataFrame()
        if not species_df.empty and "주요 SHAP 변수" in species_df.columns:
            sp = species_df.copy()
            sp["__feat"] = sp["주요 SHAP 변수"].astype(str).str.lower()

            def _match_species_feat(feat: str) -> bool:
                f = str(feat).lower()
                return any(tok.lower() in f or f in tok.lower() for tok in cand_tokens if len(tok) >= 3)

            spm = sp[sp["__feat"].apply(_match_species_feat)].copy()
            if not spm.empty:
                species_actions = pd.DataFrame(
                    {
                        "risk_signal": spm.get("신호 의미", ""),
                        "shap_feature_group": spm.get("주요 SHAP 변수", ""),
                        "recommended_action": spm.get("권고 대응 액션", ""),
                        "responsible_unit": "수질관리/댐운영/정수장",
                        "urgency_level": "긴급",
                    }
                )
                species_actions = species_actions.drop_duplicates(
                    subset=["recommended_action", "shap_feature_group"], keep="first"
                )
        else:
            st.warning(
                "종별 회귀 정책 매핑 파일이 없어(또는 컬럼 불일치) 추가 매핑을 생략했습니다. "
                "`outputs/species_regression/tables/policy_insight_T7.csv`를 생성해 주세요."
            )

        # 기존 대응표 + 종별 정책표 병합(중복 제거)
        merged_df = matched_df.copy()
        if not species_actions.empty:
            merged_df = pd.concat([species_actions, matched_df], ignore_index=True, sort=False)
            merged_df = merged_df.drop_duplicates(
                subset=[c for c in ["recommended_action", "responsible_unit", "urgency_level"] if c in merged_df.columns],
                keep="first",
            )

        st.caption(
            f"SHAP 상위 요인 기반 매핑: {len(matched_df)}개"
            + (f" + 종별 회귀 정책매핑: {len(species_actions)}개" if not species_actions.empty else "")
        )
        if cols:
            st.dataframe(merged_df[cols], width="stretch", hide_index=True)
        else:
            st.dataframe(merged_df, width="stretch", hide_index=True)

        st.markdown("#### ✅ 이번 케이스 체크리스트 (선택)")
        n_show = min(8, len(merged_df))
        for i in range(n_show):
            row = merged_df.iloc[i].to_dict()
            label = f"[{row.get('urgency_level','')}] {row.get('recommended_action','')} ({row.get('responsible_unit','')})"
            if st.checkbox(label, value=(i < 3), key=f"audit_action_pick_{site}_{action_lead}_{i}"):
                selected_actions.append(
                    {
                        "lead_time": action_lead,
                        "risk_signal": row.get("risk_signal", ""),
                        "shap_feature_group": row.get("shap_feature_group", ""),
                        "recommended_action": row.get("recommended_action", ""),
                        "responsible_unit": row.get("responsible_unit", ""),
                        "urgency_level": row.get("urgency_level", ""),
                    }
                )

    report["권장대응"] = selected_actions

def render_report_tab() -> None:
    st.subheader("📄 사후 감사 보고서 출력")
    st.caption("사후 감사 탭에서 실행한 결과가 여기에 누적됩니다.")

    history: list[dict[str, Any]] = st.session_state.get(_SESSION_REPORT, [])
    if not history:
        st.info("아직 실행된 사후 감사 결과가 없습니다. '사후 감사' 탭에서 먼저 실행하세요.")
        return

    st.markdown(f"총 **{len(history)}건**의 감사 결과가 누적되어 있습니다.")

    all_rows: list[dict[str, Any]] = []
    for rpt in history:
        for lt in LEAD_TIMES:
            m1 = rpt.get("모델1_사후확인", {}).get(lt, {})
            m2 = rpt.get("모델2_사전예측", {}).get(lt, {})
            all_rows.append(
                {
                    "기준일": rpt.get("기준일", ""),
                    "채수위치": rpt.get("채수위치", ""),
                    "리드타임": lt,
                    "모델1_확률": m1.get("확률", np.nan),
                    "모델1_판정": m1.get("판정", ""),
                    "종합판정": rpt.get("종합판정", ""),
                    "권장대응_선택수": len(rpt.get("권장대응", []) or []),
                }
            )

    full_df = pd.DataFrame(all_rows)

    st.markdown("#### 최근 감사 결과")
    latest = history[-1]
    st.markdown(
        f"**기준일:** {latest['기준일']}  |  **지점:** {latest['채수위치']}  |  **종합판정:** {latest['종합판정']}"
    )
    st.dataframe(export_report_df(latest), width="stretch", hide_index=True)

    if latest.get("권장대응"):
        st.markdown("#### ✅ 선택된 권장 대응")
        st.dataframe(pd.DataFrame(latest["권장대응"]), width="stretch", hide_index=True)

    if len(history) > 1:
        st.markdown("#### 전체 감사 이력")
        st.dataframe(full_df, width="stretch", hide_index=True)

    st.markdown("#### 📥 다운로드")
    col1, col2 = st.columns(2)
    with col1:
        csv_buf = io.StringIO()
        full_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            "📊 전체 보고서 CSV 다운로드",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name=f"algae_audit_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        lines = ["=" * 60, "대청호 조류경보 사후 감사 보고서", "=" * 60, ""]
        for rpt in history:
            lines += [
                f"기준일: {rpt['기준일']}  |  지점: {rpt['채수위치']}",
                f"종합판정: {rpt['종합판정']}",
                "",
            ]
            for lt in LEAD_TIMES:
                m1 = rpt.get("모델1_사후확인", {}).get(lt, {})
                m2 = rpt.get("모델2_사전예측", {}).get(lt, {})
                if "모델2_사전예측" in rpt:
                    match = rpt.get("일치여부", {}).get(lt, "")
                    interp = rpt.get("해석", {}).get(lt, "")
                    lines.append(
                        f"  {lt}: 모델2={m2.get('확률', 0):.1%}({m2.get('판정', '')})"
                        f" | 모델1={m1.get('확률', 0):.1%}({m1.get('판정', '')})"
                        f" | {match} → {interp}"
                    )
                else:
                    lines.append(
                        f"  {lt}: 모델1={m1.get('확률', 0):.1%}({m1.get('판정', '')})"
                        f" | threshold={m1.get('threshold', 0):.2f}"
                    )
            lines += ["", "-" * 60, ""]
        txt = "\n".join(lines)
        st.download_button(
            "📝 텍스트 보고서 다운로드",
            data=txt.encode("utf-8"),
            file_name=f"algae_audit_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            width="stretch",
        )

    if st.button("🗑️ 감사 이력 초기화", type="secondary"):
        st.session_state[_SESSION_REPORT] = []
        st.session_state.pop("audit_last_display", None)
        st.rerun()
