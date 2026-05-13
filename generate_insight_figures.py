from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams


def main() -> None:
    sns.set_theme(style="whitegrid")
    # 한글 폰트 우선 적용 (환경에 없으면 기본 폰트 유지)
    for fname in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if any(fname in f.name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = fname
            break
    rcParams["axes.unicode_minus"] = False

    base = Path("outputs") / "modeling_env"
    tbl = base / "tables"
    out = base / "figures" / "insight_solution"
    out.mkdir(parents=True, exist_ok=True)

    best_path = tbl / "best_model_summary.csv"
    shap_path = tbl / "shap_top_features.csv"
    rec_path = tbl / "scenario_recommendation.csv"

    missing = [str(p) for p in [best_path, shap_path, rec_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다: " + ", ".join(missing))

    best = pd.read_csv(best_path)
    shap = pd.read_csv(shap_path)
    rec = pd.read_csv(rec_path)

    lead_order = ["T+1", "T+3", "T+7", "T+10"]
    for df in [best, shap, rec]:
        if "lead_time" in df.columns:
            df["lead_time"] = pd.Categorical(df["lead_time"], categories=lead_order, ordered=True)

    # Figure 1: 리드타임별 성능
    metric_col = "pr_auc" if "pr_auc" in best.columns else ("f1" if "f1" in best.columns else "roc_auc")
    fig1 = best.sort_values("lead_time")[["lead_time", metric_col]].dropna()
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=fig1, x="lead_time", y=metric_col, color="#4C78A8")
    ax.set_title(f"리드타임별 모델 성능 ({metric_col.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("리드타임")
    ax.set_ylabel(metric_col.upper())
    for i, v in enumerate(fig1[metric_col].tolist()):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig1_path = out / "fig1_leadtime_risk_performance.png"
    plt.savefig(fig1_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Figure 2: SHAP 상위 신호
    shap2 = shap.sort_values(["lead_time", "rank"]).copy()
    if "mean_abs_shap" not in shap2.columns:
        raise ValueError("shap_top_features.csv에 mean_abs_shap 컬럼이 없습니다.")
    shap2 = shap2.groupby("lead_time", observed=False).head(3)
    shap2["feature_short"] = shap2["feature"].astype(str).str.slice(0, 28)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=shap2, x="lead_time", y="mean_abs_shap", hue="feature_short")
    ax.set_title("리드타임별 SHAP 상위 기여 신호 (Top3)", fontsize=12, fontweight="bold")
    ax.set_xlabel("리드타임")
    ax.set_ylabel("Mean |SHAP|")
    ax.legend(title="feature", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig2_path = out / "fig2_shap_top_signals.png"
    plt.savefig(fig2_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Figure 3: 해결책 매핑 강도
    rec2 = rec.copy()
    for c in ["lead_time", "risk_signal", "recommended_action"]:
        if c not in rec2.columns:
            raise ValueError(f"scenario_recommendation.csv에 {c} 컬럼이 없습니다.")
    heat = (
        rec2.groupby(["lead_time", "risk_signal"], observed=False)
        .size()
        .reset_index(name="n")
        .pivot(index="risk_signal", columns="lead_time", values="n")
        .fillna(0)
    )
    plt.figure(figsize=(8, max(4, 0.45 * len(heat))))
    ax = sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "매핑 건수"})
    ax.set_title("리드타임 × 위험신호군 해결책 매핑", fontsize=12, fontweight="bold")
    ax.set_xlabel("리드타임")
    ax.set_ylabel("위험신호군")
    plt.tight_layout()
    fig3_path = out / "fig3_solution_map_by_leadtime.png"
    plt.savefig(fig3_path, dpi=160, bbox_inches="tight")
    plt.close()

    print("저장 완료:")
    print("-", fig1_path)
    print("-", fig2_path)
    print("-", fig3_path)


if __name__ == "__main__":
    main()
