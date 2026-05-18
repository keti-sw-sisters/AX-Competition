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

    _try_generate_insight3_figures(tbl, out, lead_order)
    _try_generate_accuracy_recall_tradeoff(tbl, out, lead_order)

    print("저장 완료:")
    print("-", fig1_path)
    print("-", fig2_path)
    print("-", fig3_path)


def _try_generate_insight3_figures(tbl: Path, out: Path, lead_order: list[str]) -> None:
    """
    인사이트 3용: 라벨 이동(fig8), 테스트 채수위치별 F1(fig9).
    `train_environmental_models.py`의 진단 CSV가 있을 때만 생성.
    """
    shift_path = tbl / "test_train_valid_test_label_shift.csv"
    slice_path = tbl / "test_insight_slices.csv"
    if not shift_path.exists() or not slice_path.exists():
        print("(선택) fig8·fig9 스킵: test_train_valid_test_label_shift.csv 또는 test_insight_slices.csv 없음.")
        return

    sh = pd.read_csv(shift_path)
    sh["lead_time"] = pd.Categorical(sh["lead_time"], categories=lead_order, ordered=True)
    sh["split"] = pd.Categorical(sh["split"], categories=["train", "valid", "test"], ordered=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(data=sh, x="lead_time", y="positive_rate", hue="split", ax=ax, palette="muted")
    ax.set_title("인사이트 3 — 구간별 양성비(라벨 이동): train → valid → test", fontsize=12, fontweight="bold")
    ax.set_xlabel("리드타임")
    ax.set_ylabel("양성 비율 (이진 발령)")
    ax.set_ylim(0, max(0.45, sh["positive_rate"].max() * 1.12))
    ax.legend(title="구간", loc="upper left")
    plt.tight_layout()
    fig8_path = out / "fig8_label_shift_train_valid_test.png"
    plt.savefig(fig8_path, dpi=160, bbox_inches="tight")
    plt.close()

    sl = pd.read_csv(slice_path)
    site = sl[sl["slice_dim"] == "site"].copy()
    if site.empty or "f1" not in site.columns:
        print("(선택) fig9 스킵: site 슬라이스 없음.")
        print("-", fig8_path)
        return
    site["lead_time"] = pd.Categorical(site["lead_time"], categories=lead_order, ordered=True)
    pivot = site.pivot_table(index="slice_value", columns="lead_time", values="f1", aggfunc="first")

    plt.figure(figsize=(7.5, 3.8))
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "F1 (테스트)"})
    ax.set_title("인사이트 3 — 채수위치별 테스트 F1 (임계 적용)", fontsize=12, fontweight="bold")
    ax.set_xlabel("리드타임")
    ax.set_ylabel("채수위치")
    plt.tight_layout()
    fig9_path = out / "fig9_test_f1_by_site_and_lead.png"
    plt.savefig(fig9_path, dpi=160, bbox_inches="tight")
    plt.close()

    print("-", fig8_path)
    print("-", fig9_path)


def _try_generate_accuracy_recall_tradeoff(tbl: Path, out: Path, lead_order: list[str]) -> None:
    """
    인사이트 9용: 불균형 데이터에서 Accuracy만으로는 미탐(저 Recall)을 가리기 어렵다는 근거.
    `model_results.csv`의 env_with_chla 테스트 행만 사용.
    """
    mr_path = tbl / "model_results.csv"
    if not mr_path.exists():
        print("(선택) fig11 스킵: model_results.csv 없음.")
        return

    mr = pd.read_csv(mr_path)
    sub = mr[(mr["experiment"] == "env_with_chla") & (mr["dataset"] == "test")].copy()
    if sub.empty:
        print("(선택) fig11 스킵: env_with_chla 테스트 행 없음.")
        return

    sub["lead_time"] = pd.Categorical(sub["lead_time"], categories=lead_order, ordered=True)
    recall_cut = 0.75
    y_min = max(0.78, float(sub["accuracy"].min()) - 0.04)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes_list = axes.ravel()

    for idx, lead in enumerate(lead_order):
        ax = axes_list[idx]
        d = sub[sub["lead_time"] == lead].copy()
        if d.empty:
            ax.set_visible(False)
            continue

        pr = float(d["positive_rate"].iloc[0])
        lazy_acc = 1.0 - pr  # 양성(발령)이 소수일 때, 항상 비발령 예측의 Accuracy

        ax.axvspan(0.0, recall_cut, ymin=0, ymax=1, alpha=0.14, color="#e74c3c", zorder=0)
        ax.axhline(lazy_acc, color="#555555", linestyle="--", linewidth=1.2, zorder=1)

        sns.scatterplot(
            data=d,
            x="recall",
            y="accuracy",
            hue="model_name",
            s=200,
            ax=ax,
            zorder=4,
            legend=idx == 0,
        )

        ax.set_title(lead, fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall (재현율 · 미탐↓)")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(y_min, 1.01)
        ax.grid(True, alpha=0.35)

        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

    if axes_list[0].get_legend() is not None:
        axes_list[0].legend(title="모델", loc="lower right", fontsize=8)

    fig.suptitle(
        "테스트: Accuracy vs Recall (env_with_chla)\n"
        "점선: ‘항상 비발령’만으로도 얻는 Accuracy — 고 Accuracy가 곧 안전하지 않음",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        f"붉은 음영: Recall < {recall_cut:.2f} (미탐지 위험 구간 안내) · 본 분석은 PR-AUC·Recall 중심으로 모델을 비교함",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig11_path = out / "fig11_accuracy_vs_recall_imbalance.png"
    plt.savefig(fig11_path, dpi=160, bbox_inches="tight")
    plt.close()
    print("-", fig11_path)


if __name__ == "__main__":
    main()
