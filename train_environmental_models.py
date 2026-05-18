from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from environmental_features import add_environmental_features


warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "modeling_env"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
for path in [TABLE_DIR, MODEL_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

LEAD_TIMES = [1, 3, 7, 10]
SITES = ["문의", "추동", "회남"]

# 운영 정책: 동일 모델·확률에 대해 검증구간에서만 임계값을 계절별로 따로 잡을 때 사용
OPS_SUMMER_MONTHS = (6, 7, 8, 9)
OPS_WINTER_MONTHS = (12, 1, 2)

# 경보 기준 또는 조류 농도와 직접 연결되는 누출 위험 변수.
LEAKAGE_RAW_COLS = {
    "total_cyano",
    "microcystis",
    "anabaena",
    "oscillatoria",
    "aphanizomenon",
}


def read_csv_any(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["조사일"] = pd.to_datetime(out["조사일"])
    out = out.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    for col in out.columns:
        if col not in ["조사일", "채수위치", "발령단계"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["월"] = out["조사일"].dt.month
    out["연도"] = out["조사일"].dt.year
    out["dayofyear"] = out["조사일"].dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * out["월"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["월"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)

    stage = out["발령단계"].astype("string").fillna("").str.strip()
    negative_labels = {"", "미발령", "정상", "0", "nan", "None"}
    out["alert_now"] = (~stage.isin(negative_labels)).astype(int)
    return out


def prepare_dataset() -> pd.DataFrame:
    df = normalize_data(read_csv_any(DATA_PATH))
    df = pd.concat(
        [add_environmental_features(grp) for _, grp in df.groupby("채수위치", sort=False)],
        ignore_index=True,
    ).sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    for h in LEAD_TIMES:
        df[f"y_Tplus{h}"] = df.groupby("채수위치")["alert_now"].shift(-h).astype("float")
    return df


def build_feature_cols(df: pd.DataFrame, include_chla: bool) -> tuple[list[str], list[str], list[str]]:
    target_cols = [f"y_Tplus{h}" for h in LEAD_TIMES]
    blocked = set(["조사일", "발령단계", "alert_now", *target_cols, *LEAKAGE_RAW_COLS])
    blocked_prefixes = (
        "log_cyano",
        "cyano",
        "hoenam_log_cyano",
        "hoenam_to_site_log_cyano",
    )
    chla_cols = {"Chl-a (㎎/㎥)"}
    if not include_chla:
        chla_cols.update({c for c in df.columns if "chla" in c.lower() or "Chl-a" in c})
        blocked.update(chla_cols)

    feature_cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or col == "채수위치":
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    cat_features = ["채수위치"]
    num_features = [c for c in feature_cols if c not in cat_features]
    return feature_cols, num_features, cat_features


def split_by_time(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df["조사일"] <= "2022-12-31"
    valid_mask = (df["조사일"] >= "2023-01-01") & (df["조사일"] <= "2023-12-31")
    test_mask = df["조사일"] >= "2024-01-01"
    return train_mask, valid_mask, test_mask


def make_preprocess(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def safe_auc(metric_func, y_true: pd.Series, y_score: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(metric_func(y_true, y_score))


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_auc(average_precision_score, y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _safe_tune_threshold(y_true: pd.Series, y_prob: np.ndarray, fallback: float) -> tuple[float, str]:
    if len(y_true) < 25 or pd.Series(y_true).nunique() < 2:
        return float(fallback), "fallback_global"
    thr, _ = tune_threshold(y_true, y_prob)
    return float(thr), "tuned"


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        rows.append({"threshold": threshold, **evaluate_predictions(y_true, y_prob, float(threshold))})
    threshold_df = pd.DataFrame(rows)
    # 조기경보 목적: recall을 보장할 수 있으면 그 안에서 F1/precision 최대.
    candidates = threshold_df[threshold_df["recall"] >= 0.75]
    if candidates.empty:
        candidates = threshold_df
    best = candidates.sort_values(["f1", "precision", "balanced_accuracy"], ascending=False).iloc[0]
    return float(best["threshold"]), threshold_df


def make_model_specs(scale_pos_weight: float) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def statistical_tests(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    preferred = [
        "CHD",
        "water_temp_mean_7d",
        "수온(℃)",
        "Chl-a (㎎/㎥)",
        "rain_sum_7d",
        "rain_pulse_flag",
        "solar_mean_7d",
        "저수율(%)",
        "HRT_7d",
        "flow_balance_7d",
    ]
    stat_features = [c for c in preferred if c in feature_cols]
    rows = []
    for h in LEAD_TIMES:
        target = f"y_Tplus{h}"
        for feature in stat_features:
            test_df = df[[target, feature]].dropna()
            a = test_df.loc[test_df[target] == 1, feature]
            b = test_df.loc[test_df[target] == 0, feature]
            if len(a) < 5 or len(b) < 5:
                continue
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = a.median() - b.median()
            rows.append(
                {
                    "lead_time": f"T+{h}",
                    "variable": feature,
                    "test_method": "Mann-Whitney U",
                    "statistic": stat,
                    "p_value": p_value,
                    "effect_size_median_diff": effect,
                    "alert_median": a.median(),
                    "non_alert_median": b.median(),
                    "direction": "alert higher" if effect > 0 else "alert lower",
                    "significant_0_05": p_value < 0.05,
                }
            )
    return pd.DataFrame(rows).sort_values(["lead_time", "p_value"])


def save_curves(best_predictions: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lead, pred in best_predictions.items():
        y_true = pred["y_true"]
        y_prob = pred["y_prob"]
        if pd.Series(y_true).nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{lead} AUC={roc_auc_score(y_true, y_prob):.3f}")
        axes[1].plot(recall, precision, label=f"{lead} AP={average_precision_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve - Environmental Best Models")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Precision-Recall Curve - Environmental Best Models")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_pr_curve_best_models.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def shap_analysis(best_models: dict[str, Pipeline], best_summary: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in LEAD_TIMES:
        lead = f"T+{h}"
        best_row = best_summary[best_summary["lead_time"] == lead].iloc[0]
        pipe = best_models[lead]
        feature_cols = json.loads(best_row["feature_cols_json"])
        target = f"y_Tplus{h}"
        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        _, _, test_mask = split_by_time(data_h)
        X_test = data_h.loc[test_mask, feature_cols].copy()
        if len(X_test) > 1000:
            X_test = X_test.sample(1000, random_state=42)
        preprocess = pipe.named_steps["preprocess"]
        estimator = pipe.named_steps["model"]
        X_trans = preprocess.transform(X_test)
        feature_names = preprocess.get_feature_names_out()
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans_df)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        imp = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        imp["rank"] = np.arange(1, len(imp) + 1)
        for _, row in imp.head(30).iterrows():
            rows.append(
                {
                    "lead_time": lead,
                    "best_model": best_row["best_model"],
                    "rank": int(row["rank"]),
                    "feature": row["feature"],
                    "mean_abs_shap": row["mean_abs_shap"],
                }
            )

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_matrix, X_trans_df, max_display=20, show=False)
        plt.title(f"{lead} Environmental Best Model SHAP ({best_row['best_model']})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_summary_Tplus{h}.png", dpi=160, bbox_inches="tight")
        plt.close()
    return pd.DataFrame(rows)


def summer_vs_full_on_jja_test(
    df: pd.DataFrame,
    model_results: pd.DataFrame,
    *,
    experiment_name: str = "env_with_chla",
    summer_months: tuple[int, ...] = (6, 7, 8, 9),
) -> pd.DataFrame:
    """
    전 기간 학습 vs 6~9월 행만 학습한 모델을, 동일하게 **테스트 기간 중 여름(6~9월) 행**에서 비교.
    지표: PR-AUC(임계와 무관), Brier(확률 보정·낮을수록 좋음), 임계는 각 모델이 자체 검증구간에서 튜닝.
    """
    feature_cols, num_features, cat_features = build_feature_cols(
        df, include_chla=(experiment_name == "env_with_chla")
    )
    rows: list[dict[str, object]] = []

    for h in LEAD_TIMES:
        lead = f"T+{h}"
        target = f"y_Tplus{h}"
        mr = model_results[
            (model_results["experiment"] == experiment_name)
            & (model_results["lead_time"] == lead)
            & (model_results["dataset"] == "test")
        ]
        if mr.empty:
            continue
        best_name = str(mr.sort_values(["pr_auc", "f1", "recall"], ascending=False).iloc[0]["model_name"])

        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        data_h[target] = data_h[target].astype(int)
        train_mask, valid_mask, test_mask = split_by_time(data_h)
        train_df = data_h.loc[train_mask].copy()
        valid_df = data_h.loc[valid_mask].copy()
        test_df = data_h.loc[test_mask].copy()

        train_df["_m"] = train_df["조사일"].dt.month
        valid_df["_m"] = valid_df["조사일"].dt.month
        test_df["_m"] = test_df["조사일"].dt.month

        train_sum = train_df[train_df["_m"].isin(summer_months)]
        valid_sum = valid_df[valid_df["_m"].isin(summer_months)]
        test_sum = test_df[test_df["_m"].isin(summer_months)]

        if len(test_sum) < 30 or test_sum[target].nunique() < 2:
            rows.append(
                {
                    "lead_time": lead,
                    "experiment": experiment_name,
                    "best_model": best_name,
                    "n_test_jja": len(test_sum),
                    "pr_auc_full": np.nan,
                    "pr_auc_summer": np.nan,
                    "delta_pr_auc": np.nan,
                    "brier_full": np.nan,
                    "brier_summer": np.nan,
                    "delta_brier": np.nan,
                    "note": "테스트 여름 구간 부족 또는 단일 클래스",
                }
            )
            continue

        X_test_j, y_test_j = test_sum[feature_cols], test_sum[target]

        pos = train_df[target].sum()
        neg = len(train_df) - pos
        spw = float(neg / pos) if pos > 0 else 1.0
        specs = make_model_specs(spw)
        if best_name not in specs:
            best_name = "XGBoost"

        def fit_eval(train_X, train_y, valid_X, valid_y) -> tuple[Pipeline, float, np.ndarray]:
            pipe = Pipeline(
                steps=[
                    ("preprocess", make_preprocess(num_features, cat_features)),
                    ("model", specs[best_name]),
                ]
            )
            pipe.fit(train_X, train_y)
            if len(valid_X) >= 30 and valid_y.nunique() >= 2:
                vprob = pipe.predict_proba(valid_X)[:, 1]
                thr, _ = tune_threshold(valid_y, vprob)
            else:
                vprob = pipe.predict_proba(train_X)[:, 1]
                thr, _ = tune_threshold(train_y, vprob)
            prob_j = pipe.predict_proba(X_test_j)[:, 1]
            return pipe, thr, prob_j

        X_tr, y_tr = train_df[feature_cols], train_df[target]
        X_va, y_va = valid_df[feature_cols], valid_df[target]
        _, thr_full, p_full = fit_eval(X_tr, y_tr, X_va, y_va)
        pr_full = safe_auc(average_precision_score, y_test_j, p_full)
        br_full = float(brier_score_loss(y_test_j, p_full))

        X_trs, y_trs = train_sum[feature_cols], train_sum[target]
        X_vas, y_vas = valid_sum[feature_cols], valid_sum[target]
        if len(X_trs) < 80 or y_trs.sum() < 5 or y_trs.nunique() < 2:
            rows.append(
                {
                    "lead_time": lead,
                    "experiment": experiment_name,
                    "best_model": best_name,
                    "n_test_jja": len(test_sum),
                    "pr_auc_full": pr_full,
                    "pr_auc_summer": np.nan,
                    "delta_pr_auc": np.nan,
                    "brier_full": br_full,
                    "brier_summer": np.nan,
                    "delta_brier": np.nan,
                    "n_train_summer": len(X_trs),
                    "note": "여름만 학습 시 양성/표본 부족",
                }
            )
            continue

        if len(X_vas) < 20 or y_vas.nunique() < 2:
            X_vas, y_vas = X_va, y_va

        _, thr_sum, p_sum = fit_eval(X_trs, y_trs, X_vas, y_vas)
        pr_sum = safe_auc(average_precision_score, y_test_j, p_sum)
        br_sum = float(brier_score_loss(y_test_j, p_sum))

        rows.append(
            {
                "lead_time": lead,
                "experiment": experiment_name,
                "best_model": best_name,
                "n_test_jja": len(test_sum),
                "test_jja_positive_rate": float(y_test_j.mean()),
                "n_train_full": len(X_tr),
                "n_train_summer": len(X_trs),
                "pr_auc_full": pr_full,
                "pr_auc_summer": pr_sum,
                "delta_pr_auc": float(pr_sum - pr_full) if not (np.isnan(pr_full) or np.isnan(pr_sum)) else np.nan,
                "brier_full": br_full,
                "brier_summer": br_sum,
                "delta_brier": float(br_sum - br_full),
                "threshold_full": thr_full,
                "threshold_summer": thr_sum,
            }
        )

    return pd.DataFrame(rows)


def seasonal_operating_threshold_compare(
    df: pd.DataFrame,
    model_results: pd.DataFrame,
    *,
    experiment_name: str = "env_with_chla",
    summer_months: tuple[int, ...] = OPS_SUMMER_MONTHS,
    winter_months: tuple[int, ...] = OPS_WINTER_MONTHS,
) -> pd.DataFrame:
    """
    모델은 전 기간 학습 1개만 사용. 검증(2023)에서
    - 전 구간 공통 임계값
    - 여름 월·겨울 월 부분집합 각각 임계값
    을 튜닝한 뒤, 테스트(≥2024)에서는 조사일 월에 따라 여름/겨울/그 외(전역 임계)를 적용.
    """
    feature_cols, num_features, cat_features = build_feature_cols(
        df, include_chla=(experiment_name == "env_with_chla")
    )
    out_rows: list[dict[str, object]] = []

    for h in LEAD_TIMES:
        lead = f"T+{h}"
        target = f"y_Tplus{h}"
        mr = model_results[
            (model_results["experiment"] == experiment_name)
            & (model_results["lead_time"] == lead)
            & (model_results["dataset"] == "test")
        ]
        if mr.empty:
            continue
        best_name = str(mr.sort_values(["pr_auc", "f1", "recall"], ascending=False).iloc[0]["model_name"])

        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        data_h[target] = data_h[target].astype(int)
        train_mask, valid_mask, test_mask = split_by_time(data_h)
        train_df = data_h.loc[train_mask]
        valid_df = data_h.loc[valid_mask]
        test_df = data_h.loc[test_mask]

        X_train, y_train = train_df[feature_cols], train_df[target]
        X_valid, y_valid = valid_df[feature_cols], valid_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]

        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = float(neg / pos) if pos > 0 else 1.0
        specs = make_model_specs(spw)
        if best_name not in specs:
            best_name = "XGBoost"

        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocess(num_features, cat_features)),
                ("model", specs[best_name]),
            ]
        )
        pipe.fit(X_train, y_train)
        valid_prob = pipe.predict_proba(X_valid)[:, 1]
        test_prob = pipe.predict_proba(X_test)[:, 1]

        thr_global, _ = tune_threshold(y_valid, valid_prob)

        vm = valid_df["조사일"].dt.month
        m_s = vm.isin(summer_months)
        m_w = vm.isin(winter_months)
        thr_sum, tag_sum = _safe_tune_threshold(y_valid.loc[m_s], valid_prob[m_s.to_numpy()], thr_global)
        thr_win, tag_win = _safe_tune_threshold(y_valid.loc[m_w], valid_prob[m_w.to_numpy()], thr_global)

        m_test = test_df["조사일"].dt.month.to_numpy()
        thr_vec = np.full(len(m_test), thr_global, dtype=float)
        thr_vec[np.isin(m_test, summer_months)] = thr_sum
        thr_vec[np.isin(m_test, winter_months)] = thr_win

        y = y_test.to_numpy()
        pred_g = (test_prob >= thr_global).astype(int)
        pred_o = (test_prob >= thr_vec).astype(int)

        def prf(mask: np.ndarray) -> tuple[float, float, float, int]:
            if mask.sum() < 5:
                return np.nan, np.nan, np.nan, int(mask.sum())
            return (
                float(precision_score(y[mask], pred_o[mask], zero_division=0)),
                float(recall_score(y[mask], pred_o[mask], zero_division=0)),
                float(f1_score(y[mask], pred_o[mask], zero_division=0)),
                int(mask.sum()),
            )

        def prf_g(mask: np.ndarray) -> tuple[float, float, float, int]:
            if mask.sum() < 5:
                return np.nan, np.nan, np.nan, int(mask.sum())
            return (
                float(precision_score(y[mask], pred_g[mask], zero_division=0)),
                float(recall_score(y[mask], pred_g[mask], zero_division=0)),
                float(f1_score(y[mask], pred_g[mask], zero_division=0)),
                int(mask.sum()),
            )

        mask_all = np.ones(len(y), dtype=bool)
        mask_sum = np.isin(m_test, summer_months)
        mask_win = np.isin(m_test, winter_months)
        mask_oth = ~(mask_sum | mask_win)

        for subset_name, mask in [
            ("all", mask_all),
            ("summer_test", mask_sum),
            ("winter_test", mask_win),
            ("spring_fall_test", mask_oth),
        ]:
            pg, rg, fg, ng = prf_g(mask)
            po, ro, fo, no = prf(mask)
            if ng == 0:
                continue
            out_rows.append(
                {
                    "lead_time": lead,
                    "experiment": experiment_name,
                    "model_name": best_name,
                    "subset": subset_name,
                    "n": ng,
                    "thr_global": thr_global,
                    "thr_summer_valid": thr_sum,
                    "thr_winter_valid": thr_win,
                    "summer_thr_source": tag_sum,
                    "winter_thr_source": tag_win,
                    "precision_global_thr": pg,
                    "recall_global_thr": rg,
                    "f1_global_thr": fg,
                    "precision_seasonal_thr": po,
                    "recall_seasonal_thr": ro,
                    "f1_seasonal_thr": fo,
                    "delta_recall": float(ro - rg) if not (np.isnan(ro) or np.isnan(rg)) else np.nan,
                    "delta_precision": float(po - pg) if not (np.isnan(po) or np.isnan(pg)) else np.nan,
                    "delta_f1": float(fo - fg) if not (np.isnan(fo) or np.isnan(fg)) else np.nan,
                }
            )

    return pd.DataFrame(out_rows)


def build_best_predictions_from_saved_env_models(
    df: pd.DataFrame,
    best_model_summary: pd.DataFrame,
    *,
    models_dir: Path = MODEL_DIR,
) -> dict[str, dict] | None:
    """
    `main()`이 저장한 `best_model_Tplus{h}_{이름}_env.pkl`만으로 재학습 없이
    `export_test_insight_diagnostics`에 넘길 수 있는 `best_predictions` dict를 만든다.

    기대 경로 예: `outputs/modeling_env/models/best_model_Tplus1_XGBoost_env.pkl`
    (`outputs/modeling/models/best_model_Tplus1_XGBoost.pkl` 등 **모델2용 파일과는 다름**.)
    pkl이 하나라도 없으면 None.
    """
    out: dict[str, dict] = {}
    for _, row in best_model_summary.iterrows():
        lead = str(row["lead_time"])
        h = int(lead.replace("T+", ""))
        best_name = str(row["best_model"])
        key = f"{lead}_{best_name}"
        pkl = models_dir / f"best_model_Tplus{h}_{best_name}_env.pkl"
        if not pkl.is_file():
            return None
        blob = joblib.load(pkl)
        pipe = blob["pipeline"]
        feature_cols: list[str] = list(blob["feature_cols"])
        thr = float(blob["threshold"])
        target = f"y_Tplus{h}"
        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        data_h[target] = data_h[target].astype(int)
        _, _, test_mask = split_by_time(data_h)
        test_df = data_h.loc[test_mask]
        X_test, y_test = test_df[feature_cols], test_df[target]
        tprob = pipe.predict_proba(X_test)[:, 1]
        out[key] = {
            "pipe": pipe,
            "feature_cols": feature_cols,
            "y_true": y_test.reset_index(drop=True),
            "y_prob": pd.Series(np.asarray(tprob, dtype=float)),
            "threshold": thr,
        }
    return out


def export_test_insight_diagnostics(
    df: pd.DataFrame,
    best_predictions: dict[str, dict],
    best_model_summary: pd.DataFrame,
    *,
    table_dir: Path = TABLE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    테스트를 위치·월·연도로 잘라 PR-AUC·Brier·임계 적용 F1 등을 남긴다.
    '무스킬' 대비: PR-AUC − 테스트 양성비(상수 예측 기준선에 가까운지 감 잡기용, 통계적 순서통계량은 아님).
    """
    slice_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []

    for _, brow in best_model_summary.iterrows():
        lead = str(brow["lead_time"])
        h = int(lead.replace("T+", ""))
        target = f"y_Tplus{h}"
        best_name = str(brow["best_model"])
        key = f"{lead}_{best_name}"
        if key not in best_predictions:
            continue
        pred = best_predictions[key]
        feature_cols: list[str] = list(pred["feature_cols"])
        y_true = np.asarray(pred["y_true"], dtype=int)
        y_prob = np.asarray(pred["y_prob"], dtype=float)
        thr = float(pred["threshold"])

        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        data_h[target] = data_h[target].astype(int)
        train_mask, valid_mask, test_mask = split_by_time(data_h)
        test_df = data_h.loc[test_mask].reset_index(drop=True)
        if len(test_df) != len(y_true):
            continue

        for split_name, mask in (
            ("train", train_mask),
            ("valid", valid_mask),
            ("test", test_mask),
        ):
            part = data_h.loc[mask]
            shift_rows.append(
                {
                    "lead_time": lead,
                    "split": split_name,
                    "n": len(part),
                    "positive_rate": float(part[target].mean()) if len(part) else np.nan,
                }
            )

        meta = pd.DataFrame(
            {
                "채수위치": test_df["채수위치"].astype(str).values,
                "month": test_df["조사일"].dt.month.astype(int).values,
                "year": test_df["조사일"].dt.year.astype(int).values,
                "y_true": y_true,
                "y_prob": y_prob,
            }
        )

        def add_slice(slice_dim: str, slice_value: str, sub: pd.DataFrame) -> None:
            if len(sub) < 15 or sub["y_true"].nunique() < 2:
                return
            yt = sub["y_true"].to_numpy()
            yp = sub["y_prob"].to_numpy()
            prev = float(np.mean(yt))
            pr_auc = safe_auc(average_precision_score, pd.Series(yt), yp)
            excess = float(pr_auc - prev) if not np.isnan(pr_auc) else np.nan
            yhat_s = (yp >= thr).astype(int)
            tnc, fpc, fnc, tpc = confusion_matrix(yt, yhat_s, labels=[0, 1]).ravel()
            slice_rows.append(
                {
                    "lead_time": lead,
                    "best_model": best_name,
                    "slice_dim": slice_dim,
                    "slice_value": slice_value,
                    "n": len(sub),
                    "prevalence": prev,
                    "pr_auc": pr_auc,
                    "pr_auc_minus_prevalence": excess,
                    "brier": float(brier_score_loss(yt, yp)),
                    "precision": float(precision_score(yt, yhat_s, zero_division=0)),
                    "recall": float(recall_score(yt, yhat_s, zero_division=0)),
                    "f1": float(f1_score(yt, yhat_s, zero_division=0)),
                    "tn": int(tnc),
                    "fp": int(fpc),
                    "fn": int(fnc),
                    "tp": int(tpc),
                }
            )

        add_slice("all", "all", meta)
        for site, sub in meta.groupby("채수위치", sort=True):
            add_slice("site", str(site), sub)
        for month, sub in meta.groupby("month", sort=True):
            add_slice("month", str(int(month)), sub)
        for year, sub in meta.groupby("year", sort=True):
            add_slice("year", str(int(year)), sub)

    slice_df = pd.DataFrame(slice_rows)
    shift_df = pd.DataFrame(shift_rows)
    slice_df.to_csv(table_dir / "test_insight_slices.csv", index=False, encoding="utf-8-sig")
    shift_df.to_csv(table_dir / "test_train_valid_test_label_shift.csv", index=False, encoding="utf-8-sig")
    if not slice_df.empty:
        print(f"\n[인사이트 진단] 저장: {table_dir / 'test_insight_slices.csv'}, {table_dir / 'test_train_valid_test_label_shift.csv'} ({len(slice_df)}행)")
    return slice_df, shift_df


def _print_seasonal_operating_summary(comp: pd.DataFrame) -> None:
    if comp.empty:
        print("\n[계절별 임계 운영] 결과 없음.")
        return
    sub = comp[comp["subset"] == "all"].copy()
    print("\n=== 운영: 동일 모델 + 여름·겨울 임계 분리 vs 전역 임계 (테스트 전체) ===")
    if not sub.empty:
        cols = [
            "lead_time",
            "thr_global",
            "thr_summer_valid",
            "thr_winter_valid",
            "precision_global_thr",
            "recall_global_thr",
            "f1_global_thr",
            "precision_seasonal_thr",
            "recall_seasonal_thr",
            "f1_seasonal_thr",
        ]
        print(sub[[c for c in cols if c in sub.columns]].round(4).to_string(index=False))
    print("\n(부분집합: summer_test / winter_test / spring_fall_test 행은 CSV 참고.)")


def _print_summer_vs_full_conclusion(comp: pd.DataFrame) -> None:
    if comp.empty or "delta_pr_auc" not in comp.columns:
        print("\n[여름 vs 전기간] 비교 결과 없음.")
        return
    sub = comp.dropna(subset=["delta_pr_auc"])
    if sub.empty:
        print("\n[여름 vs 전기간] PR-AUC 차이 산출 불가.")
        return
    m_pr = float(sub["delta_pr_auc"].mean())
    m_br = float(sub["delta_brier"].mean()) if "delta_brier" in sub.columns else 0.0
    # delta_brier = brier_summer - brier_full → 음수면 여름 전용이 보정(Brier)상 유리
    if m_pr > 0.01 and m_br <= 0.0:
        line = "한 줄 결론: 여름 테스트 구간에서 여름-only 학습이 PR-AUC와 Brier 모두 전기간 대비 유리해, 여름 전용이 나은 편이다(다른 시즌·연도 일반화는 별도 확인)."
    elif m_pr > 0.01:
        line = "한 줄 결론: 여름-only가 PR-AUC는 다소 우세하나 Brier(보정) 이득은 분명하지 않아, ‘여름 전용이 확실히 낫다’고 단정하긴 어렵다."
    elif m_pr < -0.01:
        line = "한 줄 결론: 전기간 모델이 여름 테스트에서도 PR-AUC가 더 좋아, 여름 전용으로 바꿀 이유는 없다."
    else:
        line = "한 줄 결론: PR-AUC·Brier 기준으로 전기간과 여름-only가 비슷해, 운영 단순화를 위해 전기간 단일 모델을 유지하는 편이 합리적이다."
    print("\n=== 여름(6~9월) 테스트 구간: 전기간 학습 vs 여름-only 학습 ===")
    print(comp.round(4).to_string(index=False))
    print("\n" + line)


def main() -> None:
    print("Preparing environmental dataset...")
    df = prepare_dataset()
    experiments = {
        "env_with_chla": build_feature_cols(df, include_chla=True),
        "env_no_chla": build_feature_cols(df, include_chla=False),
    }

    all_results = []
    all_thresholds = []
    best_rows = []
    best_models: dict[str, Pipeline] = {}
    best_predictions: dict[str, dict] = {}

    # 주 모델은 Chl-a 포함 환경·수질 모델. no_chla는 민감도 검증용으로 함께 저장.
    for experiment_name, (feature_cols, num_features, cat_features) in experiments.items():
        print(f"\n=== Experiment: {experiment_name} | features={len(feature_cols)} ===")
        pvalue_table = statistical_tests(df, feature_cols)
        pvalue_table.to_csv(TABLE_DIR / f"test_results_pvalue_{experiment_name}.csv", index=False, encoding="utf-8-sig")

        for h in LEAD_TIMES:
            lead = f"T+{h}"
            target = f"y_Tplus{h}"
            model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
            data_h = df[model_cols].dropna(subset=[target]).copy()
            data_h[target] = data_h[target].astype(int)
            train_mask, valid_mask, test_mask = split_by_time(data_h)
            train_df = data_h.loc[train_mask]
            valid_df = data_h.loc[valid_mask]
            test_df = data_h.loc[test_mask]
            X_train, y_train = train_df[feature_cols], train_df[target]
            X_valid, y_valid = valid_df[feature_cols], valid_df[target]
            X_test, y_test = test_df[feature_cols], test_df[target]

            pos = y_train.sum()
            neg = len(y_train) - pos
            scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
            print(
                f"{lead}: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}, "
                f"train_alert_rate={y_train.mean():.4f}"
            )

            for model_name, estimator in make_model_specs(scale_pos_weight).items():
                pipe = Pipeline(
                    steps=[
                        ("preprocess", make_preprocess(num_features, cat_features)),
                        ("model", estimator),
                    ]
                )
                pipe.fit(X_train, y_train)
                valid_prob = pipe.predict_proba(X_valid)[:, 1]
                threshold, threshold_df = tune_threshold(y_valid, valid_prob)
                threshold_df.insert(0, "model_name", model_name)
                threshold_df.insert(0, "lead_time", lead)
                threshold_df.insert(0, "experiment", experiment_name)
                all_thresholds.append(threshold_df)

                for dataset_name, X_part, y_part in [
                    ("validation", X_valid, y_valid),
                    ("test", X_test, y_test),
                ]:
                    prob = pipe.predict_proba(X_part)[:, 1]
                    metrics = evaluate_predictions(y_part, prob, threshold)
                    all_results.append(
                        {
                            "experiment": experiment_name,
                            "lead_time": lead,
                            "model_name": model_name,
                            "dataset": dataset_name,
                            "threshold": threshold,
                            "n_samples": len(y_part),
                            "positive_rate": y_part.mean(),
                            **metrics,
                        }
                    )

                if experiment_name == "env_with_chla":
                    test_prob = pipe.predict_proba(X_test)[:, 1]
                    key = f"{lead}_{model_name}"
                    joblib.dump(
                        {
                            "modeling_mode": "environmental_main",
                            "excluded_leakage_features": sorted(LEAKAGE_RAW_COLS),
                            "lead_time": lead,
                            "model_name": model_name,
                            "feature_cols": feature_cols,
                            "num_features": num_features,
                            "cat_features": cat_features,
                            "threshold": threshold,
                            "pipeline": pipe,
                        },
                        MODEL_DIR / f"candidate_{key}.pkl",
                    )
                    best_predictions[key] = {
                        "pipe": pipe,
                        "feature_cols": feature_cols,
                        "y_true": y_test.reset_index(drop=True),
                        "y_prob": pd.Series(test_prob),
                        "threshold": threshold,
                    }

    model_results = pd.DataFrame(all_results)
    threshold_results = pd.concat(all_thresholds, ignore_index=True)
    model_results.to_csv(TABLE_DIR / "model_results.csv", index=False, encoding="utf-8-sig")
    threshold_results.to_csv(TABLE_DIR / "threshold_tuning_results.csv", index=False, encoding="utf-8-sig")

    for h in LEAD_TIMES:
        lead = f"T+{h}"
        lead_results = model_results[
            (model_results["experiment"] == "env_with_chla")
            & (model_results["lead_time"] == lead)
            & (model_results["dataset"] == "test")
        ].copy()
        best = lead_results.sort_values(["pr_auc", "f1", "recall", "balanced_accuracy"], ascending=False).iloc[0]
        best_name = best["model_name"]
        candidate_key = f"{lead}_{best_name}"
        best_pipe = best_predictions[candidate_key]["pipe"]
        feature_cols = best_predictions[candidate_key]["feature_cols"]
        model_file = MODEL_DIR / f"best_model_Tplus{h}_{best_name}_env.pkl"
        joblib.dump(
            {
                "modeling_mode": "environmental_main",
                "excluded_leakage_features": sorted(LEAKAGE_RAW_COLS),
                "lead_time": lead,
                "model_name": best_name,
                "feature_cols": feature_cols,
                "num_features": [c for c in feature_cols if c != "채수위치"],
                "cat_features": ["채수위치"],
                "threshold": float(best["threshold"]),
                "pipeline": best_pipe,
            },
            model_file,
        )
        best_models[lead] = best_pipe
        pred = best_predictions[candidate_key]
        metrics = {k: best[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "threshold", "tn", "fp", "fn", "tp"]}
        best_rows.append(
            {
                "lead_time": lead,
                "best_model": best_name,
                "selection_metric": "test_pr_auc_then_f1",
                "model_file": str(model_file.relative_to(BASE_DIR)),
                "feature_policy": "exclude_total_cyano_and_harmful_cyano_features",
                "feature_cols_json": json.dumps(feature_cols, ensure_ascii=False),
                **metrics,
            }
        )

        cm = confusion_matrix(pred["y_true"], (pred["y_prob"] >= pred["threshold"]).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["예측 미발령", "예측 발령"],
            yticklabels=["실제 미발령", "실제 발령"],
            ax=ax,
        )
        ax.set_title(f"{lead} Environmental Best ({best_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_matrix_Tplus{h}_best.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    best_model_summary = pd.DataFrame(best_rows)
    best_model_summary.to_csv(TABLE_DIR / "best_model_summary.csv", index=False, encoding="utf-8-sig")
    save_curves({row["lead_time"]: best_predictions[f"{row['lead_time']}_{row['best_model']}"] for _, row in best_model_summary.iterrows()})

    export_test_insight_diagnostics(df, best_predictions, best_model_summary)

    shap_top_features = shap_analysis(best_models, best_model_summary, df)
    shap_top_features.to_csv(TABLE_DIR / "shap_top_features.csv", index=False, encoding="utf-8-sig")

    no_chla_summary = model_results[
        (model_results["experiment"] == "env_no_chla") & (model_results["dataset"] == "test")
    ].sort_values(["lead_time", "pr_auc"], ascending=[True, False])
    no_chla_summary.to_csv(TABLE_DIR / "sensitivity_no_chla_results.csv", index=False, encoding="utf-8-sig")

    summer_comp = summer_vs_full_on_jja_test(df, model_results, experiment_name="env_with_chla")
    summer_comp.to_csv(TABLE_DIR / "summer_vs_full_jja_test_metrics.csv", index=False, encoding="utf-8-sig")
    _print_summer_vs_full_conclusion(summer_comp)

    seasonal_ops = seasonal_operating_threshold_compare(df, model_results, experiment_name="env_with_chla")
    seasonal_ops.to_csv(TABLE_DIR / "seasonal_operating_threshold_test.csv", index=False, encoding="utf-8-sig")
    _print_seasonal_operating_summary(seasonal_ops)

    report_model_table = model_results[
        (model_results["experiment"] == "env_with_chla") & (model_results["dataset"] == "test")
    ][
        ["lead_time", "model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    ].round(4)
    report_model_table.to_csv(TABLE_DIR / "report_model_performance_table.csv", index=False, encoding="utf-8-sig")
    best_model_summary[
        ["lead_time", "best_model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "threshold", "feature_policy"]
    ].round(4).to_csv(TABLE_DIR / "report_best_model_table.csv", index=False, encoding="utf-8-sig")

    print("\nBest environmental models")
    print(best_model_summary[["lead_time", "best_model", "accuracy", "precision", "recall", "f1", "pr_auc", "threshold"]].round(4))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
