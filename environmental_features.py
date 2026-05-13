"""학습 스크립트 `train_environmental_models.py`와 동일한 환경·수문 파생 피처."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_environmental_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("조사일").copy()
    rain_col = "강우량(mm)" if "강우량(mm)" in grp.columns else "일강수량(mm)"
    temp_col = "수온(℃)"
    air_col = "평균기온(°C)"
    solar_col = "합계 일사량(MJ/m2)"
    inflow_col = "유입량(㎥/s)"
    outflow_col = "총방류량(㎥/s)"
    volume_col = "저수량(백만㎥)"
    chla_col = "Chl-a (㎎/㎥)"

    if temp_col in grp.columns:
        hot = (grp[temp_col] > 25).fillna(False)
        runs, count = [], 0
        for is_hot in hot:
            count = count + 1 if is_hot else 0
            runs.append(count)
        grp["CHD"] = runs
        grp["water_temp_mean_3d"] = grp[temp_col].rolling(3, min_periods=1).mean()
        grp["water_temp_mean_7d"] = grp[temp_col].rolling(7, min_periods=1).mean()
        grp["water_temp_mean_14d"] = grp[temp_col].rolling(14, min_periods=1).mean()
        for lag in [1, 3, 7, 10, 14]:
            grp[f"water_temp_lag{lag}"] = grp[temp_col].shift(lag)

    if air_col in grp.columns:
        grp["air_temp_mean_3d"] = grp[air_col].rolling(3, min_periods=1).mean()
        grp["air_temp_mean_7d"] = grp[air_col].rolling(7, min_periods=1).mean()
        grp["air_temp_mean_14d"] = grp[air_col].rolling(14, min_periods=1).mean()
        grp["heat_degree_7d"] = (grp[air_col] - 25).clip(lower=0).rolling(7, min_periods=1).sum()

    if rain_col in grp.columns:
        grp["rain_sum_3d"] = grp[rain_col].rolling(3, min_periods=1).sum()
        grp["rain_sum_7d"] = grp[rain_col].rolling(7, min_periods=1).sum()
        grp["rain_sum_14d"] = grp[rain_col].rolling(14, min_periods=1).sum()
        grp["rain_sum_30d"] = grp[rain_col].rolling(30, min_periods=1).sum()
        dry_runs, count = [], 0
        for rain in grp[rain_col].fillna(0):
            count = count + 1 if rain <= 1 else 0
            dry_runs.append(count)
        grp["dry_days"] = dry_runs
        grp["rain_pulse_flag"] = ((grp["dry_days"].shift(1) >= 5) & (grp[rain_col] >= 10)).astype(int)

    if solar_col in grp.columns:
        grp["solar_mean_3d"] = grp[solar_col].rolling(3, min_periods=1).mean()
        grp["solar_mean_7d"] = grp[solar_col].rolling(7, min_periods=1).mean()
        grp["solar_mean_14d"] = grp[solar_col].rolling(14, min_periods=1).mean()

    if inflow_col in grp.columns and volume_col in grp.columns:
        safe_inflow = grp[inflow_col].replace(0, np.nan)
        grp["HRT"] = grp[volume_col] * 1e6 / (safe_inflow * 86400)
        grp["HRT_7d"] = grp["HRT"].rolling(7, min_periods=1).mean()
        grp["HRT_14d"] = grp["HRT"].rolling(14, min_periods=1).mean()

    if outflow_col in grp.columns and inflow_col in grp.columns:
        grp["flow_balance"] = grp[inflow_col] - grp[outflow_col]
        grp["flow_balance_7d"] = grp["flow_balance"].rolling(7, min_periods=1).mean()
        grp["flow_balance_14d"] = grp["flow_balance"].rolling(14, min_periods=1).mean()

    if chla_col in grp.columns:
        for lag in [1, 3, 7, 10, 14]:
            grp[f"chla_lag{lag}"] = grp[chla_col].shift(lag)
        grp["chla_roll7"] = grp[chla_col].shift(1).rolling(7, min_periods=1).mean()
        grp["chla_roll14"] = grp[chla_col].shift(1).rolling(14, min_periods=1).mean()

    if {"CHD", "solar_mean_7d", "HRT_7d"}.issubset(grp.columns):
        grp["BGI_env"] = grp["CHD"] * grp["solar_mean_7d"] / grp["HRT_7d"].replace(0, np.nan)

    return grp
