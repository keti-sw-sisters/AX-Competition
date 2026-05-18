from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests


KMA_ASOS_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
KMA_VILAGE_FCST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KMA_MID_TA_URL = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"

# 대청댐 인근 단기예보 격자 (기본값 — 대시보드 TAB3와 동일)
DEFAULT_KMA_SHORT_NX = 67
DEFAULT_KMA_SHORT_NY = 100
# 중기기온 예보구역 (충북 대덕·대청 인근)
DEFAULT_KMA_MID_REGID = "11C20401"

# 기상청 ASOS 지점번호: 청주 131, 대전 133, 보은 226
DEFAULT_KMA_STATIONS = {
    "청주": "131",
    "대전": "133",
    "보은": "226",
}


@dataclass
class ApiFetchResult:
    ok: bool
    message: str
    data: pd.DataFrame


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_kma_asos_daily(
    service_key: str,
    start_date: date,
    end_date: date,
    stations: dict[str, str] | None = None,
    timeout: int = 30,
) -> ApiFetchResult:
    """Fetch KMA ASOS daily weather and return columns compatible with final_data.csv.

    The Korean public data portal key must be supplied by the operator. Daily ASOS
    data are often finalized after the observation day, so today's row may not be
    available until later.
    """
    if not service_key:
        return ApiFetchResult(False, "기상청 API 인증키가 필요합니다.", pd.DataFrame())

    stations = stations or DEFAULT_KMA_STATIONS
    rows: list[dict[str, Any]] = []
    for station_name, station_id in stations.items():
        params = {
            "serviceKey": service_key,
            "pageNo": 1,
            "numOfRows": 999,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": start_date.strftime("%Y%m%d"),
            "endDt": end_date.strftime("%Y%m%d"),
            "stnIds": station_id,
        }
        try:
            response = requests.get(KMA_ASOS_DAILY_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            body = payload.get("response", {}).get("body", {})
            items = body.get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            for item in items:
                rows.append(
                    {
                        "조사일": item.get("tm"),
                        "기상관측소": station_name,
                        "평균기온(°C)": _as_float(item.get("avgTa")),
                        "최저기온(°C)": _as_float(item.get("minTa")),
                        "최고기온(°C)": _as_float(item.get("maxTa")),
                        "일강수량(mm)": _as_float(item.get("sumRn")),
                        "강우량(mm)": _as_float(item.get("sumRn")),
                        "평균 풍속(m/s)": _as_float(item.get("avgWs")),
                        "평균 상대습도(%)": _as_float(item.get("avgRhm")),
                        "합계 일조시간(hr)": _as_float(item.get("sumSsHr")),
                        "합계 일사량(MJ/m2)": _as_float(item.get("sumGsr")),
                        "평균 전운량(1/10)": _as_float(item.get("avgTca")),
                    }
                )
        except Exception as exc:
            return ApiFetchResult(False, f"기상청 API 호출 실패({station_name}): {exc}", pd.DataFrame())

    if not rows:
        return ApiFetchResult(False, "기상청 API 응답에 데이터가 없습니다.", pd.DataFrame())

    station_df = pd.DataFrame(rows)
    station_df["조사일"] = pd.to_datetime(station_df["조사일"])
    weather_cols = [
        "평균기온(°C)",
        "최저기온(°C)",
        "최고기온(°C)",
        "일강수량(mm)",
        "강우량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "합계 일조시간(hr)",
        "합계 일사량(MJ/m2)",
        "평균 전운량(1/10)",
    ]
    daily_df = station_df.groupby("조사일", as_index=False)[weather_cols].mean(numeric_only=True)
    return ApiFetchResult(True, f"기상청 일자료 {len(daily_df):,}일 수집 완료", daily_df)


def normalize_kma_service_key(service_key: str) -> str:
    """공공데이터포털 serviceKey: URL 인코딩 문자열이면 디코딩 후 전달."""
    key = (service_key or "").strip()
    if "%" in key:
        try:
            key = unquote(key)
        except Exception:
            pass
    return key


def parse_kma_pcp_mm(val: Any) -> float:
    """기상청 PCP(강수량) 표기 → mm."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if s in ("", "강수없음", "0", "0mm", "None"):
        return 0.0
    s = s.replace("mm", "").replace("미만", "").replace("이하", "").strip()
    if "~" in s:
        s = s.split("~")[0].strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _latest_village_base(now: datetime | None = None) -> tuple[date, str]:
    """가장 최근 단기예보 base_date / base_time."""
    now = now or datetime.now()
    slots = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
    for days_back in range(5):
        day = (now - timedelta(days=days_back)).date()
        for bt in reversed(slots):
            hh, mm = int(bt[:2]), int(bt[2:4])
            tpub = datetime.combine(day, datetime.min.time()) + timedelta(hours=hh, minutes=mm + 40)
            if tpub <= now:
                return day, bt
    return (now.date() - timedelta(days=1)), "1100"


def _mid_tmfc_candidates(now: datetime | None = None) -> list[str]:
    """중기예보 발표시각(tmFc) 후보: 최근 발표부터."""
    now = now or datetime.now()
    pairs: list[tuple[datetime, str]] = []
    for days_back in range(6):
        d = (now.date() - timedelta(days=days_back))
        for hh in (18, 6):
            tpub = datetime.combine(d, datetime.min.time()) + timedelta(hours=hh, minutes=15)
            if tpub <= now:
                pairs.append((tpub, d.strftime("%Y%m%d") + f"{hh:02d}00"))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in pairs]


def _kma_header_ok(payload: dict) -> bool:
    header = (payload.get("response") or {}).get("header") or {}
    return str(header.get("resultCode", "")).strip() in ("00", "0")


def fetch_kma_vilage_forecast_daily(
    service_key: str,
    nx: int = DEFAULT_KMA_SHORT_NX,
    ny: int = DEFAULT_KMA_SHORT_NY,
    timeout: int = 25,
) -> ApiFetchResult:
    """
    기상청 단기예보(getVilageFcst) → 일별 평균기온(TMP 시간평균), 일강수(PCP 합).
    """
    if not service_key:
        return ApiFetchResult(False, "기상청 API 인증키가 필요합니다.", pd.DataFrame())

    key = normalize_kma_service_key(service_key)
    candidates = [_latest_village_base()]
    now = datetime.now()
    for d in (1, 2):
        for bt in ("1100", "0500", "0200"):
            candidates.append(((now - timedelta(days=d)).date(), bt))

    last_err = ""
    for day, btime in candidates:
        params = {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": day.strftime("%Y%m%d"),
            "base_time": btime,
            "nx": int(nx),
            "ny": int(ny),
        }
        try:
            response = requests.get(KMA_VILAGE_FCST_URL, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            last_err = str(exc)
            continue
        if not _kma_header_ok(data):
            last_err = str((data.get("response") or {}).get("header", {}))
            continue
        items = ((data.get("response") or {}).get("body") or {}).get("items") or {}
        it = items.get("item")
        if not it:
            last_err = "item 없음"
            continue
        if isinstance(it, dict):
            it = [it]

        tmp_by_date: dict[str, list[float]] = defaultdict(list)
        rain_by_date: dict[str, float] = defaultdict(float)
        for row in it:
            ds = str(row.get("fcstDate", ""))
            if len(ds) != 8:
                continue
            cat = row.get("category")
            val = row.get("fcstValue")
            if cat == "TMP":
                try:
                    tmp_by_date[ds].append(float(val))
                except (TypeError, ValueError):
                    pass
            elif cat == "PCP":
                rain_by_date[ds] += parse_kma_pcp_mm(val)

        if not tmp_by_date and not rain_by_date:
            last_err = "TMP/PCP 파싱 결과 없음"
            continue

        rows: list[dict[str, Any]] = []
        for ds in sorted(set(tmp_by_date.keys()) | set(rain_by_date.keys())):
            dtp = pd.Timestamp(f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}")
            tlist = tmp_by_date.get(ds, [])
            tmean = float(np.mean(tlist)) if tlist else float("nan")
            rows.append(
                {
                    "예보일": dtp.normalize(),
                    "기온(°C)": tmean,
                    "강수(mm)": float(rain_by_date.get(ds, 0.0)),
                    "기온자료": "기상청 단기(TMP시간평균)",
                    "강수자료": "기상청 단기(PCP일합)",
                    "출처": "기상청 단기예보",
                    "base_date": day.strftime("%Y-%m-%d"),
                    "base_time": btime,
                    "nx": nx,
                    "ny": ny,
                }
            )
        out = pd.DataFrame(rows).sort_values("예보일").reset_index(drop=True)
        msg = (
            f"단기예보 {len(out)}일 수집 (발표 {day} {btime}, 격자 nx={nx} ny={ny})"
        )
        return ApiFetchResult(True, msg, out)

    return ApiFetchResult(
        False,
        f"단기예보 API 응답 없음. 마지막 오류: {last_err or '알 수 없음'}",
        pd.DataFrame(),
    )


def fetch_kma_mid_ta_forecast_daily(
    service_key: str,
    reg_id: str = DEFAULT_KMA_MID_REGID,
    timeout: int = 25,
) -> ApiFetchResult:
    """
    기상청 중기기온(getMidTa) → n일 후 최저·최고 평균기온.
    강수 필드는 API에 없어 비움.
    """
    if not service_key:
        return ApiFetchResult(False, "기상청 API 인증키가 필요합니다.", pd.DataFrame())
    rid = (reg_id or "").strip()
    if not rid:
        return ApiFetchResult(False, "중기 예보구역 regId가 필요합니다.", pd.DataFrame())

    key = normalize_kma_service_key(service_key)
    last_err = ""
    for tmfc in _mid_tmfc_candidates():
        params = {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "JSON",
            "regId": rid,
            "tmFc": tmfc,
        }
        try:
            response = requests.get(KMA_MID_TA_URL, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            last_err = str(exc)
            continue
        if not _kma_header_ok(data):
            last_err = str((data.get("response") or {}).get("header", {}))
            continue
        items = ((data.get("response") or {}).get("body") or {}).get("items") or {}
        it = items.get("item")
        if not it:
            last_err = "item 없음"
            continue
        if isinstance(it, list):
            it = it[0]
        try:
            base_date = datetime.strptime(tmfc[:8], "%Y%m%d").date()
        except ValueError:
            continue

        rows: list[dict[str, Any]] = []
        for k, v in it.items():
            m = re.match(r"^taMin(\d+)$", str(k))
            if not m:
                continue
            n = int(m.group(1))
            tk = f"taMax{n}"
            if tk not in it:
                continue
            try:
                tmean = (float(v) + float(it[tk])) / 2.0
            except (TypeError, ValueError):
                continue
            fdate = pd.Timestamp(base_date + timedelta(days=n))
            rows.append(
                {
                    "예보일": fdate.normalize(),
                    "기온(°C)": tmean,
                    "최저기온(°C)": float(v),
                    "최고기온(°C)": float(it[tk]),
                    "강수(mm)": float("nan"),
                    "기온자료": "기상청 중기(최저·최고 평균)",
                    "강수자료": "— (중기 API 미제공)",
                    "출처": "기상청 중기기온",
                    "tmFc": tmfc,
                    "regId": rid,
                }
            )
        if rows:
            out = pd.DataFrame(rows).sort_values("예보일").reset_index(drop=True)
            return ApiFetchResult(
                True,
                f"중기기온 {len(out)}일 수집 (발표 tmFc={tmfc}, regId={rid})",
                out,
            )
        last_err = "taMin/taMax 파싱 결과 없음"

    return ApiFetchResult(
        False,
        f"중기기온 API 응답 없음. 마지막 오류: {last_err or '알 수 없음'}",
        pd.DataFrame(),
    )


def fetch_generic_json_records(
    url: str,
    api_key: str | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    records_path: str | None = None,
    timeout: int = 30,
) -> ApiFetchResult:
    """Fetch records from an organization-specific JSON API.

    records_path can be a dot-separated path such as "response.body.items.item".
    If omitted, the function accepts either a JSON list or the top-level "data"/"items" list.
    """
    if not url:
        return ApiFetchResult(False, "API URL이 필요합니다.", pd.DataFrame())

    params = dict(params or {})
    headers = dict(headers or {})
    if api_key:
        # Most Korean OpenAPI endpoints use serviceKey in the query string.
        params.setdefault("serviceKey", api_key)

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return ApiFetchResult(False, f"API 호출 실패: {exc}", pd.DataFrame())

    records: Any = payload
    if records_path:
        for part in records_path.split("."):
            records = records.get(part, {}) if isinstance(records, dict) else {}
    elif isinstance(payload, dict):
        records = payload.get("data", payload.get("items", payload))

    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        return ApiFetchResult(False, "API 응답에서 레코드를 찾지 못했습니다.", pd.DataFrame())

    return ApiFetchResult(True, f"API 레코드 {len(records):,}건 수집 완료", pd.DataFrame(records))


def merge_weather_into_observations(observations: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Fill weather columns in observation rows by date."""
    if observations.empty or weather.empty:
        return observations
    out = observations.copy()
    out["조사일"] = pd.to_datetime(out["조사일"])
    weather_copy = weather.copy()
    weather_copy["조사일"] = pd.to_datetime(weather_copy["조사일"])
    weather_cols = [c for c in weather_copy.columns if c != "조사일"]
    merged = out.merge(weather_copy, on="조사일", how="left", suffixes=("", "_api"))
    for col in weather_cols:
        api_col = f"{col}_api"
        if api_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[api_col])
            else:
                merged[col] = merged[api_col]
            merged = merged.drop(columns=api_col)
    return merged

