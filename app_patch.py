#!/usr/bin/env python3
"""
app_patch.py
============
기존 app.py에 What-if 시뮬레이션 탭 3개를 추가하는 패치 스크립트.

실행:
    python app_patch.py

결과:
    app.py 가 백업(app_backup.py)된 후
    What-if / 사후감사 / 보고서 탭이 추가된 새 app.py 생성.

주의: ``st.tabs`` / ``from api_clients`` 문자열이 아래 OLD_* 와 정확히 일치하는
레거시 ``app.py`` 용입니다. 이미 통합된 트리에서는 각 단계가 스킵됩니다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
BACKUP = ROOT / "app_backup.py"

# ─── 1. import 블록 ───────────────────────────────────────────────────────────
OLD_IMPORT = "from api_clients import ("
NEW_IMPORT_WITH_TAB = """from whatif_simulation import load_all_bundles, MODEL1_DIR, MODEL2_DIR
from whatif_tab import render_whatif_tab, render_audit_tab, render_report_tab
from api_clients import ("""
NEW_IMPORT_SIM_ONLY = """from whatif_simulation import load_all_bundles, MODEL1_DIR, MODEL2_DIR
from api_clients import ("""

# ─── 2. 탭 선언부 (레거시 6탭) ───────────────────────────────────────────────
OLD_TABS = (
    'tab_dashboard, tab_predict, tab_api, tab_reason, tab_action, tab_performance = st.tabs(\n'
    '    ["대시보드", "예측 실행", "API 수집", "위험 원인", "대응 시나리오", "모델 성능"]\n'
    ')'
)
NEW_TABS = (
    'tab_dashboard, tab_predict, tab_api, tab_reason, tab_action, tab_performance, \\\n'
    '    tab_whatif, tab_audit, tab_report = st.tabs([\n'
    '    "대시보드", "예측 실행", "API 수집", "위험 원인", "대응 시나리오", "모델 성능",\n'
    '    "🔮 What-if 시뮬레이션", "🔍 사후 감사", "📄 보고서 출력",\n'
    '])'
)

# ─── 3. 모델 번들 캐시 (ensure_operational_data 앞) ─────────────────────────
OLD_ENSURE = "def ensure_operational_data() -> None:"
NEW_BUNDLE_LOADER = '''
@st.cache_resource(show_spinner=False)
def load_model1_bundles() -> dict:
    """모델 1 (조류 모니터링 포함 보조 모델) 번들 로드."""
    try:
        return load_all_bundles(MODEL1_DIR)
    except Exception as e:
        st.warning(f"모델 1 로드 실패: {e}")
        return {}


@st.cache_resource(show_spinner=False)
def load_model2_bundles() -> dict:
    """모델 2 (환경·수문 기반 사전예측 모델) 번들 로드."""
    try:
        return load_all_bundles(MODEL2_DIR)
    except Exception as e:
        st.warning(f"모델 2 로드 실패: {e}")
        return {}


def ensure_operational_data() -> None:'''

# ─── 4. 파일 끝에 추가할 탭 본문 ────────────────────────────────────────────
WHATIF_SECTION = '''

# ═══════════════════════════════════════════════════════
# What-if 시뮬레이션 탭 (모델 2 사전예측)
# ═══════════════════════════════════════════════════════
with tab_whatif:
    _bundles2 = load_model2_bundles()
    # selected_sites 는 multiselect → 단일 지점 선택용 selectbox 추가
    _whatif_site = st.sidebar.selectbox(
        "What-if 기준 지점", options=selected_sites,
        key="whatif_site_select"
    ) if len(selected_sites) > 1 else (selected_sites[0] if selected_sites else "문의")
    if _bundles2:
        render_whatif_tab(
            feature_df=feature_df,
            target_date=target_date,
            site=_whatif_site,
            bundles2=_bundles2,
            final_data=_read_csv_any(DATA_PATH),
        )
    else:
        st.error("모델 2 로드 실패. outputs/modeling_env/ 경로를 확인하세요.")

# ═══════════════════════════════════════════════════════
# 사후 감사 탭 (모델 1 조류 포함)
# ═══════════════════════════════════════════════════════
with tab_audit:
    _bundles1 = load_model1_bundles()
    _bundles2_audit = load_model2_bundles()
    if _bundles1 and _bundles2_audit:
        render_audit_tab(
            feature_df=feature_df,
            target_date=target_date,
            site=_whatif_site,
            bundles1=_bundles1,
            bundles2=_bundles2_audit,
            final_data=_read_csv_any(DATA_PATH),
        )
    else:
        st.error("모델 1 또는 모델 2 로드 실패. outputs/ 경로를 확인하세요.")

# ═══════════════════════════════════════════════════════
# 보고서 출력 탭
# ═══════════════════════════════════════════════════════
with tab_report:
    render_report_tab()
'''


def main() -> None:
    if not APP.exists():
        raise SystemExit(f"app.py 가 없습니다: {APP}")

    shutil.copy(APP, BACKUP)
    print(f"✅ 백업 완료: {BACKUP}")

    src = APP.read_text(encoding="utf-8")

    # 1. import
    if "from whatif_simulation import" not in src:
        if OLD_IMPORT not in src:
            print("⚠️  `from api_clients import (` 를 찾을 수 없습니다 — import 단계 스킵")
        elif "from whatif_tab import" in src:
            src = src.replace(OLD_IMPORT, NEW_IMPORT_SIM_ONLY, 1)
            print("✅ whatif_simulation import 추가 완료 (whatif_tab 은 기존 유지)")
        else:
            src = src.replace(OLD_IMPORT, NEW_IMPORT_WITH_TAB, 1)
            print("✅ import 블록 수정 완료")
    else:
        print("ℹ️  import 이미 존재 — 스킵")

    # 2. tabs
    if "tab_whatif" not in src:
        if OLD_TABS not in src:
            print("⚠️  레거시 6탭 선언을 찾을 수 없습니다 — 탭 선언 단계 스킵")
        else:
            src = src.replace(OLD_TABS, NEW_TABS, 1)
            print("✅ 탭 선언부 교체 완료")
    else:
        print("ℹ️  탭 이미 존재 — 스킵")

    # 3. bundle loaders
    if "load_model1_bundles" not in src:
        if OLD_ENSURE not in src:
            print("⚠️  `def ensure_operational_data()` 를 찾을 수 없습니다 — 번들 함수 스킵")
        else:
            src = src.replace(OLD_ENSURE, NEW_BUNDLE_LOADER, 1)
            print("✅ 모델 번들 캐시 함수 추가 완료")
    else:
        print("ℹ️  번들 함수 이미 존재 — 스킵")

    # 4. append tab bodies
    # 스펙의 (tab_whatif not in src or render_whatif not in src) 는 탭 선언 직후
    # 항상 거짓이 되어 본문이 붙지 않는 버그가 있어, `with tab_whatif:` 로 판별합니다.
    if "with tab_whatif:" not in src:
        src = src.rstrip() + "\n" + WHATIF_SECTION
        print("✅ What-if 탭 렌더링 코드 추가 완료")
    else:
        print("ℹ️  `with tab_whatif:` 블록 이미 존재 — 파일 끝 append 스킵")

    APP.write_text(src, encoding="utf-8")

    print()
    print("✅ app.py 패치 완료!")
    print()
    print("⚠️  주의사항:")
    print("   app.py 내에서 feature_df / target_date / selected_sites 변수명이")
    print("   실제 코드와 다를 경우 아래 부분을 수동 수정하세요:")
    print()
    print("   with tab_whatif:")
    print("       render_whatif_tab(")
    print("           feature_df=<실제 변수명>,   # engineer_features() 결과")
    print("           target_date=<실제 변수명>,  # 기준일 Timestamp")
    print("           site=<실제 변수명>,         # 채수위치 str")
    print("           ...")


if __name__ == "__main__":
    main()
