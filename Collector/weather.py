# 기상청 초단기실황 수집
# 이미지 수집기가 매시 한 번 호출한다. 초단기실황은 최근 1일치만 제공되어 백필이 불가능하므로,
# 이미지와 마찬가지로 지나가면 영구 손실이다. 그래서 야간에도 거른 없이 받는다.
import csv
import logging
import math
import time
from datetime import timedelta

import requests

URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# 초단기실황 관측 항목 (저장 순서)
CATEGORIES = ["T1H", "REH", "WSD", "VEC", "PTY", "RN1", "UUU", "VVV"]
CSV_FIELDS = ["base_date", "base_time", "nx", "ny"] + CATEGORIES + ["fetched_at", "error"]

log = logging.getLogger("collector")


def latlon_to_grid(lat, lon):
    """위경도 → 기상청 격자(nx, ny). Lambert Conformal Conic.
    서울 종로(60,127) · 인천 미추홀(54,124) · 제주(53,38)로 검증함."""
    RE, GRID = 6371.00877, 5.0                      # 지구 반경(km), 격자 간격(km)
    SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180.0
    re, slat1, slat2 = RE / GRID, SLAT1 * D, SLAT2 * D
    olon, olat = OLON * D, OLAT * D
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / \
         math.log(math.tan(math.pi * .25 + slat2 * .5) / math.tan(math.pi * .25 + slat1 * .5))
    sf = (math.tan(math.pi * .25 + slat1 * .5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi * .25 + olat * .5) ** sn)
    ra = re * sf / (math.tan(math.pi * .25 + lat * D * .5) ** sn)
    theta = (lon * D - olon + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + .5), int(ro - ra * math.cos(theta) + YO + .5)


def grids_of(cameras):
    """카메라 목록에서 중복을 제거한 격자 집합. 카메라 20대가 격자 13~14개로 줄어든다."""
    return sorted({latlon_to_grid(c["lat"], c["lon"]) for c in cameras})


def fetch_ncst(key, nx, ny, when, timeout=20):
    """해당 정시의 초단기실황 1건. 성공하면 {항목: 값}, 실패하면 None과 사유."""
    r = requests.get(URL, params={"serviceKey": key, "dataType": "JSON", "numOfRows": 10,
                                  "pageNo": 1, "base_date": f"{when:%Y%m%d}",
                                  "base_time": f"{when:%H}00", "nx": nx, "ny": ny}, timeout=timeout)
    r.raise_for_status()
    body = r.json()["response"]
    if body["header"]["resultCode"] != "00":
        return None, body["header"]["resultMsg"]
    return {i["category"]: i["obsrValue"] for i in body["body"]["items"]["item"]}, None


def collect(key, cameras, csv_path, slot, now_fn, retry_wait=10):
    """격자별로 slot 시각의 실황을 받아 CSV에 쌓는다. 반환: (성공 수, 전체 수)

    자료는 매시 정시 기준이고 게시는 15~40분 사이다. 이 함수는 :40 회차에 호출되므로
    같은 시각 자료를 바로 받을 수 있고, 아직 없으면 직전 시간으로 한 번 더 시도한다."""
    if not key:
        log.warning("기상 수집 건너뜀 — 인증키 없음")
        return 0, 0

    targets = grids_of(cameras)
    rows, ok = [], 0
    for nx, ny in targets:
        data, err, base = None, None, slot
        for base in (slot, slot - timedelta(hours=1)):
            try:
                data, err = fetch_ncst(key, nx, ny, base)
            except Exception as e:
                data, err = None, f"{type(e).__name__}: {e}"
            if data:
                break
            log.warning(f"기상 {nx},{ny} {base:%m-%d %H}시 실패: {err}")
            time.sleep(1)

        row = {"base_date": f"{base:%Y%m%d}", "base_time": f"{base:%H}00", "nx": nx, "ny": ny,
               "fetched_at": f"{now_fn():%Y-%m-%d %H:%M:%S}", "error": "" if data else err}
        row.update({c: (data or {}).get(c, "") for c in CATEGORIES})
        rows.append(row)
        ok += bool(data)

    try:
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerows(rows)
    except OSError as e:
        log.error(f"기상 CSV 기록 실패({type(e).__name__}: {e})")

    if ok:
        sample = next((r for r in rows if not r["error"]), {})
        log.info(f"기상 수집 {ok}/{len(targets)} 격자 | {slot:%m-%d %H}시 | "
                 f"예시 기온 {sample.get('T1H')}℃ 습도 {sample.get('REH')}% 강수형태 {sample.get('PTY')}")
    else:
        log.error(f"기상 수집 전멸 — {len(targets)}개 격자 전부 실패")
    return ok, len(targets)
