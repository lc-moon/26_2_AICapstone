# 베이징 외부 데이터셋(Feng et al. 2021)을 인천 labels.csv와 같은 13칸 표로 만든다.
#
#   python Tools/build_labels_beijing.py
#   python Tools/build_labels_beijing.py --src 경로 --images 경로 --review 경로
#
# 입력 (저장소 밖, 기본값 C:\26_2_AICapstone_external\beijing)
#   Photo_Beijing\*.jpg          원본 사진. 파일명이 촬영 시각(베이징 현지), 90° 누워 있다
#   PhotoRGB_AQI_Met.xlsx         시간별 PM2.5·PM10 (Feng 저장소)
#   noaa_isd_54511_full.csv       관측소 54511 30분 관측 (NOAA ISD, UTC): 기온·이슬점·바람·현재 날씨 코드
# 결과
#   data/labels/labels_beijing.csv     인천 labels.csv와 같은 칸. image는 사진 폴더 기준 상대 경로
#   {images}/bj/YYYY-MM-DD/HHMM.jpg    똑바로 돌려 540×960으로 줄인 사본 (JPEG 95)
#   {review}/YYYYMMDD_HHMM.jpg         눈으로 검토할 사본 (날짜 구분 없이 한 폴더)
#   검토가 끝나 bj 폴더를 data/images/ 아래로 옮겨도 표는 그대로 맞는다
#
# 붙이는 규칙 (docs/현황.md §5, 2026-10-10 결정)
#   낮만    태양고도 0° 초과. 좌표는 저자가 기상을 뽑은 奥体中心 39.98°N 116.40°E (카메라 위치는 추정)
#   PM2.5  H시대(H:00~H:59) 사진에 엑셀 H+1 행. 중국 실시간 보고도 끝 시각 표기라서 (HJ 633-2012 §5.1.2·5.1.3)
#   기상    54511 관측을 촬영 시각 기준으로 앞뒤 관측 사이 선형 보간. 앞뒤 간격이 30분을 넘거나 한쪽이 없으면 결측
#          시각이 정확히 같은 값만 쓰는 §2 원칙의 예외다 — 1분 자료가 없고, 인천 1분 자료로 재 본 보간 오차가 중앙값 0.75%p
#   비      앞뒤 관측 중 하나라도 현재 날씨 코드 50~99(이슬비·비·눈·소나기·뇌우, WMO 4677)면 1
#          연무(05)·박무(10)는 먼지·습기 자체라 비로 보지 않는다
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Collector"))
from collect_cctv import sun_altitude  # noqa: E402

EXT = Path(r"C:\26_2_AICapstone_external")
CST = timezone(timedelta(hours=8))
LAT, LON = 39.98, 116.40
CAMERA = "bj"
STATION, PM_STATION = "54511", "奥体中心(추정)"
SIZE = (540, 960)                         # 돌린 뒤 가로×세로. 인천 960×540과 화소 수가 같다
MAX_GAP = timedelta(minutes=30)
FIELDS = ["camera_id", "image", "captured_at", "sun_alt", "pm25", "pm10", "pm_station", "pm_time",
          "temperature", "humidity", "wind_speed", "rain", "aws_station"]


def isd_value(s, scale=10):
    """ISD 'value,quality' → 숫자. 9999는 결측"""
    if not isinstance(s, str):
        return np.nan
    v = s.split(",")[0]
    return np.nan if v.lstrip("+-") in ("9999", "999") else int(v) / scale


def load_weather(path):
    """30분 관측표 (베이징 현지 시각). 같은 시각에 SYNOP(FM-12)과 METAR(FM-15)이 겹치면 METAR를 쓴다 —
    METAR가 :00·:30 둘 다 있어 간격이 고르다. 비 코드는 어느 보고든 하나라도 있으면 그 시각의 코드로 본다."""
    d = pd.read_csv(path, dtype=str)
    d["t"] = pd.to_datetime(d.DATE).dt.tz_localize("UTC").dt.tz_convert(CST).dt.tz_localize(None)
    d["T"] = d.TMP.map(isd_value)
    d["Td"] = d.DEW.map(isd_value)
    d["WS"] = d.WND.map(lambda s: isd_value(s.split(",")[3] + ",1") if isinstance(s, str) else np.nan)
    d["code"] = d.MW1.map(lambda s: int(s.split(",")[0]) if isinstance(s, str) else np.nan)
    d["metar"] = d.REPORT_TYPE.str.strip() == "FM-15"
    wet = d.groupby("t").code.apply(lambda c: bool(((c >= 50) & (c <= 99)).any()))
    obs = d.dropna(subset=["T", "Td"]).sort_values("metar").groupby("t").last()[["T", "Td", "WS"]]
    es = lambda T: 6.112 * np.exp(17.62 * T / (243.12 + T))
    obs["RH"] = (100 * es(obs.Td) / es(obs["T"])).clip(upper=100)
    obs["wet"] = wet.reindex(obs.index).fillna(False)
    return obs.sort_index()


def interp(obs, t):
    """촬영 시각 t의 기온·습도·풍속(보간)과 비 여부. 앞뒤 관측이 없거나 간격이 30분을 넘으면 None"""
    i = obs.index.searchsorted(t)
    if i < len(obs) and obs.index[i] == t:
        r = obs.iloc[i]
        return r["T"], r.RH, r.WS, int(r.wet)
    if i == 0 or i == len(obs):
        return None
    a, b = obs.iloc[i - 1], obs.iloc[i]
    ta, tb = obs.index[i - 1], obs.index[i]
    if tb - ta > MAX_GAP:
        return None
    w = (t - ta) / (tb - ta)
    lin = lambda k: a[k] + (b[k] - a[k]) * w
    return lin("T"), lin("RH"), lin("WS"), int(a.wet or b.wet)


def convert(args):
    src, dst, review = args
    if dst.exists() and review.exists():
        return
    img = cv2.imread(str(src))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)          # 원본은 오른쪽이 하늘 → 위로
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    for p in (dst, review):
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    ap = argparse.ArgumentParser(description="베이징 데이터셋을 인천 labels.csv 양식으로")
    ap.add_argument("--src", default=str(EXT / "beijing"), help="Photo_Beijing·엑셀·ISD가 있는 폴더")
    ap.add_argument("--images", default=str(EXT / "beijing" / "images"), help="돌린 사진을 둘 사진 폴더 (그 아래 bj/)")
    ap.add_argument("--review", default=str(EXT / "beijing_review"), help="검토용 사본 폴더")
    ap.add_argument("--out", default=str(ROOT / "data/labels/labels_beijing.csv"))
    args = ap.parse_args()
    src, images, review = Path(args.src), Path(args.images), Path(args.review)

    pm = pd.read_excel(src / "PhotoRGB_AQI_Met.xlsx", sheet_name=0)
    pm["Time"] = pd.to_datetime(pm["Time"])
    pm = pm.set_index("Time")[["PM2.5", "PM10"]]
    obs = load_weather(src / "noaa_isd_54511_full.csv")

    rows, jobs, night = [], [], 0
    for f in sorted((src / "Photo_Beijing").glob("*.jpg")):
        t = datetime.strptime(f.name[:14], "%Y%m%d%H%M%S")
        alt = sun_altitude(LAT, LON, t.replace(tzinfo=CST))
        if alt <= 0:
            night += 1
            continue
        rel = f"{CAMERA}/{t:%Y-%m-%d}/{t:%H%M}.jpg"
        end = pd.Timestamp(t).floor("h") + pd.Timedelta(hours=1)
        p25, p10 = (pm.loc[end, "PM2.5"], pm.loc[end, "PM10"]) if end in pm.index else (np.nan, np.nan)
        w = interp(obs, pd.Timestamp(t))
        rows.append({"camera_id": CAMERA, "image": rel, "captured_at": f"{t:%Y-%m-%d %H:%M:%S}", "sun_alt": round(alt, 1),
                     "pm25": p25, "pm10": p10, "pm_station": PM_STATION, "pm_time": f"{end:%Y-%m-%d %H:%M}",
                     "temperature": None if w is None else round(w[0], 1), "humidity": None if w is None else round(w[1], 1),
                     "wind_speed": None if w is None else round(w[2], 1), "rain": None if w is None else float(w[3]),
                     "aws_station": STATION})
        jobs.append((f, images / rel, review / f"{t:%Y%m%d_%H%M}.jpg"))
    if len({r["image"] for r in rows}) != len(rows):
        sys.exit("같은 분에 찍힌 사진이 있어 파일명이 겹칩니다")

    with ProcessPoolExecutor() as ex:
        list(ex.map(convert, jobs, chunksize=16))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: ("" if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in r.items()} for r in rows)

    d = pd.DataFrame(rows)
    print(f"사진 {len(rows) + night}장 중 낮 {len(rows)}장 (밤 {night}장 제외) → {out}")
    for k in ("pm25", "humidity", "rain"):
        print(f"  {k:10} 빈칸 {int(d[k].isna().sum())}")
    print(f"  비 감지 {int((d.rain == 1).sum())}장, PM2.5 75 초과 {int((d.pm25 > 75).sum())}장")
    print(f"  사진 폴더 {images}, 검토용 {review}")


if __name__ == "__main__":
    main()
