# 수집 로그의 사진마다 기상값과 PM2.5를 붙여 학습용 표(labels.csv)를 만든다.
#
#   python Tools/build_labels.py                                  data/logs/capture_log.csv 사용
#   python Tools/build_labels.py --log 경로 --check-weather 경로    weather.csv와 대조까지
#
# 결과: data/labels/labels.csv (사진 1장 = 1행), data/labels/stations.csv (카메라별 관측소·측정소)
# 몇 번을 다시 돌려도 된다. 받은 응답은 data/labels/raw/에 남기고 같은 것을 다시 호출하지 않는다.
# PM2.5는 3개월이 지나면 다시 받을 수 없으므로 raw/airkorea/는 지우지 말 것.
#
# 붙이는 규칙 (docs/현황.md §2·§6·§8)
#   기상  — 사진을 실제로 찍은 '분'의 AWS 1분값. 파일명 시각(회차)이 아니라 로그의 captured_at을 쓴다
#           관측소는 기온·습도·강수감지를 모두 갖춘 곳 중 최근접
#   PM2.5 — 찍은 시각이 속한 1시간의 값. 에어코리아는 끝 시각으로 적으므로 11:21 사진은 dataTime 12:00
#   값이 없으면 빈칸으로 둔다. 다른 시각·다른 관측소 값으로 메우지 않는다
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Collector"))
from collect_cctv import KST, sun_altitude  # noqa: E402

OUT = ROOT / "data/labels"
RAW = OUT / "raw"
AWS_MIN = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
AWS_STN = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
AK_STN = "http://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getMsrstnList"
AK_PM = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
AWS_COLS = ["tm", "stn", "WD1", "WS1", "WDS", "WSS", "WD10", "WS10", "TA", "RE",
            "RN15", "RN60", "RN12H", "RNDAY", "HM", "PA", "PS", "TD"]
NEED = ("TA", "HM", "RE")   # 습도·강수 필터에 필요해서, 셋 중 하나라도 없는 관측소는 쓰지 않는다
NEED_RATIO = 0.9            # 하루 중 이 비율 이상 값이 있어야 '갖췄다'고 본다.
                            # 양도(500)는 기온·습도가 하루 179분만 나왔다 — 센서가 있어도 쓸 수 없다
FIELDS = ["camera_id", "image", "captured_at", "sun_alt", "pm25", "pm10", "pm_station", "pm_time",
          "temperature", "humidity", "wind_speed", "rain", "aws_station"]


def km(lat1, lon1, lat2, lon2):
    return math.hypot(lat1 - lat2, (lon1 - lon2) * math.cos(math.radians(lat1))) * 111.0


def get(url, params, key, what):
    """API 호출. 504는 잠시 뒤 재시도하고, 그 밖의 실패는 인증키를 가린 메시지로 멈춘다."""
    for wait in (2, 4, 8, None):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            msg = str(e)
            for k in (key, requests.utils.quote(key, safe="")):
                msg = msg.replace(k, "***")
            sys.exit(f"{what} 호출 실패: {msg}")
        if r.status_code != 504 and "SERVICETIMEOUT" not in r.text:
            if r.status_code != 200:
                sys.exit(f"{what} 호출 실패: HTTP {r.status_code}")
            return r
        if wait is None:
            sys.exit(f"{what} 호출 실패: 504가 계속됨")
        time.sleep(wait)


def kma_text(url, params, key, what):
    """API허브 응답 본문. 응답이 중간에 끊겨 오는 일이 있어(끝 표시 7777END가 없음) 그때는 다시 받는다.
    끊긴 응답이 저장본으로 남으면 그날 오후가 통째로 결측처럼 보인다 (2026-10-01 실제로 9건)."""
    for wait in (5, 10, 20, None):
        text = get(url, {**params, "authKey": key}, key, what).content.decode("euc-kr", "replace")
        if "7777END" in text:
            return text
        if wait is None:
            sys.exit(f"{what}: 응답이 계속 중간에 끊김")
        time.sleep(wait)


def airkorea_body(text, what):
    """에어코리아 응답 본문(body). 인증 오류는 JSON이 아니라 XML로 오므로 여기서 걸러진다."""
    try:
        data = json.loads(text)["response"]
    except (ValueError, KeyError):
        sys.exit(f"{what} 비정상 응답: {text[:200]}")
    if data["header"].get("resultCode") != "00":
        sys.exit(f"{what} API 오류: {data['header']}")
    return data["body"]


def cached(path, fetch):
    """path에 저장본이 있으면 그것을, 없으면 fetch()로 받아 저장한 뒤 돌려준다."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def load_cameras():
    """cameras.toml에서 켜진 카메라 번호 + url_cache.json의 이름·좌표"""
    import tomllib
    cfg = tomllib.loads((ROOT / "Collector/cameras.toml").read_text(encoding="utf-8-sig"))
    cache = json.loads((ROOT / "Collector/url_cache.json").read_text(encoding="utf-8"))
    return {str(c["id"]): {"name": cache[str(c["id"])]["name"],
                           "lat": float(cache[str(c["id"])]["lat"]),
                           "lon": float(cache[str(c["id"])]["lon"])}
            for c in cfg["cameras"] if c["enabled"]}


# ───────── 기상 (기상청 API허브 AWS 매분자료) ─────────
class Aws:
    def __init__(self, key):
        self.key, self.days = key, {}

    def stations(self):
        text = cached(RAW / "aws_stations.txt", lambda: kma_text(
            AWS_STN, {"inf": "AWS", "help": "1"}, self.key, "AWS 지점정보"))
        out = []
        for line in text.splitlines():
            p = line.split()
            if p and p[0].isdigit():
                out.append({"id": p[0], "lon": float(p[1]), "lat": float(p[2]), "name": p[8]})
        return out

    def day(self, stn, day):
        """관측소 하루치 1분값 {분: {TA, HM, RE, WS1}}. -50 이하는 결측이라 None으로 둔다."""
        if (stn, day) not in self.days:
            text = cached(RAW / "aws" / f"{stn}_{day:%Y%m%d}.txt", lambda: kma_text(
                AWS_MIN, {"tm1": f"{day:%Y%m%d}0000", "tm2": f"{day:%Y%m%d}2359", "stn": stn,
                          "disp": "0", "help": "0"}, self.key, f"AWS {stn} {day}"))
            rows = {}
            for line in text.splitlines():
                if line[:2] == "20":
                    v = dict(zip(AWS_COLS, line.split()))
                    t = datetime.strptime(v["tm"], "%Y%m%d%H%M").replace(tzinfo=KST)
                    rows[t] = {k: (float(v[k]) if float(v[k]) > -50 else None)
                               for k in ("TA", "HM", "RE", "WS1")}
            self.days[(stn, day)] = rows
        return self.days[(stn, day)]

    def pick(self, cams, sample_day):
        """카메라마다 기온·습도·강수감지를 모두 갖춘 관측소 중 최근접. 센서 유무는 sample_day 하루로 본다."""
        stations, complete, pick = self.stations(), {}, {}
        for cid, c in cams.items():
            for s in sorted(stations, key=lambda s: km(c["lat"], c["lon"], s["lat"], s["lon"]))[:10]:
                if s["id"] not in complete:
                    rows = list(self.day(s["id"], sample_day).values())
                    complete[s["id"]] = bool(rows) and all(
                        sum(r[k] is not None for r in rows) >= NEED_RATIO * len(rows) for k in NEED)
                if complete[s["id"]]:
                    pick[cid] = (s["id"], s["name"], km(c["lat"], c["lon"], s["lat"], s["lon"]))
                    break
        return pick


# ───────── PM2.5 (에어코리아) ─────────
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):     # "-"·빈값은 측정 안 됨
        return None


def end_time(data_time):
    """dataTime("2026-09-28 13:00", "... 24:00")을 그 1시간이 끝나는 시각으로"""
    d, hm = data_time.split()
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=KST) + timedelta(hours=int(hm[:2]))


class AirKorea:
    def __init__(self, key):
        self.key = key

    def pick(self, cams):
        """카메라마다 최근접 도시대기 측정소. 경계 근처 카메라가 있어 경기도 측정소도 후보에 넣는다."""
        def fetch(addr):
            text = get(AK_STN, {"serviceKey": self.key, "returnType": "json", "addr": addr,
                                "numOfRows": 500, "pageNo": 1}, self.key, f"측정소 목록 {addr}").text
            airkorea_body(text, f"측정소 목록 {addr}")      # 오류 응답이 저장본으로 남지 않게 먼저 확인
            return text

        stations = []
        for addr in ("인천", "경기"):
            text = cached(RAW / f"airkorea_stations_{addr}.json", lambda: fetch(addr))
            stations += [s for s in airkorea_body(text, addr)["items"] if s["mangName"] == "도시대기"]
        return {cid: min(((s["stationName"], km(c["lat"], c["lon"], float(s["dmX"]), float(s["dmY"])))
                          for s in stations), key=lambda x: x[1])
                for cid, c in cams.items()}

    def hours(self, name, term):
        """측정소 시간값 {1시간이 끝나는 시각: (pm25, pm10)}.
        오늘 받은 것과 예전에 받은 파일을 날짜순으로 합친다 — 3개월이 지난 구간은 예전 파일이 유일한 기록이다."""
        folder = RAW / "airkorea"
        today = folder / f"{name}_{datetime.now(KST):%Y%m%d}.json"
        if not today.exists():
            items, page = [], 1
            while True:
                body = airkorea_body(get(AK_PM, {
                    "serviceKey": self.key, "returnType": "json", "stationName": name,
                    "dataTerm": term, "ver": "1.5", "numOfRows": 100, "pageNo": page,
                }, self.key, f"PM2.5 {name}").text, f"PM2.5 {name}")
                items += body["items"]
                if page * 100 >= body["totalCount"]:
                    break
                page += 1
                time.sleep(1)
            folder.mkdir(parents=True, exist_ok=True)
            today.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        out = {}
        for f in sorted(folder.glob(f"{name}_*.json")):     # 나중에 받은 값이 앞의 값을 덮는다
            for it in json.loads(f.read_text(encoding="utf-8")):
                out[end_time(it["dataTime"])] = (num(it.get("pm25Value")), num(it.get("pm10Value")))
        return out


# ───────── 대조 ─────────
def check_weather(path, aws, stations, before):
    """수집기의 weather.csv(초단기실황 격자값)가 어느 관측소의 정각 1분값과 같은지 격자마다 대조한다.
    실황 격자값은 관측소 한 곳의 정각 값이므로(2026-10-01 확인) 정상이면 어떤 관측소와 100% 일치한다."""
    grids = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        t = datetime.strptime(r["base_date"] + r["base_time"], "%Y%m%d%H%M").replace(tzinfo=KST)
        if r["T1H"] and t < before:
            grids.setdefault((r["nx"], r["ny"]), []).append((t, float(r["T1H"]), float(r["REH"])))
    print(f"\n[weather.csv 대조] 격자 {len(grids)}개 — 격자마다 가장 잘 맞는 관측소")
    for g, rows in sorted(grids.items()):
        best = (None, 0, 0)
        for stn in stations:
            hit = n = 0
            for t, ta, hm in rows:
                v = aws.day(stn, t.date()).get(t)
                if v and v["TA"] is not None and v["HM"] is not None:
                    n += 1
                    hit += abs(ta - v["TA"]) < 0.05 and abs(hm - round(v["HM"])) <= 1
            if n and hit / n > (best[1] / best[2] if best[2] else -1):
                best = (stn, hit, n)
        stn, hit, n = best
        flag = "" if n and hit == n else "  ← 확인 필요"
        print(f"  {g}  관측소 {stn}  {hit}/{n}시간 일치{flag}")


def main():
    ap = argparse.ArgumentParser(description="수집 사진에 기상·PM2.5를 붙여 labels.csv를 만든다")
    ap.add_argument("--log", default=str(ROOT / "data/logs/capture_log.csv"), help="수집기의 capture_log.csv")
    ap.add_argument("--check-weather", help="수집기의 weather.csv. 주면 정각 값끼리 대조한다")
    args = ap.parse_args()

    env = dotenv_values(ROOT / ".env")
    if not env.get("KMA_HUB_KEY") or not env.get("AIRKOREA_API_KEY"):
        sys.exit(".env에 KMA_HUB_KEY와 AIRKOREA_API_KEY가 모두 있어야 합니다")
    ak_key = env["AIRKOREA_API_KEY"]
    aws, ak = Aws(env["KMA_HUB_KEY"]), AirKorea(unquote(ak_key) if "%" in ak_key else ak_key)
    cams = load_cameras()

    # 오늘 찍은 사진은 빼고 간다. 오늘 하루치 기상·PM2.5가 아직 다 나오지 않아
    # 반쪽짜리 응답이 저장본으로 남으면 다시 받지 않게 되기 때문이다.
    today = datetime.now(KST).replace(hour=0, minute=0, second=0, microsecond=0)
    shots, later = [], 0
    for r in csv.DictReader(open(args.log, encoding="utf-8-sig")):
        if r["result"] != "success" or r["cctv_id"] not in cams:
            continue
        t = datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        if t >= today:
            later += 1
            continue
        shots.append((r["cctv_id"], r["file_name"], t))
    if not shots:
        sys.exit("붙일 사진이 없습니다")
    first = min(t for _, _, t in shots)
    print(f"사진 {len(shots)}장 ({first:%m-%d} ~ {max(t for _, _, t in shots):%m-%d})"
          + (f", 오늘 찍은 {later}장은 제외" if later else ""))

    aws_pick, pm_pick = aws.pick(cams, first.date()), ak.pick(cams)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stations.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "name", "aws_station", "aws_name", "aws_km", "pm_station", "pm_km"])
        for cid, c in cams.items():
            a, p = aws_pick.get(cid, ("", "", 0)), pm_pick[cid]
            w.writerow([cid, c["name"], a[0], a[1], round(a[2], 1), p[0], round(p[1], 1)])
    print(f"관측소 {len(set(a[0] for a in aws_pick.values()))}곳 (최장 {max(a[2] for a in aws_pick.values()):.1f}km), "
          f"측정소 {len(set(p[0] for p in pm_pick.values()))}곳 (최장 {max(p[1] for p in pm_pick.values()):.1f}km)")

    # 에어코리아 조회 기간은 가장 오래된 사진이 들어가는 만큼만. 3MONTH는 측정소당 20쪽이 넘어 일일 한도(500)를 먹는다
    days = (today - first).days + 1
    term = "MONTH" if days <= 28 else "3MONTH"
    pm = {name: ak.hours(name, term) for name in sorted({p[0] for p in pm_pick.values()})}

    rows = []
    for cid, image, t in shots:
        c, stn = cams[cid], aws_pick.get(cid, ("",))[0]
        w = aws.day(stn, t.date()).get(t.replace(second=0)) if stn else None
        end = t.replace(minute=0, second=0) + timedelta(hours=1)
        pm25, pm10 = pm[pm_pick[cid][0]].get(end, (None, None))
        rows.append({
            "camera_id": cid, "image": image, "captured_at": f"{t:%Y-%m-%d %H:%M:%S}",
            "sun_alt": round(sun_altitude(c["lat"], c["lon"], t), 1),
            "pm25": pm25, "pm10": pm10, "pm_station": pm_pick[cid][0], "pm_time": f"{end:%Y-%m-%d %H:%M}",
            "temperature": w and w["TA"], "humidity": w and w["HM"],
            "wind_speed": w and w["WS1"], "rain": w and w["RE"], "aws_station": stn,
        })
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)

    n = len(rows)
    has = lambda k: sum(r[k] is not None for r in rows)
    print(f"\n{OUT / 'labels.csv'}  {n}행")
    for k in ("pm25", "temperature", "humidity", "rain"):
        print(f"  {k:12} {has(k):5} / {n}  (빈칸 {n - has(k)})")
    print(f"  강수 감지된 사진 {sum(r['rain'] == 1 for r in rows)}장")

    if args.check_weather:
        check_weather(args.check_weather, aws, sorted({a[0] for a in aws_pick.values()}), today)


if __name__ == "__main__":
    main()
