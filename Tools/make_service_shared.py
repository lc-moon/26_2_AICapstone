# 조원(백엔드 A·프론트 B)에게 줄 공통 자료와 API 응답 예시를 만든다.
#
#   python Tools/make_service_shared.py --estimates data/estimates/estimates_vm.csv
#
# 입력: VM의 data/estimates/estimates.csv 사본, Web/mockup/incheon_inland.js(경계), model_output/cameras.json,
#       data/stations.json(에어코리아 측정소 목록)
# 결과: data/service_shared/ — 새 저장소의 shared/로 옮긴다. 규칙은 docs/서비스_구상.md, Notion 역할 분배 가이드
#
# 격자·보간은 Web/mockup/index.html과 같은 식이다. A는 mock/ingest_estimates.json을 넣었을 때
# mock/frames_latest.json과 같은 값이 나오면 계산이 맞은 것이다.
import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/service_shared"
KST = timezone(timedelta(hours=9))
KX, KY = 111.32 * math.cos(math.radians(37.45)), 110.57     # 도 → km
SX, SY = 1 / KX, 1 / KY                                       # 1km를 도로
MODEL_SWITCH = datetime(2026, 10, 15, 11, 25, 22, tzinfo=KST)  # 이 시각부터 인천 ver (docs/현황.md §7)
LATEST = "2026-10-16T13:20:00+09:00"                          # 예시로 쓰는 회차
GRADES = [(15, "좋음"), (35, "보통"), (75, "나쁨"), (10**9, "매우나쁨")]


def km(a, b):
    return math.hypot((a[0] - b[0]) * KX, (a[1] - b[1]) * KY)


def idw(p, pts):
    """pts: [(lon, lat, 값)]. 가중치 1/(d²+0.05), d는 km"""
    w = [1 / (km(p, q) ** 2 + 0.05) for q in pts]
    return sum(wi * q[2] for wi, q in zip(w, pts)) / sum(w)


def grade(v):
    return next(name for top, name in GRADES if round(v) <= top)


def in_ring(p, ring):
    c, j = False, len(ring) - 1
    for i in range(len(ring)):
        (xi, yi), (xj, yj) = ring[i], ring[j]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


def iso(t):
    return t.isoformat(timespec="seconds")


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


SEED = """
-- estimates.model이 참조하므로 먼저 넣어 둔다. 수치는 교차검증·시험 A의 MAE (docs/현황.md §7)
INSERT INTO model_versions (model, deployed_at, cv_mae, test_mae, note) VALUES
  ('mixed_top60_w-hum', NULL, 3.54, 2.48, '인천+베이징 ver. 10/13~10/15 11:25 서비스'),
  ('incheon_log_top60', '2026-10-15 11:25:22+09', 3.37, 2.46, '인천 ver');
"""

README = """# shared — 공통 자료

A(백엔드)·B(프론트)가 같이 쓰는 파일입니다. 규칙과 할 일은 [AGENTS.md](../AGENTS.md) 또는 [Notion 가이드](https://app.notion.com/p/A-B-3fd4a3212e4f81458cd8f40cff41cad5)를 보세요.
팀장 저장소의 `Tools/make_service_shared.py`가 만든 것이라 **직접 고치지 말고** 필요하면 팀장에게 말해 주세요.

| 파일 | 내용 |
|---|---|
| `schema.sql` | DB 테이블 (PostgreSQL) |
| `boundary.geojson` | 인천 내륙 + 송도 경계(강화·옹진·영종 제외). `kind`가 `area`(범위), `outline`(바깥 테두리 선), `simple`(꼭짓점을 줄인 것, 지도 글자 거르기용) |
| `grid_cells.geojson` | 1km 격자 388칸. `grid_id` 순서가 API `values` 배열의 순서 |
| `cameras.json` | 카메라 20대. `grid_id`가 null인 103번(강화)은 화면 범위 밖이지만 보간 입력으로는 쓴다 |
| `stations.json` | 인천과 인접 경기·서울의 도시대기 측정소. `use_for_compare`는 인천 내륙 안에 있는 곳 |
| `mock/` | API 응답 예시 (아래) |
| `mockup/` | 참고용 화면 시안. `index.html`을 더블클릭하면 열림 (인터넷 필요) |

## mock/

| 파일 | API |
|---|---|
| `grid.json` | `GET /api/grid` |
| `frames_latest.json` | `GET /api/frames/latest` — 낮 (10/16 13:20 실제 추정값) |
| `frames_latest_night.json` | `GET /api/frames/latest` — 밤 (측정소 기준, **값은 가짜**) |
| `frames_2026-10-16.json` | `GET /api/frames?from=&to=` — 10/16 낮 33회차 (실제) |
| `point_inside.json` · `point_outside.json` | `GET /api/point` — 측정소 실측 `pm25`는 **예시값** |
| `health.json` · `health_stale.json` | `GET /api/health` |
| `ingest_estimates.json` | `POST /api/ingest/estimates` 요청 본문 (10/16 13:20 실제) |
| `ingest_2026-10-16.json` | 같은 형식으로 10/16 하루치 (실제). 여러 회차·늦게 오는 값을 시험할 때 |

**A 계산 확인법**: `ingest_estimates.json`을 넣고 계산한 격자값이 `frames_latest.json`의 `values`와 같으면(소수 첫째 자리) 맞습니다.
124번 카메라는 이 회차에 사진이 없어서 빠져 있습니다. 실제로도 이렇게 빠지는 회차가 있습니다.

## 알아둘 것
- 시각은 모두 한국 시간(`+09:00`), 농도는 ㎍/㎥
- 경계와 `gu`(구 이름)는 2026-07-01 행정구역 개편(서해구·검단구·제물포구·영종구)이 반영된 2026-10-01판입니다
- 경계를 화면에 쓰면 출처를 표기합니다: "행정경계: 통계청 SGIS (vuski/admdongkor)" (CC BY 4.0)
- CCTV 사진은 어디에도 내보내지 않습니다 (번호판)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", required=True, help="VM data/estimates/estimates.csv 사본")
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)

    # ── 경계와 격자 ──
    inc = json.loads((ROOT / "Web/mockup/incheon_inland.js").read_text(encoding="utf-8")
                     .split("window.INCHEON = ", 1)[1].rstrip().rstrip(";"))
    rings, gus = inc["rings"], inc["gu"]
    boundary = {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}
    dump(OUT / "boundary.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"kind": "area", "note": "인천 내륙 + 송도 (행정경계 2026-10-01판, 통계청 SGIS). 범위 판정·가림막용"},
         "geometry": boundary},
        {"type": "Feature", "properties": {"kind": "outline", "note": "바깥 테두리 선만. 구 사이 경계는 뺐다"},
         "geometry": {"type": "MultiLineString", "coordinates": inc["edge"]}},
        {"type": "Feature", "properties": {"kind": "simple", "note": "꼭짓점을 줄인 경계(약 150m). 지도 글자 거르기(within)용 — 원본은 느리다"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in inc["simple"]]}}]})

    xs = [x for r in rings for x, _ in r]
    ys = [y for r in rings for _, y in r]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cells, row = [], 0
    lat = y0 + SY / 2
    while lat < y1:
        col, lon = 0, x0 + SX / 2
        while lon < x1:
            gu = next((g for g, r in zip(gus, rings) if in_ring((lon, lat), r)), None)
            if gu:
                cells.append({"grid_id": len(cells), "row": row, "col": col, "gu": gu,
                              "center": [round(lon, 6), round(lat, 6)]})
            col += 1
            lon += SX
        row += 1
        lat += SY

    def square(c):
        x, y = c["center"]
        return [[[round(x + dx * SX / 2, 6), round(y + dy * SY / 2, 6)]
                 for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]]
    dump(OUT / "grid_cells.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": c, "geometry": {"type": "Polygon", "coordinates": square(c)}} for c in cells]})

    def cell_of(p):
        best = min(cells, key=lambda c: km(p, c["center"]))
        return best if km(p, best["center"]) <= 0.75 and any(in_ring(p, r) for r in rings) else None

    # ── 카메라와 측정소 ──
    cams = json.loads((ROOT / "model_output/cameras.json").read_text(encoding="utf-8"))["cameras"]
    for c in cams:
        cell = cell_of((c["lon"], c["lat"]))
        c["grid_id"] = cell["grid_id"] if cell else None   # 103(강화)은 범위 밖이라 None. 보간 입력으로는 쓴다
    dump(OUT / "cameras.json", {"cameras": cams})
    cam_at = {c["camera_id"]: (c["lon"], c["lat"]) for c in cams}

    stations = []
    for s in json.loads((ROOT / "data/stations.json").read_text(encoding="utf-8")):
        lon, lat = float(s["dmY"]), float(s["dmX"])
        if s["mangName"] != "도시대기" or not (x0 - 0.1 < lon < x1 + 0.1 and y0 - 0.1 < lat < y1 + 0.1):
            continue
        sido = next((k for k in ("인천", "경기", "서울") if s["addr"].startswith(k)), s["addr"].split()[0])
        stations.append({"station_name": s["stationName"], "sido": sido, "network": s["mangName"],
                         "lon": lon, "lat": lat, "use_for_compare": any(in_ring((lon, lat), r) for r in rings)})
    dump(OUT / "stations.json", {"stations": stations})

    # ── 추정값 → 회차별 프레임 ──
    rows = [r for r in csv.DictReader(open(args.estimates, encoding="utf-8-sig")) if r["mode"] == "live"]
    by_slot = {}
    for r in rows:
        t = datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        est = datetime.strptime(r["estimated_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        slot = t.replace(minute=t.minute // 20 * 20, second=0)
        by_slot.setdefault(iso(slot), []).append({
            "camera_id": r["camera_id"], "slot": iso(slot), "captured_at": iso(t),
            "pm25_est": float(r["pm25_est"]) if r["pm25_est"] else None, "status": r["status"],
            "humidity": float(r["humidity"]) if r["humidity"] else None,
            "rain_60m": float(r["rain"]) if r["rain"] else None,
            "model": "incheon_log_top60" if est >= MODEL_SWITCH else "mixed_top60_w-hum",
            "estimated_at": iso(est)})

    def frame(slot):
        got = [r for r in by_slot[slot] if r["pm25_est"] is not None]
        pts = [(*cam_at[r["camera_id"]], r["pm25_est"]) for r in got]
        return {"slot": slot, "source": "cctv", "basis_time": None, "n_inputs": len(pts),
                "values": [round(idw(c["center"], pts), 1) for c in cells],
                "updated_at": max(r["estimated_at"] for r in got)}

    mock = OUT / "mock"
    dump(mock / "ingest_estimates.json", {"rows": by_slot[LATEST]})
    dump(mock / "ingest_2026-10-16.json", {"rows": [r for s in sorted(by_slot) if s.startswith(LATEST[:10]) for r in by_slot[s]]})
    latest = frame(LATEST)
    dump(mock / "frames_latest.json", latest)
    day = sorted(s for s in by_slot if s.startswith(LATEST[:10]))
    dump(mock / "frames_2026-10-16.json", {"frames": [{k: f[k] for k in ("slot", "source", "values")}
                                                      for f in map(frame, day)]})

    # 밤 예시: 측정소 값은 가짜. 형태만 보여 준다
    night_pts = [(s["lon"], s["lat"], 10 + i % 7) for i, s in enumerate(stations)]
    dump(mock / "frames_latest_night.json", {
        "slot": "2026-10-16T21:00:00+09:00", "source": "station", "basis_time": "2026-10-16T21:00:00+09:00",
        "n_inputs": len(night_pts), "values": [round(idw(c["center"], night_pts), 1) for c in cells],
        "updated_at": "2026-10-16T21:25:00+09:00"})

    dump(mock / "grid.json", {"cell_size_m": 1000, "count": len(cells),
                              "cells": [{k: c[k] for k in ("grid_id", "row", "col", "center", "gu")} for c in cells],
                              "boundary": boundary})

    p = (126.6395, 37.3935)                     # 송도 센트럴파크
    cell = cell_of(p)
    st = min((s for s in stations if s["use_for_compare"]), key=lambda s: km(p, (s["lon"], s["lat"])))
    v = latest["values"][cell["grid_id"]]
    dump(mock / "point_inside.json", {
        "inside": True, "gu": cell["gu"], "grid_id": cell["grid_id"],
        "estimate": {"pm25": v, "grade": grade(v), "slot": LATEST, "source": "cctv",
                     "nearest_cctv_km": round(min(km(p, q) for q in cam_at.values()), 1)},
        "station": {"name": st["station_name"], "distance_km": round(km(p, (st["lon"], st["lat"])), 1),
                    "pm25": 11, "data_time": "2026-10-16T13:00:00+09:00", "period": "12~13시 평균"}})
    dump(mock / "point_outside.json", {"inside": False})
    dump(mock / "health.json", {"last_slot": LATEST, "last_received_at": latest["updated_at"], "stale": False})
    dump(mock / "health_stale.json", {"last_slot": "2026-10-16T13:20:00+09:00",
                                      "last_received_at": latest["updated_at"], "stale": True})

    # ── DB와 시안 ──
    md = (ROOT / "docs/서비스_구상.md").read_text(encoding="utf-8")
    sql = md.split("```sql\n", 1)[1].split("```", 1)[0]
    (OUT / "schema.sql").write_text("-- 인천 초미세먼지 지도 DB (PostgreSQL)\n" + sql + SEED, encoding="utf-8")
    shutil.copytree(ROOT / "Web/mockup", OUT / "mockup")
    (OUT / "README.md").write_text(README, encoding="utf-8")

    print(f"격자 {len(cells)}칸, 카메라 {len(cams)}대, 측정소 {len(stations)}곳, "
          f"10/16 회차 {len(day)}개 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
