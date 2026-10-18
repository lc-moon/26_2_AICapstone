# 서비스 화면 시안(Web/mockup/)이 쓰는 인천 내륙 경계를 만든다.
#
#   python Tools/make_mockup_boundary.py
#
# 입력: 통계청 2013 시군구 경계 (southkorea/southkorea-maps, 약 55MB). data/mockup/에 한 번 받아 두고 다시 받지 않는다
# 결과: Web/mockup/incheon_inland.js — 구별 폴리곤(가림막·클릭 판정용)과 바깥 테두리 선(구 사이 경계는 뺀다)
#
# 범위는 인천 내륙 + 송도 (docs/서비스_구상.md §2). 강화·옹진은 통째로, 중구의 영종·용유와 작은 섬은 뺀다.
# 2013 경계라 송도 매립지 일부가 빠져 있고 2026 행정구역 개편 전 구 이름이다. 실서비스에는 최신 경계를 쓸 것
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC_URL = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo.json"
SRC = ROOT / "data/mockup/muni2013.json"
OUT = ROOT / "Web/mockup/incheon_inland.js"

KEEP = {"23010": "중구", "23020": "동구", "23030": "미추홀구", "23040": "연수구", "23050": "남동구",
        "23060": "부평구", "23070": "계양구", "23080": "서구"}   # 23030은 2013 자료에서 '남구'
MIN_LON = 126.585    # 이보다 서쪽에 중심이 있는 조각은 영종·용유
MIN_AREA = 2e-4      # 도(deg²). 약 2km²보다 작은 섬은 뺀다


def area(ring):
    return abs(sum((ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1])
                   for i, j in zip(range(len(ring)), [len(ring) - 1] + list(range(len(ring) - 1)))) / 2)


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        SRC.write_bytes(requests.get(SRC_URL, timeout=120).content)
    feats = json.loads(SRC.read_text(encoding="utf-8"))["features"]

    parts = []   # (구 이름, 바깥 고리)
    for f in feats:
        code = str(f["properties"]["code"])
        if code not in KEEP:
            continue
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            ring = [[round(x, 5), round(y, 5)] for x, y in poly[0]]
            if sum(p[0] for p in ring) / len(ring) > MIN_LON and area(ring) > MIN_AREA:
                parts.append((KEEP[code], ring))

    # 바깥 테두리: 두 구가 함께 쓰는 변은 두 번 나오므로, 한 번만 나오는 변만 이어 붙인다
    def key(a, b):
        return tuple(sorted((tuple(a), tuple(b))))
    count = {}
    for _, r in parts:
        for a, b in zip(r, r[1:]):
            count[key(a, b)] = count.get(key(a, b), 0) + 1
    edge = []
    for _, r in parts:
        line = []
        for a, b in zip(r, r[1:]):
            if count[key(a, b)] == 1:
                line = line or [a]
                line.append(b)
            elif line:
                edge.append(line)
                line = []
        if len(line) > 1:
            edge.append(line)

    data = {"gu": [g for g, _ in parts], "rings": [r for _, r in parts], "edge": edge}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// Tools/make_mockup_boundary.py가 만든 파일. 통계청 2013 시군구 경계(southkorea-maps)에서 인천 내륙만\n"
                   "window.INCHEON = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"구 조각 {len(parts)}개, 테두리 선 {len(edge)}개 → {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
