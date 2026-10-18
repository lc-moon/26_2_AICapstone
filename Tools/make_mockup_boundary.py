# 서비스 시안(Web/mockup/)과 조원 공통 자료가 쓰는 인천 내륙 경계를 만든다.
#
#   python Tools/make_mockup_boundary.py
#
# 입력: 행정동 경계 2026-10-01판 (vuski/admdongkor, 통계청 SGIS 원자료, CC BY 4.0, 약 35MB).
#       data/mockup/에 한 번 받아 두고 다시 받지 않는다
# 결과: Web/mockup/incheon_inland.js — 구별 폴리곤(가림막·클릭 판정용), 줄인 폴리곤(글자 거르기용),
#       바깥 테두리 선(구 사이 경계는 뺀다)
#
# 범위는 인천 내륙 + 송도 (docs/서비스_구상.md §2). 2026-07-01 개편 후 구 기준으로 강화군·옹진군·영종구를 빼고,
# 남은 구의 작은 섬도 뺀다. 출처 표기: "행정경계: 통계청 SGIS (vuski/admdongkor)"
import json
from pathlib import Path

import requests
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
SRC_URL = "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20261001/HangJeongDong_ver20261001.geojson"
SRC = ROOT / "data/mockup/HangJeongDong_ver20261001.geojson"
OUT = ROOT / "Web/mockup/incheon_inland.js"

KEEP = ["검단구", "계양구", "서해구", "부평구", "제물포구", "미추홀구", "남동구", "연수구"]   # 강화군·옹진군·영종구 제외
MIN_AREA = 2e-4      # 도(deg²). 약 2km²보다 작은 섬은 뺀다
SIMPLE_TOL = 0.0015  # 도. 약 150m. 글자 거르기용으로만 쓰므로 거칠어도 된다


def parts(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def coords(ring):
    return [[round(x, 5), round(y, 5)] for x, y in ring.coords]


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        SRC.write_bytes(requests.get(SRC_URL, timeout=300).content)
    feats = [f for f in json.loads(SRC.read_text(encoding="utf-8"))["features"]
             if f["properties"]["sidonm"] == "인천광역시"]

    gus, rings, kept = [], [], []
    for gu in KEEP:
        dongs = [shape(f["geometry"]) for f in feats if f["properties"]["sggnm"] == gu]
        assert dongs, f"{gu}가 경계 자료에 없다"
        for p in parts(unary_union(dongs)):
            if p.area > MIN_AREA:
                gus.append(gu)
                rings.append(coords(p.exterior))
                kept.append(p)

    inland = unary_union(kept)
    simple = inland.simplify(SIMPLE_TOL)
    edge = [coords(r) for p in parts(inland) for r in [p.exterior, *p.interiors]]
    data = {"gu": gus, "rings": rings, "simple": [coords(p.exterior) for p in parts(simple)], "edge": edge}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// Tools/make_mockup_boundary.py가 만든 파일. 행정경계: 통계청 SGIS (vuski/admdongkor 2026-10-01판), 인천 내륙만\n"
                   "window.INCHEON = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"구 조각 {len(rings)}개 (꼭짓점 {sum(map(len, rings))} → 줄인 것 {sum(map(len, data['simple']))}), "
          f"테두리 선 {len(edge)}개 → {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
