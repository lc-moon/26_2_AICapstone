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
#
# 행정경계에는 아직 어느 동에도 들지 않은 매립지(경인항 부근, 인천항 부두, 송도 남쪽 등)가 빠져 있다.
# 지도(OSM)에는 땅으로 그려져 사용자가 누르게 되므로, OSM 바다가 아닌 땅 중 어느 행정동에도 없고
# 인천 내륙에 붙어 있는 조각을 가장 가까운 구에 붙인다 (10/26 결정). OSM 바다는 OpenFreeMap 벡터 타일의
# water 층(class=ocean)에서 한 번 받아 data/mockup/에 둔다. 출처 표기: "© OpenStreetMap" (이미 지도에 있음)
import json
import math
from pathlib import Path

import requests
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
SRC_URL = "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20261001/HangJeongDong_ver20261001.geojson"
SRC = ROOT / "data/mockup/HangJeongDong_ver20261001.geojson"
OCEAN = ROOT / "data/mockup/osm_ocean_z13.geojson"
OUT = ROOT / "Web/mockup/incheon_inland.js"

KEEP = ["검단구", "계양구", "서해구", "부평구", "제물포구", "미추홀구", "남동구", "연수구"]   # 강화군·옹진군·영종구 제외
MIN_AREA = 2e-4      # 도(deg²). 약 2km²보다 작은 섬은 뺀다
SIMPLE_TOL = 0.0015  # 도. 약 150m. 글자 거르기용으로만 쓰므로 거칠어도 된다
BBOX = (126.50, 37.30, 126.82, 37.66)   # OSM 바다를 받을 범위 (인천 내륙보다 조금 넓게)
ZOOM = 13                               # 타일 한 칸 약 4km, 좌표 해상도 약 1m
FREE_MIN = 1e-6      # 도². 약 0.01km²보다 작은 매립지 조각은 붙이지 않는다
FREE_TOUCH = 0.003   # 도. 약 300m 안에서 내륙에 닿는 조각만 붙인다


def parts(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def _varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _fields(b):
    """protobuf 메시지를 (번호, 값) 목록으로. 값은 varint면 int, 길이 지정이면 bytes"""
    i, out = 0, []
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"protobuf wire type {wt}")
        out.append((num, v))
    return out


def _packed(b):
    i, out = 0, []
    while i < len(b):
        v, i = _varint(b, i)
        out.append(v)
    return out


def ocean_polygons(tile, x, y, z):
    """Mapbox 벡터 타일(pbf)의 water 층에서 class=ocean 다각형을 경위도로. 외부 패키지 없이 읽는다"""
    polys = []
    for num, layer in _fields(tile):
        if num != 3:
            continue
        lf = _fields(layer)
        if next(v for n, v in lf if n == 1).decode() != "water":
            continue
        keys = [v.decode() for n, v in lf if n == 3]
        vals = [next((sv.decode() for sn, sv in _fields(v) if sn == 1), None) for n, v in lf if n == 4]
        ext = next((v for n, v in lf if n == 5), 4096)
        for n, feat in lf:
            if n != 2:
                continue
            ff = _fields(feat)
            tags = _packed(next((v for k, v in ff if k == 2), b""))
            props = {keys[tags[j]]: vals[tags[j + 1]] for j in range(0, len(tags), 2)}
            if props.get("class") != "ocean" or next((v for k, v in ff if k == 3), 0) != 3:
                continue
            geo, rings, ring, cx, cy, j = _packed(next(v for k, v in ff if k == 4)), [], [], 0, 0, 0
            while j < len(geo):
                cmd, cnt = geo[j] & 7, geo[j] >> 3
                j += 1
                if cmd == 7:
                    rings.append(ring)
                    ring = []
                    continue
                for _ in range(cnt):
                    dx, dy = geo[j], geo[j + 1]
                    j += 2
                    cx += (dx >> 1) ^ -(dx & 1)
                    cy += (dy >> 1) ^ -(dy & 1)
                    if cmd == 1:
                        ring = []
                    ring.append(((x + cx / ext) / 2 ** z * 360 - 180,
                                 math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + cy / ext) / 2 ** z))))))
            # 타일 좌표(y 아래로)에서 넓이가 양수인 고리가 바깥 고리, 음수는 구멍
            shell = None
            for r in rings:
                if len(r) < 3:
                    continue
                area = sum(r[k - 1][0] * r[k][1] - r[k][0] * r[k - 1][1] for k in range(len(r)))
                if area < 0:            # 경위도로 바꾸면 y가 뒤집혀 부호도 뒤집힌다
                    if shell:
                        polys.append(Polygon(*shell))
                    shell = (r, [])
                elif shell:
                    shell[1].append(r)
            if shell:
                polys.append(Polygon(*shell))
    return polys


def load_ocean():
    if OCEAN.exists():
        return unary_union([shape(f["geometry"]) for f in json.loads(OCEAN.read_text(encoding="utf-8"))["features"]])
    url = requests.get("https://tiles.openfreemap.org/planet", timeout=60).json()["tiles"][0]
    tx = lambda lon: int((lon + 180) / 360 * 2 ** ZOOM)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** ZOOM)
    polys = []
    for x in range(tx(BBOX[0]), tx(BBOX[2]) + 1):
        for y in range(ty(BBOX[3]), ty(BBOX[1]) + 1):
            r = requests.get(url.format(z=ZOOM, x=x, y=y), timeout=60)
            r.raise_for_status()
            polys += ocean_polygons(r.content, x, y, ZOOM)
    ocean = unary_union([p.buffer(0) for p in polys])
    OCEAN.write_text(json.dumps({"type": "FeatureCollection", "source": url, "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in parts(ocean)]}), encoding="utf-8")
    return ocean


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

    # 어느 행정동에도 없는 육지(매립지) 중 내륙에 붙은 조각을 가장 가까운 구에 붙인다
    region = box(*BBOX)
    adm = unary_union([g for f in json.loads(SRC.read_text(encoding="utf-8"))["features"]
                       for g in [shape(f["geometry"]).buffer(0)] if g.intersects(region)])
    free = region.difference(load_ocean()).difference(adm)
    inland_parts = list(kept)
    inland0 = unary_union(kept)
    added = [p for p in parts(free) if p.area > FREE_MIN and p.distance(inland0) < FREE_TOUCH]
    for p in added:                     # 구 조각 하나로 따로 넣는다 (gu는 가장 가까운 구)
        gus.append(gus[min(range(len(inland_parts)), key=lambda k: inland_parts[k].distance(p))])
        rings.append(coords(p.exterior))
        kept.append(p)
    print(f"행정경계 밖 매립지 {len(added)}조각 {sum(p.area for p in added) * 111.32 * 0.7944 * 110.57:.1f}km² 추가")

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
