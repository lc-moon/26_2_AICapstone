# 10/1 프로토타입용 데이터 두 개를 만든다.
#
#   model_output/cameras.json    카메라 20대의 번호·이름·좌표 (고정값. 백엔드가 격자 매핑에 쓴다)
#   model_output/values.json     2026-09-22 12:00 시점의 카메라별 추정 농도 (가상값)
#
# 카메라 번호·이름·좌표는 실제 값이고, 농도만 가상이다.
# 실행: python Tools/make_prototype_data.py
import json
import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model_output"
SLOT = "2026-09-22T12:00:00+09:00"


def load_cameras():
    """cameras.toml의 번호 + url_cache.json의 이름·좌표."""
    cfg = tomllib.loads((ROOT / "Collector/cameras.toml").read_text(encoding="utf-8-sig"))
    cache = json.loads((ROOT / "Collector/url_cache.json").read_text(encoding="utf-8"))
    out = []
    for cam in cfg["cameras"]:
        if not cam["enabled"]:
            continue
        info = cache[str(cam["id"])]
        out.append({"camera_id": str(cam["id"]), "name": info["name"],
                    "lat": round(float(info["lat"]), 6), "lon": round(float(info["lon"]), 6)})
    return out


def mock_pm25(cams):
    """가상 농도. '좋음'(≤15)과 '보통'(16~35)에 고루 퍼지게 만든다.

    무작위로 흩뿌리면 지도에서 옆 칸끼리 색이 튀어 어색하므로,
    서쪽 해안이 낮고 동쪽 내륙(남동공단 쪽)이 높은 완만한 기울기를 준다.
    값 자체에 의미는 없고 색이 골고루 칠해지는지 보기 위한 것이다.
    """
    # 남동공단 부근을 '높은 곳'으로 두고 거기서 멀어질수록 낮아지게 한다.
    # 가까운 카메라끼리 값이 비슷해야 지도에서 옆 격자끼리 색이 튀지 않는다.
    # 실제 거리에서 나온 값을 20대의 최소·최대로 정규화해 좋음·보통에 고루 퍼지게 맞춘다.
    src_lat, src_lon = 37.4225, 126.7094          # 남동공단입구사거리
    dist = {c["camera_id"]: math.hypot(c["lat"] - src_lat,
                                       (c["lon"] - src_lon) * math.cos(math.radians(src_lat)))
            for c in cams}
    # 거리 '순위'로 값을 배분한다. 거리 값을 그대로 쓰면 멀리 떨어진 카메라 한 대가
    # 정규화를 먹어버려 나머지 19대가 한 가지 색이 된다.
    near_first = sorted(dist, key=dist.get)
    last = len(near_first) - 1
    return {cid: round(30.0 - 21.0 * (i / last) + (int(cid) % 5) * 0.3 - 0.6, 1)
            for i, cid in enumerate(near_first)}


def main():
    cams = load_cameras()
    pm25 = mock_pm25(cams)
    OUT.mkdir(exist_ok=True)

    (OUT / "cameras.json").write_text(json.dumps(
        {"cameras": cams}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (OUT / "values.json").write_text(json.dumps({
        "time": SLOT,
        "unit": "ug/m3",
        "note": "10/1 프로토타입용 가상값. 카메라 번호는 실제, 농도는 가상",
        "data": [{"camera_id": c["camera_id"], "pm25": pm25[c["camera_id"]]} for c in cams],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    vals = sorted(pm25.values())
    good = sum(v <= 15 for v in vals)
    print(f"model_output/cameras.json  카메라 {len(cams)}대")
    print(f"model_output/values.json   {SLOT}")
    print(f"  농도 {vals[0]} ~ {vals[-1]} ㎍/㎥ · 좋음 {good}대 / 보통 {len(vals) - good}대")


if __name__ == "__main__":
    main()
