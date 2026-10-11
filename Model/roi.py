# ROI(관심 영역) — 카메라마다 하늘·먼 건물/산 상자를 사람이 그리고, CNN에 넣을 ROI 후보 5개를 정한다.
#
#   python Model/roi.py                    상자 그리기 페이지 만들기 → data/review/roi.html (브라우저로 열어 그린 뒤 [저장])
#   python Model/roi.py --apply 경로.csv    페이지에서 저장한 roi.csv를 Model/roi.csv로 반영
#
# 후보 (2026-10-11 결정, docs/현황.md §7)
#   sky    ① 하늘                    직접 그림
#   far    ② 먼 건물·산               직접 그림
#   both   ③ 하늘+먼 건물·산          ①②를 감싸는 상자 (자동)
#   top3   ④ 위쪽 1/3                모든 카메라 같음 (자동)
#   top60  ⑤ 위쪽 60% (아래 도로 빼기) 모든 카메라 같음 (자동)
# 상자가 "없음"인 카메라는 그 후보에서 사진 전체를 쓴다 — 카메라를 빼면 후보마다 채점 사진이 달라져 비교가 공정하지 않다.
# 좌표는 사진 크기에 대한 비율(0~1)이라 해상도가 바뀐 카메라(10042번, 10/8)에도 그대로 쓴다.
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from baseline import HUMIDITY_MAX, LABELS
from view_check import load_review

ROOT = Path(__file__).resolve().parent.parent
BJ_LABELS = ROOT / "data/labels/labels_beijing.csv"
IMAGES = ROOT / "data/images"
CAMERAS = ROOT / "model_output/cameras.json"
TEMPLATE = Path(__file__).with_name("roi_page.html")
PAGE = ROOT / "data/review/roi.html"
ROI = Path(__file__).with_name("roi.csv")
AUTO = Path(__file__).with_name("roi_auto.csv")     # ⑥ 자동 상자 (roi_auto.py). 연습시험 회차(0·1·2)와 최종(all)마다 다르다
CANDIDATES = ["sky", "far", "both", "top3", "top60"]
FULL = (0.0, 0.0, 1.0, 1.0)


def load_roi():
    """Model/roi.csv → {카메라: {"sky": (x0,y0,x1,y1) 또는 None(없음), "far": ...}}"""
    if not ROI.exists():
        return {}
    out = {}
    for r in pd.read_csv(ROI, dtype={"camera": str}).itertuples():
        out.setdefault(r.camera, {})[r.kind] = None if r.x0 == "none" else tuple(float(v) for v in (r.x0, r.y0, r.x1, r.y1))
    return out


def load_auto():
    """Model/roi_auto.csv → {(카메라, 회차): (x0,y0,x1,y1)}. 회차는 "0"·"1"·"2"·"all" """
    t = pd.read_csv(AUTO, dtype={"camera": str, "fold": str})
    return {(r.camera, r.fold): (r.x0, r.y0, r.x1, r.y1) for r in t.itertuples()}


def box(roi, camera, candidate):
    """카메라·후보의 상자 (비율 좌표). 직접 그린 상자가 없음이면 사진 전체"""
    if candidate == "top3":
        return (0.0, 0.0, 1.0, 1 / 3)
    if candidate == "top60":
        return (0.0, 0.0, 1.0, 0.6)
    b = roi[camera]
    if candidate in ("sky", "far"):
        return b[candidate] or FULL
    drawn = [v for v in (b["sky"], b["far"]) if v]
    if not drawn:
        return FULL
    return (min(v[0] for v in drawn), min(v[1] for v in drawn), max(v[2] for v in drawn), max(v[3] for v in drawn))


def samples():
    """카메라마다 그리기 바탕이 될 맑은 날·탁한 날 사진 (해가 높은 시간, 검수 통과, 비·고습도 아닌 사진 중 PM2.5 최저·최고)"""
    names = {c["camera_id"]: c["name"] for c in json.loads(CAMERAS.read_text(encoding="utf-8"))["cameras"]}
    names["bj"] = "베이징 (외부 데이터)"
    d = pd.concat([pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"camera_id": str}),
                   pd.read_csv(BJ_LABELS, encoding="utf-8-sig", dtype={"camera_id": str})], ignore_index=True)
    d = d[(d.image.map(load_review()) == "keep") & d.pm25.notna() & (d.rain != 1) & (d.humidity < HUMIDITY_MAX) & (d.sun_alt > 25)]
    # 10042번은 해상도가 바뀌기 전(1280×720) 사진으로 그린다. 화각은 같고 비율 좌표라 바뀐 뒤에도 그대로 맞는다
    d = d[~((d.camera_id == "10042") & (d.captured_at >= "2026-10-08 08:41"))]
    cams = []
    for cam, g in d.groupby("camera_id", sort=False):
        lo, hi = g.loc[g.pm25.idxmin()], g.loc[g.pm25.idxmax()]
        w, h = Image.open(IMAGES / lo.image).size          # 상자 모양 경고(가로:세로)를 실제 픽셀로 계산하려고
        cams.append({"cam": cam, "name": names.get(cam, ""), "clear": lo.image, "hazy": hi.image,
                     "clear_pm": int(lo.pm25), "hazy_pm": int(hi.pm25), "w": w, "h": h})
    return sorted(cams, key=lambda c: (c["cam"] == "bj", int(c["cam"]) if c["cam"].isdigit() else 0))


def build_page():
    saved = {cam: {k: ("none" if v is None else list(v)) for k, v in kinds.items()} for cam, kinds in load_roi().items()}
    page = TEMPLATE.read_text(encoding="utf-8").replace("__CAMS__", json.dumps(samples(), ensure_ascii=False)).replace(
        "__SAVED__", json.dumps(saved, ensure_ascii=False))
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page, encoding="utf-8")
    print(f"상자 그리기 페이지: {PAGE}\n  브라우저로 열어 그린 뒤 [저장] → python Model/roi.py --apply 내려받기\\roi.csv")


def apply(path):
    t = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if list(t.columns) != ["camera", "kind", "x0", "y0", "x1", "y1"]:
        sys.exit(f"roi.csv 칸이 다릅니다: {list(t.columns)}")
    blank = t.x0.isna()
    if blank.any():
        print(f"  ※ 아직 안 그린 상자 {int(blank.sum())}개: " + ", ".join(t[blank].camera + "/" + t[blank].kind))
    t = t[~blank]
    drawn = t[t.x0 != "none"]
    xy = drawn[["x0", "y0", "x1", "y1"]].astype(float)
    if ((xy < 0) | (xy > 1)).any().any() or (xy.x0 >= xy.x1).any() or (xy.y0 >= xy.y1).any():
        sys.exit("좌표가 0~1 밖이거나 뒤집힌 상자가 있습니다")
    t.to_csv(ROI, index=False)
    print(f"반영: {ROI} — 상자 {len(drawn)}개, 없음 {int((t.x0 == 'none').sum())}개, 카메라 {t.camera.nunique()}대")


def main():
    ap = argparse.ArgumentParser(description="ROI 상자 그리기 페이지 / 결과 반영")
    ap.add_argument("--apply", help="페이지에서 저장한 roi.csv")
    args = ap.parse_args()
    if args.apply:
        apply(args.apply)
    else:
        build_page()


if __name__ == "__main__":
    main()
