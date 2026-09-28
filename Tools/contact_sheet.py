# 수집된 이미지를 한 장으로 붙여 본다 (공유·검수용)
#
#   python Tools/contact_sheet.py --date 2026-09-28 --time 1200   한 시각의 카메라 전체
#   python Tools/contact_sheet.py --date 2026-09-28 --camera 113  한 카메라의 하루
#
# 결과는 data/contact_sheets/ 에 jpg 한 장으로 저장된다.
# 글자는 OpenCV가 한글을 못 그려서 카메라 번호와 시각(ASCII)만 적는다.
# 번호와 이름의 대응은 Collector/cameras.toml 또는 model_output/cameras.json 참고.
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/images"
OUT_DIR = ROOT / "data/contact_sheets"


def tile(path, w, h, label):
    """이미지 한 장을 정해진 크기로 줄이고 왼쪽 위에 라벨을 얹는다."""
    img = cv2.imread(str(path)) if path and path.exists() else None
    if img is None:
        img = np.full((h, w, 3), 40, np.uint8)          # 없는 프레임은 어두운 칸
        cv2.putText(img, "NO IMAGE", (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 200), 2)
    else:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)  # 라벨이 배경에 묻히지 않도록
    cv2.putText(img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    return img


def grid(tiles, cols):
    """타일을 격자로 쌓는다. 마지막 줄이 모자라면 빈 칸으로 채운다."""
    h, w = tiles[0].shape[:2]
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rows[-1] += [np.full((h, w, 3), 25, np.uint8)] * (cols - len(rows[-1]))
    return np.vstack([np.hstack(r) for r in rows])


def by_time(date, hhmm):
    """그 시각에 찍힌 카메라 전체. 번호 순으로 놓는다."""
    # 수집기가 --once로 만드는 _test 폴더가 섞여 있을 수 있어 숫자 폴더만 고른다
    cams = sorted((d.name for d in IMAGES.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
    if not cams:
        raise SystemExit(f"{IMAGES}에 카메라 폴더가 없습니다")
    tiles = [tile(IMAGES / c / date / f"{hhmm}.jpg", 480, 270, f"{c}  {hhmm[:2]}:{hhmm[2:]}")
             for c in cams]
    return grid(tiles, 5), f"{date}_{hhmm}_all"


def by_camera(date, cam):
    """그 카메라의 하루. 사진이 있는 시각만 시간순으로 놓는다."""
    day = IMAGES / str(cam) / date
    shots = sorted(day.glob("*.jpg")) if day.exists() else []
    if not shots:
        raise SystemExit(f"{day}에 사진이 없습니다")
    tiles = [tile(p, 320, 180, f"{p.stem[:2]}:{p.stem[2:]}") for p in shots]
    return grid(tiles, 6), f"{date}_cam{cam}"


def main():
    ap = argparse.ArgumentParser(description="수집 이미지 컨택트 시트")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--time", help="HHMM — 이 시각의 카메라 전체")
    ap.add_argument("--camera", help="카메라 번호 — 이 카메라의 하루 전체")
    args = ap.parse_args()

    if bool(args.time) == bool(args.camera):
        raise SystemExit("--time 또는 --camera 중 하나만 지정하세요")

    sheet, name = (by_time(args.date, args.time) if args.time
                   else by_camera(args.date, args.camera))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.jpg"
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise SystemExit("jpg 인코딩 실패")
    out.write_bytes(buf.tobytes())

    h, w = sheet.shape[:2]
    print(f"{out}  ({w}x{h}, {out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
