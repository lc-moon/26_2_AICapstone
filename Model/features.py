# labels.csv의 사진마다 베이스라인용 이미지 통계를 계산해 features.csv를 만든다.
#
#   python Model/features.py
#
# 결과: data/model/features.csv (사진 1장 = 1행, image 열로 labels.csv와 붙인다)
# 이미 계산한 사진은 건너뛰므로 새 사진이 쌓인 뒤 다시 돌리면 그만큼만 계산한다.
#
# 특징 (docs/현황.md §7 베이스라인)
#   ROI를 정하기 전이라 사진 전체(all)와 위쪽 1/3 띠(top)에서 같은 값을 따로 잰다.
#   하늘·원경은 대부분 위쪽 1/3에 있고, 아래쪽은 도로·차량·자막이다.
#   해상도가 640×360·960×540·1280×720으로 섞여 있어 모두 640×360으로 줄인 뒤 잰다.
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/labels/labels.csv"
IMAGES = ROOT / "data/images"
OUT = ROOT / "data/model/features.csv"
SIZE = (640, 360)
DARK_PATCH = 15      # 암채널 최소값 필터 크기 (He et al. 2009의 기본값)


def stats(img, prefix):
    """BGR 0~1 영역 하나의 통계"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = cv2.erode(img.min(axis=2), np.ones((DARK_PATCH, DARK_PATCH), np.uint8))
    mean = gray.mean()
    return {
        f"{prefix}_bright": mean,
        f"{prefix}_contrast": gray.std() / mean if mean > 0 else 0.0,     # RMS 대비
        f"{prefix}_sat": hsv[..., 1].mean(),
        f"{prefix}_dark": dark.mean(),
        f"{prefix}_sharp": cv2.Laplacian(gray, cv2.CV_32F).var(),
        f"{prefix}_blue_red": img[..., 0].mean() / max(img[..., 2].mean(), 1e-6),
    }


def measure(image):
    img = cv2.imread(str(IMAGES / image))
    if img is None:
        return image, None
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    h = SIZE[1]
    top, mid = img[: h // 3], img[h // 3: 2 * h // 3]
    row = {**stats(img, "all"), **stats(top, "top")}
    # 하늘-원경 밝기차 대용: 위쪽 띠와 가운데 띠의 밝기 차. 탁할수록 원경이 하늘 밝기에 가까워진다
    row["top_mid_diff"] = row["top_bright"] - cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).mean()
    return image, row


def main():
    images = [r["image"] for r in csv.DictReader(open(LABELS, encoding="utf-8-sig"))]
    done, old = set(), []
    if OUT.exists():
        old = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
        done = {r["image"] for r in old}
    todo = [i for i in images if i not in done]
    print(f"사진 {len(images)}장 중 새로 계산 {len(todo)}장")

    rows, missing = [], []
    with ProcessPoolExecutor() as ex:
        for n, (image, row) in enumerate(ex.map(measure, todo, chunksize=32), 1):
            if row is None:
                missing.append(image)
            else:
                rows.append({"image": image, **{k: round(float(v), 6) for k, v in row.items()}})
            if n % 2000 == 0:
                print(f"  {n}/{len(todo)}")
    if missing:
        print(f"파일 없음·읽기 실패 {len(missing)}장 (예: {missing[0]})")
    if not rows and not old:
        sys.exit("계산된 사진이 없습니다")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image"] + [k for k in (rows or old)[0] if k != "image"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old + rows)
    print(f"{OUT}  {len(old) + len(rows)}행")


if __name__ == "__main__":
    main()
