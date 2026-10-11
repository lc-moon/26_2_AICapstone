# 자동 ROI — "먼지 민감도 지도"로 카메라마다 PM2.5에 따라 가장 많이 흐려지는 곳에 상자를 잡는다 (ROI 후보 ⑥).
#
#   python Model/roi_auto.py      지도·상자 계산 → Model/roi_auto.csv, 확인용 data/review/roi_auto_*.jpg
#
# 지도: 사진을 30칸 × 17칸으로 나누고, 칸마다 윤곽 세기(밝기 기울기 평균)를 잰다. 수백 장에 걸쳐 칸의 윤곽 세기와 PM2.5의
#       순위 상관(스피어만)을 구해, 음수가 클수록(탁할수록 흐려질수록) 민감한 칸으로 본다. 해가 낮은 사진(15° 미만)과 글자 자리는 뺀다.
# 상자: 폭 1/4 이상, 가로:세로 1:3~3:1인 직사각형 중 민감도 평균이 가장 큰 것.
# 점수가 부풀려지지 않게 연습시험 회차(0·1·2)마다 그 회차의 공부용 날짜로만 지도를 만든다(채점 날짜는 안 씀).
# 최종 모델용(fold = all)은 개발 날짜 전체로. 시험지 A·B 날짜는 어디에도 쓰지 않는다 (2026-10-11 결정).
# 베이징도 같은 방식(베이징 학습 날짜의 3묶음).
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cnn
import text_mask

OUT = Path(__file__).with_name("roi_auto.csv")
SHEETS = cnn.ROOT / "data/review"
GX, GY = 30, 17                     # 칸 수 (가로, 세로)
MIN_SUN = 15
MIN_W = GX // 4                     # 폭 1/4 이상


def cell_edges(path, scene):
    """사진 → 칸마다 윤곽 세기 (GY × GX). 글자 자리는 NaN"""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = text_mask.load(scene)
    keep = np.ones_like(mag) if m is None else (~m).astype(np.float32)
    # 칸마다 (글자 자리를 뺀) 평균 = 칸 합 / 칸 안 남은 픽셀 수. 면적 평균 축소로 한 번에 계산
    s = cv2.resize(mag * keep, (GX, GY), interpolation=cv2.INTER_AREA)
    k = cv2.resize(keep, (GX, GY), interpolation=cv2.INTER_AREA)
    return np.where(k > 0.5, s / np.maximum(k, 1e-6), np.nan).astype(np.float32)


def sensitivity(E, pm):
    """칸마다 -스피어만 상관 (탁할수록 흐려지면 양수)"""
    s = np.zeros((GY, GX), np.float32)
    for i in range(GY):
        for j in range(GX):
            v = E[:, i, j]
            ok = np.isfinite(v)
            if ok.sum() > 30:
                s[i, j] = -spearmanr(v[ok], pm[ok]).statistic
    return np.nan_to_num(s)


def best_box(s, W, H):
    """민감도 평균이 가장 큰 직사각형 (칸 단위) → 비율 좌표. W·H는 사진 크기 (가로:세로 비율 조건을 실제 픽셀로 따지려고)"""
    ii = np.pad(s, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, arg = -9.0, None
    for y0 in range(GY):
        for y1 in range(y0 + 1, GY + 1):
            for x0 in range(GX):
                for x1 in range(x0 + MIN_W, GX + 1):
                    bw, bh = (x1 - x0) / GX * W, (y1 - y0) / GY * H
                    if not (1 / 3 <= bw / bh <= 3):
                        continue
                    v = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((y1 - y0) * (x1 - x0))
                    if v > best:
                        best, arg = v, (x0, y0, x1, y1)
    x0, y0, x1, y1 = arg
    return (x0 / GX, y0 / GY, x1 / GX, y1 / GY), best


def main():
    d = cnn.load()
    d = d[d.part.isin(["f0", "f1", "f2", "b0", "b1", "b2"]) & (d.sun_alt >= MIN_SUN)]
    rows, tiles = [], []
    for cam, g in d.groupby("camera_id", sort=False):
        E = np.stack([cell_edges(cnn.IMAGES / p, s) for p, s in zip(g.image, g.scene)])
        W, H = (540, 960) if cam == "bj" else (16, 9)
        pm = g.pm25.values
        num = g.part.str[1].values                     # 회차 번호 0·1·2
        maps = {}
        for fold in ("0", "1", "2", "all"):
            use = num != fold if fold != "all" else np.ones(len(g), bool)
            s = sensitivity(E[use], pm[use])
            box, score = best_box(s, W, H)
            maps[fold] = (s, box)
            rows.append({"camera": cam, "fold": fold, "x0": round(box[0], 3), "y0": round(box[1], 3),
                         "x1": round(box[2], 3), "y1": round(box[3], 3), "score": round(float(score), 3), "photos": int(use.sum())})
        s, box = maps["all"]
        print(f"  {cam:6} 사진 {len(g):5}  상자 {box[0]:.2f}~{box[2]:.2f} × {box[1]:.2f}~{box[3]:.2f}  민감도 {rows[-1]['score']:.2f}", flush=True)
        # 확인용: 맑은 날 사진 위에 지도(빨강 = 민감) + 최종 상자(초록) + 회차별 상자(노랑)
        ref = g.loc[g.pm25.idxmin()].image
        im = cv2.imread(str(cnn.IMAGES / ref))
        h, w = im.shape[:2]
        # 먼지가 끼면 화면 전체가 조금씩 흐려져 지도가 거의 다 양수다. 카메라 안에서 상위 25% 칸만 진하게 보이게 순위로 칠한다
        lo, hi = np.quantile(s, 0.75), s.max()
        heat = cv2.resize(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1), (w, h), interpolation=cv2.INTER_NEAREST)
        im = (im * (1 - 0.5 * heat[..., None]) + np.array([0, 0, 255]) * 0.5 * heat[..., None]).astype(np.uint8)
        for f in ("0", "1", "2"):
            b = maps[f][1]
            cv2.rectangle(im, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w) - 1, int(b[3] * h) - 1), (0, 255, 255), max(1, w // 400))
        cv2.rectangle(im, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w) - 1, int(box[3] * h) - 1), (0, 255, 0), max(2, w // 200))
        im = cv2.resize(im, (480, 270) if cam != "bj" else (152, 270))
        if cam == "bj":
            im = np.hstack([im, np.zeros((270, 328, 3), np.uint8)])
        cv2.putText(im, cam, (5, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(im)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    for k in range(0, len(tiles), 9):
        part = tiles[k:k + 9]
        part += [np.zeros_like(tiles[0])] * (-len(part) % 3)
        cv2.imwrite(str(SHEETS / f"roi_auto_{k // 9}.jpg"), np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"저장: {OUT}\n확인용: {SHEETS}\\roi_auto_*.jpg (빨강 = 그 카메라에서 탁할수록 가장 많이 흐려지는 상위 25% 칸, 초록 = 최종 상자, 노랑 = 회차별 상자)")


if __name__ == "__main__":
    sys.exit(main())
