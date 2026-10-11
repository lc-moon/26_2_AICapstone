# Grad-CAM — 학습한 CNN이 PM2.5를 부를 때 화면 어디를 봤는지 색칠한다 (확인·발표용).
#
#   python Model/gradcam.py                             1단계(글자 가림) 두 버전
#   python Model/gradcam.py --runs incheon_log_masked   고른 모델만
#
# 카메라마다 개발 날짜 사진 30장의 지도를 평균해, 맑은 날 사진 위에 겹친다(빨강 = 많이 봄).
# 비교용으로 사람이 그린 상자(파랑 = 하늘, 주황 = 먼 건물·산)와 자동 상자(초록)를 같이 그린다.
# 상자를 고르는 데는 쓰지 않는다 — 최종 모델은 개발 날짜 전체의 정답을 보고 배웠으므로, 이 지도로 고른 상자를
# 같은 날짜로 채점하면 점수가 부풀려진다(2026-10-11 결정). 시험지 A·B 사진은 쓰지 않는다.
# GPU를 쓰므로 cnn.py 학습과 동시에 돌리지 않는다.
import argparse
import json
import sys

import cv2
import numpy as np
import torch

import cnn
from roi import box as roi_box, load_auto, load_roi

PER_CAMERA = 30
SHEETS = cnn.ROOT / "data/review"


def cam_maps(model, arr, rows):
    """사진들 → Grad-CAM 지도 (장 × 12 × 12, 장마다 최댓값 1로)"""
    acts, grads = {}, {}
    layer = model.stages[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = []
    for i in range(0, len(rows), 8):
        x = cnn.batch(arr, rows[i:i + 8]).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()
        a, g = acts["a"].detach(), grads["g"].detach()
        cam = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1))          # 예측을 올리는 쪽으로 기여한 곳
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
        out.append(cam.float().cpu().numpy())
    h1.remove()
    h2.remove()
    return np.concatenate(out)


def draw(im, b, color, w, h):
    cv2.rectangle(im, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w) - 1, int(b[3] * h) - 1), color, max(2, w // 300))


def main():
    ap = argparse.ArgumentParser(description="학습한 CNN이 본 곳 (Grad-CAM)")
    ap.add_argument("--runs", nargs="+", default=["incheon_log_masked", "mixed_masked"])
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    d = cnn.load()
    rois, auto = load_roi(), load_auto()
    dev = d[d.part.isin(["f0", "f1", "f2", "b0", "b1", "b2"])]
    for run in args.runs:
        info = json.loads((cnn.OUT / run / "info.json").read_text(encoding="utf-8"))
        arr = cnn.inputs(d, info["input"])["all"]
        model = cnn.make_model(pretrained=False)
        model.load_state_dict(torch.load(cnn.OUT / run / "model.pt", map_location=cnn.DEV))
        model.eval()
        tiles = []
        for cam, g in dev.groupby("camera_id", sort=False):
            rows = g.sample(min(PER_CAMERA, len(g)), random_state=0).index.values
            m = cam_maps(model, arr, rows).mean(0)
            ref = g.loc[g.pm25.idxmin()]
            im = cv2.imread(str(cnn.IMAGES / ref.image))
            tm = cnn._mask(ref.scene)
            if tm is not None:
                im[tm] = cnn.FILL
            h, w = im.shape[:2]
            heat = cv2.resize(m / max(m.max(), 1e-8), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            im = (0.55 * im + 0.45 * color).astype(np.uint8)
            if cam in rois:
                draw(im, roi_box(rois, cam, "sky"), (255, 140, 30), w, h)
                draw(im, roi_box(rois, cam, "far"), (0, 140, 255), w, h)
            draw(im, auto[(cam, "all")], (0, 220, 0), w, h)
            im = cv2.resize(im, (480, 270) if cam != "bj" else (152, 270))
            if cam == "bj":
                im = np.hstack([im, np.zeros((270, 328, 3), np.uint8)])
            cv2.putText(im, cam, (5, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(im)
            print(f"  {run} {cam}", flush=True)
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
        for k in range(0, len(tiles), 9):
            part = tiles[k:k + 9]
            part += [np.zeros_like(tiles[0])] * (-len(part) % 3)
            cv2.imwrite(str(SHEETS / f"gradcam_{run}_{k // 9}.jpg"),
                        np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        del model
        torch.cuda.empty_cache()
    print(f"확인용: {SHEETS}\\gradcam_*.jpg (빨강 = 많이 봄, 파랑 상자 = 하늘, 주황 = 먼 건물·산, 초록 = 자동)")


if __name__ == "__main__":
    sys.exit(main())
