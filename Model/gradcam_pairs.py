# 전시용 Grad-CAM 그림 (M6) — 서비스 모델이 같은 카메라의 맑은 날·탁한 날에 화면 어디를 봤는지 나란히 그린다.
#
#   python Model/gradcam_pairs.py              → data/review/m6/{카메라}.png (20대) + data/review/m6/index.html (한눈에 고르기용)
#   python Model/gradcam_pairs.py --run 모델    다른 통합 모델로 (기본: 서비스 모델 ⓑ'+습도 물리식)
#
# 사진 고르기: 시험 기간(10/20~11/1, 모델이 배우지 않은 날)에서 카메라마다 실측이 가장 높은 사진을 "탁한 날"로 고르고,
# 그 시각 ±1시간 안의 다른 날 사진 중 실측이 가장 낮은 것을 "맑은 날"로 고른다(해 높이·그림자를 비슷하게).
# 그림: 위쪽 60%(모델이 보는 곳)에 지도를 겹치고, 아래 40%는 서비스 화면과 같은 굵은 모자이크(번호판)로 가린다.
# 화면 위 글자는 모델이 본 대로 평균색으로 칠한다. 빨강 = 예측을 올리는 쪽으로 많이 본 곳.
# **밖에 쓰기 전에 위쪽 60%에 읽히는 번호판이 없는지 눈으로 확인한다**(docs/전시_준비.md 사진 규칙). data/라 저장소에는 올리지 않는다.
# GPU를 쓰므로 학습과 동시에 돌리지 않는다.
import argparse
import base64
import json
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cnn  # noqa: E402
from gradcam import cam_maps  # noqa: E402
from roi import box as roi_box  # noqa: E402

SERVICE = "incheon_log_top60_h-phys_1019_lr3e-05"
OUT = cnn.ROOT / "data/review/m6"
MOSAIC = 24        # 640px 기준 블록 크기 (Estimator/estimate.py display_copy와 같게)
plt.rcParams["font.family"] = "Malgun Gothic"            # 윈도우 한글 글꼴 (이 PC에서만 돌린다)
plt.rcParams["axes.unicode_minus"] = False


def pick(test):
    """카메라마다 (맑은 날 행, 탁한 날 행)"""
    out = {}
    for c, g in test.groupby("camera_id"):
        hazy = g.loc[g.pm25.idxmax()]
        tod = lambda s: s.str[11:13].astype(int) * 60 + s.str[14:16].astype(int)   # noqa: E731
        t0 = int(hazy.captured_at[11:13]) * 60 + int(hazy.captured_at[14:16])
        near = g[(abs(tod(g.captured_at) - t0) <= 60) & (g.captured_at.str[:10] != hazy.captured_at[:10])]
        if len(near):
            out[c] = (near.loc[near.pm25.idxmin()], hazy)
    return out


def picture(row, heat, b):
    """사진 + 위쪽 60% 지도 + 아래 40% 모자이크 → RGB"""
    im = cv2.imread(str(cnn.IMAGES / row.image))
    tm = cnn._mask(row.scene)
    if tm is not None:
        im[tm] = cnn.FILL
    h, w = im.shape[:2]
    x0, y0, x1, y1 = round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h)
    if heat is not None:
        hm = cv2.resize(heat / max(heat.max(), 1e-8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        im[y0:y1, x0:x1] = (0.55 * im[y0:y1, x0:x1] + 0.45 * color).astype(np.uint8)
    blk = max(2, round(MOSAIC * w / 640))
    bot = im[y1:]
    small = cv2.resize(bot, (max(1, w // blk), max(1, -(-(h - y1) // blk))), interpolation=cv2.INTER_AREA)
    im[y1:] = cv2.resize(small, (w, h - y1), interpolation=cv2.INTER_NEAREST)
    return im[:, :, ::-1]


def main():
    ap = argparse.ArgumentParser(description="전시용 Grad-CAM: 맑은 날·탁한 날 나란히 (M6)")
    ap.add_argument("--run", default=SERVICE)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    info = json.loads((cnn.OUT / args.run / "info.json").read_text(encoding="utf-8"))
    hum = info.get("hum")
    model = cnn.make_model(pretrained=False)
    if hum:
        model = cnn.HumNet(model, hum).to(cnn.DEV)
    model.load_state_dict(torch.load(cnn.OUT / args.run / "model.pt", map_location=cnn.DEV))
    model.eval()

    d = cnn.load()
    arr = cnn.inputs(d, info["input"])["all"]
    pairs = pick(d[d.part == "A"])
    rows = np.array([r.name for pr in pairs.values() for r in pr])
    maps = cam_maps(getattr(model, "net", model), arr, rows)          # 습도 칸은 사진과 무관한 덧셈이라 지도는 사진 모델로
    preds = cnn.predict(model, arr, rows, info["log_target"], cnn.hum_input(d, hum))
    got = {r: (m, p) for r, m, p in zip(rows, maps, preds)}
    b = roi_box({}, None, info["input"])

    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for c, (clear, hazy) in sorted(pairs.items()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), dpi=110)
        for ax, r, title in zip(axes, (clear, hazy), ("맑은 날", "탁한 날")):
            m, p = got[r.name]
            ax.imshow(picture(r, m, b))
            ax.set_title(f"{title} · {r.captured_at[5:16]}   실측 {r.pm25:.0f}  ·  추정 {p:.1f} ㎍/㎥  ·  습도 {r.humidity:.0f}%", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{c}번 카메라 — 모델이 많이 본 곳(빨강). 아래 40%는 모델이 보지 않는 곳(가림)", fontsize=11)
        fig.tight_layout()
        path = OUT / f"{c}.png"
        fig.savefig(path)
        plt.close(fig)
        cards.append(f'<figure><img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{c}번</figcaption></figure>')
        print(f"{c}: 맑은 날 {clear.captured_at[5:16]} 실측 {clear.pm25:.0f} 추정 {got[clear.name][1]:.1f} / "
              f"탁한 날 {hazy.captured_at[5:16]} 실측 {hazy.pm25:.0f} 추정 {got[hazy.name][1]:.1f}", flush=True)
    (OUT / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>M6 Grad-CAM</title><style>body{{font-family:sans-serif;margin:16px}}
img{{width:100%;max-width:1320px}} figure{{margin:0 0 24px}}</style><h1>M6 — {args.run}</h1>
<p>밖에 쓰기 전에 위쪽 60%에 읽히는 번호판이 없는지 확인할 것</p>{''.join(cards)}""", encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
