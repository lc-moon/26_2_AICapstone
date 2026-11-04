# Grad-CAM 나란히 보기 (M3 지름길 검사) — 같은 사진에서 모델마다 화면 어디를 보고 높게 불렀는지 비교한다.
#
#   python Model/gradcam_compare.py      → data/review/gradcam_compare.html (원본 사진이 들어 있으니 저장소·외부에 올리지 말 것)
#
# 카메라 20대 모두, 시험 기간(10/20~11/1) 사진 중 실측이 가장 낮은 것과 가장 높은 것 1장씩 (docs/전시_준비.md M3).
# 모델은 통합 ⓐ·ⓑ·C와 카메라별 ①·②·③(전체·마지막 층) 중 저장된 것만. 카메라별 모델은 그 카메라의 모델로 그린다.
# 모델은 위쪽 60%만 보므로 지도도 그 영역에만 겹치고, 보지 않는 아래 40%는 어둡게 한다. 화면 위 글자는 모델이 본 대로 가린다.
# 빨강 = 예측을 올리는 쪽으로 많이 본 곳. 하늘·먼 건물이 아니라 그림자·가까운 나무·특정 물체에 몰리면 지름길로 의심한다.
# GPU를 쓰므로 학습과 동시에 돌리지 않는다.
import base64
import json
import sys

import cv2
import numpy as np
import torch

import cnn
import per_camera
from gradcam import cam_maps
from roi import box as roi_box

MODELS = [("ⓐ", "incheon_log_top60"), ("ⓑ", "incheon_log_top60_1019"), ("C", "incheon_log_top60_c14"),
          ("①", "percam_hf"), ("②-전체", "percam_from_a"), ("②-마지막 층", "percam_from_a_head"),
          ("③-전체", "percam_from_b"), ("③-마지막 층", "percam_from_b_head")]
OUT = cnn.ROOT / "data/review/gradcam_compare.html"
WIDTH = 320


def models():
    """(표시 이름, 카메라 → 모델을 주는 함수) 중 저장된 것만"""
    found = []
    for label, run in MODELS:
        if run.startswith("percam_"):
            name = run.removeprefix("percam_")
            if (cnn.OUT / run / "info.json").exists():
                cams = set(json.loads((cnn.OUT / run / "info.json").read_text(encoding="utf-8"))["cameras"])
                found.append((label, lambda c, n=name, cams=cams: per_camera.load_cam_model(n, c) if c in cams else None))
        elif (cnn.OUT / run / "model.pt").exists():
            m = cnn.make_model(pretrained=False)
            m.load_state_dict(torch.load(cnn.OUT / run / "model.pt", map_location=cnn.DEV))
            found.append((label, lambda c, m=m: m))
    return found


def tile(row, heat, b):
    """사진 한 장 + 위쪽 60%에 지도를 겹친 그림 → JPEG base64"""
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
    im[y1:] = (im[y1:] * 0.35).astype(np.uint8)
    im = cv2.resize(im, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


def main():
    torch.cuda.set_per_process_memory_fraction(0.8)
    d = cnn.load()
    arr = cnn.inputs(d, "top60")["all"]
    test = d[d.part == "A"]
    picks = []
    for c, g in test.groupby("camera_id"):
        g = g.sort_values("captured_at")
        picks += [(c, "가장 낮음", g.loc[g.pm25.idxmin()]), (c, "가장 높음", g.loc[g.pm25.idxmax()])]

    ms = models()
    print("모델:", ", ".join(label for label, _ in ms))
    cells = {}          # (카메라, 낮음/높음, 모델) → (지도, 예측)
    for label, get in ms:
        for c in sorted({p[0] for p in picks}):
            model = get(c)
            if model is None:
                continue
            model.eval()
            rows = np.array([r.name for cc, _, r in picks if cc == c])
            maps = cam_maps(model, arr, rows)
            preds = cnn.predict(model, arr, rows, True)
            for (cc, kind, _), mp, pr in zip([p for p in picks if p[0] == c], maps, preds):
                cells[(cc, kind, label)] = (mp, float(pr))
        print(f"  {label} 끝", flush=True)
        torch.cuda.empty_cache()

    b = roi_box({}, None, "top60")
    head = "".join(f"<th>{label}</th>" for label, _ in ms)
    body = []
    for c, kind, r in picks:
        tds = [f'<td><img src="data:image/jpeg;base64,{tile(r, None, b)}"><br>{r.captured_at[5:16]} · 실측 {r.pm25:.0f}</td>']
        for label, _ in ms:
            if (c, kind, label) in cells:
                mp, pr = cells[(c, kind, label)]
                tds.append(f'<td><img src="data:image/jpeg;base64,{tile(r, mp, b)}"><br>예측 {pr:.1f} ({pr - r.pm25:+.1f})</td>')
            else:
                tds.append("<td>—</td>")
        body.append(f"<tr><th>{c}번<br>{kind}</th>{''.join(tds)}</tr>")
    OUT.write_text(f"""<!doctype html><meta charset="utf-8"><title>Grad-CAM 비교</title><style>
body{{font-family:sans-serif;margin:16px;background:#f4f4f4}} table{{border-collapse:collapse;background:#fff}}
td,th{{border:1px solid #ddd;padding:4px;font-size:12px;text-align:center;vertical-align:top}} img{{display:block;width:{WIDTH}px}}
thead th{{position:sticky;top:0;background:#fff}}</style>
<h1>Grad-CAM 비교 — 시험 기간, 카메라마다 실측이 가장 낮은·높은 사진</h1>
<p>빨강 = 예측을 올리는 쪽으로 많이 본 곳. 모델은 위쪽 60%만 보고(아래는 어둡게), 화면 위 글자는 가린 채로 본다. 원본 사진 — 저장소·외부에 올리지 말 것</p>
<table><thead><tr><th></th><th>사진</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>""", encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
