# CNN — 사전학습 ConvNeXt V2-Tiny를 미세조정해 사진만 보고 PM2.5를 맞힌다.
#
#   python Model/cnn.py --version incheon        인천 ver. 연습시험으로 바퀴 수를 정하고, 개발 날짜 전체로 최종 모델을 만든다
#   python Model/cnn.py --version incheon_log    인천 ver과 같고 log(1+PM2.5)를 배운다 (로그의 효과만 따로 보려고)
#   python Model/cnn.py --version mixed          인천+베이징 ver. 베이징 학습 날짜를 섞고 log(1+PM2.5)를 배운다
#   python Model/cnn.py --version mixed --input sky    입력 종류 (기본 masked = 사진 전체 + 글자 가림). INPUTS 참고
#   python Model/cnn.py --test                   저장된 최종 모델 전부를 시험지 A(인천)·B(베이징)로 채점. 모든 작업이 끝난 뒤 한 번만
#   여러 개를 이어서 돌릴 때는 Model/run_queue.py
#
# 입력: data/labels/labels.csv·labels_beijing.csv, data/images/, 거르기는 베이스라인과 같다 (view_check.py·view_review.csv)
#       글자 자리 Model/text_mask/ (text_mask.py), ROI 상자 Model/roi.csv (roi.py)
# 결과: data/model/cnn/{버전}_{입력}/  epochs.csv  바퀴별 연습시험 점수 (인천·베이징)
#                                      pred_cv.csv 연습시험 사진별 예측 (그 날이 빠진 묶음 모델의 예측, 고른 바퀴 기준)
#                                      model.pt    최종 모델   info.json 설정과 고른 바퀴
#       data/model/cnn_cache/{입력}_384.npy  가리고 자르고 384×384로 줄인 사진 (사진 목록이 같으면 다시 쓴다)
#
# 결정 사항 (docs/현황.md §7, 2026-10-11)
#   - 사진 전체(또는 ROI 상자)를 384×384로 늘이거나 줄여 맞춘다. 사전학습 때와 같은 방식(crop_mode squash)
#   - 화면 위 글자(시각·지명)는 평균색으로 가린다. 시각을 읽는 지름길을 막으려고 (10/11 오후부터. full 입력만 예외)
#   - 카메라·도시 정보는 주지 않는다. "베이징이면 높게" 같은 지름길을 배우지 않게
#   - 증강은 살짝 잘라내기·좌우 뒤집기만. 색·밝기는 정답 신호라 건드리지 않는다
#   - 인천+베이징 ver은 log(1+PM2.5)를 배운다. 베이징 200대의 큰 오차가 학습을 장악하지 않게. 점수는 원래 농도로 낸다
#   - 연습시험: 인천 개발 14일과 베이징 학습 206일을 각각 날짜 번호 % 4 (0·1·2)로 3묶음. 회차마다 같은 번호의
#     인천·베이징 묶음을 함께 빼고 채점한다. 베이징을 배우지 않는 버전도 베이징 묶음을 채점해 기록한다
#   - 바퀴 수: 최대 15바퀴 동안 매 바퀴 연습시험. 인천 ver은 인천 오차가 가장 작은 바퀴.
#     인천+베이징 ver은 인천 오차가 최고 바퀴와 0.1 이내인 바퀴들 중 베이징 오차가 가장 작은 바퀴 (인천은 지키고 베이징을 올린다)
#     최종 모델은 같은 학습 일정으로 그 바퀴에서 멈춘다
#   - 시험지 B = 베이징 첫날(2019-05-19)부터 센 날짜 번호 % 4 == 3인 날. 어느 버전도 학습에 쓰지 않는다
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from timm.layers.grn import GlobalResponseNorm

import text_mask
from baseline import HUMIDITY_MAX, LABELS, WEATHER, fold_of, is_test_day, scores, view_info
from roi import FULL, box as roi_box, load_auto, load_roi
from view_check import load_review

ROOT = Path(__file__).resolve().parent.parent
BJ_LABELS = ROOT / "data/labels/labels_beijing.csv"
IMAGES = ROOT / "data/images"
CACHE = ROOT / "data/model/cnn_cache"
OUT = ROOT / "data/model/cnn"

BJ_FIRST_DAY = date(2019, 5, 19)
NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"     # 가중치 CC-BY-NC 4.0 (Meta), 코드 timm MIT
SIZE = 384
EPOCHS = 15
BATCH = 16             # 8GB에서 5.8GB. 32는 넘친다
LR = 1e-4
WEIGHT_DECAY = 0.05
DROP_PATH = 0.1
SEED = 0
TOLERANCE = 0.1        # 인천 오차 차이가 이 안이면 같은 수준으로 본다
VERSIONS = {           # 이름: (베이징을 배우나, 로그로 배우나)
    "incheon": (False, False),
    "incheon_log": (False, True),
    "mixed": (True, True),
}
INPUTS = {             # 이름: 설명. full 말고는 모두 화면 위 글자를 가린다 (text_mask.py)
    "full": "사진 전체 (글자 그대로 — 10/11 오전까지의 1단계)",
    "masked": "사진 전체 + 글자 가림",
    "sky": "ROI ① 하늘", "far": "ROI ② 먼 건물·산", "both": "ROI ③ 하늘+먼 건물·산",
    "top3": "ROI ④ 위쪽 1/3", "top60": "ROI ⑤ 위쪽 60%",
    "auto": "ROI ⑥ 자동 (먼지 민감도 지도, 연습시험 회차마다 공부 날짜로 만든 상자)",
}
FILL = (104, 116, 124)  # 가린 자리를 칠할 색 (BGR). 사전학습 입력의 평균색이라 표준화하면 0 = "정보 없음"
DEV = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def grn_fast(self, x):
    """timm GRN과 같은 식 x + b + w·(x·n). 원본을 fp32로 복사하지 않고 크기만 fp32로 재며, 곱·더하기를 한 번에 한다.
    3060 Ti 학습 속도 초당 63 → 90장, 결과는 fp32 대비 차이가 원래 방식과 같다(2.2%, 1위 일치 100%). 2026-10-11"""
    g = torch.linalg.vector_norm(x, dim=self.spatial_dim, keepdim=True, dtype=torch.float32)
    n = (g / (g.mean(dim=self.channel_dim, keepdim=True) + self.eps)).to(x.dtype)
    w = self.weight.view(self.wb_shape).to(x.dtype)
    b = self.bias.view(self.wb_shape).to(x.dtype)
    return torch.addcmul(x + b, x, w * n)


GlobalResponseNorm.forward = grn_fast


def load():
    """두 도시 사진에 베이스라인과 같은 거르기를 하고, 각 사진이 어느 묶음·시험지인지 붙인다"""
    inc = pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"camera_id": str})
    bad, inc["scene"] = view_info(inc)
    inc["bad_view"] = bad
    bj = pd.read_csv(BJ_LABELS, encoding="utf-8-sig", dtype={"camera_id": str})
    decided = bj.image.map(load_review())
    if decided.isna().any():
        sys.exit(f"검수하지 않은 베이징 사진 {int(decided.isna().sum())}장. view_check.py --import-folders 로 넣으세요")
    bj["bad_view"] = decided == "exclude"
    bj["scene"] = "bj"
    d = pd.concat([inc, bj], ignore_index=True)
    d["city"] = np.where(d.camera_id == "bj", "bj", "inc")
    d["day"] = pd.to_datetime(d.captured_at).dt.date

    steps = [("PM2.5 결측", d.pm25.isna()),
             ("강수 감지", d.rain == 1),
             (f"습도 {HUMIDITY_MAX}% 이상", d.humidity >= HUMIDITY_MAX),
             ("기상 결측", d[WEATHER].isna().any(axis=1)),
             ("화각 변경·깨짐", d.bad_view)]
    keep = pd.Series(True, index=d.index)
    city = d.city == "inc"
    print(f"사진 인천 {int(city.sum())}장, 베이징 {int((~city).sum())}장")
    for name, bad in steps:
        print(f"  - {name:14} 인천 {int((bad & keep & city).sum()):5}  베이징 {int((bad & keep & ~city).sum()):5}장 제외")
        keep &= ~bad
    d = d[keep].reset_index(drop=True)

    inc = d.city == "inc"
    # 인천 f0·f1·f2 / 시험지 A, 베이징 b0·b1·b2 / 시험지 B. 번호가 같은 묶음끼리 같은 회차에 빠진다
    bj_n = d.day.map(lambda x: (x - BJ_FIRST_DAY).days % 4)
    d["part"] = np.where(inc, d.day.map(lambda x: "A" if is_test_day(x) else f"f{fold_of(x)}"),
                         np.where(bj_n == 3, "B", "b" + bj_n.astype(str)))
    for name, m in [("인천 개발", inc & (d.part != "A")), ("시험지 A", d.part == "A"),
                    ("베이징 학습", ~inc & (d.part != "B")), ("시험지 B", d.part == "B")]:
        print(f"  = {name:8} {d[m].day.nunique():3}일 {int(m.sum()):5}장, PM2.5 평균 {d[m].pm25.mean():5.1f} 최대 {d[m].pm25.max():4.0f}")
    return d


@lru_cache(maxsize=None)
def _mask(scene):
    return text_mask.load(scene)


def _resized(job):
    """사진 한 장 → (글자 가림) → (ROI 상자로 자르기) → 384×384 RGB"""
    path, scene, masked, b = job
    img = cv2.imread(str(path))
    m = _mask(scene) if masked else None
    if m is not None:
        img[m] = FILL
    h, w = img.shape[:2]
    img = img[round(b[1] * h):round(b[3] * h), round(b[0] * w):round(b[2] * w)]
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def cache(d, kind):
    """사진을 입력 종류대로 가리고 잘라 384×384로 줄여 한 파일(uint8, 장×세로×가로×RGB)에 모은다. 매 바퀴 원본을 다시 줄이지 않게.
    사진 목록이나 상자가 바뀌면 다시 만든다(1.6만 장에 1분 남짓). 목록 파일은 다 만든 뒤에 써서, 중간에 끊기면 다음에 다시 만든다.
    kind가 auto0·auto1·auto2·autoall이면 ⑥ 자동 상자의 그 회차 상자를 쓴다"""
    images = d.image.tolist()
    if kind in ("full", "masked"):
        boxes = [FULL] * len(d)
    elif kind.startswith("auto"):
        auto, fold = load_auto(), kind[4:]
        boxes = [auto[(c, fold)] for c in d.camera_id]
    else:
        rois = load_roi()
        missing = sorted(set(d.camera_id) - set(rois)) if kind in ("sky", "far", "both") else []
        if missing:
            sys.exit(f"ROI 상자가 없는 카메라: {missing}. python Model/roi.py 로 그린 뒤 --apply 하세요")
        boxes = [roi_box(rois, c, kind) for c in d.camera_id]
    table = pd.DataFrame({"image": images, "box": [",".join(f"{v:.3f}" for v in b) for b in boxes]})
    arr, idx = CACHE / f"{kind}_{SIZE}.npy", CACHE / f"{kind}_{SIZE}.csv"
    if arr.exists() and idx.exists():
        old = pd.read_csv(idx, dtype=str)
        if old.image.tolist() == images and ("box" not in old or old.box.tolist() == table.box.tolist()):
            return np.load(arr, mmap_mode="r")
    jobs = [(IMAGES / p, s, kind != "full", b) for p, s, b in zip(images, d.scene, boxes)]
    print(f"사진 {len(images)}장을 입력 '{kind}'로 만드는 중 → {arr}")
    CACHE.mkdir(parents=True, exist_ok=True)
    idx.unlink(missing_ok=True)
    out = np.lib.format.open_memmap(arr, mode="w+", dtype=np.uint8, shape=(len(images), SIZE, SIZE, 3))
    with ProcessPoolExecutor(8) as ex:
        for i, img in enumerate(ex.map(_resized, jobs, chunksize=32)):
            out[i] = img
    out.flush()
    del out
    table.to_csv(idx, index=False)
    return np.load(arr, mmap_mode="r")


def make_model(bias=0.0, pretrained=True):
    m = timm.create_model(NAME, pretrained=pretrained, num_classes=1, drop_path_rate=DROP_PATH)
    torch.nn.init.constant_(m.head.fc.bias, bias)       # 처음부터 평균 근처를 부르게
    return m.to(DEV).to(memory_format=torch.channels_last)


def batch(arr, rows):
    x = torch.from_numpy(arr[rows]).to(DEV).permute(0, 3, 1, 2).float()
    return ((x - MEAN.to(DEV)) / STD.to(DEV)).contiguous(memory_format=torch.channels_last)


def augment(x):
    """사진마다 살짝 잘라내기(가로·세로 같은 비율로 최대 8%, 위치 무작위)와 반 확률 좌우 뒤집기. 색·밝기는 그대로"""
    n = len(x)
    s = torch.empty(n, device=DEV).uniform_(0.92, 1.0)
    theta = torch.zeros(n, 2, 3, device=DEV)
    theta[:, 0, 0] = s * torch.where(torch.rand(n, device=DEV) < 0.5, -1.0, 1.0)
    theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(n, 2, device=DEV) * 2 - 1) * (1 - s)[:, None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=False).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, arr, rows, log):
    model.eval()
    out = []
    for i in range(0, len(rows), 32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(batch(arr, rows[i:i + 32])).squeeze(1).float().cpu())
    p = torch.cat(out).numpy()
    return np.expm1(p) if log else p


def train(arr, d, train_rows, log, label, eval_rows=None, stop=None):
    """train_rows로 EPOCHS바퀴 일정(1바퀴 준비 + 코사인 감소)을 따라 학습한다.
    eval_rows가 있으면 매 바퀴 그 사진들의 예측(원래 농도)을 모아 돌려주고, stop이 있으면 그 바퀴에서 멈춘다"""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    y = d.pm25.values.astype(np.float32)
    target = np.log1p(y) if log else y
    model = make_model(float(target[train_rows].mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = len(train_rows) // BATCH
    total = EPOCHS * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / steps) * 0.5 * (1 + math.cos(math.pi * i / total)))
    preds = []
    for ep in range(1, (stop or EPOCHS) + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(train_rows)
        loss_sum = torch.zeros((), device=DEV)
        for i in range(steps):
            rows = order[i * BATCH:(i + 1) * BATCH]
            x = augment(batch(arr, rows))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x).squeeze(1).float()
            loss = F.l1_loss(out, torch.from_numpy(target[rows]).to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        msg = f"  [{label}] {ep:2}바퀴 학습 손실 {loss_sum.item() / steps:.3f}"
        if eval_rows is not None:
            preds.append(predict(model, arr, eval_rows, log))
            err, inc = np.abs(preds[-1] - y[eval_rows]), d.city.values[eval_rows] == "inc"
            msg += f"  연습시험 MAE 인천 {err[inc].mean():.2f} 베이징 {err[~inc].mean():.2f}"
        print(f"{msg}  ({time.time() - t0:.0f}초)", flush=True)
    return model, preds


def pick_epoch(table, use_bj):
    """인천 ver: 인천 오차가 가장 작은 바퀴. 인천+베이징 ver: 인천 오차가 최고 바퀴와 TOLERANCE 이내인 바퀴들 중 베이징 오차가 가장 작은 바퀴"""
    if not use_bj:
        return int(table.MAE.idxmin())
    ok = table[table.MAE <= table.MAE.min() + TOLERANCE]
    return int(ok["베이징MAE"].idxmin())


def inputs(d, kind):
    """회차("0"·"1"·"2"·"all") → 사진 사본. ⑥ 자동은 회차마다 상자가 달라 사본도 따로, 나머지는 하나를 같이 쓴다"""
    if kind == "auto":
        return {f: cache(d, f"auto{f}") for f in ("0", "1", "2", "all")}
    arr = cache(d, kind)
    return {f: arr for f in ("0", "1", "2", "all")}


def run(d, arrs, version, kind):
    use_bj, log = VERSIONS[version]
    out = OUT / f"{version}_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    inc = d[(d.city == "inc") & (d.part != "A")]
    bj = d[(d.city == "bj") & (d.part != "B")]
    t_start = time.time()

    print(f"\n[연습시험] 날짜 3묶음 × 최대 {EPOCHS}바퀴 (학습: 인천 2묶음{' + 베이징 2묶음' if use_bj else ''}, 채점: 인천·베이징 1묶음씩)")
    ev = pd.concat([inc, bj])
    P = np.zeros((EPOCHS, len(ev)))
    for f in ("0", "1", "2"):
        val = ev.part.isin([f"f{f}", f"b{f}"]).values
        learn = ev[~val & ((ev.city == "inc") | use_bj).values]
        _, preds = train(arrs[f], d, learn.index.values, log, f"묶음 {f}", eval_rows=ev.index[val].values)
        P[:, val] = np.array(preds)
    ev = ev.reset_index(drop=True)
    is_inc = (ev.city == "inc").values
    rows = []
    for e in range(EPOCHS):
        s = scores(ev[is_inc].reset_index(drop=True), P[e, is_inc])
        s.update({f"베이징{k}": v for k, v in scores(ev[~is_inc].reset_index(drop=True), P[e, ~is_inc]).items()})
        rows.append(s)
    table = pd.DataFrame(rows, index=pd.RangeIndex(1, EPOCHS + 1, name="epoch"))
    best = pick_epoch(table, use_bj)
    table.round(3).to_csv(out / "epochs.csv", encoding="utf-8-sig")
    print(table.round(2).to_string())
    print(f"→ {best}바퀴 (인천 MAE {table.MAE[best]:.2f}, 베이징 MAE {table['베이징MAE'][best]:.2f}"
          f"{', 인천 최고와 ' + str(TOLERANCE) + ' 이내 중 베이징이 가장 좋은 바퀴' if use_bj else ', 인천이 가장 좋은 바퀴'})")
    pred = ev[["image", "camera_id", "scene", "captured_at", "pm_time", "pm25", "humidity", "sun_alt"]].copy()
    pred["pred"] = P[best - 1].round(2)
    pred.to_csv(out / "pred_cv.csv", index=False, encoding="utf-8-sig")

    learn = pd.concat([inc, bj]) if use_bj else inc
    print(f"\n[최종 모델] 인천 개발 {inc.day.nunique()}일{f' + 베이징 학습 {bj.day.nunique()}일' if use_bj else ''}로 {best}바퀴")
    model, _ = train(arrs["all"], d, learn.index.values, log, "최종", stop=best)
    torch.save(model.state_dict(), out / "model.pt")
    info = {"version": version, "input": kind, "model": NAME, "beijing_train": use_bj, "log_target": log, "best_epoch": best,
            "cv": table.loc[best].round(3).to_dict(), "train_photos": int(len(learn)),
            "batch": BATCH, "lr": LR, "max_epochs": EPOCHS, "minutes": round((time.time() - t_start) / 60, 1),
            "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out} ({info['minutes']}분)")


def test(d):
    runs = sorted(p.parent for p in OUT.glob("*/model.pt"))
    if not runs:
        sys.exit("저장된 최종 모델이 없습니다")
    rows, preds = {}, []
    for r in runs:
        info = json.loads((r / "info.json").read_text(encoding="utf-8"))
        arr = inputs(d, info["input"])["all"]
        model = make_model(pretrained=False)
        model.load_state_dict(torch.load(r / "model.pt", map_location=DEV))
        for sheet in ("A", "B"):
            part = d[d.part == sheet]
            p = predict(model, arr, part.index.values, info["log_target"])
            rows[(r.name, sheet)] = scores(part.reset_index(drop=True), p)
            preds.append(part[["image", "camera_id", "captured_at", "pm_time", "pm25"]].assign(run=r.name, sheet=sheet, pred=p.round(2)))
    t = pd.DataFrame(rows).T
    t.index.names = ["모델", "시험지"]
    print("\n시험지 A = 인천 (본 성적), B = 베이징 고농도 (참고)")
    print(t.round(2).to_string())
    pd.concat(preds).to_csv(OUT / "pred_test.csv", index=False, encoding="utf-8-sig")
    print(f"사진별 예측: {OUT / 'pred_test.csv'}")


def main():
    ap = argparse.ArgumentParser(description="ConvNeXt V2-Tiny 미세조정 → PM2.5")
    ap.add_argument("--version", choices=list(VERSIONS), help="incheon: 인천만, incheon_log: 인천만·로그, mixed: 인천+베이징·로그")
    ap.add_argument("--input", choices=list(INPUTS), default="masked",
                    help="; ".join(f"{k}: {v}" for k, v in INPUTS.items()).replace("%", "%%") + " (기본 masked)")
    ap.add_argument("--test", action="store_true", help="저장된 최종 모델을 시험지 A·B로 채점 (모든 작업이 끝난 뒤 한 번만)")
    args = ap.parse_args()
    if not args.test and not args.version:
        ap.error("--version 또는 --test 를 주세요")

    # 메모리가 넘치면 Windows는 오류 대신 PC 메모리를 빌려 수십 배 느려진다. 그 전에 오류가 나게 상한을 둔다
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True
    d = load()
    if args.test:
        test(d)
    else:
        print(f"입력: {args.input} — {INPUTS[args.input]}")
        run(d, inputs(d, args.input), args.version, args.input)


if __name__ == "__main__":
    main()
