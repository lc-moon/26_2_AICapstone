# CNN — 사전학습 ConvNeXt V2-Tiny를 미세조정해 사진만 보고 PM2.5를 맞힌다.
#
#   python Model/cnn.py --version incheon    인천 ver. 연습시험으로 바퀴 수를 정하고, 개발 날짜 전체로 최종 모델을 만든다
#   python Model/cnn.py --version mixed      인천+베이징 ver. 위와 같고 베이징 학습 날짜를 섞는다
#   python Model/cnn.py --test               저장된 최종 모델 전부를 시험지 A(인천)·B(베이징)로 채점. 모든 작업이 끝난 뒤 한 번만
#
# 입력: data/labels/labels.csv·labels_beijing.csv, data/images/, 거르기는 베이스라인과 같다 (view_check.py·view_review.csv)
# 결과: data/model/cnn/{버전}_full/  epochs.csv  바퀴별 연습시험 점수
#                                    pred_cv.csv 개발 날짜 사진별 예측 (그 날이 빠진 묶음 모델의 예측, 고른 바퀴 기준)
#                                    model.pt    최종 모델   info.json 설정과 고른 바퀴
#       data/model/cnn_cache/        384×384로 줄인 사진 (사진 목록이 같으면 다시 쓴다)
#
# 결정 사항 (docs/현황.md §7, 2026-10-11)
#   - 사진 전체를 384×384로 늘이거나 줄여 맞춘다. 사전학습 때와 같은 방식(crop_mode squash)
#   - 카메라·도시 정보는 주지 않는다. "베이징이면 높게" 같은 지름길을 배우지 않게
#   - 증강은 살짝 잘라내기·좌우 뒤집기만. 색·밝기는 정답 신호라 건드리지 않는다
#   - 인천+베이징 ver은 log(1+PM2.5)를 배운다. 베이징 200대의 큰 오차가 학습을 장악하지 않게. 점수는 원래 농도로 낸다
#   - 연습시험: 개발 14일을 날짜 번호 % 4 (0·1·2)로 3묶음. 베이징 학습 날짜는 모든 묶음에서 학습에만 쓰고 채점은 인천만
#   - 바퀴 수: 최대 15바퀴 동안 매 바퀴 연습시험 → 오차가 가장 작은 바퀴. 최종 모델은 같은 학습 일정으로 그 바퀴에서 멈춘다
#   - 시험지 B = 베이징 첫날(2019-05-19)부터 센 날짜 번호 % 4 == 3인 날. 어느 버전도 학습에 쓰지 않는다
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from timm.layers.grn import GlobalResponseNorm

from baseline import HUMIDITY_MAX, LABELS, WEATHER, fold_of, is_test_day, scores, view_info
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
    d["part"] = np.where(inc, d.day.map(lambda x: "A" if is_test_day(x) else f"f{fold_of(x)}"),
                         d.day.map(lambda x: "B" if (x - BJ_FIRST_DAY).days % 4 == 3 else "bj"))
    for name, m in [("인천 개발", inc & (d.part != "A")), ("시험지 A", d.part == "A"),
                    ("베이징 학습", d.part == "bj"), ("시험지 B", d.part == "B")]:
        print(f"  = {name:8} {d[m].day.nunique():3}일 {int(m.sum()):5}장, PM2.5 평균 {d[m].pm25.mean():5.1f} 최대 {d[m].pm25.max():4.0f}")
    return d


def _resized(path):
    return cv2.resize(cv2.imread(str(path)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def cache(images):
    """사진을 384×384로 줄여 한 파일(uint8, 장×세로×가로×RGB)에 모은다. 매 바퀴 원본을 다시 줄이지 않게.
    사진 목록이 바뀌면 다시 만든다(1.6만 장에 1분 남짓). 목록 파일은 다 만든 뒤에 써서, 중간에 끊기면 다음에 다시 만든다"""
    arr, idx = CACHE / f"full_{SIZE}.npy", CACHE / f"full_{SIZE}.csv"
    if arr.exists() and idx.exists() and pd.read_csv(idx).image.tolist() == list(images):
        return np.load(arr, mmap_mode="r")
    print(f"사진 {len(images)}장을 {SIZE}×{SIZE}로 줄이는 중 → {arr}")
    CACHE.mkdir(parents=True, exist_ok=True)
    idx.unlink(missing_ok=True)
    out = np.lib.format.open_memmap(arr, mode="w+", dtype=np.uint8, shape=(len(images), SIZE, SIZE, 3))
    with ProcessPoolExecutor(8) as ex:
        for i, img in enumerate(ex.map(_resized, [IMAGES / p for p in images], chunksize=32)):
            out[i] = img
    out.flush()
    del out
    pd.DataFrame({"image": list(images)}).to_csv(idx, index=False)
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
            msg += f"  연습시험 MAE {np.abs(preds[-1] - y[eval_rows]).mean():.2f}"
        print(f"{msg}  ({time.time() - t0:.0f}초)", flush=True)
    return model, preds


def run(d, arr, version):
    log = version == "mixed"
    out = OUT / f"{version}_full"
    out.mkdir(parents=True, exist_ok=True)
    dev = d[(d.city == "inc") & (d.part != "A")]
    extra = d.index[d.part == "bj"].values if version == "mixed" else np.array([], dtype=int)
    t_start = time.time()

    print(f"\n[연습시험] 날짜 3묶음 × 최대 {EPOCHS}바퀴 (학습: 인천 2묶음{' + 베이징 ' + str(len(extra)) + '장' if len(extra) else ''})")
    P = np.zeros((EPOCHS, len(dev)))
    for f in sorted(dev.part.unique()):
        val = (dev.part == f).values
        rows = np.concatenate([dev.index[~val].values, extra])
        _, preds = train(arr, d, rows, log, f"묶음 {f[1]}", eval_rows=dev.index[val].values)
        P[:, val] = np.array(preds)
    ev = dev.reset_index(drop=True)
    table = pd.DataFrame([scores(ev, P[e]) for e in range(EPOCHS)], index=pd.RangeIndex(1, EPOCHS + 1, name="epoch"))
    best = int(table.MAE.idxmin())
    table.round(3).to_csv(out / "epochs.csv", encoding="utf-8-sig")
    print(table.round(2).to_string())
    print(f"→ {best}바퀴가 가장 좋음 (MAE {table.MAE[best]:.2f}, 시간 MAE {table['시간MAE'][best]:.2f})")
    pred = ev[["image", "camera_id", "scene", "captured_at", "pm_time", "pm25", "humidity", "sun_alt"]].copy()
    pred["pred"] = P[best - 1].round(2)
    pred.to_csv(out / "pred_cv.csv", index=False, encoding="utf-8-sig")

    print(f"\n[최종 모델] 인천 개발 14일{' + 베이징' if len(extra) else ''}로 {best}바퀴")
    model, _ = train(arr, d, np.concatenate([dev.index.values, extra]), log, "최종", stop=best)
    torch.save(model.state_dict(), out / "model.pt")
    info = {"version": version, "input": "full", "model": NAME, "log_target": log, "best_epoch": best,
            "cv": table.loc[best].round(3).to_dict(), "train_photos": int(len(dev) + len(extra)),
            "batch": BATCH, "lr": LR, "max_epochs": EPOCHS, "minutes": round((time.time() - t_start) / 60, 1),
            "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out} ({info['minutes']}분)")


def test(d, arr):
    runs = sorted(p.parent for p in OUT.glob("*/model.pt"))
    if not runs:
        sys.exit("저장된 최종 모델이 없습니다")
    rows, preds = {}, []
    for r in runs:
        info = json.loads((r / "info.json").read_text(encoding="utf-8"))
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
    ap.add_argument("--version", choices=["incheon", "mixed"], help="incheon: 인천만, mixed: 인천+베이징")
    ap.add_argument("--test", action="store_true", help="저장된 최종 모델을 시험지 A·B로 채점 (모든 작업이 끝난 뒤 한 번만)")
    args = ap.parse_args()
    if not args.test and not args.version:
        ap.error("--version 또는 --test 를 주세요")

    # 메모리가 넘치면 Windows는 오류 대신 PC 메모리를 빌려 수십 배 느려진다. 그 전에 오류가 나게 상한을 둔다
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True
    d = load()
    arr = cache(d.image)
    if args.test:
        test(d, arr)
    else:
        run(d, arr, args.version)


if __name__ == "__main__":
    main()
