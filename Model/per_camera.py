# 카메라별 모델 (M3) — 카메라마다 따로 학습한 모델이 통합 모델 1개보다 나은지 본다.
#
#   python Model/per_camera.py hf                 ① Hugging Face에서 출발, 카메라마다 그 카메라의 9/22~10/19 사진만 (약 2시간)
#   python Model/per_camera.py from_a             ②-전체: ⓐ에서 출발, ⓐ가 배우지 않은 14일로 몸통까지 (약 1시간)
#   python Model/per_camera.py from_a_head        ②-마지막 층: ⓐ에서 출발, 같은 14일로 마지막 층만 (몇십 분)
#   python Model/per_camera.py from_b from_b_head  ③ "한꺼번에": ⓑ에서 출발, ⓑ가 배운 9/22~10/19 그대로 (cnn.py --tag folds --folds-only 먼저)
#   python Model/per_camera.py hf from_a from_a_head   이어서
#   python Model/per_camera.py --lr-range from_b from_b_head   학습률 범위 시험 (후보를 안 주면 전부)
#   python Model/per_camera.py --test             저장된 후보를 시험 데이터로 채점 (ⓐ·ⓑ·보정과 나란히)
#
# 결과: data/model/cnn/percam_{후보}/  epochs.csv  바퀴별 연습시험 점수 (20대 예측을 합쳐서)
#                                     pred_cv.csv 연습시험 사진별 예측   info.json 설정과 고른 바퀴
#                                     cam_{카메라}.pt 카메라별 최종 모델 (마지막 층만이면 heads.pt 하나)
#       data/model/cnn/pred_test_percam.csv  시험 사진별 예측
#
# 규칙 (docs/전시_준비.md "M3에서 시험 점수를 보기 전에 정한 것", 2026-11-03)
#   - 입력·증강·글자 가림·거르기·log 정답은 통합 모델(ⓐ·ⓑ)과 같다. 다른 것은 "그 카메라 사진만 배운다"와 출발점
#   - 바퀴 수는 후보의 학습 기간을 날짜 번호 % 3으로 나눈 연습시험으로, 20대 공통 하나(20대 예측을 합친 MAE가 가장 작은 바퀴)
#   - ②는 ⓐ가 배우지 않은 학습 기간 날(M2 보정과 같은 14일)만 쓴다. ⓐ가 외운 날로는 카메라 버릇이 드러나지 않는다
#   - ③은 몸통이 이미 본 날로 다시 배운다(2026-11-04 추가, ②의 2단계와 비교). 연습시험은 그 묶음을 안 본 ⓑ 묶음별 몸통(FOLDS)에서 출발해
#     이미 본 날로 채점하는 일을 피한다. 최종 모델만 ⓑ에서 출발한다
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn
from baseline import fold_of, scores
from calibrate import bias_table, day_compare

BASE = "incheon_log_top60"     # ⓐ 지금 서비스 모델
RETRAIN = "incheon_log_top60_1019"   # ⓑ
FOLDS = "incheon_log_top60_folds"    # ⓑ와 같은 조리법의 연습시험 묶음별 몸통 (fold_0~2.pt)
INPUT = "top60"
CANDIDATES = {                 # 이름: (출발점, 학습 기간, 마지막 층만, 학습률, 최대 바퀴, 설명)
    # 학습률은 범위 시험(--lr-range, 2026-11-03)에서 손실이 치솟기 시작하는 곳보다 충분히 작은 값으로 정했다.
    # ①은 처음 정한 1e-4가 62·143번이 오르기 시작하는 경계라 3e-5로, ②-마지막 층은 1e-3이 경계 바로 앞이라 3e-4로 낮췄다.
    # ②-전체는 손실이 5e-4~1e-3에서 치솟는데 처음 정한 1e-5를 그대로 썼다(11/3 실수, 규칙대로면 5e-5~1.7e-4). 11/4에 1e-4로 다시 학습하고 옛것은 from_a_1e5로 남긴다
    "hf": ("hf", "all", False, 3e-5, 12, "① Hugging Face에서 출발, 9/22~10/19"),
    "from_a": ("a", "unseen", False, 1e-4, 10, "②-전체 ⓐ에서 출발, ⓐ가 배우지 않은 14일, 몸통까지"),
    "from_a_1e5": ("a", "unseen", False, 1e-5, 10, "②-전체 (학습률 1e-5, 11/3 실수판)"),
    "from_a_head": ("a", "unseen", True, 3e-4, 10, "②-마지막 층 ⓐ에서 출발, 같은 14일, 마지막 층만"),
    # ③의 학습률은 범위 시험 뒤 확정한다. 지금 값은 ②의 짝과 같은 임시값
    "from_b": ("b", "all", False, 1e-4, 10, "③-전체 ⓑ에서 출발, 9/22~10/19, 몸통까지"),
    "from_b_head": ("b", "all", True, 3e-4, 10, "③-마지막 층 ⓑ에서 출발, 9/22~10/19, 마지막 층만"),
}


def unseen_days(d):
    """ⓐ가 배우지 않은 학습 기간 날 (ⓐ 연습시험 예측에 없는 인천 날짜)"""
    cv = pd.read_csv(cnn.OUT / BASE / "pred_cv.csv", encoding="utf-8-sig", dtype={"camera_id": str})
    learned = set(pd.to_datetime(cv.captured_at).dt.date)
    dev = d[(d.city == "inc") & (d.part != "A")]
    return sorted(set(dev.day) - learned)


def new_model(start, head_only, y_mean, fold=None):
    """fold: 연습시험 묶음 번호. ③은 그 묶음을 안 본 몸통에서 출발한다 (최종 모델은 None → ⓑ)"""
    if start == "hf":
        return cnn.make_model(y_mean)
    if start == "a":
        path = cnn.OUT / BASE / "model.pt"
    else:
        path = cnn.OUT / FOLDS / f"fold_{fold}.pt" if fold is not None else cnn.OUT / RETRAIN / "model.pt"
    m = cnn.make_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=cnn.DEV))
    if head_only:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.head.fc.parameters():
            p.requires_grad = True
    return m


def train(arr, d, rows, start, head_only, lr, epochs, eval_rows=None, stop=None, seed=cnn.SEED, fold=None):
    """cnn.train과 같은 일정(1바퀴 준비 + 코사인 감소)·증강·절대오차. 출발점과 학습할 부분만 다르다.
    마지막 층만 학습할 때는 몸통을 평가 모드로 두어 특징이 흔들리지 않게 한다"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target = np.log1p(d.pm25.values.astype(np.float32))
    model = new_model(start, head_only, float(target[rows].mean()), fold)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cnn.WEIGHT_DECAY)
    steps = max(1, len(rows) // cnn.BATCH)
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / steps) * 0.5 * (1 + math.cos(math.pi * i / total)))
    preds = []
    for ep in range(1, (stop or epochs) + 1):
        model.train(not head_only)
        order = rng.permutation(rows)
        for i in range(steps):
            r = order[i * cnn.BATCH:(i + 1) * cnn.BATCH]
            x = cnn.augment(cnn.batch(arr, r))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x).squeeze(1).float()
            loss = F.l1_loss(out, torch.from_numpy(target[r]).to(cnn.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if eval_rows is not None:
            preds.append(cnn.predict(model, arr, eval_rows, True))
    return model, preds


def candidate_rows(d, name):
    dev = d[(d.city == "inc") & (d.part != "A")]
    return dev[dev.day.isin(unseen_days(d))] if CANDIDATES[name][1] == "unseen" else dev


def lr_range(d, arr, names, cams=("128", "143", "62"), steps=150):
    """학습률 범위 시험: 한 번 돌리는 동안 학습률을 1e-7에서 1e-1까지 키우며 손실을 기록한다. 손실이 가장 빨리 줄어드는 구간을 눈으로 고른다.
    카메라는 M2에서 버릇이 뚜렷했던 128(높게)·143(낮게)·62(기간마다 바뀜)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    lrs = np.geomspace(1e-7, 1e-1, steps)
    names = names or list(CANDIDATES)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5), squeeze=False)
    target = np.log1p(d.pm25.values.astype(np.float32))
    for ax, name in zip(axes[0], names):
        start, _, head_only, lr0, _, desc = CANDIDATES[name]
        dev = candidate_rows(d, name)
        for c in cams:
            rows = dev[dev.camera_id == c].index.values
            torch.manual_seed(cnn.SEED)
            rng = np.random.default_rng(cnn.SEED)
            model = new_model(start, head_only, float(target[rows].mean()))
            model.train(not head_only)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lrs[0], weight_decay=cnn.WEIGHT_DECAY)
            losses, smooth = [], None
            for i, lr in enumerate(lrs):
                for g in opt.param_groups:
                    g["lr"] = lr
                r = rng.choice(rows, cnn.BATCH)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(cnn.augment(cnn.batch(arr, r))).squeeze(1).float()
                loss = F.l1_loss(out, torch.from_numpy(target[r]).to(cnn.DEV))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
                losses.append(smooth)
                if i > 20 and smooth > 3 * min(losses):
                    break
            ax.plot(lrs[:len(losses)], losses, label=f"{c}번")
            print(f"  {name} {c}번 {len(losses)}걸음, 가장 낮은 손실 {min(losses):.3f} (학습률 {lrs[int(np.argmin(losses))]:.0e})", flush=True)
            del model, opt
            torch.cuda.empty_cache()
        ax.axvline(lr0, color="gray", ls="--", label=f"처음 정한 값 {lr0:.0e}")
        ax.set(xscale="log", title=desc, xlabel="학습률", ylabel="손실 (log 정답 절대오차, 이동평균)")
        ax.legend()
    fig.tight_layout()
    path = cnn.ROOT / f"data/review/lr_range{'_' + '_'.join(names) if len(names) < len(CANDIDATES) else ''}.png"
    fig.savefig(path, dpi=110)
    print(f"저장: {path}")


def run(d, arr, name):
    start, period, head_only, lr, epochs, desc = CANDIDATES[name]
    out = cnn.OUT / f"percam_{name}"
    out.mkdir(parents=True, exist_ok=True)
    dev = candidate_rows(d, name)
    fold = dev.day.map(fold_of)
    cams = sorted(dev.camera_id.unique())
    t0 = time.time()
    print(f"\n[{desc}] {dev.day.nunique()}일 {len(dev)}장, 카메라 {len(cams)}대, 학습률 {lr}, 최대 {epochs}바퀴")

    print("[연습시험] 카메라마다 날짜 3묶음 (날짜 번호 % 3)")
    P = np.full((epochs, len(dev)), np.nan)
    pos = pd.Series(np.arange(len(dev)), index=dev.index)
    for c in cams:
        for f in (0, 1, 2):
            m = dev.camera_id == c
            learn, val = dev[m & (fold != f)].index.values, dev[m & (fold == f)].index.values
            if len(val) == 0 or len(learn) == 0:
                continue
            _, preds = train(arr, d, learn, start, head_only, lr, epochs, eval_rows=val, fold=f)
            P[:, pos[val].values] = np.array(preds)
        done = ~np.isnan(P[0])
        print(f"  {c:>6} 끝 ({time.time() - t0:.0f}초, 지금까지 마지막 바퀴 MAE {np.abs(P[-1, done] - dev.pm25.values[done]).mean():.2f})", flush=True)
    ev = dev.reset_index(drop=True)
    table = pd.DataFrame([scores(ev, P[e]) for e in range(epochs)], index=pd.RangeIndex(1, epochs + 1, name="epoch"))
    best = int(table.MAE.idxmin())
    table.round(3).to_csv(out / "epochs.csv", encoding="utf-8-sig")
    print(table.round(2).to_string())
    print(f"→ {best}바퀴 (20대 합친 연습시험 MAE {table.MAE[best]:.2f})")
    pred = ev[["image", "camera_id", "captured_at", "pm_time", "pm25"]].assign(pred=P[best - 1].round(2))
    pred.to_csv(out / "pred_cv.csv", index=False, encoding="utf-8-sig")

    print(f"[최종 모델] 카메라마다 학습 기간 전체로 {best}바퀴")
    heads = {}
    for c in cams:
        model, _ = train(arr, d, dev[dev.camera_id == c].index.values, start, head_only, lr, epochs, stop=best)
        if head_only:
            heads[c] = {k: v.cpu() for k, v in model.head.fc.state_dict().items()}
        else:
            torch.save(model.state_dict(), out / f"cam_{c}.pt")
    if head_only:
        torch.save(heads, out / "heads.pt")
    info = {"candidate": name, "description": desc, "start": start, "period": period, "head_only": head_only,
            "days": sorted(str(x) for x in dev.day.unique()), "photos": int(len(dev)), "cameras": cams,
            "lr": lr, "max_epochs": epochs, "best_epoch": best, "cv": table.loc[best].round(3).to_dict(),
            "input": INPUT, "minutes": round((time.time() - t0) / 60, 1), "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out} ({info['minutes']}분)")


def load_cam_model(name, c):
    out = cnn.OUT / f"percam_{name}"
    if CANDIDATES[name][2]:
        m = new_model(CANDIDATES[name][0], True, 0.0)
        m.head.fc.load_state_dict(torch.load(out / "heads.pt")[c])
        return m
    m = cnn.make_model(pretrained=False)
    m.load_state_dict(torch.load(out / f"cam_{c}.pt", map_location=cnn.DEV))
    return m


def round_corr(t, col):
    """같은 회차(10분) 카메라끼리 예측과 실측의 상관 평균. 시간대·날씨는 회차 안에서 같으니, 이게 높으면 카메라 사이의 먼지 차이를 읽는 것"""
    g = t.assign(slot=pd.to_datetime(t.captured_at).dt.floor("10min")).groupby("slot")
    r = [x[col].corr(x.pm25) for _, x in g if len(x) >= 10 and x.pm25.nunique() > 1]
    return float(np.nanmean(r)), len(r)


def test(d, arr):
    names = [n for n in CANDIDATES if (cnn.OUT / f"percam_{n}" / "info.json").exists()]
    if not names:
        sys.exit("저장된 카메라별 모델이 없습니다")
    part = d[d.part == "A"]
    t = part[["image", "camera_id", "captured_at", "pm_time", "pm25", "humidity"]].copy()
    for n in names:
        trained = json.loads((cnn.OUT / f"percam_{n}" / "info.json").read_text(encoding="utf-8"))["cameras"]
        p = pd.Series(np.nan, index=part.index)
        for c, g in part.groupby("camera_id"):
            if c in trained:
                p[g.index] = cnn.predict(load_cam_model(n, c), arr, g.index.values, True)
        t[n] = p.round(2).values
    t = t.reset_index(drop=True)

    # 비교 상대: ⓐ·ⓑ (pred_test.csv), 보정 ② (pred_test_calib.csv)
    ref = pd.read_csv(cnn.OUT / "pred_test.csv", encoding="utf-8-sig")
    for run, col in [(BASE, "ⓐ"), (RETRAIN, "ⓑ")]:
        r = ref[(ref.run == run) & (ref.sheet == "A")].set_index("image").pred
        if len(r):
            t[col] = t.image.map(r)
    for run, col in [(BASE, "ⓐ+②보정"), (RETRAIN, "ⓑ+②보정")]:
        f = cnn.OUT / run / "pred_test_calib.csv"
        if f.exists():
            t[col] = t.image.map(pd.read_csv(f, encoding="utf-8-sig").set_index("image").calib2)
    cols = {c: c for c in ["ⓐ", "ⓐ+②보정", "ⓑ", "ⓑ+②보정", *names] if c in t and t[c].notna().all()}
    t.to_csv(cnn.OUT / "pred_test_percam.csv", index=False, encoding="utf-8-sig")

    print(f"\n[시험 데이터] 10/20~11/1 {t.captured_at.str[:10].nunique()}일 {len(t)}장")
    rows = {}
    for k in cols:
        s = scores(t, t[k].values)
        s["회차상관"], s["회차수"] = round_corr(t, k)
        rows[k] = s
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\n습도별 치우침 (예측 − 정답 평균)")
    print(bias_table(t, cols).round(2).to_string())
    for n in names:
        ref_col = "ⓐ+②보정" if CANDIDATES[n][0] == "a" else "ⓑ+②보정"
        if ref_col in cols:
            print(f"\n날짜별: {n} vs {ref_col}")
            day_compare(t, {ref_col: ref_col, "ⓐ": "ⓐ", n: n})
    print(f"\n저장: {cnn.OUT / 'pred_test_percam.csv'}")


def main():
    ap = argparse.ArgumentParser(description="카메라별 모델 (M3)")
    ap.add_argument("names", nargs="*", help=f"학습할 후보 ({', '.join(CANDIDATES)})")
    ap.add_argument("--test", action="store_true", help="저장된 후보를 시험 데이터로 채점")
    ap.add_argument("--lr-range", nargs="*", help="학습률 범위 시험 그래프 (data/review/lr_range*.png, 후보당 약 3분). 후보를 안 주면 전부")
    ap.add_argument("--lr", nargs="*", default=[], help="후보별 학습률 바꾸기 (예: --lr hf=3e-5 from_a=1e-5)")
    args = ap.parse_args()
    for kv in args.lr:
        k, v = kv.split("=")
        CANDIDATES[k] = (*CANDIDATES[k][:3], float(v), *CANDIDATES[k][4:])
    if args.lr_range is not None:
        if set(args.lr_range) - set(CANDIDATES):
            ap.error(f"후보는 {list(CANDIDATES)} 중에서")
        torch.cuda.set_per_process_memory_fraction(0.8)
        d = cnn.load()
        lr_range(d, cnn.inputs(d, INPUT)["all"], args.lr_range)
        return
    if not args.names and not args.test:
        ap.error("후보 이름 또는 --test 를 주세요")
    if set(args.names) - set(CANDIDATES):
        ap.error(f"후보는 {list(CANDIDATES)} 중에서")
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True
    d = cnn.load()
    arr = cnn.inputs(d, INPUT)["all"]
    if args.test:
        test(d, arr)
    for n in args.names:
        run(d, arr, n)


if __name__ == "__main__":
    main()
