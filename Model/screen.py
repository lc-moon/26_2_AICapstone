# CNN 후보 거르기(예선) — 사전학습 ConvNeXt V2가 사진에서 뽑은 특징(768개)에 Ridge 회귀를 얹어 입력 후보를 빠르게 비교한다.
#
#   python Model/screen.py                         ROI 후보 6개(⑥ 자동 포함) + 비교용(사진 전체·글자 가림)을 두 버전으로
#   python Model/screen.py --inputs masked sky     고른 입력만
#
# 미세조정(cnn.py, 후보당 1~2시간) 대신 사전학습 모델을 그대로 두고 특징만 뽑아(후보당 1~2분) 같은 연습시험 3묶음으로 채점한다.
# 평균 숫자 몇 개만 보는 베이스라인 Ridge보다 실제 CNN 결과에 가깝지만 같지는 않다 — 결선 후보를 고르는 데만 쓴다(2026-10-11 결정).
# 버전·묶음·채점·로그 여부는 cnn.py와 같다. 인천+베이징 ver은 베이징 점수도 낸다.
# GPU를 쓰므로 cnn.py 학습과 동시에 돌리지 않는다 (메모리가 넘친다).
#
# 결과: data/model/screen.csv (입력 × 버전의 인천·베이징 점수), data/model/screen_feat/{입력}.npy (뽑은 특징, 다시 쓴다)
import argparse
import sys

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cnn
from baseline import scores

FEAT = cnn.ROOT / "data/model/screen_feat"
OUT = cnn.ROOT / "data/model/screen.csv"
ALPHAS = np.logspace(-1, 4, 11)       # 규제 세기는 학습 묶음 안에서 자동으로 고른다 (채점 묶음은 안 봄)


@torch.no_grad()
def features(d, kind):
    """입력 종류대로 만든 384 사진 → 사전학습 모델의 마지막 평균 특징 768개"""
    path = FEAT / f"{kind}.npy"
    idx = FEAT / f"{kind}.csv"
    if path.exists() and idx.exists() and pd.read_csv(idx).image.tolist() == d.image.tolist():
        return np.load(path)
    arr = cnn.cache(d, kind)
    model = timm.create_model(cnn.NAME, pretrained=True, num_classes=0).to(cnn.DEV).eval().to(memory_format=torch.channels_last)
    out = []
    for i in range(0, len(arr), 64):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(cnn.batch(arr, np.arange(i, min(i + 64, len(arr))))).float().cpu().numpy())
    f = np.concatenate(out)
    FEAT.mkdir(parents=True, exist_ok=True)
    np.save(path, f)
    pd.DataFrame({"image": d.image}).to_csv(idx, index=False)
    return f


def evaluate(d, f, version):
    """f: 특징 (사진 × 768). ⑥ 자동처럼 회차마다 입력이 다르면 {회차: 특징}"""
    use_bj, log = cnn.VERSIONS[version]
    inc = d[(d.city == "inc") & (d.part != "A")]
    bj = d[(d.city == "bj") & (d.part != "B")]
    ev = pd.concat([inc, bj])
    y = d.pm25.values
    pred = pd.Series(np.nan, index=ev.index)
    for k in ("0", "1", "2"):
        F = f[k] if isinstance(f, dict) else f
        val = ev.part.isin([f"f{k}", f"b{k}"])
        learn = ev[~val & ((ev.city == "inc") | use_bj)]
        t = np.log1p(y[learn.index]) if log else y[learn.index]
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(F[learn.index], t)
        p = reg.predict(F[ev.index[val]])
        pred[ev.index[val]] = np.expm1(p) if log else p
    is_inc = (ev.city == "inc").values
    s = scores(ev[is_inc].reset_index(drop=True), pred.values[is_inc])
    b = scores(ev[~is_inc].reset_index(drop=True), pred.values[~is_inc])
    return {"인천MAE": s["MAE"], "인천시간MAE": s["시간MAE"], "베이징MAE": b["MAE"]}


def main():
    ap = argparse.ArgumentParser(description="사전학습 특징 + Ridge로 CNN 입력 후보 거르기")
    ap.add_argument("--inputs", nargs="+", choices=list(cnn.INPUTS), default=list(cnn.INPUTS))
    ap.add_argument("--versions", nargs="+", choices=list(cnn.VERSIONS), default=["incheon_log", "mixed"])
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    d = cnn.load()
    rows = []
    for kind in args.inputs:
        f = {k: features(d, f"auto{k}") for k in ("0", "1", "2")} if kind == "auto" else features(d, kind)
        for v in args.versions:
            r = evaluate(d, f, v)
            rows.append({"입력": kind, "설명": cnn.INPUTS[kind], "버전": v, **{k: round(x, 3) for k, x in r.items()}})
            print(f"  {kind:7} {v:12} 인천 {r['인천MAE']:.2f}  베이징 {r['베이징MAE']:.2f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\n" + t.pivot(index="입력", columns="버전", values=["인천MAE", "베이징MAE"]).round(2).to_string())
    print(f"\n저장: {OUT}  (예선 점수 — 실제 미세조정 점수와 다르다. 결선 후보를 고르는 데만 쓴다)")


if __name__ == "__main__":
    sys.exit(main())
