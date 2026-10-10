# 베이스라인 — 이미지 통계와 기상으로 PM2.5를 회귀해, "이미지에 PM2.5 신호가 있는가"를 확인한다.
#
#   python Model/baseline.py           개발 점수 (시험지 A를 뺀 날짜로 날짜 단위 교차검증)
#   python Model/baseline.py --test    시험지 A 점수. 모델을 다 고른 뒤에만 볼 것
#
# 입력: data/labels/labels.csv (build_labels.py), data/model/features.csv (features.py),
#       data/model/view_check.csv·Model/view_review.csv (view_check.py — 화각 변경·깨진 화면 거르기)
# 결과: data/model/baseline_pred.csv — 개발 날짜 사진마다 정답과 모델별 예측 (교차검증에서 그 날을 뺀 모델의 예측)
#
# 평가 규칙 (docs/현황.md §7)
#   - 날짜 단위로 나눈다. 같은 날 사진이 학습과 시험에 걸치면 점수가 부풀려진다
#   - 시험지 A = 9/22부터 센 날짜 번호를 4로 나눈 나머지가 3인 날. 데이터가 늘어도 규칙이 같아 흔들리지 않는다
#   - 평균만 찍는 모델 대비 개선폭을 같이 낸다
#   - 사진 1장 단위와, 같은 카메라·같은 1시간 사진들의 예측 평균을 1시간 정답과 비교하는 단위를 둘 다 낸다
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from view_check import OUT as VIEW_CHECK, flags, load_review

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/labels/labels.csv"
FEATURES = ROOT / "data/model/features.csv"
PRED = ROOT / "data/model/baseline_pred.csv"

FIRST_DAY = date(2026, 9, 22)
HUMIDITY_MAX = 95        # 이 습도 이상은 안개·박무로 흐려져 먼지 신호와 구분할 수 없다고 보고 뺀다
WEATHER = ["humidity", "sun_alt", "temperature", "wind_speed"]
IMAGE = ["all_bright", "all_contrast", "all_sat", "all_dark", "all_sharp", "all_blue_red",
         "top_bright", "top_contrast", "top_sat", "top_dark", "top_sharp", "top_blue_red", "top_mid_diff"]


def is_test_day(d):
    return (d - FIRST_DAY).days % 4 == 3


def bad_view(d):
    """화각이 바뀌었거나 깨진 사진 (view_check.py). 사람이 검토한 결정을 따르고,
    아직 검토하지 않은 후보는 기본 기준(20픽셀 초과·깨진 화면)으로 거른다."""
    if not VIEW_CHECK.exists():
        sys.exit("data/model/view_check.csv가 없습니다. python Model/view_check.py 를 먼저 돌리세요")
    v = d[["image"]].merge(pd.read_csv(VIEW_CHECK, encoding="utf-8-sig"), on="image", how="left")
    if v["shift"].isna().any():
        sys.exit(f"화각을 재지 않은 사진 {int(v['shift'].isna().sum())}장. python Model/view_check.py 를 먼저 돌리세요")
    broken, candidate, default = flags(v)
    decided = v.image.map(load_review())
    unreviewed = candidate & decided.isna()
    if unreviewed.any():
        print(f"  ※ 검토하지 않은 화각 후보 {int(unreviewed.sum())}장은 기본 기준으로 거릅니다 (view_check.py로 검토 가능)")
    return (broken | (decided == "exclude") | (decided.isna() & default)).values


def load():
    d = pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"camera_id": str})
    d = d.merge(pd.read_csv(FEATURES, encoding="utf-8-sig"), on="image")
    d["day"] = pd.to_datetime(d["captured_at"]).dt.date
    n = len(d)
    steps = [("PM2.5 결측", d["pm25"].isna()),
             ("강수 감지", d["rain"] == 1),
             (f"습도 {HUMIDITY_MAX}% 이상", d["humidity"] >= HUMIDITY_MAX),
             ("기상 결측", d[WEATHER].isna().any(axis=1)),
             ("화각 변경·깨짐", pd.Series(bad_view(d), index=d.index))]
    keep = pd.Series(True, index=d.index)
    print(f"사진 {n}장")
    for name, bad in steps:
        print(f"  - {name:14} {int((bad & keep).sum()):5}장 제외")
        keep &= ~bad
    d = d[keep].reset_index(drop=True)
    print(f"  = 남은 사진 {len(d)}장, {d['day'].nunique()}일")
    return d


def design(train, rest, cols, mode):
    """카메라마다 장면이 달라 특징 값의 크기를 카메라끼리 비교할 수 없다. 이를 다루는 방식 셋:
      both  — 카메라마다 학습 구간의 평균·표준편차로 표준화 + 카메라 더미 (기본)
      camz  — 카메라별 표준화만. 카메라마다 평소 농도가 다르다는 정보는 빠진다
      dummy — 전체 평균·표준편차로 표준화 + 카메라 더미. 카메라마다 값이 출렁이는 폭은 맞추지 않는다
    19일치 교차검증에서 both·dummy는 거의 같았고(MAE 4.86·4.81) camz만 뒤졌다(5.05). 2026-10-10"""
    if not cols:
        cols = ["_none"]
        train, rest = train.assign(_none=0.0), rest.assign(_none=0.0)
    if mode == "dummy":
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        scale = lambda df: (df[cols] - mu) / sd
    else:
        g = train.groupby("camera_id")[cols]
        mu, sd = g.mean(), g.std().replace(0, 1)
        scale = lambda df: (df[cols] - mu.loc[df["camera_id"]].values) / sd.loc[df["camera_id"]].values
    out = []
    for df in (train, rest):
        x = scale(df).fillna(0).values
        if mode != "camz":
            dummies = pd.get_dummies(df["camera_id"]).reindex(columns=sorted(train["camera_id"].unique()), fill_value=0)
            x = np.hstack([x, dummies.values.astype(float)])
        out.append(x)
    return out


MODELS = {      # 이름: (입력 열, 카메라 처리 방식)
    "평균만": (None, None),
    "카메라별 평균": ([], "both"),
    "기상만": (WEATHER, "both"),
    "이미지만": (IMAGE, "both"),
    "이미지+기상": (IMAGE + WEATHER, "both"),
    "이미지+기상 (표준화만)": (IMAGE + WEATHER, "camz"),
    "이미지+기상 (전체표준화+카메라)": (IMAGE + WEATHER, "dummy"),
}


def predict(train, rest, spec):
    cols, mode = spec
    if cols is None:
        return np.full(len(rest), train["pm25"].mean())
    xtr, xte = design(train, rest, cols, mode)
    return Ridge(alpha=1.0).fit(xtr, train["pm25"]).predict(xte)


def scores(df, pred):
    err = pred - df["pm25"].values
    hourly = df.assign(pred=pred).groupby(["camera_id", "pm_time"]).agg(pred=("pred", "mean"), pm25=("pm25", "first"))
    herr = hourly["pred"] - hourly["pm25"]
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "시간MAE": np.abs(herr).mean(), "시간RMSE": np.sqrt((herr ** 2).mean())}


def report(results):
    t = pd.DataFrame(results).T
    base = t.loc["평균만", "MAE"]
    t["MAE개선"] = (1 - t["MAE"] / base) * 100
    print(t.round(2).to_string(formatters={"MAE개선": "{:+.1f}%".format}))


def main():
    ap = argparse.ArgumentParser(description="이미지 통계·기상 → PM2.5 베이스라인")
    ap.add_argument("--test", action="store_true", help="시험지 A로 평가 (모델을 다 고른 뒤에만)")
    args = ap.parse_args()

    d = load()
    test = d["day"].map(is_test_day)
    dev, held = d[~test].reset_index(drop=True), d[test].reset_index(drop=True)
    print(f"개발 {dev['day'].nunique()}일 {len(dev)}장 / 시험지 A {held['day'].nunique()}일 {len(held)}장 "
          f"({', '.join(f'{x:%m/%d}' for x in sorted(held['day'].unique()))})")
    print(f"PM2.5 개발 구간 평균 {dev['pm25'].mean():.1f}, 표준편차 {dev['pm25'].std():.1f}, 최대 {dev['pm25'].max():.0f}")

    if args.test:
        print("\n[시험지 A] 개발 날짜 전체로 학습")
        report({name: scores(held, predict(dev, held, spec)) for name, spec in MODELS.items()})
        return

    print("\n[개발] 하루씩 빼고 나머지로 학습 (날짜 단위 교차검증)")
    preds = {name: np.zeros(len(dev)) for name in MODELS}
    for day in sorted(dev["day"].unique()):
        fold = dev["day"] == day
        for name, spec in MODELS.items():
            preds[name][fold.values] = predict(dev[~fold], dev[fold], spec)
    report({name: scores(dev, p) for name, p in preds.items()})

    # 사진마다 정답과 모델별 예측. 어느 날·카메라·시간대에서 틀리는지 볼 때, 나중에 CNN과 같은 사진끼리 비교할 때 쓴다
    out = dev[["image", "camera_id", "captured_at", "pm_time", "pm25", "humidity", "sun_alt"]].copy()
    for name, p in preds.items():
        out[name] = p.round(2)
    out.to_csv(PRED, index=False, encoding="utf-8-sig")
    print(f"\n사진별 예측: {PRED}")


if __name__ == "__main__":
    main()
