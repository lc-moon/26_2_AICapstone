# 카메라별 보정 (M2) — 저장된 CNN 예측 위에 카메라마다 절편(또는 기울기+절편)을 얹고 시험 데이터로 채점한다.
#
#   python Model/cnn.py --test incheon_log_top60      먼저 시험 예측을 만든다 (data/model/cnn/pred_test.csv)
#   python Model/calibrate.py incheon_log_top60       ⓐ 지금 서비스 모델
#   python Model/calibrate.py incheon_log_top60_1019  ⓑ 10/19까지 다시 학습한 모델
#
# 결과: data/model/cnn/{모델}/calib.csv  카메라별 ①의 a, ②의 a·b (보정값 = a + b × 예측, ①은 b = 1)
#                             pred_unseen.csv  최종 모델이 배우지 않은 학습 기간 날의 예측 (있을 때만)
#                             pred_test_calib.csv  시험 사진별 보정 전후 예측
#
# 규칙 (docs/전시_준비.md "M2에서 시험 점수를 보기 전에 정한 것", 2026-11-03)
#   - 보정은 시험 데이터로 맞추지 않는다. 학습 기간(9/22~10/19) 중 최종 모델이 배우지 않은 날이 있으면 그 날의 최종 모델 예측으로,
#     없으면 연습시험 예측(pred_cv.csv, 그 날이 빠진 묶음 모델의 예측)으로 맞춘다
#   - ① 절편만 ② 기울기+절편. 사진 1장 단위로 절대오차가 가장 작게(중앙값 회귀). 둘 다 시험 데이터로 채점하고
#     ②가 0.1 넘게 좋을 때만 ② (11/3 바꿈. 처음엔 맞추는 기간 안 하루씩 빼기로 하나만 골랐다. 그 결과는 참고로 출력한다)
#   - 맞추는 기간에 없는 카메라는 보정하지 않는다
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn
from baseline import scores

TOLERANCE = 0.1        # ②가 이보다 더 좋아야 ②를 쓴다
HUMID = 70             # 교체 기준의 "습한 날"


def fit_one(p, y, kind):
    """보정값 = a + b × 예측의 (a, b). ① 절편만은 b = 1, a = 오차의 중앙값"""
    if kind == "①":
        return float(np.median(y - p)), 1.0
    q = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(p.reshape(-1, 1), y)
    return float(q.intercept_), float(q.coef_[0])


def fit(df, kind):
    return {c: fit_one(g.pred.values, g.pm25.values, kind) for c, g in df.groupby("camera_id")}


def apply(df, coef):
    ab = np.array([coef.get(c, (0.0, 1.0)) for c in df.camera_id])
    return ab[:, 0] + ab[:, 1] * df.pred.values


def fit_source(run, d):
    """맞출 예측. 최종 모델이 배우지 않은 학습 기간 날이 있으면 그 날을 최종 모델로 추정하고, 없으면 연습시험 예측을 쓴다"""
    out = cnn.OUT / run
    cv = pd.read_csv(out / "pred_cv.csv", encoding="utf-8-sig", dtype={"camera_id": str})
    cv = cv[cv.camera_id != "bj"]
    learned = set(pd.to_datetime(cv.captured_at).dt.date)
    dev = d[(d.city == "inc") & (d.part != "A")]
    unseen = dev[~dev.day.isin(learned)]
    if unseen.empty:
        print(f"맞출 예측: 연습시험 예측 {cv.captured_at.str[:10].nunique()}일 {len(cv)}장 (pred_cv.csv)")
        return cv[["image", "camera_id", "captured_at", "pm_time", "pm25", "pred"]]

    saved = out / "pred_unseen.csv"
    if saved.exists():               # 사진 목록이 같으면 GPU로 다시 추정하지 않는다 (다른 학습이 GPU를 쓰는 중일 수 있다)
        src = pd.read_csv(saved, encoding="utf-8-sig", dtype={"camera_id": str})
        if sorted(src.image) == sorted(unseen.image):
            print(f"맞출 예측: 최종 모델이 배우지 않은 {unseen.day.nunique()}일 {len(src)}장 (저장된 {saved.name})")
            return src
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    weather = info.get("weather", [])
    arr = cnn.inputs(d, info["input"])["all"]
    model = cnn.make_model(pretrained=False, n_weather=len(weather))
    model.load_state_dict(cnn.torch.load(out / "model.pt", map_location=cnn.DEV))
    W = d[weather].values.astype(np.float32) if weather else None
    p = cnn.predict(model, arr, unseen.index.values, info["log_target"], W)
    src = unseen[["image", "camera_id", "captured_at", "pm_time", "pm25"]].assign(pred=p.round(2))
    src.to_csv(out / "pred_unseen.csv", index=False, encoding="utf-8-sig")
    print(f"맞출 예측: 최종 모델이 배우지 않은 {unseen.day.nunique()}일 {len(src)}장 "
          f"({', '.join(f'{x:%m/%d}' for x in sorted(unseen.day.unique()))}) → {out / 'pred_unseen.csv'}")
    return src


def leave_one_day_out(src):
    days = src.captured_at.str[:10]
    rows = {}
    for kind in ("①", "②"):
        p = np.empty(len(src))
        for day in days.unique():
            held = (days == day).values
            p[held] = apply(src[held], fit(src[~held], kind))
        rows[kind] = scores(src, p)
    rows["보정 없음"] = scores(src, src.pred.values)
    return pd.DataFrame(rows).T.loc[["보정 없음", "①", "②"]]


def bias_table(t, cols):
    bins = pd.cut(t.humidity, [0, 50, HUMID, 85, 95], right=False, labels=["<50", f"50~{HUMID}", f"{HUMID}~85", "85~95"])
    rows = {k: (t[c] - t.pm25).groupby(bins, observed=True).mean() for k, c in cols.items()}
    out = pd.DataFrame(rows)
    out.loc[f"{HUMID}% 이상"] = [(t[c] - t.pm25)[t.humidity >= HUMID].mean() for c in cols.values()]
    out["장"] = list(bins.value_counts(sort=False)) + [int((t.humidity >= HUMID).sum())]
    return out


def day_compare(t, cols, n_boot=5000):
    """날짜별 MAE와, 시험 날짜를 무작위로 다시 뽑았을 때(부트스트랩) MAE 차이의 95% 범위.
    보정에는 학습의 흔들림이 없어 0.1 규칙의 근거(다시 학습한 흔들림 0.03~0.04)가 맞지 않는다. 대신 "하필 이 13일"의 흔들림을 잰다"""
    day = t.captured_at.str[5:10]
    err = pd.DataFrame({k: (t[c] - t.pm25).abs() for k, c in cols.items()})
    by_day = err.groupby(day)
    print(by_day.mean().round(2).to_string())
    sums, counts = by_day.sum().values, by_day.size().values
    names = list(cols)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(counts), (n_boot, len(counts)))
    mae = sums[pick].sum(1) / counts[pick].sum(1)[:, None]          # 뽑을 때마다 사진 수만큼 가중한 MAE
    daily = by_day.mean()
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        lo, hi = np.percentile(mae[:, i] - mae[:, j], [2.5, 97.5])
        wins = int((daily[names[i]] < daily[names[j]]).sum())
        print(f"{names[i]} − {names[j]}: 이긴 날 {wins}/{len(daily)}, MAE 차이 95% 범위 {lo:+.2f} ~ {hi:+.2f}")


def main():
    ap = argparse.ArgumentParser(description="카메라별 보정 (M2)")
    ap.add_argument("run", help="data/model/cnn/ 아래 모델 폴더 이름 (예: incheon_log_top60)")
    args = ap.parse_args()

    test_csv = cnn.OUT / "pred_test.csv"
    test = pd.read_csv(test_csv, encoding="utf-8-sig", dtype={"camera_id": str}) if test_csv.exists() else pd.DataFrame(columns=["run"])
    test = test[(test.run == args.run) & (test.sheet == "A")].reset_index(drop=True)
    if test.empty:
        sys.exit(f"{test_csv}에 {args.run}의 시험 예측이 없습니다. python Model/cnn.py --test {args.run} 를 먼저 돌리세요")

    d = cnn.load()
    src = fit_source(args.run, d).reset_index(drop=True)

    print("\n[참고] 맞추는 기간 안에서 하루씩 빼고 (시험 데이터는 쓰지 않음)")
    print(leave_one_day_out(src).round(2).to_string())

    coefs = {k: fit(src, k) for k in ("①", "②")}
    n = src.camera_id.value_counts()
    pd.DataFrame([(c, coefs["①"][c][0], *coefs["②"][c], int(n[c])) for c in coefs["①"]],
                 columns=["camera_id", "a_①", "a_②", "b_②", "photos"]) \
        .round(3).to_csv(cnn.OUT / args.run / "calib.csv", index=False, encoding="utf-8-sig")
    missing = sorted(set(test.camera_id) - set(coefs["①"]))
    if missing:
        print(f"맞추는 기간에 없어 보정하지 않는 카메라: {missing}")

    cols = {"보정 전": "pred", "①": "calib1", "②": "calib2"}
    test["calib1"] = apply(test, coefs["①"]).round(2)
    test["calib2"] = apply(test, coefs["②"]).round(2)
    test = test.merge(d[["image", "humidity"]], on="image", how="left")
    print(f"\n[시험 데이터] {args.run} — 10/20~11/1 {test.captured_at.str[:10].nunique()}일 {len(test)}장")
    t = pd.DataFrame({k: scores(test, test[c].values) for k, c in cols.items()}).T
    print(t.round(2).to_string())
    kind = "②" if t.MAE["②"] < t.MAE["①"] - TOLERANCE else "①"
    print(f"→ {kind} ({'기울기+절편' if kind == '②' else '절편만'}{'' if kind == '②' else ', ②가 0.1 넘게 좋지 않으면 더 단순한 ①'})")

    print("\n날짜별 MAE")
    day_compare(test, cols)

    print("\n습도별 치우침 (예측 − 정답 평균)")
    print(bias_table(test, cols).round(2).to_string())

    err = {k: test[c] - test.pm25 for k, c in cols.items()}
    cam = pd.DataFrame({"장": test.camera_id.value_counts()})
    for k, e in err.items():
        cam[f"치우침_{k}"] = e.groupby(test.camera_id).mean()
    for k, e in err.items():
        cam[f"MAE_{k}"] = e.abs().groupby(test.camera_id).mean()
    cam["a_①"] = pd.Series({c: a for c, (a, _) in coefs["①"].items()})
    cam[["a_②", "b_②"]] = pd.DataFrame(coefs["②"], index=["a_②", "b_②"]).T
    print("\n카메라별")
    print(cam.sort_index().round(2).to_string())

    test.to_csv(cnn.OUT / args.run / "pred_test_calib.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {cnn.OUT / args.run / 'calib.csv'}, pred_test_calib.csv")


if __name__ == "__main__":
    main()
