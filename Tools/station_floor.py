# 왜 MAE 3 아래로 못 내려가나 (11/6): 정답(측정소) 흔들림과 모델 오차 나누기. 화면에만 출력하고 아무것도 저장하지 않는다.
# 원리 페이지에 인용한 숫자의 출처: 이웃 측정소 실측을 그대로 답으로 내면 MAE 3.24, 2.6~5km 떨어진 측정소끼리 같은 시각 값 차이 평균 약 3.1.
#   python Tools/station_floor.py
import glob
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

D = str(Path(__file__).resolve().parents[1] / "data")
RUNS = {"ⓐ": "incheon_log_top60", "ⓑ'+물리식": "incheon_log_top60_h-phys_1019_lr3e-05"}

# --- 측정소 시간값 (시험 기간) ---
rows = []
for f in glob.glob(D + r"\labels\raw\airkorea\*.json"):
    for r in json.load(open(f, encoding="utf-8")):
        v = r.get("pm25Value")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        rows.append((r["stationName"], r["dataTime"], v))
st = pd.DataFrame(rows, columns=["st", "t", "v"]).drop_duplicates(["st", "t"])
st = st[(st.t >= "2026-10-20 08:00") & (st.t <= "2026-11-02 00:00")]
w = st.pivot(index="t", columns="st", values="v")
hour = np.array([int(t[11:13]) for t in w.index])
w = w[(hour >= 8) & (hour <= 18)]           # 사진이 있는 낮 시간(끝 시각 표기)
loc = {}
js = json.load(open(D + r"\labels\raw\airkorea_stations_인천.json", encoding="utf-8"))["response"]["body"]["items"]
for s in js:
    loc[s["stationName"]] = (float(s["dmX"]), float(s["dmY"]))


def km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    return 6371 * math.hypot(math.radians(la2 - la1), math.radians(lo2 - lo1) * math.cos(math.radians((la1 + la2) / 2)))


names = [s for s in w.columns if s in loc]
pairs = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        x = (w[a] - w[b]).abs().dropna()
        pairs.append((a, b, km(loc[a], loc[b]), x.mean(), len(x)))
pp = pd.DataFrame(pairs, columns=["a", "b", "km", "mad", "n"]).sort_values("km")
print("측정소 쌍(시험 기간 낮 시간값) 같은 시각 값 차이 평균:")
for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 30)]:
    g = pp[(pp.km >= lo) & (pp.km < hi)]
    if len(g):
        print(f"  {lo}~{hi}km {len(g):2d}쌍: {g.mad.mean():.2f} (쌍별 {g.mad.min():.2f}~{g.mad.max():.2f})")
print("  가장 가까운 5쌍:", "; ".join(f"{r.a}-{r.b} {r.km:.1f}km {r.mad:.2f}" for r in pp.head(5).itertuples()))

# 한 측정소의 바로 앞 시간과의 차이 (1시간 안 변화의 크기 감)
dh = w.diff().abs()
print(f"같은 측정소 한 시간 전 값과의 차이 평균: {np.nanmean(dh.values):.2f}")

# --- 이웃 측정소를 예측값으로 쓰면 (사진 대신 '2~5km 떨어진 실제 측정기') ---
cams = {c["camera_id"]: (c["lat"], c["lon"]) for c in json.load(open(D + r"\service_shared\cameras.json", encoding="utf-8"))["cameras"]}
p = pd.read_csv(D + r"\model\cnn\pred_test.csv", encoding="utf-8-sig", dtype={"camera_id": str})
lab = pd.read_csv(D + r"\labels\labels.csv", encoding="utf-8-sig", dtype={"camera_id": str}, usecols=["image", "pm_station"])
p = p.merge(lab, on="image")
nb = {}
for c, xy in cams.items():
    own = p.loc[p.camera_id == c, "pm_station"].iloc[0] if (p.camera_id == c).any() else None
    cand = sorted((km(xy, loc[s]), s) for s in names if s != own)
    nb[c] = cand[0]
base = p[p.run == RUNS["ⓐ"]].copy()
base["nb"] = [w.at[t, nb[c][1]] if t in w.index else np.nan for c, t in zip(base.camera_id, base.pm_time)]
ok = base.dropna(subset=["nb"])
print(f"\n이웃 측정소(자기 측정소 다음으로 가까운 곳, 카메라에서 평균 {np.mean([v[0] for v in nb.values()]):.1f}km) 값을 그대로 예측으로 쓰면: "
      f"MAE {(ok.nb - ok.pm25).abs().mean():.2f} ({len(ok)}장)")

# --- 모델 오차 나누기 ---
p["day"] = p.captured_at.str[:10]
for lab_, run in RUNS.items():
    g = p[p.run == run].copy()
    g["e"] = g.pred - g.pm25
    m0 = g.e.abs().mean()
    d1 = (g.e - g.groupby("day").e.transform("median")).abs().mean()
    d2 = (g.e - g.groupby("pm_time").e.transform("median")).abs().mean()
    d3 = (g.e - g.groupby(["camera_id", "day"]).e.transform("median")).abs().mean()
    print(f"\n[{lab_}] 시험 MAE {m0:.2f} (평균 PM {g.pm25.mean():.1f}, 상대 {m0 / g.pm25.mean() * 100:.0f}%)")
    print(f"  날마다 공통 치우침을 알고 뺐다면 {d1:.2f} / 같은 시각(회차) 공통 치우침까지 {d2:.2f} / 카메라·날마다 치우침까지 {d3:.2f}")
    bins = [0, 10, 15, 25, 35, 100]
    g["b"] = pd.cut(g.pm25, bins, right=False)
    t = g.groupby("b", observed=True).agg(n=("e", "size"), mae=("e", lambda x: x.abs().mean()), bias=("e", "mean"), pm=("pm25", "mean"))
    t["오차몫%"] = (g.groupby("b", observed=True).e.apply(lambda x: x.abs().sum()) / g.e.abs().sum() * 100).round(0)
    t["상대%"] = (t.mae / t.pm * 100).round(0)
    print(t.round(2).to_string())
