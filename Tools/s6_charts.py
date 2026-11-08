# 원리 페이지(S6) 결과 절 그래프 2장: 오차 비교 막대, 습도 구간별 치우침.
# 시험 데이터 10/20~11/1 예측(data/model/cnn/pred_test.csv)으로 그린다 → data/review/s6/
# 서비스 저장소 web/about/img/에는 2_error.png(1_오차비교), 3_humidity.png(2_습도치우침) 이름으로 넣었다(2026-11-08).
#   python Tools/s6_charts.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[1] / "data"
OUT = D / "review" / "s6"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b2"

p = pd.read_csv(D / "model" / "cnn" / "pred_test.csv", encoding="utf-8-sig")
p = p[p.sheet == "A"].merge(pd.read_csv(D / "labels" / "labels.csv", encoding="utf-8-sig", usecols=["image", "humidity"]), on="image")
A, F = "incheon_log_top60", "incheon_log_top60_h-phys_1019_lr3e-05"   # 처음 모델 ⓐ, 서비스 모델 ⓑ'+습도 물리식
a, f = p[p.run == A], p[p.run == F]
mae = lambda g: (g.pred - g.pm25).abs().mean()
mean_only = (f.pm25 - 13.2).abs().mean()      # 학습 기간 평균(13.2)만 늘 부르기
NEIGHBOR = 3.24                               # 이웃 측정소 실측을 그대로 답으로 낸 MAE — Tools/station_floor.py
print(f"MAE: 평균만 {mean_only:.2f}, 처음 모델 {mae(a):.2f}, 최종 {mae(f):.2f}")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=12)


# 1. 얼마나 맞히나 — 막대 하나 계열(값 직접 표시, 최종만 강조)
rows = [("늘 평균(13)만 부르기", mean_only, GRAY), ("처음 모델 (10/9까지 배움)", mae(a), GRAY),
        ("4km 떨어진 이웃 측정소 실측", NEIGHBOR, GRAY), ("최종 모델 (28일 + 습도)", mae(f), BLUE)]
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150, facecolor=SURF)
style(ax)
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
for yi, (name, v, _) in zip(y, rows):
    ax.text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=13, color=INK, fontweight="bold" if "최종" in name else "normal")
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(0, 6.2)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("사진 한 장당 평균 오차 (㎍/㎥, 작을수록 좋음)", color=INK2, fontsize=11)
ax.set_title("처음 보는 13일(10/20~11/1) 사진 6,852장으로 채점", loc="left", color=INK, fontsize=14, pad=12)
fig.tight_layout()
fig.savefig(OUT / "1_오차비교.png", facecolor=SURF)

# 2. 습한 날 착각 — 습도 구간별 치우침(예측 − 실측), 처음 모델 · 28일 · 최종
bins, labels = [0, 50, 70, 85, 95], ["50% 미만", "50~70%", "70~85%", "85~95%"]
bias = lambda g: g.groupby(pd.cut(g.humidity, bins, right=False, labels=labels), observed=True).apply(lambda x: (x.pred - x.pm25).mean(), include_groups=False)
n = f.groupby(pd.cut(f.humidity, bins, right=False, labels=labels), observed=True).size()
b2 = p[p.run == "incheon_log_top60_1019_lr3e-05"]                 # ⓑ': 28일, 습도 없음 — 날 수와 습도의 몫을 가르려고 가운데에 둔다
series = [("처음 모델 (14일)", bias(a), ORANGE), ("28일로 다시 배움", bias(b2), "#1baf7a"), ("28일 + 습도 (최종)", bias(f), BLUE)]
print(pd.DataFrame({name: s for name, s, _ in series} | {"장": n}).round(2))
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=SURF)
style(ax)
x = np.arange(len(labels))
w = 0.26
for k, (name, s, col) in enumerate(series):
    xs = x + (k - 1) * (w + 0.02)
    ax.bar(xs, s.values, w, color=col, label=name)
    for xi, v in zip(xs, s.values):
        ax.text(xi, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x, [f"{l}\n({k:,}장)" for l, k in zip(labels, n.values)])
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("평균 치우침 (예측 - 실측, ㎍/㎥)", color=INK2, fontsize=11)
lo = min(s.min() for _, s, _ in series)
hi = max(s.max() for _, s, _ in series)
ax.set_ylim(lo - 0.7, hi + 0.8)
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.set_title("습할수록 뿌연 화면을 먼지로 착각하던 버릇", loc="left", color=INK, fontsize=14, pad=12)
fig.tight_layout()
fig.savefig(OUT / "2_습도치우침.png", facecolor=SURF)
print("저장:", OUT)
