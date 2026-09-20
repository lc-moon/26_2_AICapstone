"""raw 폴더의 저장본(API 응답 원문)만으로 숭의 측정소 PM2.5 등급 분포를 분석한다.

산출물: data/songui_pm25_hourly.csv, charts/*.png, summary.md
API는 호출하지 않는다 (조회는 fetch_songui.py 담당).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"
CHART_DIR = BASE_DIR / "charts"

# 등급 기준 (환경부, ㎍/㎥): 좋음 0~15 / 보통 16~35 / 나쁨 36~75 / 매우나쁨 76 이상
GRADES = ["좋음", "보통", "나쁨", "매우나쁨"]
BINS = [-float("inf"), 15, 35, 75, float("inf")]

# 색상: 메인 네이비, 강조 오렌지, 등급색은 채도를 낮춰 구분만 되게
NAVY = "#1B3A5C"
ORANGE = "#F2994A"
GRADE_COLORS = {"좋음": "#5B8DB8", "보통": "#7FB77E", "나쁨": ORANGE, "매우나쁨": "#C8553D"}
SOURCE = "자료: 한국환경공단 에어코리아 (실시간 미확정 자료)"

# 한글 폰트와 마이너스 기호 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_raw():
    """ 저장된 모든 페이지를 읽어 필요한 열만 DataFrame으로 합친다. """
    rows = []
    for path in sorted(RAW_DIR.glob("page_*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))["response"]["body"]["items"]
        rows += [{k: it.get(k) for k in ("dataTime", "pm25Value", "pm25Flag")} for it in items]
    return pd.DataFrame(rows)


def parse_time(s):
    """ KST 'YYYY-MM-DD HH:MM'을 시각으로 변환한다. '24:00'은 다음 날 00:00으로 바꾼다. """
    if s.endswith("24:00"):
        return pd.Timestamp(s[:10]) + pd.Timedelta(days=1)
    return pd.Timestamp(s)


def clean(raw):
    """ 시각 변환, 결측 분류, 등급 부여를 한다. """
    df = pd.DataFrame()
    df["datetime"] = raw["dataTime"].map(parse_time)
    df["pm25_flag"] = raw["pm25Flag"]
    # 결측: 플래그가 있거나 값이 '-'·빈 값인 경우
    value = raw["pm25Value"].fillna("").str.strip()
    df["missing"] = raw["pm25Flag"].notna() | value.isin(["-", ""])
    df["pm25"] = pd.to_numeric(value.where(~df["missing"]), errors="coerce")
    df["grade"] = pd.cut(df["pm25"], bins=BINS, labels=GRADES, right=True)
    df["date"] = df["datetime"].dt.date
    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    df = df.sort_values("datetime").reset_index(drop=True)
    # 중복 시각이 있으면 분석이 왜곡되므로 중단
    assert not df["datetime"].duplicated().any(), "중복 시각이 있습니다"
    return df[["datetime", "date", "month", "pm25", "grade", "missing", "pm25_flag"]]


def md_table(frame):
    """ DataFrame을 마크다운 표 문자열로 만든다 (tabulate 의존성 없이). """
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def add_source(fig):
    """ 그래프 하단에 출처 문구를 넣는다. """
    fig.text(0.01, 0.015, SOURCE, fontsize=13, color="#555555", ha="left", va="bottom")


def chart_grade_bar(dist, period, path):
    """ 전체 기간 등급별 비율 막대그래프 (16:9, 200dpi) """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    bars = ax.bar(GRADES, dist["비율(%)"], color=[GRADE_COLORS[g] for g in GRADES], width=0.6)
    # 막대 위에 비율과 시간 수 표기
    for bar, (_, r) in zip(bars, dist.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{r['비율(%)']:.1f}%\n({r['시간 수']:,}시간)",
                ha="center", va="bottom", fontsize=20, color=NAVY, fontweight="bold")
    ax.set_ylim(0, max(dist["비율(%)"].max() * 1.25, 10))
    ax.set_ylabel("비율 (%)", fontsize=18, color=NAVY)
    ax.set_title(f"숭의 측정소 PM2.5 등급별 비율 ({period})", fontsize=26, color=NAVY,
                 fontweight="bold", pad=20)
    ax.tick_params(axis="x", labelsize=20, colors=NAVY)
    ax.tick_params(axis="y", labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    add_source(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def chart_daily_line(daily, period, path):
    """ 일평균 PM2.5 추이 선그래프 + 등급 구간 배경 (16:9, 200dpi) """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ymax = max(85, daily["일평균"].max() * 1.1)
    # 등급 구간 배경색과 경계선(15, 35, 75)
    edges = [0, 15, 35, 75, ymax]
    for g, lo, hi in zip(GRADES, edges[:-1], edges[1:]):
        ax.axhspan(lo, hi, color=GRADE_COLORS[g], alpha=0.12, lw=0)
        ax.text(1.005, (lo + hi) / 2, g, transform=ax.get_yaxis_transform(),
                fontsize=15, color=GRADE_COLORS[g], va="center", fontweight="bold")
    for y in (15, 35, 75):
        ax.axhline(y, color="#888888", lw=1, ls="--")
    ax.plot(pd.to_datetime(daily["날짜"]), daily["일평균"], color=NAVY, lw=2.5)
    ax.set_ylim(0, ymax)
    ax.set_xlim(pd.to_datetime(daily["날짜"]).min(), pd.to_datetime(daily["날짜"]).max())
    ax.set_ylabel("일평균 PM2.5 (㎍/㎥)", fontsize=18, color=NAVY)
    ax.set_title(f"숭의 측정소 일평균 PM2.5 추이 ({period})", fontsize=26, color=NAVY,
                 fontweight="bold", pad=20)
    ax.tick_params(labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    add_source(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    raw = load_raw()
    midnight = raw["dataTime"].str.endswith("24:00").sum()  # 24:00 표기 건수
    df = clean(raw)
    df.to_csv(DATA_DIR / "songui_pm25_hourly.csv", index=False, encoding="utf-8-sig")

    valid = df[~df["missing"]]
    n_total, n_valid, n_missing = len(df), len(valid), int(df["missing"].sum())
    start, end = df["datetime"].min(), df["datetime"].max()
    period = f"{start:%Y.%m.%d}~{end:%Y.%m.%d}"
    flag_counts = df.loc[df["missing"], "pm25_flag"].fillna("(플래그 없음)").value_counts()

    # 1) 기간·건수 요약
    overview = pd.DataFrame({
        "항목": ["첫 시각 (KST)", "마지막 시각 (KST)", "전체 시간 수", "유효 시간 수", "결측 시간 수",
               "24:00 표기 변환 건수"],
        "값": [f"{start:%Y-%m-%d %H:%M}", f"{end:%Y-%m-%d %H:%M}", f"{n_total:,}", f"{n_valid:,}",
              f"{n_missing:,} (" + ", ".join(f"{k} {v}" for k, v in flag_counts.items()) + ")",
              f"{midnight:,}"],
    })

    # 2) 전체 등급 분포 (비율 분모 = 유효 시간 수)
    counts = valid["grade"].value_counts().reindex(GRADES, fill_value=0)
    dist = pd.DataFrame({"등급": GRADES, "시간 수": counts.values,
                         "비율(%)": (counts.values / n_valid * 100).round(1)})
    assert dist["시간 수"].sum() == n_valid  # 등급 합계 = 유효 시간 수 검증

    # 3) 월별 등급 분포 (비율 %, 괄호 안은 시간 수)
    ct = pd.crosstab(valid["month"], valid["grade"]).reindex(columns=GRADES, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    monthly = pd.DataFrame({"월": ct.index, "유효 시간": ct.sum(axis=1).values})
    for g in GRADES:
        monthly[g] = [f"{p:.1f}% ({c})" for p, c in zip(pct[g], ct[g])]

    # 4) 최고 농도, 나쁨 이상 시각
    max_val = valid["pm25"].max()
    max_times = valid.loc[valid["pm25"] == max_val, "datetime"]
    bad = valid[valid["grade"].isin(["나쁨", "매우나쁨"])]
    n_bad = len(bad)

    # 5) 일평균 (유효 시간만 평균, 해당 날짜의 유효 시간 수도 함께 기록)
    daily = valid.groupby("date")["pm25"].agg(["mean", "count"]).reset_index()
    daily.columns = ["날짜", "일평균", "유효 시간"]
    daily["일평균"] = daily["일평균"].round(1)
    max_day = daily.loc[daily["일평균"].idxmax()]

    # 그래프 저장
    chart_grade_bar(dist, period, CHART_DIR / "01_grade_ratio_bar.png")
    chart_daily_line(daily, period, CHART_DIR / "02_daily_mean_line.png")

    # 발표용 핵심 문장 3개
    good_normal = dist.loc[dist["등급"].isin(["좋음", "보통"]), "비율(%)"].sum()
    headline = [
        f"최근 3개월({period}) 숭의 측정소 PM2.5 나쁨 이상은 {n_bad:,}시간({n_bad / n_valid * 100:.1f}%)에 불과했다"
        f" (유효 {n_valid:,}시간 기준).",
        f"같은 기간 측정 시간의 {good_normal:.1f}%가 '좋음' 또는 '보통' 등급이었고, "
        f"'좋음'만 {dist.loc[0, '비율(%)']:.1f}%였다.",
        f"일평균 최고치도 {max_day['일평균']:.1f}㎍/㎥({max_day['날짜']:%m월 %d일})로 "
        + ("나쁨 기준(36㎍/㎥)에 미치지 못해, 여름·가을철에는 고농도(나쁨 이상) 데이터가 거의 쌓이지 않는다."
           if max_day["일평균"] <= 35 else "나쁨 수준에 도달한 날이 있었다."),
    ]

    bad_table = pd.DataFrame({"시각 (KST)": bad["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
                              "PM2.5 (㎍/㎥)": bad["pm25"].astype(int), "등급": bad["grade"]})
    daily_md = daily.assign(날짜=daily["날짜"].astype(str))

    md = [
        "# 숭의 측정소 최근 3개월 PM2.5 등급 분포",
        "",
        "## 발표용 핵심 문장",
        *[f"{i}. {s}" for i, s in enumerate(headline, 1)],
        "",
        "> 자료: 한국환경공단 에어코리아 실시간 측정값(미확정). 비율의 분모는 유효 시간 수이며, "
        "시각은 KST(24:00은 다음 날 00:00으로 변환).",
        "",
        "## 1. 조회 기간과 데이터 수", md_table(overview), "",
        "## 2. 전체 기간 등급 분포", "등급 기준(㎍/㎥): 좋음 0~15 / 보통 16~35 / 나쁨 36~75 / 매우나쁨 76 이상", "",
        md_table(dist), "",
        "## 3. 월별 등급 분포", "비율(%) (시간 수). 첫 달과 마지막 달은 일부 기간만 포함.", "",
        md_table(monthly), "",
        "## 4. 최고 농도와 나쁨 이상 발생 시각",
        f"- 최고 농도: **{int(max_val)}㎍/㎥** — " + ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in max_times),
        f"- 나쁨 이상 발생: {n_bad}시간", "",
        md_table(bad_table) if n_bad else "(없음)", "",
        "## 5. 일평균 PM2.5 추이", "유효 시간이 적은 날(첫날·마지막 날, 결측 발생일)은 평균의 대표성이 낮음.", "",
        md_table(daily_md), "",
    ]
    (BASE_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # 콘솔 요약
    print(md_table(overview), md_table(dist), md_table(monthly), sep="\n\n")
    print(f"\n최고 {max_val} @ {list(max_times.astype(str))}, 나쁨 이상 {n_bad}시간")
    print(f"일평균 최고 {max_day['일평균']} ({max_day['날짜']}), 유효시간<12인 날 {int((daily['유효 시간'] < 12).sum())}일")
    print("\n".join(headline))


if __name__ == "__main__":
    main()
