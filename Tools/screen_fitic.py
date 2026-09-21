# fitic(인천교통정보센터) CCTV 화각 자동 스크리닝
# 흐름: 카메라 목록 받기 → 프레임 1장씩 캡처 → 하늘/원경 지표 계산 → 점수순 CSV + 썸네일 저장
#
# 사용법
#   python screen_fitic.py --limit 10      # 10대만 테스트
#   python screen_fitic.py                 # 전체 (237대)
#   python screen_fitic.py --list-only     # 목록만 확인하고 종료
#
# 기준은 일부러 널널하게 잡았습니다. 최종 판단은 육안으로 합니다.
import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

# ───────── 설정 ─────────
LIST_URL = "https://www.fitic.go.kr/gis/selectListData.do"
OLD_HOST = "http://61.40.94.13:1935"          # 목록이 돌려주는 원본 호스트
NEW_HOST = "https://cctv.fitic.go.kr"         # 사이트 자체 JS가 https에서 바꿔 쓰는 주소

ROOT = Path(__file__).resolve().parents[1]   # 리포 루트
OUT_DIR = ROOT / "data" / "screening"
OPEN_TIMEOUT = 8          # 스트림 열기 타임아웃(초). 짧게 잡아야 실패가 전체를 붙잡지 않음
READ_FRAMES = 2           # 읽어볼 프레임 수. 마지막 성공분을 사용

# 지표 임계값 (널널하게)
SKY_ROW_SPREAD = 28       # 이 값보다 균일한 행은 하늘로 간주 (p85-p15)
SKY_MIN_ROWS = 8          # 하늘로 인정할 최소 행 수
SKY_MIN_RATIO = 5.0       # 하늘이 화면에서 차지해야 할 최소 비율(%)
SKY_MAX_SPREAD = 15       # 하늘 가로 균일도 상한 (이보다 얼룩지면 기준으로 못 씀)
SAT_LEVEL = 240           # 포화로 보는 밝기
LOWRES_W = 960            # 이 폭 미만은 원경 디테일이 부족해 감점


# ───────── 1. 카메라 목록 ─────────
def fetch_cameras():
    """fitic 지도 페이지가 쓰는 내부 엔드포인트에서 카메라 목록을 받아온다."""
    r = requests.post(
        LIST_URL,
        data={"type": "cctv", "keyword": ""},
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        timeout=20,
    )
    r.raise_for_status()
    body = r.json()                 # 응답이 이중 인코딩(문자열 안에 JSON)이라 두 번 푼다
    if isinstance(body, str):
        body = json.loads(body)
    cams = []
    for c in body["result"]:
        cams.append({
            "id": c["CCTV_MNGM_NMBR"],
            "name": c["CCTV_NM"],
            "lon": c["X_CRDN"],
            "lat": c["Y_CRDN"],
            "url": c["HTTPADDR"].replace(OLD_HOST, NEW_HOST),
        })
    return cams


# ───────── 2. 프레임 1장 캡처 ─────────
def grab_frame(url):
    """HLS 스트림에서 프레임 한 장. 실패하면 None (재시도 없음 — 전수 훑기라 속도 우선)"""
    opts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT * 1000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, OPEN_TIMEOUT * 1000]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, opts)
    frame = None
    try:
        if cap.isOpened():
            for _ in range(READ_FRAMES):
                ok, f = cap.read()
                if not ok:
                    break
                frame = f
    finally:
        cap.release()
    return frame


# ───────── 3. 지표 계산 ─────────
def spread(a):
    """p85-p15. 표준편차보다 작은 오버레이에 덜 흔들린다."""
    return float(np.percentile(a, 85) - np.percentile(a, 15))


def find_sky_band(g):
    """상단 45% 안에서 균일한 행이 가장 길게 이어지는 구간을 하늘로 본다.
    맨 위부터 세지 않는 이유: 타임스탬프·안내문구 같은 오버레이가 1행만 걸쳐도 전부 놓치기 때문."""
    h, w = g.shape
    x0, x1 = int(w * 0.08), int(w * 0.72)
    flat = [spread(g[y, x0:x1]) <= SKY_ROW_SPREAD for y in range(int(h * 0.45))]

    best = (0, 0)                       # (시작행, 길이)
    start = None
    for y, f in enumerate(flat + [False]):
        if f and start is None:
            start = y
        elif not f and start is not None:
            if y - start > best[1]:
                best = (start, y - start)
            start = None
    return best                          # 길이 0이면 하늘 없음


def measure(frame):
    """프레임 한 장에서 하늘·원경 지표를 뽑는다."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = g.shape
    x0, x1 = int(w * 0.08), int(w * 0.72)

    sky_y0, sky_h = find_sky_band(g)
    m = {"w": w, "h": h, "sky_y0": sky_y0, "sky_h": sky_h,
         "sky_ratio": round(sky_h / h * 100, 1)}

    if sky_h < SKY_MIN_ROWS:                       # 하늘 없음 → 기준을 못 만든다
        m.update(sky_mean=0, sky_spread=0, sky_sat=0,
                 c_min=0, c_max=0, c_spread=0, score=0, note="하늘없음")
        return m, None

    sky = g[sky_y0:sky_y0 + sky_h, x0:x1]
    sky_mean = float(sky.mean())

    if m["sky_ratio"] < SKY_MIN_RATIO:              # 하늘이 너무 얇으면 기준으로 못 쓴다
        m.update(sky_mean=round(sky_mean, 1), sky_spread=0, sky_sat=0,
                 c_min=0, c_max=0, c_spread=0, score=0, note="하늘부족")
        return m, None

    # 하늘은 아래쪽 지면보다 밝다. 이 조건이 없으면 균일한 아스팔트 노면을 하늘로 오인한다
    ground_mean = float(g[int(h * 0.5):, x0:x1].mean())
    if sky_mean < ground_mean + 10:
        m.update(sky_mean=round(sky_mean, 1), sky_spread=0, sky_sat=0,
                 c_min=0, c_max=0, c_spread=0, score=0, note="하늘아님(노면추정)")
        return m, None

    m["sky_mean"] = round(sky_mean, 1)
    # 균일도는 '행별 가로 균일도의 중앙값'으로 잰다.
    # 2D 전체로 재면 하늘의 자연스러운 위아래 그라데이션까지 불균일로 잡혀 멀쩡한 카메라가 깎인다
    m["sky_spread"] = round(float(np.median([spread(r) for r in sky])), 1)
    m["sky_sat"] = round(float((sky >= SAT_LEVEL).mean()) * 100, 2)

    if m["sky_spread"] > SKY_MAX_SPREAD:            # 하늘이라기엔 가로로 너무 얼룩짐
        m.update(c_min=0, c_max=0, c_spread=0, score=0, note="하늘불균일")
        return m, None

    # 하늘 바로 아래 띠 = 원경 밴드. 좌→우 8칸으로 쪼개 대비를 잰다
    band_h = max(30, int(h * 0.10))
    y0, y1 = sky_y0 + sky_h, min(h, sky_y0 + sky_h + band_h)
    band = g[y0:y1, x0:x1]
    cols = np.array_split(band, 8, axis=1)
    contrasts = [(sky_mean - c.mean()) / sky_mean * 100 for c in cols if c.size]
    m["c_min"] = round(min(contrasts), 1)
    m["c_max"] = round(max(contrasts), 1)
    m["c_spread"] = round(max(contrasts) - min(contrasts), 1)

    # 점수 (0~100). 널널한 가중합이며 순서 정렬용일 뿐, 합격선이 아님
    s_area = min(sky_h / h, 0.20) / 0.20 * 25                      # 하늘 면적
    s_unif = max(0.0, 1 - m["sky_spread"] / 40) * 25               # 하늘 균일도
    s_sat = max(0.0, 1 - m["sky_sat"] / 5) * 15                    # 포화 여유
    # 소실 구간 존재: 원경 밴드가 하늘보다 어두워야(대비 양수) 정상.
    # 음수면 하늘을 잘못 잡았거나 밴드에 하늘보다 밝은 것이 들어온 것이므로 점수를 주지 않는다
    s_far = 0.0 if m["c_min"] < 0 else max(0.0, 1 - m["c_min"] / 25) * 20
    # 대비 사다리. 재질 차이(어두운 나무 vs 밝은 건물)로도 벌어지는 값이라 가중치를 낮게 둔다
    s_lad = min(m["c_spread"], 20) / 20 * 8
    score = s_area + s_unif + s_sat + s_far + s_lad
    if w < LOWRES_W:                                               # 저해상도 감점
        score *= 0.8
    m["score"] = round(score, 1)
    m["note"] = ""
    return m, (y0, y1)


# ───────── 4. 한 대 처리 ─────────
def safe_name(s):
    return re.sub(r'[\\/:*?"<>|]', "_", s)


def imwrite(path, img, quality=88):
    """cv2.imwrite는 Windows에서 한글 경로에 쓰지 못한다. 인코딩 후 직접 기록."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        Path(path).write_bytes(buf.tobytes())
    return ok


def load_frame(path):
    """한글 경로 대응 이미지 읽기"""
    buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rescore(cams):
    """이미 저장된 프레임으로 지표만 다시 계산한다. 임계값을 바꿔가며 볼 때 사용."""
    by_id = {c["id"]: c for c in cams}
    rows = []
    for p in sorted((OUT_DIR / "frames").glob("*.jpg")):
        cid = int(p.name.split("_", 1)[0])
        cam = by_id.get(cid, {"id": cid, "name": p.stem.split("_", 1)[-1],
                              "lat": "", "lon": "", "url": ""})
        frame = load_frame(p)
        row = {k: cam.get(k, "") for k in ("id", "name", "lat", "lon", "url")}
        if frame is None:
            row.update(score=0, note="읽기실패")
        else:
            m, _ = measure(frame)
            row.update(m)
        rows.append(row)
    return rows


def similarity(a, b):
    """두 프레임의 닮은 정도(-1~1). 축소·흐림 후 상관계수라 차량 통행 정도로는 잘 안 떨어지고,
    카메라가 돌아가면 크게 떨어진다."""
    def prep(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = g[:int(g.shape[0] * 0.45)]          # 상단만 본다 — 차량 통행은 아래쪽에서 일어나므로
        g = cv2.GaussianBlur(cv2.resize(g, (160, 45)), (5, 5), 0).astype(np.float64)
        return g - g.mean()
    A, B = prep(a), prep(b)
    d = np.linalg.norm(A) * np.linalg.norm(B)
    return float((A * B).sum() / d) if d else 0.0


def ptz_check(cams, workers, thresh=0.90):
    """저장된 프레임과 지금 프레임을 비교해 화각이 변한 카메라를 찾는다.
    시간 간격이 길수록 정확하다 — 처음 수집 후 몇 시간 뒤에 돌릴 것."""
    scores = {}
    csv_path = OUT_DIR / "results.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                scores[int(r["id"])] = float(r["score"] or 0)

    targets = [c for c in cams
               if scores.get(c["id"], 0) > 0
               and (OUT_DIR / "frames" / f'{c["id"]:04d}_{safe_name(c["name"])}.jpg').exists()]
    (OUT_DIR / "frames2").mkdir(parents=True, exist_ok=True)
    print(f"    대상 {len(targets)}대 (점수>0인 것만)")

    def one(cam):
        tag = f'{cam["id"]:04d}_{safe_name(cam["name"])}'
        old = load_frame(OUT_DIR / "frames" / f"{tag}.jpg")
        new = grab_frame(cam["url"])
        if new is None or old is None:
            return {"id": cam["id"], "name": cam["name"], "sim": "", "판정": "비교불가"}
        imwrite(OUT_DIR / "frames2" / f"{tag}.jpg", new, 85)
        if old.shape != new.shape:
            return {"id": cam["id"], "name": cam["name"], "sim": 0.0, "판정": "해상도변경"}
        s = similarity(old, new)
        return {"id": cam["id"], "name": cam["name"], "sim": round(s, 3),
                "판정": "화각변경 의심" if s < thresh else "고정"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, targets))
    rows.sort(key=lambda r: (r["sim"] if isinstance(r["sim"], float) else 9))

    with open(OUT_DIR / "ptz_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "name", "sim", "판정"])
        wr.writeheader()
        wr.writerows(rows)

    bad = [r for r in rows if r["판정"] != "고정"]
    print(f"\n    화각변경 의심·비교불가 {len(bad)}대 / 전체 {len(rows)}대")
    for r in bad:
        print(f'      {r["sim"]:>6} {r["id"]:>5}  {r["name"]}  ← {r["판정"]}')
    print(f'\n    {OUT_DIR / "ptz_check.csv"}')


def process(cam):
    frame = grab_frame(cam["url"])
    row = {k: cam[k] for k in ("id", "name", "lat", "lon", "url")}
    if frame is None:
        row.update(score=0, note="캡처실패")
        return row

    m, band = measure(frame)
    row.update(m)

    tag = f'{cam["id"]:04d}_{safe_name(cam["name"])}'
    imwrite(OUT_DIR / "frames" / f"{tag}.jpg", frame, 85)
    if band:                                        # 지평선 띠만 잘라 크게 — 육안 확인용
        y0, y1 = band
        strip = frame[max(0, y0 - 20):min(frame.shape[0], y1 + 20)]
        if strip.size:
            sc = 1800 / strip.shape[1]
            strip = cv2.resize(strip, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
            imwrite(OUT_DIR / "strips" / f"{tag}.jpg", strip)
    return row


# ───────── 5. 실행 ─────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="앞에서 N대만 처리 (0=전체)")
    p.add_argument("--workers", type=int, default=6, help="동시 처리 수")
    p.add_argument("--list-only", action="store_true", help="목록만 출력하고 종료")
    p.add_argument("--rescore", action="store_true",
                   help="저장된 프레임으로 점수만 다시 계산 (재캡처 없음)")
    p.add_argument("--ptz-check", action="store_true",
                   help="저장된 프레임과 비교해 화각이 변한 카메라 찾기 (몇 시간 뒤 실행할 것)")
    args = p.parse_args()

    print("[1] 카메라 목록 요청")
    cams = fetch_cameras()
    print(f"    {len(cams)}대 수신")
    if args.list_only:
        for c in cams[:20]:
            print(f"    {c['id']:>4} {c['name']}  ({c['lat']}, {c['lon']})")
        print(f"    ... 이하 생략 (총 {len(cams)}대)")
        return

    for d in ("frames", "strips"):
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.ptz_check:
        print("[2] 화각 변경 점검")
        ptz_check(cams, args.workers)
        print(f"\n[3] 완료 — {time.time()-t0:.0f}초")
        return

    if args.rescore:
        print("[2] 저장된 프레임으로 재채점")
        rows = rescore(cams)
        print(f"    {len(rows)}장 처리")
    else:
        if args.limit:
            cams = cams[:args.limit]
        print(f"[2] 프레임 캡처 및 지표 계산 — {len(cams)}대, 동시 {args.workers}")
        rows, done = [], 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for row in ex.map(process, cams):
                rows.append(row)
                done += 1
                print(f"\r    {done}/{len(cams)}  {time.time()-t0:5.0f}초", end="", flush=True)
        print()

    rows.sort(key=lambda r: r.get("score", 0), reverse=True)
    for r in rows:                                   # 해상도를 한 칸으로 합쳐 보기 쉽게
        r["res"] = f'{r["w"]}x{r["h"]}' if r.get("w") else ""
    cols = ["score", "id", "name", "res", "sky_ratio", "sky_spread", "sky_sat",
            "c_min", "c_max", "c_spread", "note", "lat", "lon", "url"]
    csv_path = OUT_DIR / "results.csv"
    for n in range(10):                              # Excel에 열려 있으면 다른 이름으로 저장
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(rows)
            break
        except PermissionError:
            csv_path = OUT_DIR / f"results_{n + 2}.csv"
    else:
        print("    CSV 저장 실패 — 열려 있는 파일을 닫고 다시 실행하세요")

    ok = [r for r in rows if r.get("score", 0) > 0]
    fail = len(rows) - len(ok)
    print(f"\n[3] 완료 — {time.time()-t0:.0f}초 | 성공 {len(ok)} / 실패·하늘없음 {fail}")
    print(f"    {csv_path}\n")
    print(f"{'점수':>5} {'번호':>5}  {'이름':<18} {'해상도':>9} {'하늘%':>6} {'균일':>5} "
          f"{'포화%':>6} {'대비min':>7} {'사다리':>6}")
    for r in ok[:25]:
        print(f'{r["score"]:>5} {r["id"]:>5}  {r["name"][:18]:<18} {r["res"]:>9} '
              f'{r["sky_ratio"]:>6} {r["sky_spread"]:>5} {r["sky_sat"]:>6} '
              f'{r["c_min"]:>7} {r["c_spread"]:>6}')


if __name__ == "__main__":
    main()
