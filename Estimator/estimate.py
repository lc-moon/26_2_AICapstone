# 추정기 — 수집기가 저장한 사진을 읽어 카메라별 PM2.5를 추정한다. 수집기와 따로 돈다(추정 쪽 오류가 수집을 멈추지 않게).
#
#   python Estimator/estimate.py            찍은 지 3분 넘은 사진 중 아직 안 한 것 처리 (systemd 타이머가 1분마다 부른다)
#                                           회차(:00/:20/:40) 5분 뒤부터 15분 뒤까지는 1분마다, 그 뒤 남은 사진은 5분마다(정각 기준) 처리한다
#   python Estimator/estimate.py --latest   가장 최근 낮 회차를 다시 처리 (밤에 동작 확인용, 결과에 "테스트" 표시)
#   python Estimator/estimate.py --check-preprocess   PC 전용. 학습 때 만든 사진 사본과 전처리가 같은지 비교
#
# 입력: data/logs/capture_log.csv·data/images/ (수집기), Model/text_mask/{장면}.png, Estimator/stations.json
#       data/model/onnx/{모델}.onnx·.json (Estimator/export_onnx.py, scp로 옮김), .env의 KMA_HUB_KEY (AWS 매분자료)
# 결과: data/estimates/estimates.csv  사진마다 한 줄 (추정값 또는 추정 안 한 사유)
#       data/estimates/index.html     카메라별 최신 값 (web 서비스가 127.0.0.1:8080으로 보여 준다. SSH 터널로 본다)
#       서비스 백엔드로 전송 (send.py, live만. .env에 주소·토큰이 있을 때. 실패해도 위 파일은 그대로)
#
# 학습과 같게 (Model/cnn.py _resized·batch): 글자 자리를 평균색으로 칠하고 → 위쪽 60% 자르고 → 384×384 INTER_AREA → RGB → 표준화.
# 습도는 학습 때와 같은 관측소(stations.json)의 사진 찍은 '분'의 1분값. 시각이 맞지 않는 값으로 메우지 않는다(현황 §2).
# 습도는 아래 거르기에 늘 쓰고, 모델 입력으로는 모델이 받을 때(.json의 weather)만 넣는다.
# 학습에서 뺀 조건(비, 습도 95% 이상)은 추정하지 않고 사유를 남긴다. 밤은 수집기가 이미 건너뛴다
# 비 판단: 60분 강수량 > 0 또는 레이더 강우강도 0.5 mm/h 이상이면 비. 학습 라벨은 강수 감지(RE)로 걸렀지만 RE는 당일 결측이라
# 실시간에는 쓸 수 없다(10/18). 60분 강수량은 1~2분 뒤 올라와 늘 있고, 레이더는 4~5분(늦으면 10분 넘게) 뒤 올라오고 빈 응답이 잦다(10/19).
# 그래서 레이더는 기다리지 않는다 — 처리할 때 와 있으면 보태고, 없으면 60분 강수량으로 판단한다.
# 기다리는 건 60분 강수량이 결측이고 레이더도 없을 때뿐이다(회차 15분 뒤까지 1분마다). 그래도 없으면 일단 추정한다(10/18 결정)
import argparse
import csv
import html
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/logs/capture_log.csv"
IMAGES = ROOT / "data/images"
MASKS = ROOT / "Model/text_mask"
STATIONS = Path(__file__).with_name("stations.json")
MODEL = "incheon_log_top60"                     # 인천 ver 최종 (⑤ 위쪽 60%, 기상 없음). 2026-10-15 인천+베이징 ver(mixed_top60_w-hum)에서 교체
                                                # — 인천에서는 인천 ver가 같거나 낫고, 베이징 학습의 이점이 인천 고농도에서 확인되지 않음(현황 §7)
                                                # 바꿀 때 send.py의 MODEL도 같이 (백엔드에 보내는 모델 이름)
ONNX_DIR = ROOT / "data/model/onnx"
OUT = ROOT / "data/estimates"
AWS_MIN = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
RADAR = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-rdr_cmp_aws_pt_data"   # AWS지점별 레이더 합성, 5분 간격
WAIT_MIN = 3          # 1분 습도는 1~2분 뒤에 올라온다(10/13 실측). 찍은 지 이만큼 지난 사진만 처리
GIVE_UP_MIN = 60      # 이만큼 지나도 습도가 없으면 '습도 결측'으로 남기고 넘어간다
START_MIN = 5         # 회차 이만큼 뒤부터 처리한다
QUICK_MIN = 15        # 회차 이만큼 뒤까지는 1분마다 다시 본다. 레이더는 15분이면 99% 올라온다(10/19 실측)
RADAR_MMH = 0.5       # 레이더 강우강도 이 이상이면 비. 10/18 비 온 날 대조에서 비 99%, 헛걸림 4.5%
HUMIDITY_MAX = 95     # 학습에서 뺀 조건 (Model/baseline.py)
TOP = 0.6             # ⑤ 위쪽 60%
FIELDS = ["captured_at", "camera_id", "name", "image", "humidity", "rain", "pm25_est", "status", "mode", "estimated_at"]


def env_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("KMA_HUB_KEY="):
            return line.split("=", 1)[1].strip().strip('"\'')
    sys.exit(".env에 KMA_HUB_KEY가 없습니다")


def preprocess(path, meta):
    """사진 한 장 → (1,3,384,384) float32. 학습 때 cnn._resized + cnn.batch와 같은 계산"""
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    scene = f"{path.parts[-3]}@{w}x{h}"
    mp = MASKS / f"{scene}.png"
    if mp.exists():
        img[cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0] = meta["fill_bgr"]
    img = img[:round(TOP * h), :w]
    img = cv2.resize(img, (meta["size"], meta["size"]), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    x = (img.astype(np.float32) - np.float32(meta["mean"]) * 255) / (np.float32(meta["std"]) * 255)
    return x.transpose(2, 0, 1)[None], scene if mp.exists() else f"{scene}(글자 가림 없음)"


class Humidity:
    """관측소·시간별 1분값을 한 번에 받아 둔다. 실패하면 None (live는 다음 번에 다시).
    API가 가끔 30초 걸려 빈 응답(머리말만)을 준다(10/13 실측). 특히 아직 오지 않은 분까지 범위에 넣으면 그렇다.
    그래서 범위를 필요한 분까지만 잡고, 자료 줄이 하나도 없으면 실패로 보고 다시 묻는다"""

    def __init__(self, key):
        self.key, self.cache = key, {}

    def get(self, stn, t):
        k, minute = (stn, t.strftime("%Y%m%d%H")), t.strftime("%Y%m%d%H%M")
        if k not in self.cache or (self.cache[k] is not None and minute not in self.cache[k]):
            t0 = t.replace(minute=0)
            tm2 = min(t0 + timedelta(minutes=59), datetime.now() - timedelta(minutes=1), max(t, t0))
            rows = None
            for attempt in range(3):
                try:
                    r = requests.get(AWS_MIN, params={"tm1": t0.strftime("%Y%m%d%H%M"), "tm2": tm2.strftime("%Y%m%d%H%M"),
                                                      "stn": stn, "disp": "0", "help": "0", "authKey": self.key}, timeout=40)
                    got = {}
                    for line in r.content.decode("euc-kr", "replace").splitlines():
                        f = line.split()
                        if len(f) >= 15 and f[0][:1].isdigit():
                            hm, rn60 = float(f[14]), float(f[11])
                            got[f[0]] = (hm if hm > -50 else None, rn60 if rn60 > -50 else None)
                    if got:
                        rows = got
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            self.cache[k] = rows
        rows = self.cache[k]
        return None if rows is None else rows.get(minute, (None, None))


def frame_time(v):
    """레이더 응답의 시각 칸(YYYYMMDDHHMI)이 32비트 정수로 넘쳐 온다 → 2^32를 더해 되돌린다(10/18 확인)"""
    v = int(v)
    while v < 200000000000:
        v += 2 ** 32
    return datetime.strptime(str(v), "%Y%m%d%H%M")


class Radar:
    """관측소별 레이더 강우강도(mm/h). 사진을 찍은 시각이 속한 5분 값을 쓴다.
    빈 응답이 잦다(재시도 없이 65%, 10/19 실측). 한 번 실행에서는 두 번만 묻고 없으면 None — 추정기가 1분마다 다시 돌므로
    여기서 오래 붙잡지 않는다(관측소 7곳에 긴 재시도를 걸면 한 번 실행이 몇 분씩 걸린다)"""

    def __init__(self, key):
        self.key, self.cache = key, {}

    def get(self, stn, t):
        frame = t.replace(minute=t.minute // 5 * 5, second=0)
        k = (stn, frame)
        if k not in self.cache:
            self.cache[k] = None
            when = frame.strftime("%Y%m%d%H%M")
            for attempt in range(2):
                try:
                    r = requests.get(RADAR, params={"tm1": when, "tm2": when, "itv": "5", "qcd": "EXT", "cmp": "HSP",
                                                    "stn": stn, "help": "0", "authKey": self.key}, timeout=5)
                    for line in r.content.decode("euc-kr", "replace").splitlines():
                        f = line.split(",")
                        if f[0][:1].isdigit() and len(f) > 4 and frame_time(f[0]) == frame:
                            v = float(f[4])
                            self.cache[k] = None if v <= -900 else max(0.0, v)   # -250은 에코 없음(비 없음)
                    if self.cache[k] is not None:
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(2)
        return self.cache[k]


def read_log():
    with open(LOG, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["result"] == "success" and r["file_name"]]


def read_done():
    p = OUT / "estimates.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        return {r["image"]: r for r in csv.DictReader(f) if r["mode"] == "live"}


def append(rows):
    p = OUT / "estimates.csv"
    new = not p.exists()
    with open(p, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def render(rows, mode):
    """카메라별 최신 한 줄씩 → index.html (숫자만)"""
    latest = {}
    for r in rows:
        latest[r["camera_id"]] = r
    cells = "".join(
        f"<tr><td>{html.escape(r['camera_id'])}</td><td>{html.escape(r['name'])}</td><td>{r['captured_at'][5:16]}</td>"
        f"<td class=v>{r['pm25_est'] or '-'}</td><td>{html.escape(r['status'])}</td><td>{r['humidity']}</td></tr>"
        for r in sorted(latest.values(), key=lambda r: r["name"]))
    note = "<p class=t>테스트: 가장 최근 낮 회차를 다시 계산한 값</p>" if mode == "test" else ""
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><meta http-equiv=refresh content=60>
<title>PM2.5 추정</title><style>body{{font-family:sans-serif;margin:16px}}td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}}
.v{{font-size:1.4em;font-weight:bold}}.t{{color:#b00}}</style>
<h1>CCTV PM2.5 추정 (㎍/㎥)</h1>{note}<p>모델 {MODEL} · 갱신 {datetime.now():%Y-%m-%d %H:%M} · 우리가 계산한 추정값이며 에어코리아 측정값이 아님</p>
<table><tr><th>카메라</th><th>위치</th><th>촬영</th><th>추정</th><th>상태</th><th>습도 %</th></tr>{cells}</table>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


def estimate(todo, mode, key):
    import onnxruntime as ort
    meta = json.loads((ONNX_DIR / f"{MODEL}.json").read_text(encoding="utf-8"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(ONNX_DIR / f"{MODEL}.onnx"), opts, providers=["CPUExecutionProvider"])
    stations = json.loads(STATIONS.read_text(encoding="utf-8"))
    hum, radar = Humidity(key), Radar(key)
    now = datetime.now()
    out = []
    for r in todo:
        t = datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S")
        late = now - datetime.strptime(r["scheduled_at"], "%Y-%m-%d %H:%M:%S") > timedelta(minutes=QUICK_MIN)
        row = {"captured_at": r["captured_at"], "camera_id": r["cctv_id"], "name": r["name"], "image": r["file_name"],
               "humidity": "", "rain": "", "pm25_est": "", "mode": mode, "estimated_at": now.strftime("%Y-%m-%d %H:%M:%S")}
        stn = stations.get(r["cctv_id"])
        v = hum.get(stn, t) if stn else (None, None)
        if v is None or v[0] is None:
            if mode == "live" and now - t < timedelta(minutes=GIVE_UP_MIN):
                continue                                   # 아직 안 올라왔거나 조회 실패 → 다음 번에 다시
            out.append({**row, "status": "습도 결측"})
            continue
        rn60 = v[1]
        rd = None if rn60 is not None and rn60 > 0 else radar.get(stn, t)     # 강수량이 이미 비면 레이더는 안 묻는다
        if rn60 is None and rd is None and mode == "live" and not late:
            continue                                       # 비를 판단할 근거가 둘 다 없다 → 회차 15분 뒤까지 1분마다 다시
        row["humidity"] = v[0]
        row["rain"] = ("60분 결측" if rn60 is None else f"60분 {rn60}mm") +             ("" if rn60 is not None and rn60 > 0 else " · 레이더 결측" if rd is None else f" · 레이더 {rd}mm/h")
        if rn60 is not None and rn60 > 0:
            out.append({**row, "status": "비(60분 강수량) — 추정 안 함"})
            continue
        if rd is not None and rd >= RADAR_MMH:
            out.append({**row, "status": "비(레이더) — 추정 안 함"})
            continue                                       # 둘 다 결측이면 확인할 수 없어 일단 추정한다(10/18 결정)
        if v[0] >= HUMIDITY_MAX:
            out.append({**row, "status": f"습도 {HUMIDITY_MAX}% 이상 — 추정 안 함"})
            continue
        x, scene = preprocess(IMAGES / r["file_name"], meta)
        if x is None:
            out.append({**row, "status": "사진 읽기 실패"})
            continue
        feeds = {"image": x}
        if meta["weather"]:                            # 습도를 입력으로 받는 모델만. 95% 거르기는 위에서 모델과 상관없이 한다
            feeds["weather"] = np.array([[v[0]]], dtype=np.float32)
        y = sess.run(None, feeds)[0][0, 0]
        pm = max(0.0, float(np.expm1(y) if meta["log_target"] else y))
        out.append({**row, "pm25_est": f"{pm:.1f}", "status": "정상" if "없음" not in scene else "정상(글자 가림 없음)"})
    return out


def check_preprocess(n=40):
    """PC 전용: 추정기 전처리 결과가 학습 때 사본(cnn_cache/top60_384.npy)과 같은지"""
    sys.path.insert(0, str(ROOT / "Model"))
    import cnn
    meta = json.loads((ONNX_DIR / f"{MODEL}.json").read_text(encoding="utf-8"))
    d = cnn.load()
    arr = cnn.cache(d, "top60")
    rows = d.index[d.city == "inc"].values[::max(1, int((d.city == "inc").sum()) // n)][:n]
    worst = 0.0
    for i in rows:
        x, _ = preprocess(IMAGES / d.image[i], meta)
        ref = ((arr[i].astype(np.float32) - np.float32(meta["mean"]) * 255) / (np.float32(meta["std"]) * 255)).transpose(2, 0, 1)
        worst = max(worst, float(np.abs(x[0] - ref).max()))
    print(f"사진 {len(rows)}장: 표준화 값 최대 차이 {worst:.6f} (0이면 학습 때와 똑같다)")


def send_live():
    """서비스 백엔드로 전송(send.py). 파일은 이미 저장했으므로 전송 쪽 문제는 로그만 남기고 넘어간다"""
    try:
        import send
        send.send_pending()
    except Exception as e:
        print(f"{datetime.now():%H:%M} 전송 오류: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="수집된 사진 → PM2.5 추정")
    ap.add_argument("--latest", action="store_true", help="가장 최근 낮 회차를 다시 처리 (동작 확인용, '테스트' 표시)")
    ap.add_argument("--check-preprocess", action="store_true", help="PC 전용: 학습 때 사본과 전처리 비교")
    args = ap.parse_args()
    if args.check_preprocess:
        return check_preprocess()
    OUT.mkdir(parents=True, exist_ok=True)
    log = read_log()
    if args.latest:
        last = max(r["scheduled_at"] for r in log)
        todo, mode = [r for r in log if r["scheduled_at"] == last], "test"
    else:
        now = datetime.now()
        done, cut = read_done(), now - timedelta(minutes=WAIT_MIN)

        def due(r):
            """회차 5~15분 뒤에는 매번(1분마다), 그 뒤 남은 사진(습도 결측 등)은 5분마다만"""
            age = now - datetime.strptime(r["scheduled_at"], "%Y-%m-%d %H:%M:%S")
            if age < timedelta(minutes=START_MIN):
                return False
            return age <= timedelta(minutes=QUICK_MIN) or now.minute % 5 == 0
        todo = [r for r in log if r["file_name"] not in done and r["scheduled_at"]
                and now - timedelta(hours=2) <= datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S") <= cut and due(r)]
        mode = "live"
    if not todo:
        if mode == "test":                             # live는 1분마다 돌아 대부분 할 일이 없다. 로그를 채우지 않게 조용히 끝낸다
            print("처리할 사진 없음")
        else:
            send_live()                                # 할 일이 없어도 못 보낸 행은 다시 보낸다
        return
    t0 = time.time()
    rows = estimate(todo, mode, env_key())
    append(rows)
    shown = rows if mode == "test" else list(read_done().values())
    render(shown, mode)
    ok = sum(r["pm25_est"] != "" for r in rows)
    print(f"{datetime.now():%H:%M} {mode}: 사진 {len(todo)}장 중 {len(rows)}장 기록 (추정 {ok}), {time.time() - t0:.1f}초")
    if mode == "live":
        send_live()


if __name__ == "__main__":
    main()
