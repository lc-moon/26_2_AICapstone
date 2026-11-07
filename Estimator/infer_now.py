# 즉석 추론: 카메라 1대를 "지금" 찍어 바로 추정한다 (전시 "직접 해보기", S5b)
#   python Estimator/infer_now.py 10042            → 표준출력에 JSON 한 줄, 종료 코드 0
#   python Estimator/infer_now.py 10042 --measure  → 단계별 걸린 시간·메모리를 표준에러에 (S5a 측정)
#
# 서비스 백엔드(POST /api/infer)가 .env의 INFER_CMD 뒤에 카메라 번호를 붙여 부른다. 약속은 서비스 저장소 server/NEXT.md 4번:
#   출력 camera_id, requested_at, captured_at, pm25(없으면 null), status, photo_url
#   종료 코드 75 = 정기 추정기(또는 다른 즉석 추론)가 돌고 있어 지금은 못 함 → 백엔드가 409. 그 밖에 0이 아니면 실패(표준에러에 이유) → 503
# 정기 추정기(estimate.py)와 같은 모델·전처리·거르기(밤, 비, 습도 95% 이상)를 쓴다. 결과는 estimates.csv·DB에 넣지 않는다(시연용)
# 가린 사진(아래 40% 모자이크, 640×360)을 {CAM_DIR}/live/{카메라}.jpg에 덮어쓴다. 원본은 VM 밖으로 나가지 않는다
#
# 습도·비: 1분값은 1~2분 뒤에 올라와 "찍은 그 분" 값은 아직 없다. 그래서 정기 추정과 달리 **최근 10분 안의 가장 늦은 1분값**을 쓰고,
# API가 5초 안에 답하지 않으면 정기 추정의 최근 1시간 안 값(estimates.csv)을 쓴다. 둘 다 없으면 추정하지 않는다.
# 이 값은 화면 시연용으로만 쓰고 저장하지 않는다. 레이더는 한 번만 짧게 묻고 없으면 60분 강수량으로만 판단한다
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = time.perf_counter()
import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Collector"))
sys.path.insert(0, str(ROOT / "Estimator"))
from collect_cctv import CACHE_PATH, capture_frame, sun_altitude   # noqa: E402
from estimate import (AWS_MIN, HUMIDITY_MAX, MODEL, ONNX_DIR, OUT, RADAR, RADAR_MMH,   # noqa: E402
                      display_copy, env_value, frame_time, preprocess)

KST = timezone(timedelta(hours=9))
LIVE = ROOT / "data/live"
LOCK = OUT / ".lock"          # estimate.py와 같은 잠금 파일. VM 메모리(1GB)에 모델 둘을 같이 올리지 않게
BUSY = 75                     # 잠금이 잡혀 있을 때 종료 코드 (백엔드가 409로 바꾼다)
TIMEOUT_SEC = 10              # 사진 받기
API_SEC = 5                   # 습도·비 API를 기다리는 시간. 넘으면 정기 추정의 최근 값
RECENT_MIN = 60               # 정기 추정 값을 대신 쓸 수 있는 나이


def mem_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None


def peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024   # 리눅스는 KB
    except ImportError:
        return None


def take_lock():
    """잡으면 파일을 돌려준다(끝날 때까지 열어 둔다). 다른 쪽이 잡고 있으면 종료 코드 75로 끝낸다. 윈도우(PC)에서는 잠그지 않는다"""
    try:
        import fcntl
    except ImportError:
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except PermissionError:
        pass
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("정기 추정 또는 다른 즉석 추론이 진행 중\n")
        sys.exit(BUSY)
    return fd


def weather_now(key, stn, now):
    """최근 10분 안의 가장 늦은 1분값 (습도, 60분 강수량, 레이더). 실패하면 None"""
    try:
        r = requests.get(AWS_MIN, params={"tm1": (now - timedelta(minutes=10)).strftime("%Y%m%d%H%M"),
                                          "tm2": (now - timedelta(minutes=1)).strftime("%Y%m%d%H%M"),
                                          "stn": stn, "disp": "0", "help": "0", "authKey": key}, timeout=API_SEC)
        rows = {}
        for line in r.content.decode("euc-kr", "replace").splitlines():
            f = line.split()
            if len(f) >= 15 and f[0][:1].isdigit() and float(f[14]) > -50:
                rows[f[0]] = (float(f[14]), float(f[11]) if float(f[11]) > -50 else None)
        if not rows:
            return None
        hm, rn60 = rows[max(rows)]
    except (requests.RequestException, ValueError):
        return None
    rd = None
    if not (rn60 is not None and rn60 > 0):           # 강수량이 이미 비면 레이더는 안 묻는다
        frame = (now - timedelta(minutes=5)).replace(second=0, microsecond=0)
        frame = frame.replace(minute=frame.minute // 5 * 5)
        try:
            r = requests.get(RADAR, params={"tm1": frame.strftime("%Y%m%d%H%M"), "tm2": frame.strftime("%Y%m%d%H%M"), "itv": "5",
                                            "qcd": "EXT", "cmp": "HSP", "stn": stn, "help": "0", "authKey": key}, timeout=3)
            for line in r.content.decode("euc-kr", "replace").splitlines():
                f = line.split(",")
                if f[0][:1].isdigit() and len(f) > 4 and frame_time(f[0]) == frame and float(f[4]) > -900:
                    rd = max(0.0, float(f[4]))
        except (requests.RequestException, ValueError):
            pass
    return hm, rn60, rd


def weather_recent(cam, now):
    """정기 추정이 최근 1시간 안에 그 카메라에 남긴 습도·비 판단. 없으면 None"""
    p = OUT / "estimates.csv"
    if not p.exists():
        return None
    last = None
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["camera_id"] == cam and r["mode"] == "live" and r["humidity"]:
                last = r
    if last is None or now - datetime.strptime(last["captured_at"], "%Y-%m-%d %H:%M:%S") > timedelta(minutes=RECENT_MIN):
        return None
    if "레이더" in last["status"]:
        return float(last["humidity"]), 0.0, RADAR_MMH
    return float(last["humidity"]), (1.0 if "비" in last["status"] else 0.0), None


def save_live(cam, frame, captured):
    out = Path(env_value("CAM_DIR") or ROOT / "data/cam") / "live"
    out.mkdir(parents=True, exist_ok=True)
    dst, tmp = out / f"{cam}.jpg", out / f"{cam}.tmp"
    ok, buf = cv2.imencode(".jpg", display_copy(frame.copy()), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    tmp.write_bytes(buf.tobytes())
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)                               # 브라우저가 반쯤 쓴 파일을 받지 않게
    return f"/cam/live/{cam}.jpg?v={captured:%Y%m%d%H%M%S}"


def main():
    if len(sys.argv) < 2:
        sys.exit("사용법: infer_now.py 카메라번호 [--measure]")
    cam = sys.argv[1]
    measure = "--measure" in sys.argv
    steps, mem = {"import": time.perf_counter() - T0}, {"available_start_mb": mem_available_mb()}
    cams = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if cam not in cams:
        sys.exit(f"모르는 카메라: {cam}")
    info = cams[cam]
    requested = datetime.now()
    out = {"camera_id": cam, "requested_at": requested.isoformat(timespec="seconds") + "+09:00",
           "captured_at": None, "pm25": None, "status": "", "photo_url": None}

    def done():
        print(json.dumps(out, ensure_ascii=False))
        if measure:
            steps["total"] = time.perf_counter() - T0
            print(json.dumps({"steps_s": {k: round(v, 2) for k, v in steps.items()}, "memory": mem}, ensure_ascii=False), file=sys.stderr)

    if sun_altitude(info["lat"], info["lon"], datetime.now(KST)) < 0:
        out["status"] = "밤 — 추정 안 함"                 # 어두운 사진은 배운 적이 없다(수집기도 건너뛴다)
        return done()
    lock = take_lock()                                 # noqa: F841  (끝날 때까지 쥐고 있는다)

    stn = json.loads((ROOT / "Estimator/stations.json").read_text(encoding="utf-8")).get(cam)
    key = env_value("KMA_HUB_KEY")
    pool = ThreadPoolExecutor(1)
    asked = pool.submit(weather_now, key, stn, requested) if stn and key else None   # 사진 받는 동안 같이 묻는다

    t = time.perf_counter()
    frame, err = capture_frame(info["url"], TIMEOUT_SEC)
    steps["capture"] = time.perf_counter() - t
    if frame is None:
        sys.exit(f"{cam} 촬영 실패: {err}")
    captured = datetime.now()
    out["captured_at"] = captured.isoformat(timespec="seconds") + "+09:00"
    out["photo_url"] = save_live(cam, frame, captured)

    t = time.perf_counter()
    w = asked.result() if asked else None
    if w is None:
        w = weather_recent(cam, captured)
    pool.shutdown(wait=False)
    steps["weather"] = time.perf_counter() - t
    if w is None:
        out["status"] = "습도 확인 불가 — 추정 안 함"
        return done()
    hm, rn60, rd = w
    if rn60 is not None and rn60 > 0:
        out["status"] = "비(60분 강수량) — 추정 안 함"
        return done()
    if rd is not None and rd >= RADAR_MMH:
        out["status"] = "비(레이더) — 추정 안 함"
        return done()
    if hm >= HUMIDITY_MAX:
        out["status"] = f"습도 {HUMIDITY_MAX}% 이상 — 추정 안 함"
        return done()

    # preprocess는 경로의 카메라 폴더 이름으로 글자 가림 틀을 찾으므로 같은 모양으로 둔다
    path = LIVE / cam / "now" / "frame.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)

    t = time.perf_counter()
    import onnxruntime as ort
    meta = json.loads((ONNX_DIR / f"{MODEL}.json").read_text(encoding="utf-8"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(ONNX_DIR / f"{MODEL}.onnx"), opts, providers=["CPUExecutionProvider"])
    steps["load_model"] = time.perf_counter() - t
    mem["available_after_load_mb"] = mem_available_mb()

    t = time.perf_counter()
    x, scene = preprocess(path, meta)
    if x is None:
        sys.exit(f"{cam} 사진 읽기 실패")
    feeds = {"image": x}
    if meta["weather"]:                                # 습도를 입력으로 받는 모델(11/7~ 서비스 모델)
        feeds["weather"] = np.array([[hm]], dtype=np.float32)
    y = sess.run(None, feeds)[0][0, 0]
    out["pm25"] = round(max(0.0, float(np.expm1(y) if meta["log_target"] else y)), 1)
    out["status"] = "정상" if "없음" not in scene else "정상(글자 가림 없음)"
    steps["infer"] = time.perf_counter() - t
    mem["available_end_mb"] = mem_available_mb()
    mem["peak_rss_mb"] = peak_rss_mb()
    done()


if __name__ == "__main__":
    main()
