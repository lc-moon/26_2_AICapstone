# 즉석 추론: 카메라 1대를 "지금" 찍어 바로 추정한다 (전시 "직접 해보기"용)
#   python Estimator/infer_now.py 10042            → 표준출력에 JSON 한 줄
#   python Estimator/infer_now.py 10042 --measure  → 단계별 걸린 시간·메모리를 표준에러에 (VM에서 가능 여부 측정, S5a)
# 정기 추정기(estimate.py)와 같은 모델·전처리를 쓴다. 결과는 estimates.csv·DB에 넣지 않는다(시연용)
# 아직 측정용: 비·습도 거르기, 가린 사진 저장, 잠금은 S5a 통과 뒤에 붙인다
import json
import sys
import time
from datetime import datetime
from pathlib import Path

T0 = time.perf_counter()
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Collector"))
sys.path.insert(0, str(ROOT / "Estimator"))
from collect_cctv import CACHE_PATH, capture_frame   # noqa: E402
from estimate import MODEL, ONNX_DIR, preprocess     # noqa: E402

LIVE = ROOT / "data/live"
TIMEOUT_SEC = 10


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


def main():
    cam = sys.argv[1]
    measure = "--measure" in sys.argv
    steps, mem = {}, {"available_start_mb": mem_available_mb()}
    t = time.perf_counter()
    steps["import"] = t - T0

    url = json.loads(CACHE_PATH.read_text(encoding="utf-8"))[cam]["url"]
    requested = datetime.now()
    frame, err = capture_frame(url, TIMEOUT_SEC)
    steps["capture"] = time.perf_counter() - t
    if frame is None:
        sys.exit(f"{cam} 촬영 실패: {err}")
    captured = datetime.now()

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
    x, _ = preprocess(path, meta)
    feeds = {"image": x}
    if meta["weather"]:
        sys.exit("습도를 입력으로 받는 모델은 아직 지원하지 않습니다")
    y = sess.run(None, feeds)[0][0, 0]
    pm = max(0.0, float(np.expm1(y) if meta["log_target"] else y))
    steps["infer"] = time.perf_counter() - t
    mem["available_end_mb"] = mem_available_mb()
    mem["peak_rss_mb"] = peak_rss_mb()

    print(json.dumps({"camera_id": cam, "requested_at": requested.isoformat(timespec="seconds") + "+09:00",
                      "captured_at": captured.isoformat(timespec="seconds") + "+09:00",
                      "pm25": round(pm, 1), "status": "정상"}, ensure_ascii=False))
    if measure:
        steps["total"] = time.perf_counter() - T0
        print(json.dumps({"steps_s": {k: round(v, 2) for k, v in steps.items()}, "memory": mem}, ensure_ascii=False),
              file=sys.stderr)


if __name__ == "__main__":
    main()
