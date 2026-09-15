# CCTV 이미지 수집 스크립트 (MVP)
# 흐름: 설정 읽기 → 정각 기준 10분 단위 회차 대기 → 카메라별로 스트림 접속 → 프레임 1장 캡처·저장 → CSV/텍스트 로그 기록
# 실행: python collect_cctv.py          (반복 수집, Ctrl+C로 종료)
#       python collect_cctv.py --once   (즉시 1회 테스트 캡처 후 종료)
import argparse
import csv
import logging
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

# ───────── 상수 ─────────
KST = timezone(timedelta(hours=9), "KST")  # PC 시간대 설정과 무관하게 한국 시각(UTC+09:00) 고정
CONFIG_PATH = Path(__file__).with_name("cameras.toml")  # 기본 설정 파일: 스크립트와 같은 폴더
TIME_FMT = "%Y-%m-%d %H:%M:%S"  # CSV·로그에 쓰는 시각 형식 (KST)
LATE_GRACE_SEC = 60  # 예정 시각보다 이 초 이상 늦게 도달한 회차는 "이미 지나간 회차"로 보고 건너뜀

# CSV 열 순서
CSV_FIELDS = ["scheduled_at", "captured_at", "cctv_id", "result", "file_name", "attempts", "elapsed_sec", "error"]
# 설정 파일 필수 항목
SETTING_KEYS = ["interval_minutes", "save_dir", "log_dir", "timeout_sec",
                "retry_count", "retry_wait_sec", "warn_after_failures"]
CAMERA_KEYS = ["cctv_id", "name", "agency", "stream_id", "url", "lon", "lat", "enabled"]

log = logging.getLogger("collector")


def now_kst():
    """현재 한국 시각 반환"""
    return datetime.now(KST)


# ───────── 설정·로그 준비 ─────────
def load_config(path):
    """TOML 설정을 읽고 검증. (공통 설정, 사용 중인 카메라 목록) 반환. 문제가 있으면 ValueError"""
    # utf-8-sig: 메모장 등에서 BOM 포함 UTF-8로 저장해도 읽히도록 처리
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))

    s = cfg.get("settings", {})
    missing = [k for k in SETTING_KEYS if k not in s]
    if missing:
        raise ValueError(f"[settings] 필수 항목 누락: {missing}")
    if s["retry_wait_sec"] < 10:
        raise ValueError("retry_wait_sec는 10 이상이어야 합니다 (UTIC IP 차단 기준)")
    if s["retry_count"] < 0 or s["timeout_sec"] <= 0 or s["warn_after_failures"] < 1:
        raise ValueError("retry_count는 0 이상, timeout_sec와 warn_after_failures는 1 이상이어야 합니다")
    if s["interval_minutes"] <= 0 or 60 % s["interval_minutes"]:
        raise ValueError("interval_minutes는 60의 약수여야 합니다 (정각 기준 단위)")

    ids = set()
    for i, cam in enumerate(cfg.get("cameras", []), 1):
        missing = [k for k in CAMERA_KEYS if k not in cam]
        if missing:
            raise ValueError(f"{i}번째 [[cameras]] 필수 항목 누락: {missing}")
        if cam["cctv_id"] in ids:
            raise ValueError(f"cctv_id 중복: {cam['cctv_id']}")
        ids.add(cam["cctv_id"])

    cameras = [c for c in cfg.get("cameras", []) if c["enabled"]]
    if not cameras:
        raise ValueError("사용 중(enabled = true)인 카메라가 없습니다")
    return s, cameras


def setup_logging(log_dir):
    """텍스트 로그를 파일(collector.log)과 콘솔에 동시에 출력. 로그 시각도 KST로 표시"""
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=TIME_FMT)
    formatter.converter = lambda ts: datetime.fromtimestamp(ts, KST).timetuple()  # PC 시간대 대신 KST 사용
    for handler in (logging.FileHandler(log_dir / "collector.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def append_csv(csv_path, row):
    """CSV에 한 줄 추가. 파일이 없으면 헤더부터 작성. 엑셀에서 파일을 열어둬 쓰기 실패해도 수집은 계속"""
    try:
        new_file = not csv_path.exists()
        # utf-8-sig: 엑셀 한글 깨짐 방지용 BOM. 이어쓰기(a) 모드에서는 BOM이 파일 맨 앞에만 한 번 들어감
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        log.error(f"CSV 기록 실패({type(e).__name__}: {e}) → 누락된 행: {row}")


def empty_row(scheduled_at, cid, result, error):
    """캡처를 시도하지 않았거나(skipped) 예상치 못한 예외로 끝난(fail) 회차의 CSV 한 줄"""
    return {"scheduled_at": f"{scheduled_at:{TIME_FMT}}", "captured_at": "", "cctv_id": cid, "result": result,
            "file_name": "", "attempts": 0, "elapsed_sec": 0, "error": error}


# ───────── 캡처 ─────────
def capture_frame(url, timeout_sec):
    """스트림에 새로 접속해 프레임 1장을 읽고 즉시 연결 해제 (1회 시도).
    반환: (frame, 실패사유) — 성공 시 실패사유는 None. 캡처 방식이 바뀌면 이 함수만 교체"""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_sec * 1000,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_sec * 1000]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    try:
        if not cap.isOpened():
            return None, "스트림 열기 실패"
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return None, "프레임 읽기 실패(빈 프레임)"
        return frame, None
    finally:
        cap.release()  # 성공·실패·예외와 관계없이 연결을 유지하지 않음


def save_jpg(frame, path):
    """프레임을 jpg로 저장. 'x' 모드라 같은 이름 파일이 있으면 FileExistsError (절대 덮어쓰지 않음)"""
    ok, buf = cv2.imencode(".jpg", frame)  # imwrite 대신 인코딩 후 직접 쓰기 (Windows 한글 경로 문제 회피)
    if not ok:
        raise RuntimeError("jpg 인코딩 실패")
    with open(path, "xb") as f:
        f.write(buf.tobytes())


def capture_camera(cam, s, save_dir, scheduled_at=None, deadline=None):
    """카메라 1대의 한 회차 처리: 재시도 포함 캡처 → jpg 저장. CSV 한 줄(dict) 반환.
    scheduled_at이 None이면 --once 테스트: 실제 촬영 시각 + _test 파일명, 결과는 test
    deadline(다음 회차 시각)이 주어지면, 재시도 대기 후 그 시각을 넘기게 될 때 남은 재시도를 중단"""
    test = scheduled_at is None
    cid = cam["cctv_id"]
    start = time.monotonic()
    frame, error, captured_at, attempts = None, None, None, 0

    # 1) 캡처: 최초 1회 + 재시도 최대 retry_count회, 실패 후 retry_wait_sec 간격
    for attempt in range(s["retry_count"] + 1):
        if attempt:
            if deadline and now_kst() + timedelta(seconds=s["retry_wait_sec"]) >= deadline:
                error += " (다음 회차 시각이 가까워 남은 재시도 중단)"
                log.warning(f"{cid} 다음 회차 시각이 가까워 남은 재시도 중단")
                break
            log.info(f"{cid} {s['retry_wait_sec']}초 후 재시도 ({attempt}/{s['retry_count']})")
            time.sleep(s["retry_wait_sec"])
        attempts += 1
        try:
            frame, error = capture_frame(cam["url"], s["timeout_sec"])
        except Exception as e:
            frame, error = None, f"캡처 예외: {type(e).__name__}: {e}"
            log.exception(f"{cid} 캡처 중 예외")
        captured_at = now_kst()  # 실제 촬영(또는 마지막 시도) 시각
        if frame is not None:
            break
        log.warning(f"{cid} 시도 {attempts}회차 실패: {error}")

    # 2) 저장: 예정 시각 기준 파일명 (테스트는 실제 촬영 시각 + _test)
    file_name = ""
    if frame is not None:
        name_time = f"{captured_at:%Y%m%d_%H%M%S}_test" if test else f"{scheduled_at:%Y%m%d_%H%M}"
        try:
            save_jpg(frame, save_dir / f"{cid}_{name_time}.jpg")
            file_name = f"{cid}_{name_time}.jpg"
        except FileExistsError:
            error = f"같은 이름 파일이 이미 있어 저장하지 않음: {cid}_{name_time}.jpg"
            log.warning(f"{cid} {error}")
        except Exception as e:
            error = f"저장 실패: {type(e).__name__}: {e}"
            log.exception(f"{cid} jpg 저장 중 예외")

    elapsed = round(time.monotonic() - start, 1)
    if file_name:
        h, w = frame.shape[:2]
        log.info(f"{cid}({cam['name']}) 저장 성공: {file_name} | {w}x{h} | 시도 {attempts}회 | "
                 f"{elapsed}초 | 실제 촬영 {captured_at:{TIME_FMT}}")
        result = "test" if test else "success"
    else:
        log.error(f"{cid}({cam['name']}) 회차 실패: {error} | 시도 {attempts}회 | {elapsed}초")
        result = "test" if test else "fail"

    return {
        "scheduled_at": "" if test else f"{scheduled_at:{TIME_FMT}}",
        "captured_at": f"{captured_at:{TIME_FMT}}",
        "cctv_id": cid,
        "result": result,
        "file_name": file_name,
        "attempts": attempts,
        "elapsed_sec": elapsed,
        "error": "" if file_name else error,
    }


# ───────── 스케줄 ─────────
def next_slot(t, minutes):
    """t 이후 첫 정각 기준 단위 시각 (예: 10분 단위면 12:14 → 12:20)"""
    floor = t.replace(minute=t.minute - t.minute % minutes, second=0, microsecond=0)
    return floor + timedelta(minutes=minutes)


def wait_until(target):
    """target(KST)까지 대기. 1초 단위로 나눠 자며 매번 현재 시각을 다시 계산 (시계 보정·Ctrl+C 즉시 반응)"""
    while (remaining := (target - now_kst()).total_seconds()) > 0:
        time.sleep(min(remaining, 1))


def update_streak(cam, result, fail_counts, threshold):
    """연속 실패 횟수 갱신. 기준 도달 시 경고 1회, 경고 이후 다시 성공하면 복구 기록 1회. skipped는 영향 없음"""
    cid = cam["cctv_id"]
    if result == "success":
        if fail_counts[cid] >= threshold:
            log.info(f"[복구됨] {cid}({cam['name']}) {fail_counts[cid]}회차 연속 실패 후 캡처 성공")
        fail_counts[cid] = 0
    elif result == "fail":
        fail_counts[cid] += 1
        if fail_counts[cid] == threshold:
            log.warning(f"[경고] {cid}({cam['name']}) {threshold}회차 연속 실패 → "
                        f"브라우저에서 스트림 ID 변경 여부 확인 필요 (현재 stream_id={cam['stream_id']})")


def run_slot(slot, cameras, s, save_dir, csv_path, fail_counts):
    """한 회차 처리: 카메라를 순서대로 캡처 (동시 접속 없음). 한 카메라의 실패·예외가 다른 카메라에 영향 없음"""
    deadline = slot + timedelta(minutes=s["interval_minutes"])  # 다음 회차 시각
    log.info(f"회차 시작: {slot:%Y-%m-%d %H:%M}")
    for cam in cameras:
        cid = cam["cctv_id"]
        # 앞 카메라 처리가 밀려 다음 회차 시각에 도달했으면 이 카메라는 시작하지 않음
        if now_kst() >= deadline:
            reason = "이전 카메라 처리 지연으로 다음 회차 시각 도달"
            log.warning(f"{cid} 건너뜀: {reason}")
            append_csv(csv_path, empty_row(slot, cid, "skipped", reason))
            continue
        try:
            row = capture_camera(cam, s, save_dir, slot, deadline)
        except Exception as e:
            log.exception(f"{cid} 처리 중 예상치 못한 예외")
            row = empty_row(slot, cid, "fail", f"예외: {type(e).__name__}: {e}")
        append_csv(csv_path, row)
        update_streak(cam, row["result"], fail_counts, s["warn_after_failures"])


def run_schedule(cameras, s, save_dir, csv_path):
    """정각 기준 interval 단위로 회차를 무한 반복. 이미 지나간 회차는 몰아서 찍지 않고 skipped로 기록"""
    step = timedelta(minutes=s["interval_minutes"])
    fail_counts = {c["cctv_id"]: 0 for c in cameras}  # 카메라별 연속 실패 회차 수
    slot = next_slot(now_kst(), s["interval_minutes"])
    log.info(f"첫 회차 대기: {slot:%Y-%m-%d %H:%M}")
    while True:
        wait_until(slot)
        late = (now_kst() - slot).total_seconds()
        if late > LATE_GRACE_SEC:
            # 이전 회차 처리 지연·PC 절전 등으로 예정 시각이 이미 지난 회차
            reason = f"예정 시각 {late:.0f}초 경과(이전 회차 지연 또는 PC 절전 등)"
            log.warning(f"회차 건너뜀: {slot:%Y-%m-%d %H:%M} | {reason}")
            for cam in cameras:
                append_csv(csv_path, empty_row(slot, cam["cctv_id"], "skipped", reason))
        else:
            run_slot(slot, cameras, s, save_dir, csv_path, fail_counts)
        slot += step


# ───────── 실행 ─────────
def main():
    parser = argparse.ArgumentParser(description="CCTV 이미지 수집 (HLS 스트림 프레임 캡처)")
    parser.add_argument("--once", action="store_true", help="즉시 1회만 캡처하고 종료 (파일명 _test, CSV result=test)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="설정 파일 경로 (기본: cameras.toml)")
    args = parser.parse_args()

    # 설정 오류는 수집 시작 전에 이유를 출력하고 종료
    try:
        s, cameras = load_config(args.config)
        save_dir, log_dir = Path(s["save_dir"]), Path(s["log_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(log_dir)
    csv_path = log_dir / "capture_log.csv"
    cam_names = ", ".join(f"{c['cctv_id']}({c['name']})" for c in cameras)
    log.info(f"스크립트 시작 | 모드: {'1회 테스트(--once)' if args.once else '반복 수집'} | "
             f"간격 {s['interval_minutes']}분 | 카메라 {len(cameras)}대: {cam_names}")

    try:
        if args.once:
            # 카메라는 순서대로 하나씩 처리 (동시 접속 없음)
            for cam in cameras:
                append_csv(csv_path, capture_camera(cam, s, save_dir))
        else:
            run_schedule(cameras, s, save_dir, csv_path)
    except KeyboardInterrupt:
        log.info("사용자 종료 요청(Ctrl+C)")
    except Exception:
        log.exception("예상치 못한 오류로 스크립트 중단")
    finally:
        log.info("스크립트 종료")


if __name__ == "__main__":
    main()
