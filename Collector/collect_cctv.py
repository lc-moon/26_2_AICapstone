# CCTV 이미지 수집 스크립트
# 흐름: 설정 읽기 → 카메라 주소 확보(목록 API/캐시) → 정각 기준 회차 대기
#       → 카메라별 태양고도 확인 → 프레임 1장 캡처·저장 → CSV/로그 기록 → 이상 시 Discord 알림
# 실행: python collect_cctv.py          (반복 수집, Ctrl+C로 종료)
#       python collect_cctv.py --once   (즉시 1회 테스트 캡처 후 종료, 야간 판정 무시)
import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import cv2
import requests
from dotenv import dotenv_values

import weather

# ───────── 상수 ─────────
KST = timezone(timedelta(hours=9), "KST")  # PC 시간대 설정과 무관하게 한국 시각 고정
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "cameras.toml"
CACHE_PATH = BASE_DIR / "url_cache.json"   # 목록 API 응답 보관 (조회 실패 시 대비)
ENV_PATH = BASE_DIR.parent / ".env"
TIME_FMT = "%Y-%m-%d %H:%M:%S"
LATE_GRACE_SEC = 60  # 예정 시각보다 이 초 이상 늦게 도달한 회차는 건너뜀
DISK_WARN_PCT = 80   # 일일 요약에서 디스크 사용률이 이 값을 넘으면 경고를 따로 보낸다
WEATHER_MINUTE = 40  # 기상 실황을 받는 분. 자료 게시가 매시 15~40분 사이라 이때 받는다

# 인천교통정보센터 CCTV 목록 (지도 페이지가 쓰는 엔드포인트)
LIST_URL = "https://www.fitic.go.kr/gis/selectListData.do"
OLD_HOST = "http://61.40.94.13:1935"
NEW_HOST = "https://cctv.fitic.go.kr"

CSV_FIELDS = ["scheduled_at", "captured_at", "cctv_id", "name", "result",
              "file_name", "sun_alt", "attempts", "elapsed_sec", "error"]
SETTING_KEYS = ["interval_minutes", "save_dir", "log_dir", "timeout_sec", "retry_count",
                "retry_wait_sec", "warn_after_failures", "min_sun_altitude_deg", "jpeg_quality"]

log = logging.getLogger("collector")


def now_kst():
    return datetime.now(KST)


# ───────── 설정·로그 ─────────
def load_config(path):
    """TOML 설정을 읽고 검증. (공통 설정, 사용 중인 카메라 목록) 반환"""
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))

    s = cfg.get("settings", {})
    if missing := [k for k in SETTING_KEYS if k not in s]:
        raise ValueError(f"[settings] 필수 항목 누락: {missing}")
    if not 70 <= s["jpeg_quality"] <= 100:
        raise ValueError("jpeg_quality는 70~100 사이여야 합니다 (원경 대비가 신호라 과한 압축은 금물)")
    if s["retry_wait_sec"] < 10:
        raise ValueError("retry_wait_sec는 10 이상이어야 합니다")
    if s["retry_count"] < 0 or s["timeout_sec"] <= 0 or s["warn_after_failures"] < 1:
        raise ValueError("retry_count는 0 이상, timeout_sec와 warn_after_failures는 1 이상이어야 합니다")
    if s["interval_minutes"] <= 0 or 60 % s["interval_minutes"]:
        raise ValueError("interval_minutes는 60의 약수여야 합니다")

    ids = set()
    for i, cam in enumerate(cfg.get("cameras", []), 1):
        if "id" not in cam or "enabled" not in cam:
            raise ValueError(f"{i}번째 [[cameras]]에 id 또는 enabled가 없습니다")
        if cam["id"] in ids:
            raise ValueError(f"id 중복: {cam['id']}")
        ids.add(cam["id"])

    cameras = [c for c in cfg.get("cameras", []) if c["enabled"]]
    if not cameras:
        raise ValueError("사용 중(enabled = true)인 카메라가 없습니다")
    return s, cameras


def setup_logging(log_dir):
    """텍스트 로그를 파일과 콘솔에 동시 출력. 시각은 KST"""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=TIME_FMT)
    fmt.converter = lambda ts: datetime.fromtimestamp(ts, KST).timetuple()
    for h in (logging.FileHandler(log_dir / "collector.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def append_csv(csv_path, row):
    """CSV에 한 줄 추가. 엑셀에서 열어둬 쓰기 실패해도 수집은 계속"""
    try:
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)
    except OSError as e:
        log.error(f"CSV 기록 실패({type(e).__name__}: {e}) → 누락된 행: {row}")


def disk_usage(path):
    """path가 속한 디스크의 (사용률 %, 남은 GB). df의 Use%·Avail과 같은 기준으로 계산한다.
    경로가 없으면(로컬 정리 중 삭제 등) 저장소 쪽 경로로 물러나 요약이 죽지 않게 한다."""
    total, used, free = shutil.disk_usage(path if path.exists() else BASE_DIR)
    return round(used / (used + free) * 100), round(free / 1024 ** 3, 1)


# ───────── 알림 ─────────
class Notifier:
    """Discord 웹훅 알림. URL이 없으면 조용히 아무것도 하지 않는다(로그는 그대로 남음)."""

    def __init__(self, webhook):
        self.webhook = webhook

    def send(self, text):
        if not self.webhook:
            return
        try:
            requests.post(self.webhook, json={"content": text[:1900]}, timeout=10)
        except requests.RequestException as e:
            log.warning(f"알림 전송 실패: {type(e).__name__}")  # 알림 실패가 수집을 막지 않도록


# ───────── 카메라 주소 확보 ─────────
def fetch_camera_list():
    """목록 API에서 전체 카메라 정보를 받아 {번호: {...}}로 반환. 응답이 이중 인코딩이라 두 번 푼다."""
    r = requests.post(LIST_URL, data={"type": "cctv", "keyword": ""},
                      headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
                      timeout=20)
    r.raise_for_status()
    body = r.json()
    if isinstance(body, str):
        body = json.loads(body)
    return {c["CCTV_MNGM_NMBR"]: {"name": c["CCTV_NM"], "lat": c["Y_CRDN"], "lon": c["X_CRDN"],
                                  "url": c["HTTPADDR"].replace(OLD_HOST, NEW_HOST)}
            for c in body["result"]}


def resolve_cameras(cameras, use_cache_on_fail=True):
    """설정의 카메라 번호에 이름·좌표·스트림 주소를 채운다.
    목록 조회에 실패하면 직전에 저장해둔 캐시를 쓴다 (조회 불가가 곧 수집 중단이 되지 않도록)."""
    table, source = None, "목록 API"
    try:
        table = fetch_camera_list()
        CACHE_PATH.write_text(json.dumps({str(k): v for k, v in table.items()},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning(f"목록 조회 실패({type(e).__name__}: {e})")
        if not (use_cache_on_fail and CACHE_PATH.exists()):
            raise
        table = {int(k): v for k, v in json.loads(CACHE_PATH.read_text(encoding="utf-8")).items()}
        source = f"캐시({CACHE_PATH.name})"

    resolved, missing = [], []
    for cam in cameras:
        info = table.get(cam["id"])
        if info:
            resolved.append({"id": cam["id"], **info})
        else:
            missing.append(cam["id"])
    if missing:
        log.error(f"목록에 없는 카메라 번호: {missing} → 이번 실행에서 제외")
    if not resolved:
        raise RuntimeError("주소를 확보한 카메라가 없습니다")
    log.info(f"카메라 주소 확보: {len(resolved)}대 ({source})")
    return resolved


# ───────── 태양고도 ─────────
def sun_altitude(lat, lon, when):
    """해당 시각·좌표의 태양고도(도). 간이 계산이라 오차가 0.1도 수준이지만 주야 판정에는 충분하다."""
    utc = when.astimezone(timezone.utc)
    d = (utc - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)                       # 평균 근점이각
    q = (280.459 + 0.98564736 * d) % 360                                     # 평균 황경
    lam = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    eps = math.radians(23.439 - 0.00000036 * d)                              # 황도경사
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))            # 적경
    dec = math.asin(math.sin(eps) * math.sin(lam))                           # 적위
    gmst = (18.697374558 + 24.06570982441908 * d) % 24                       # 그리니치 항성시
    ha = math.radians((gmst * 15 + lon) % 360) - ra                          # 시간각
    phi = math.radians(lat)
    alt = math.asin(math.sin(phi) * math.sin(dec) +
                    math.cos(phi) * math.cos(dec) * math.cos(ha))
    return math.degrees(alt)


# ───────── 캡처 ─────────
def capture_frame(url, timeout_sec):
    """스트림에 새로 접속해 프레임 1장을 읽고 즉시 연결 해제 (1회 시도).
    반환: (frame, 실패사유) — 성공 시 실패사유는 None"""
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
        cap.release()


def save_jpg(frame, path, quality):
    """프레임을 jpg로 저장. 'xb' 모드라 같은 이름이 있으면 FileExistsError (덮어쓰지 않음)"""
    # imwrite 대신 인코딩 후 직접 쓰기 (한글 경로 대응)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpg 인코딩 실패")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(buf.tobytes())


def image_path(save_dir, cam_id, slot):
    """images/{카메라번호}/{날짜}/{시각}.jpg — 카메라·날짜별로 나눠 한 폴더에 파일이 몰리지 않게 한다"""
    return save_dir / str(cam_id) / f"{slot:%Y-%m-%d}" / f"{slot:%H%M}.jpg"


def capture_camera(cam, s, save_dir, slot=None, deadline=None):
    """카메라 1대의 한 회차 처리: 재시도 포함 캡처 → jpg 저장. CSV 한 줄(dict) 반환"""
    test = slot is None
    cid = cam["id"]
    start = time.monotonic()
    frame, error, captured_at, attempts = None, None, None, 0

    for attempt in range(s["retry_count"] + 1):
        if attempt:
            if deadline and now_kst() + timedelta(seconds=s["retry_wait_sec"]) >= deadline:
                error = f"{error} (다음 회차가 가까워 재시도 중단)"
                break
            log.info(f"{cid} {s['retry_wait_sec']}초 후 재시도 ({attempt}/{s['retry_count']})")
            time.sleep(s["retry_wait_sec"])
        attempts += 1
        try:
            frame, error = capture_frame(cam["url"], s["timeout_sec"])
        except Exception as e:
            frame, error = None, f"캡처 예외: {type(e).__name__}: {e}"
            log.exception(f"{cid} 캡처 중 예외")
        captured_at = now_kst()
        if frame is not None:
            break
        log.warning(f"{cid} 시도 {attempts}회차 실패: {error}")

    file_name = ""
    if frame is not None:
        path = (save_dir / "_test" / f'{cid}_{captured_at:%Y%m%d_%H%M%S}.jpg' if test
                else image_path(save_dir, cid, slot))
        try:
            save_jpg(frame, path, s["jpeg_quality"])
            file_name = str(path.relative_to(save_dir))
        except FileExistsError:
            error = f"같은 이름 파일이 이미 있어 저장하지 않음: {path.name}"
            log.warning(f"{cid} {error}")
        except Exception as e:
            error = f"저장 실패: {type(e).__name__}: {e}"
            log.exception(f"{cid} jpg 저장 중 예외")

    elapsed = round(time.monotonic() - start, 1)
    if file_name:
        h, w = frame.shape[:2]
        log.info(f"{cid}({cam['name']}) 저장: {file_name} | {w}x{h} | 시도 {attempts}회 | {elapsed}초")
        result = "test" if test else "success"
    else:
        log.error(f"{cid}({cam['name']}) 실패: {error} | 시도 {attempts}회 | {elapsed}초")
        result = "test" if test else "fail"

    return {"scheduled_at": "" if test else f"{slot:{TIME_FMT}}",
            "captured_at": f"{captured_at:{TIME_FMT}}", "cctv_id": cid, "name": cam["name"],
            "result": result, "file_name": file_name, "sun_alt": "",
            "attempts": attempts, "elapsed_sec": elapsed, "error": "" if file_name else error}


def skip_row(slot, cam, reason, sun_alt=""):
    """캡처를 시도하지 않은 회차의 CSV 한 줄"""
    return {"scheduled_at": f"{slot:{TIME_FMT}}", "captured_at": "", "cctv_id": cam["id"],
            "name": cam["name"], "result": "skipped", "file_name": "", "sun_alt": sun_alt,
            "attempts": 0, "elapsed_sec": 0, "error": reason}


# ───────── 스케줄 ─────────
def next_slot(t, minutes):
    """t 이후 첫 정각 기준 단위 시각 (20분 단위면 12:14 → 12:20)"""
    floor = t.replace(minute=t.minute - t.minute % minutes, second=0, microsecond=0)
    return floor + timedelta(minutes=minutes)


def wait_until(target):
    """target까지 1초 단위로 나눠 대기 (시계 보정·Ctrl+C 즉시 반응)"""
    while (remaining := (target - now_kst()).total_seconds()) > 0:
        time.sleep(min(remaining, 1))


class Collector:
    def __init__(self, cams, s, save_dir, csv_path, notifier, kma_key=""):
        self.cams, self.s = cams, s
        self.save_dir, self.csv_path, self.notify = save_dir, csv_path, notifier
        self.kma_key = kma_key
        self.weather_csv = csv_path.parent / "weather.csv"
        self.fail_streak = {c["id"]: 0 for c in cams}   # 카메라별 연속 실패 회차 수
        self.alerted = set()                            # 이미 알림을 보낸 카메라
        self.day = None                                 # 일일 요약 기준 날짜
        self.daily = {"success": 0, "fail": 0, "skipped": 0, "weather_ok": 0, "weather_all": 0}

    # -- 카메라별 연속 실패 --
    def update_streak(self, cam, result):
        cid = cam["id"]
        if result == "success":
            if cid in self.alerted:
                log.info(f"[복구] {cid}({cam['name']}) {self.fail_streak[cid]}회차 실패 후 성공")
                self.notify.send(f"✅ 복구: {cam['name']}({cid}) — {self.fail_streak[cid]}회차 실패 후 성공")
                self.alerted.discard(cid)
            self.fail_streak[cid] = 0
        elif result == "fail":
            self.fail_streak[cid] += 1
            if self.fail_streak[cid] >= self.s["warn_after_failures"] and cid not in self.alerted:
                log.warning(f"[경고] {cid}({cam['name']}) {self.fail_streak[cid]}회차 연속 실패")
                self.notify.send(f"⚠️ {cam['name']}({cid}) {self.fail_streak[cid]}회차 연속 실패")
                self.alerted.add(cid)

    # -- 하루 한 번 요약 (수집이 조용히 멈춘 것을 알아채기 위한 생존 신호) --
    def daily_summary(self, slot):
        if self.day is None:
            self.day = slot.date()
            return
        if slot.date() != self.day:
            d = self.daily
            pct, free_gb = disk_usage(self.save_dir)
            self.notify.send(f"📊 {self.day} 수집 요약 — 이미지 성공 {d['success']} / 실패 {d['fail']} / "
                             f"건너뜀 {d['skipped']} | 기상 {d['weather_ok']}/{d['weather_all']} "
                             f"| 카메라 {len(self.cams)}대 | 디스크 {pct}% (여유 {free_gb}GB)")
            # 하루 증가량이 1%에 못 미쳐서 하루 한 번 확인으로 충분하다
            if pct >= DISK_WARN_PCT:
                self.notify.send(f"⚠️ 디스크 {pct}% 사용 — 남은 공간 {free_gb}GB. "
                                 f"가득 차면 수집이 조용히 실패한다. GCS로 옮기고 로컬을 정리할 것")
            self.day = slot.date()
            self.daily = {"success": 0, "fail": 0, "skipped": 0, "weather_ok": 0, "weather_all": 0}

    def run_slot(self, slot):
        """한 회차 처리: 카메라를 순서대로 캡처 (동시 접속 없음)"""
        self.daily_summary(slot)
        deadline = slot + timedelta(minutes=self.s["interval_minutes"])
        results = []
        for cam in self.cams:
            alt = sun_altitude(cam["lat"], cam["lon"], slot)
            if alt < self.s["min_sun_altitude_deg"]:      # 야간 — 조명 조건이 달라 학습에 쓰지 않음
                append_csv(self.csv_path, skip_row(slot, cam, "야간(태양고도 미달)", round(alt, 1)))
                results.append("skipped")
                continue
            if now_kst() >= deadline:
                log.warning(f"{cam['id']} 건너뜀: 앞 카메라 지연으로 다음 회차 도달")
                append_csv(self.csv_path, skip_row(slot, cam, "앞 카메라 지연으로 다음 회차 도달"))
                results.append("skipped")
                continue
            try:
                row = capture_camera(cam, self.s, self.save_dir, slot, deadline)
            except Exception as e:
                log.exception(f"{cam['id']} 처리 중 예상치 못한 예외")
                row = skip_row(slot, cam, f"예외: {type(e).__name__}: {e}")
                row["result"] = "fail"
            row["sun_alt"] = round(alt, 1)
            append_csv(self.csv_path, row)
            self.update_streak(cam, row["result"])
            results.append(row["result"])

        # 기상 실황은 매시 한 번만. 이미지 캡처가 끝난 뒤에 받고, 실패해도 이미지에 영향이 없도록 격리한다
        if slot.minute == WEATHER_MINUTE:
            try:
                ok_w, all_w, err_w = weather.collect(self.kma_key, self.cams, self.weather_csv,
                                                     slot.replace(minute=0), now_kst)
                self.daily["weather_ok"] += ok_w
                self.daily["weather_all"] += all_w
                # 초단기실황은 1일치만 제공되어 지나가면 복구가 안 된다. 그래서 일부 실패도 바로 알린다
                if all_w and ok_w == 0:
                    self.notify.send(f"🚨 {slot:%m-%d %H}시 기상 수집 전멸 — {all_w}개 격자 전부 실패. "
                                     f"초단기실황은 백필이 안 되므로 이 시간대는 복구 불가 ({err_w})")
                elif ok_w < all_w:
                    self.notify.send(f"⚠️ {slot:%m-%d %H}시 기상 {all_w - ok_w}/{all_w} 격자 실패 — "
                                     f"{err_w}. 백필이 안 되므로 이 격자의 이 시간은 복구 불가")
            except Exception:
                log.exception("기상 수집 중 예상치 못한 예외")

        for r in results:
            self.daily[r] = self.daily.get(r, 0) + 1
        ok = results.count("success")
        tried = ok + results.count("fail")
        log.info(f"회차 종료 {slot:%m-%d %H:%M} | 성공 {ok} / 시도 {tried} / 건너뜀 {results.count('skipped')}")

        # 시도한 카메라가 모두 실패 = 개별 카메라 문제가 아니라 네트워크·스트림 서버 문제일 가능성
        if tried and ok == 0:
            self.notify.send(f"🚨 {slot:%m-%d %H:%M} 회차 전멸 — 시도 {tried}대 전부 실패. "
                             f"네트워크 또는 스트림 서버 확인 필요")

    def run(self):
        """정각 기준 interval 단위로 회차를 무한 반복"""
        step = timedelta(minutes=self.s["interval_minutes"])
        slot = next_slot(now_kst(), self.s["interval_minutes"])
        log.info(f"첫 회차 대기: {slot:%Y-%m-%d %H:%M}")
        while True:
            wait_until(slot)
            late = (now_kst() - slot).total_seconds()
            if late > LATE_GRACE_SEC:
                reason = f"예정 시각 {late:.0f}초 경과(지연 또는 절전 등)"
                log.warning(f"회차 건너뜀: {slot:%Y-%m-%d %H:%M} | {reason}")
                for cam in self.cams:
                    append_csv(self.csv_path, skip_row(slot, cam, reason))
            else:
                self.run_slot(slot)
            slot += step


# ───────── 실행 ─────────
def main():
    ap = argparse.ArgumentParser(description="CCTV 이미지 수집 (HLS 스트림 프레임 캡처)")
    ap.add_argument("--once", action="store_true", help="즉시 1회만 캡처 후 종료 (야간 판정 무시)")
    ap.add_argument("--limit", type=int, default=0, help="--once에서 앞 N대만 (0=전체)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args()

    try:
        s, cameras = load_config(args.config)
        # 상대경로는 저장소 루트 기준으로 푼다 — 같은 설정 파일로 Windows와 Linux 양쪽에서 돌리기 위함
        root = BASE_DIR.parent
        save_dir, log_dir = (root / s["save_dir"]), (root / s["log_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(log_dir)
    env = dotenv_values(ENV_PATH)
    notifier = Notifier(env.get("DISCORD_WEBHOOK_URL", ""))
    # 공공데이터포털은 계정당 인증키가 하나라, 별도 키가 없으면 에어코리아 키를 그대로 쓴다
    kma_key = env.get("KMA_API_KEY") or env.get("AIRKOREA_API_KEY", "")
    if kma_key and "%" in kma_key:
        kma_key = unquote(kma_key)   # requests가 다시 인코딩하므로 디코딩된 형태로 넘긴다
    if not kma_key:
        log.warning("기상청 인증키가 .env에 없습니다 → 기상 수집을 건너뜁니다")
    if not notifier.webhook:
        log.warning("DISCORD_WEBHOOK_URL이 .env에 없습니다 → 알림 없이 로그만 남깁니다")

    try:
        cams = resolve_cameras(cameras)
    except Exception as e:
        log.error(f"카메라 주소 확보 실패: {type(e).__name__}: {e}")
        notifier.send(f"🚨 수집기 시작 실패 — 카메라 주소를 확보하지 못했습니다: {type(e).__name__}")
        sys.exit(1)

    log.info(f"시작 | 모드: {'1회 테스트' if args.once else '반복 수집'} | "
             f"간격 {s['interval_minutes']}분 | 카메라 {len(cams)}대 | "
             f"야간 기준 태양고도 {s['min_sun_altitude_deg']}도 | "
             f"기상 격자 {len(weather.grids_of(cams))}개 (매시 {WEATHER_MINUTE}분)")

    csv_path = log_dir / "capture_log.csv"
    try:
        if args.once:
            for cam in (cams[:args.limit] if args.limit else cams):
                alt = sun_altitude(cam["lat"], cam["lon"], now_kst())
                row = capture_camera(cam, s, save_dir)
                row["sun_alt"] = round(alt, 1)
                append_csv(csv_path, row)
        else:
            notifier.send(f"▶️ 수집 시작 — 카메라 {len(cams)}대, {s['interval_minutes']}분 간격")
            Collector(cams, s, save_dir, csv_path, notifier, kma_key).run()
    except KeyboardInterrupt:
        log.info("사용자 종료 요청(Ctrl+C)")
        notifier.send("⏹️ 수집 중단 — 사용자 종료(Ctrl+C)")
    except Exception as e:
        log.exception("예상치 못한 오류로 중단")
        notifier.send(f"🚨 수집기 비정상 종료 — {type(e).__name__}: {e}")
    finally:
        log.info("스크립트 종료")


if __name__ == "__main__":
    main()
