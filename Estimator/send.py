# 추정 결과를 서비스 백엔드(POST /api/ingest/estimates)로 보낸다. estimate.py가 live 실행 끝에 부르고, 손으로도 돌릴 수 있다.
#
#   python Estimator/send.py      아직 못 보낸 행을 보낸다
#
# 입력: data/estimates/estimates.csv (estimate.py), .env의 SERVICE_API_URL(/api 앞까지)·SERVICE_INGEST_TOKEN
# 결과: data/estimates/sent.txt  마지막으로 보내는 데 성공한 estimated_at
#
# 주소나 토큰이 비어 있으면 아무것도 하지 않는다(백엔드 배포 전).
# sent.txt 뒤의 행 중 최근 2시간(6회차)만 보낸다. 실패(연결·타임아웃·4xx·5xx)하면 sent.txt를 그대로 두어 다음 실행에 다시 보낸다.
# 같은 (camera_id, slot)은 백엔드가 덮어쓰고 그 회차 격자를 다시 계산하므로 다시 보내도 안전하다.
# 값은 CSV 그대로 옮긴다 — 빈 칸은 null. 다른 시각 값으로 메우지 않는다(현황 §2)
import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ESTIMATES = ROOT / "data/estimates/estimates.csv"
SENT = ROOT / "data/estimates/sent.txt"
MODEL = "incheon_log_top60"     # estimate.py의 MODEL과 같게. 2시간 안의 행은 늘 지금 모델 값이다
LOOKBACK = timedelta(hours=2)
KST = timezone(timedelta(hours=9))


def env(name):
    p = ROOT / ".env"
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"\'')
    return ""


def slot_of(r):
    t = datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S")
    return t.replace(minute=t.minute // 20 * 20, second=0)


def iso(t):
    return t.replace(tzinfo=KST).isoformat(timespec="seconds")


def num(v):
    return float(v) if v else None


def to_row(r):
    """CSV 한 줄 → shared/mock/ingest_estimates.json의 한 행. rain 칸('60분 0.0mm · 레이더 …')에서는 60분 강수량만 읽는다"""
    rn60 = re.search(r"60분 ([\d.]+)mm", r["rain"])
    return {"camera_id": r["camera_id"], "slot": iso(slot_of(r)),
            "captured_at": iso(datetime.strptime(r["captured_at"], "%Y-%m-%d %H:%M:%S")),
            "pm25_est": num(r["pm25_est"]), "status": r["status"], "humidity": num(r["humidity"]),
            "rain_60m": float(rn60.group(1)) if rn60 else None, "model": MODEL,
            "estimated_at": iso(datetime.strptime(r["estimated_at"], "%Y-%m-%d %H:%M:%S"))}


def send_pending(now=None):
    url, token = env("SERVICE_API_URL"), env("SERVICE_INGEST_TOKEN")
    if not url or not token or not ESTIMATES.exists():
        return
    now = now or datetime.now()
    since = SENT.read_text(encoding="utf-8").strip() if SENT.exists() else ""
    with open(ESTIMATES, encoding="utf-8-sig", newline="") as f:
        todo = [r for r in csv.DictReader(f)
                if r["mode"] == "live" and r["estimated_at"] > since and slot_of(r) >= now - LOOKBACK]
    if not todo:
        return
    try:
        res = requests.post(url.rstrip("/") + "/api/ingest/estimates", json={"rows": [to_row(r) for r in todo]},
                            headers={"Authorization": f"Bearer {token}"}, timeout=(5, 30))
        res.raise_for_status()
        got = res.json()
    except (requests.RequestException, ValueError) as e:
        why = f"{e.response.status_code} {e.response.text[:200]}" if getattr(e, "response", None) is not None else type(e).__name__
        print(f"{datetime.now():%H:%M} 전송 실패 {len(todo)}행: {why} (다음 실행에 다시)")
        return
    SENT.write_text(max(r["estimated_at"] for r in todo), encoding="utf-8")
    extra = "".join(f", {k} {','.join(got[k])}" for k in ("unknown_cameras", "new_models") if got.get(k))
    print(f"{datetime.now():%H:%M} 전송 {len(todo)}행 → 저장 {got.get('rows_upserted')}, 회차 {len(got.get('frames', []))}개{extra}")


if __name__ == "__main__":
    send_pending()
