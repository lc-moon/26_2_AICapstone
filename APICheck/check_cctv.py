# UTIC CCTV 영상 접근 방식 확인용 1회성 프로그램
# 흐름: 저장된 목록에서 CCTV 찾기 → 스트림 페이지에서 영상 주소 찾기 → 프레임 1장 jpg 저장 → 결과 출력
import re
import sys
import time
import warnings
import urllib.parse
from datetime import datetime
from pathlib import Path

import cv2
import openpyxl
import requests
from dotenv import dotenv_values

# ───────── 설정 ─────────
BASE_DIR = Path(r"C:\26_2_AICapstone")
OUT_DIR = BASE_DIR / "APICheck"
LIST_HTML = OUT_DIR / "cctvOpenData_list.html"  # 이미 받아둔 목록 페이지 (재호출 금지 → 재사용)
COORD_XLSX = OUT_DIR / "OpenDataCCTV.xlsx"       # 공식 좌표속성 파일
TARGET_NAME = "용현사거리"                        # UTIC 지도 표기 원문
PARTIAL_NAME = "용현"                            # 정확히 일치하는 이름이 없을 때 참고로 출력할 키워드

# 공식 파일에 없는 스트림 파라미터: 브라우저에서 공식 목록 화면을 클릭해 확인한 값 (2026-09-15)
BROWSER_CONFIRMED = {
    "L020172": {"KIND": "N", "CCTVIP": "0", "ID": "L204"},
}

TIMEOUT = 20          # HTTP 요청 타임아웃(초)
RETRY_MAX = 2         # 재시도 최대 횟수 (UTIC 차단 기준 준수)
RETRY_WAIT = 10       # 재시도 간격(초), 최소 10초

# .env에서 인증키 읽기
KEY = dotenv_values(BASE_DIR / ".env").get("UTIC_API_KEY", "")


def mask(text):
    """출력용 문자열에서 인증키를 *** 로 가림"""
    return text.replace(KEY, "***") if KEY else text


def stop(msg):
    """사유를 출력하고 프로그램 종료"""
    print(f"\n[중단] {msg}")
    sys.exit(1)


# ───────── 1. 목록에서 후보 CCTV 찾기 ─────────
def load_list():
    """저장된 목록 HTML에서 (CCTVID, 이름) 목록 추출. 이름 앞의 '번호.'는 제거"""
    html = LIST_HTML.read_text(encoding="utf-8")
    return [(cid, name.strip()) for cid, name in re.findall(r"test\('([^']+)'\)\">\d+\.([^<]*)</a>", html)]


def load_coords():
    """좌표속성 xlsx에서 CCTVID → (센터명, 경도, 위도) 사전 생성"""
    warnings.filterwarnings("ignore")  # 스타일 없음 경고 무시
    ws = openpyxl.load_workbook(COORD_XLSX, read_only=True).worksheets[0]
    ws.reset_dimensions()  # 파일의 잘못된 크기 정보 무시하고 전체 행 읽기
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    idx = {h: i for i, h in enumerate(header)}
    return {r[idx["CCTVID"]]: (r[idx["CENTERNAME"]], r[idx["XCOORD"]], r[idx["YCOORD"]]) for r in rows}


# ───────── 2. 스트림 페이지 요청 (안전 재시도) ─────────
def safe_get(url):
    """타임아웃·재시도 규칙을 지키는 GET. 50x는 재시도 없이 즉시 중단"""
    for attempt in range(RETRY_MAX + 1):
        if attempt:
            print(f"  {RETRY_WAIT}초 후 재시도 ({attempt}/{RETRY_MAX})")
            time.sleep(RETRY_WAIT)
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"  네트워크 오류: {type(e).__name__}")
            continue
        if r.status_code >= 500:
            stop(f"서버 50x 오류({r.status_code}) → 차단 기준 때문에 재시도하지 않고 멈춥니다.")
        return r
    stop("네트워크 오류로 스트림 페이지를 받지 못했습니다.")


def classify(html):
    """페이지 HTML에서 실제 영상 주소를 찾아 (형태, 주소) 반환"""
    m = re.search(r"videoSrc\s*=\s*'([^']+)'", html)  # HLS 플레이어가 쓰는 주소
    if m and ".m3u8" in m.group(1):
        return "HLS(.m3u8) 스트림", m.group(1)
    m = re.search(r"<img[^>]+src=['\"]([^'\"]+\.(?:jpg|jpeg|png))", html, re.I)
    if m:
        return "정지 이미지", m.group(1)
    m = re.search(r"<iframe[^>]+src=['\"]([^'\"]+)", html, re.I)
    if m:
        return "외부 웹 플레이어(iframe)", m.group(1)
    return "웹 페이지(영상 주소 미발견)", None


# ───────── 3. 프레임 1장 캡처 ─────────
def grab_frame(stream_url):
    """OpenCV(FFmpeg)로 스트림을 열어 프레임 1장 반환. 실패 시 10초 간격 최대 2회 재시도"""
    opts = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT * 1000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT * 1000]
    for attempt in range(RETRY_MAX + 1):
        if attempt:
            print(f"  {RETRY_WAIT}초 후 재시도 ({attempt}/{RETRY_MAX})")
            time.sleep(RETRY_WAIT)
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, opts)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            return frame
        print("  스트림 열기/프레임 읽기 실패")
    return None


def main():
    if not KEY:
        stop(".env에 UTIC_API_KEY가 없습니다.")

    # 1) 후보 CCTV 찾기
    cctvs = load_list()
    coords = load_coords()
    print(f"[1] 목록(저장본) CCTV {len(cctvs)}개 중 '{TARGET_NAME}' 검색")
    matches = [(cid, name) for cid, name in cctvs if name == TARGET_NAME]

    if len(matches) > 1:  # 같은 이름 여러 개 → 모두 출력하고 멈춤
        for cid, name in matches:
            center, x, y = coords.get(cid, ("?", "?", "?"))
            print(f"  - {name} | ID {cid} | {center} | 경도 {x}, 위도 {y}")
        stop("같은 이름이 여러 개입니다. 자동으로 고르지 않습니다.")
    if not matches:  # 없음 → 부분 일치 이름만 출력하고 멈춤
        for cid, name in cctvs:
            if PARTIAL_NAME in name:
                print(f"  - {name} | ID {cid}")
        stop(f"'{TARGET_NAME}'과 정확히 일치하는 CCTV가 없습니다.")

    cctv_id, cctv_name = matches[0]
    center, x, y = coords.get(cctv_id, ("?", "?", "?"))
    print(f"  찾음: {cctv_name} | ID {cctv_id} | {center} | 경도 {x}, 위도 {y}")

    # 2) 스트림 페이지 주소 구성 → 요청 → 실제 영상 주소 찾기
    params = BROWSER_CONFIRMED.get(cctv_id)
    if not params:
        stop(f"{cctv_id}의 스트림 파라미터(KIND/ID 등)가 공식 파일에 없고, 브라우저 확인값도 없습니다.")
    name_enc = urllib.parse.quote(urllib.parse.quote(cctv_name, safe=""), safe="")  # 브라우저와 같은 이중 인코딩
    page_url = (f"http://www.utic.go.kr/jsp/map/openDataCctvStream.jsp?key={KEY}&cctvid={cctv_id}"
                f"&cctvName={name_enc}&kind={params['KIND']}&cctvip={params['CCTVIP']}"
                f"&cctvch=undefined&id={params['ID']}&cctvpasswd=undefined&cctvport=undefined")
    print(f"\n[2] 스트림 페이지 요청: {mask(page_url)}")
    r = safe_get(page_url)
    kind, stream_url = classify(r.text)
    print(f"  응답 {r.status_code} | 주소 형태: {kind}")

    shot_path, shape, fail = None, None, None
    if not stream_url:
        fail = ("페이지에서 영상 주소를 찾지 못함 → 가능성: ① 신청 IP와 현재 PC IP 불일치(키+IP 인증) "
                "② 지자체 영상 중단/점검 ③ 스트림 파라미터 변경")
    elif kind != "HLS(.m3u8) 스트림":
        fail = f"이번 확인 범위(HLS) 밖의 형태: {mask(stream_url)}"
    else:
        print(f"  실제 영상 주소: {mask(stream_url)}")

        # 3) 프레임 1장 저장
        print("\n[3] 프레임 캡처")
        frame = grab_frame(stream_url)
        if frame is None:
            fail = "스트림에서 프레임을 읽지 못함 → 가능성: 지자체 영상 중단/점검, 네트워크 문제"
        else:
            shot_path = OUT_DIR / f"{cctv_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if cv2.imwrite(str(shot_path), frame):
                shape = frame.shape
            else:
                fail, shot_path = "jpg 파일 저장 실패", None

    # 4) 결과 요약
    print("\n========== 결과 ==========")
    print(f"CCTV     : {cctv_name} ({cctv_id}), {center}")
    print(f"주소 형태: {kind}")
    print(f"영상 주소: {mask(stream_url) if stream_url else '-'}")
    print(f"캡처     : {'성공' if shot_path else '실패'}")
    if shot_path:
        print(f"해상도   : {shape[1]}x{shape[0]}")
        print(f"저장 파일: {shot_path}")
    if fail:
        print(f"실패 사유: {fail}")


if __name__ == "__main__":
    main()
