"""에어코리아 숭의 측정소 최근 3개월 실시간 측정값을 페이지 단위로 받아 raw 폴더에 저장한다.

사용법
    python fetch_songui.py --probe   # 1페이지만 호출해 응답 구조 확인
    python fetch_songui.py           # 전체 페이지 조회 (저장본이 있는 페이지는 건너뜀)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from dotenv import dotenv_values

# 경로 설정: 스크립트 위치 기준
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
ENV_PATH = BASE_DIR.parents[1] / ".env"  # C:\26_2_AICapstone\.env

# API 설정 (팀원이 검증한 값)
URL = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
NUM_OF_ROWS = 100      # 크게 하면 504가 잦으므로 100 유지
MAX_RETRY = 3          # 504 발생 시 최대 재시도 횟수
REQUEST_INTERVAL = 1   # 요청 사이 간격(초)


def load_key():
    """ .env에서 인증키를 읽는다. 인코딩된 키(%2B 등)면 디코딩해 이중 인코딩을 막는다. """
    key = dotenv_values(ENV_PATH).get("AIRKOREA_API_KEY")
    if not key:
        sys.exit("AIRKOREA_API_KEY가 .env에 없습니다.")
    return unquote(key) if "%" in key else key


def mask(text, key):
    """ 출력 문자열에서 인증키(원문·URL 인코딩 형태)를 ***로 가린다. """
    for k in {key, requests.utils.quote(key, safe="")}:
        text = text.replace(k, "***")
    return text


def page_path(page):
    """ 페이지 번호별 저장 파일 경로 """
    return RAW_DIR / f"page_{page:03d}.json"


def fetch_page(page, key):
    """ 한 페이지를 받아 JSON(dict)으로 돌려준다. 저장본이 있으면 API를 호출하지 않는다. """
    path = page_path(page)
    if path.exists():
        print(f"[저장본 사용] {path.name}")
        return json.loads(path.read_text(encoding="utf-8")), False

    params = {
        "serviceKey": key,
        "returnType": "json",
        "stationName": "숭의",
        "dataTerm": "3MONTH",
        "ver": "1.5",
        "numOfRows": NUM_OF_ROWS,
        "pageNo": page,
    }
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(URL, params=params, timeout=60)
        except requests.RequestException as e:
            # 예외 메시지에 URL(키 포함)이 들어갈 수 있으므로 가려서 종료
            sys.exit(f"[요청 실패] page {page}: {mask(str(e), key)}")

        # 504 또는 본문의 SERVICETIMEOUT이면 지수 백오프(2, 4, 8초) 후 재시도
        if resp.status_code == 504 or "SERVICETIMEOUT" in resp.text:
            if attempt < MAX_RETRY:
                wait = 2 ** (attempt + 1)
                print(f"[504] page {page} - {wait}초 후 재시도 ({attempt + 1}/{MAX_RETRY})")
                time.sleep(wait)
                continue
            sys.exit(f"[504] page {page}: 재시도 {MAX_RETRY}회 모두 실패")

        # 정상 JSON이 아니면(인증 오류 XML 등) 저장하지 않고 종료해 잘못된 저장본을 막는다
        try:
            data = resp.json()
            header = data["response"]["header"]
        except (ValueError, KeyError):
            sys.exit(f"[비정상 응답] page {page} HTTP {resp.status_code}: {mask(resp.text[:300], key)}")
        if header.get("resultCode") != "00":
            sys.exit(f"[API 오류] page {page}: {header}")

        # 응답 원문을 그대로 저장
        path.write_text(resp.text, encoding="utf-8")
        print(f"[저장] {path.name} ({len(data['response']['body']['items'])}건)")
        return data, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="1페이지만 조회해 구조 확인")
    args = parser.parse_args()

    RAW_DIR.mkdir(exist_ok=True)
    key = load_key()

    # 1페이지로 전체 건수 파악
    first, _ = fetch_page(1, key)
    body = first["response"]["body"]
    total = body["totalCount"]
    pages = math.ceil(total / NUM_OF_ROWS)
    print(f"totalCount={total}, 페이지 수={pages}")

    if args.probe:
        # 응답 구조 확인용 요약 출력
        items = body["items"]
        print("필드:", list(items[0].keys()))
        for it in items[:3]:
            print({k: it.get(k) for k in ("dataTime", "pm25Value", "pm25Flag", "pm25Grade")})
        print("24:00 표기 개수(1페이지):", sum(it["dataTime"].endswith("24:00") for it in items))
        return

    # 나머지 페이지 조회 (API를 실제 호출한 경우에만 1초 대기)
    for page in range(2, pages + 1):
        _, called = fetch_page(page, key)
        if called:
            time.sleep(REQUEST_INTERVAL)
    print("전체 조회 완료")


if __name__ == "__main__":
    main()
