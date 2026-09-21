# 26-2 AI 캡스톤: CCTV 영상으로 PM2.5 추정

CCTV 영상에서 이미지를 주기적으로 모아, 미세먼지(PM2.5) 농도 추정 모델의 학습 데이터로 쓰기 위한 프로젝트입니다.
대기 중 입자에 의한 빛 산란은 관측 거리가 길수록 누적되므로, **원거리 피사체의 대비 저하**가 농도의 시각적 신호가 된다는 가설에 기반합니다.

## 문서

| 문서 | 언제 읽나 |
|---|---|
| [docs/프로젝트.md](docs/프로젝트.md) | 이 프로젝트가 무엇이고 어떤 데이터를 쓰는지 |
| [docs/현황.md](docs/현황.md) | **지금 어떤 상태인지.** 확정된 사양과 사실, 남은 작업 |
| [docs/운영.md](docs/운영.md) | 서버에 올려 돌리는 방법, 문제가 생겼을 때 |
| [docs/기록/](docs/기록/) | 주차별 진행 기록. 순서대로 읽으면 프로젝트가 어떻게 굴러왔는지 보임 |

**앞의 셋은 갱신되는 문서**라 항상 최신 내용만 담고, **`기록/`은 과거 시점이라 갱신하지 않습니다.**

## 폴더 구조

```
C:\26_2_AICapstone\
├─ README.md            이 설명서
├─ .env                 인증키 (공유·업로드 금지)
│
├─ docs\                문서
│  ├─ 프로젝트.md          배경·가설·데이터 소스 명세
│  ├─ 현황.md              지금 상태·확정된 사실·남은 작업
│  ├─ 운영.md              배포 절차·운영·문제 대응
│  └─ 기록\               주차별 진행 기록 (날짜순)
│
├─ Collector\           상시 실행 — 이미지 수집기
│  ├─ collect_cctv.py
│  ├─ cameras.toml         수집 설정과 카메라 목록
│  └─ url_cache.json       목록 API 응답 캐시 (자동 생성)
│
├─ Tools\               1회성 스크립트
│  ├─ screen_fitic.py      CCTV 화각 자동 스크리닝 (카메라 선정용)
│  ├─ fetch_songui.py      에어코리아 PM2.5 조회
│  └─ analyze_songui.py    PM2.5 등급 분포 분석
│
├─ data\                모든 생성물 (git 제외, 스크립트로 재생성 가능)
│  ├─ images\              수집 이미지
│  ├─ logs\                수집 로그
│  ├─ screening\           화각 스크리닝 산출물
│  └─ songui_pm25\         PM2.5 원본·정제·차트
│
└─ _archive\            역할이 끝난 자료 (git 제외)
   ├─ APICheck\            UTIC 접근 방식 확인용 코드·원본 자료
   └─ TestPic\             9/15~16 초기 수집분 119장 (폐기 결정)
```

---

## `Collector\`: 이미지 수집

인천교통정보센터(fitic) CCTV **20대**에서 20분 간격으로 프레임 1장씩 캡처합니다.

- 카메라는 `cameras.toml`에 **번호만** 적습니다. 이름·좌표·스트림 주소는 실행할 때 목록 API로 조회하므로, 스트림 주소가 바뀌어도 설정을 고칠 필요가 없습니다.
- **태양고도가 0도 미만이면 건너뜁니다.** 야간 이미지는 조명 조건이 달라 학습에 쓰지 않습니다.
- 저장 경로: `data\images\{카메라번호}\{YYYY-MM-DD}\{HHMM}.jpg`
- 회차 결과는 `data\logs\capture_log.csv`, 텍스트 로그는 `collector.log`에 쌓입니다.
- `.env`에 `DISCORD_WEBHOOK_URL`이 있으면 연속 실패·회차 전멸·비정상 종료·일일 요약을 알립니다.

```
python C:\26_2_AICapstone\Collector\collect_cctv.py            반복 수집 (Ctrl+C로 종료)
python C:\26_2_AICapstone\Collector\collect_cctv.py --once     1회 테스트 캡처
```

---

## `Tools\`: 1회성 스크립트

### `screen_fitic.py` — 카메라 선정
인천 CCTV 237대의 프레임을 받아 하늘 면적·균일도·포화율·원경 대비를 계산해 점수를 매깁니다.
`cameras.toml`의 20대가 이 도구로 추려낸 결과입니다.
(자동 23대 → 화각이 도는 PTZ 2대와 측정소가 11km 떨어진 1대를 제외)

```
python Tools\screen_fitic.py                전체 촬영 후 채점 (약 6분)
python Tools\screen_fitic.py --rescore      저장된 프레임으로 재채점 (약 5초)
python Tools\screen_fitic.py --ptz-check    화각이 돌아가는 카메라 찾기
```

### `fetch_songui.py` / `analyze_songui.py` — PM2.5 라벨 분석
에어코리아에서 측정소 데이터를 받아 농도 등급 분포를 집계합니다.
고농도 구간 데이터가 부족하다는 구조적 리스크(`docs/현황.md` §5)의 근거입니다.

---

## `.env`

```
UTIC_API_KEY=...          UTIC 오픈API (현재 수집에는 사용하지 않음)
AIRKOREA_API_KEY=...      에어코리아 대기오염정보
DISCORD_WEBHOOK_URL=...   수집 장애 알림 (선택)
```

**인증키가 들어 있으므로 공유하거나 GitHub에 올리면 안 됩니다.**

---

## 출처 표기 의무

결과보고서·발표자료에 다음을 반영해야 합니다.

- CCTV 영상: 인천교통정보센터
- 대기오염 정보: 한국환경공단 에어코리아, 공공누리 제3유형(출처표시 + 변경금지)
- 에어코리아 자료는 **"인증을 받지 않은 실시간자료"**이므로 값이 다를 수 있음을 명시해야 합니다 (이용약관 제8조)
