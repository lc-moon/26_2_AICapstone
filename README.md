# 26-2 AI 캡스톤: CCTV 영상으로 PM2.5 추정

CCTV 영상에서 이미지를 주기적으로 모아, 미세먼지(PM2.5) 농도 추정 모델의 학습 데이터로 쓰기 위한 프로젝트입니다.
대기 중 입자에 의한 빛 산란은 관측 거리가 길수록 누적되므로, **원거리 피사체의 대비 저하**가 농도의 시각적 신호가 된다는 가설에 기반합니다.

## 문서

| 문서 | 언제 읽나 |
|---|---|
| [docs/프로젝트.md](docs/프로젝트.md) | 이 프로젝트가 무엇이고 어떤 데이터를 쓰는지 |
| [docs/현황.md](docs/현황.md) | **지금 어떤 상태인지.** 확정된 사양과 사실, 남은 작업 |
| [docs/운영.md](docs/운영.md) | 서버에 올려 돌리는 방법, 문제가 생겼을 때 |
| [docs/모델_이해하기.md](docs/모델_이해하기.md) | **AI를 처음 접하는 사람용.** 모델을 무엇을 왜 이렇게 만드는지 쉬운 말로. 용어 사전 포함 |
| [docs/인터페이스_모델출력.md](docs/인터페이스_모델출력.md) | 조원(백엔드·프론트)에게 넘기는 데이터 형식 |
| [docs/기록/](docs/기록/) | 주차별 진행 기록. 순서대로 읽으면 프로젝트가 어떻게 굴러왔는지 보임 |

**앞의 넷은 갱신되는 문서**라 항상 최신 내용만 담고, **`기록/`은 과거 시점이라 갱신하지 않습니다.**

## 폴더 구조

```
C:\26_2_AICapstone\
├─ README.md            이 설명서
├─ CLAUDE.md            Claude Code가 매 세션 읽는 작업 규칙
├─ .env                 인증키 (공유·업로드 금지)
│
├─ .claude\             Claude Code 설정
│  └─ skills\기록\      작업 기록 작성 스킬 (/기록)
│
├─ docs\                문서
│  ├─ 프로젝트.md          배경·가설·데이터 소스 명세
│  ├─ 현황.md              지금 상태·확정된 사실·남은 작업
│  ├─ 운영.md              배포 절차·운영·문제 대응
│  ├─ 모델_이해하기.md      모델 설명 (AI 비전공자용, 용어 사전)
│  ├─ 인터페이스_모델출력.md  조원에게 넘기는 데이터 형식
│  └─ 기록\               주차별 진행 기록 (날짜순)
│
├─ Collector\           상시 실행 — 이미지 수집기
│  ├─ collect_cctv.py
│  ├─ weather.py           기상청 초단기실황 수집 (수집기가 매시 호출, 학습에는 안 씀·검증용)
│  ├─ cameras.toml         수집 설정과 카메라 목록
│  └─ url_cache.json       목록 API 응답 캐시 (자동 생성)
│
├─ Tools\               1회성 스크립트
│  ├─ screen_fitic.py          CCTV 화각 자동 스크리닝 (카메라 선정용)
│  ├─ build_labels.py          사진마다 기상(1분)·PM2.5를 붙여 학습용 표 생성
│  ├─ build_labels_beijing.py  베이징 외부 데이터를 같은 양식의 표로 (사진 회전·축소 포함)
│  ├─ contact_sheet.py         수집 이미지를 한 장으로 붙여 보기
│  ├─ make_prototype_data.py   조원에게 넘길 모델 출력 샘플 생성
│  ├─ fetch_songui.py          에어코리아 PM2.5 조회
│  └─ analyze_songui.py        PM2.5 등급 분포 분석
│
├─ Model\               로컬 학습 (RTX 3060 Ti). VM에는 올리지 않음
│  ├─ features.py          사진마다 이미지 통계 계산 → data\model\features.csv
│  ├─ view_check.py        화각 변경·깨진 화면 찾기 + 검토 페이지·폴더 검수 결과 반영
│  ├─ view_ref.csv         장면(카메라@해상도)마다 "평소 모습" 기준 사진 (한 번 정하면 고정)
│  ├─ view_mask\           장면마다 화면 위 글자(지명 표시) 자리 — 특징점을 찾을 때 가린다
│  ├─ view_review.csv      사람이 검수한 결정 (사진 경로·제외/정상·사유)
│  ├─ baseline.py          거르기·날짜 분할·회귀·평가 → data\model\baseline_pred.csv
│  ├─ cnn.py               ConvNeXt V2-Tiny 미세조정 (인천 ver / 인천 log ver / 인천+베이징 ver) → data\model\cnn\
│  ├─ run_queue.py         cnn.py 여러 개를 이어서 실행 (실패해도 다음 진행, 절전 방지, 기록 파일)
│  ├─ text_mask.py         화면 위 글자(시각·지명) 자리 찾기 → text_mask\ (CNN 입력에서 평균색으로 가림)
│  ├─ text_mask\           장면마다 글자 자리 (한 번 만들고 눈으로 확인한 것)
│  ├─ roi.py               ROI 상자 그리기 페이지 생성·결과 반영, 후보 5개(하늘·먼 건물·둘 다·위 1/3·위 60%)의 상자
│  ├─ roi_page.html        상자 그리기 페이지 틀 (roi.py가 사진 목록을 넣어 data\review\roi.html로 만든다)
│  ├─ roi.csv              카메라마다 하늘·먼 건물 상자 (비율 좌표, 10/11 확정 — Claude 초안을 사용자가 검토·수정)
│  ├─ roi_auto.py          ROI ⑥ 자동 — 먼지 민감도 지도로 카메라마다 탁할수록 가장 흐려지는 곳에 상자 → roi_auto.csv
│  ├─ roi_auto.csv         자동 상자 (연습시험 회차 0·1·2는 그 회차 공부 날짜로만, all은 개발 날짜 전체로 만든 것)
│  ├─ screen.py            CNN 입력 후보 예선 — 사전학습 특징 + Ridge로 후보당 1~2분에 채점 (기상 조합도)
│  ├─ gradcam.py           학습한 CNN이 화면 어디를 봤는지 색칠 (확인·발표용, 상자 고르기엔 안 씀)
│  └─ requirements.txt     학습용 패키지 (수집기용과 별도)
│
├─ Estimator\           추정기 — VM에서 수집된 사진으로 PM2.5 추정 (수집기와 따로 돈다)
│  ├─ export_onnx.py       PC에서 최종 모델을 ONNX로 바꾸고 PyTorch와 결과가 같은지 확인 → data\model\onnx\ (VM에는 scp)
│  ├─ estimate.py          찍은 지 3분 넘은 사진 → 1분 습도 조회 → 추정 → data\estimates\ (--latest: 최근 낮 회차로 확인)
│  └─ stations.json        카메라마다 습도를 받는 AWS 관측소 (학습 라벨과 같은 곳)
│
├─ deploy\              서버 설치 스크립트
│  ├─ install_service.sh   수집기 systemd 서비스 등록 (docs/운영.md 참고)
│  └─ install_estimator.sh 추정기 5분 타이머 + 결과 페이지(127.0.0.1:8080) 등록
│
├─ model_output\        조원에게 넘기는 데이터 (git 포함)
│  ├─ cameras.json         카메라 번호·이름·좌표
│  └─ values.json          시각별 추정 농도
│
├─ data\                모든 생성물 (git 제외, 스크립트로 재생성 가능)
│  ├─ images\              수집 이미지 (bj\는 베이징 외부 데이터, 돌려서 줄인 사본)
│  ├─ logs\                수집 로그·기상 CSV (vm\는 VM에서 받은 사본)
│  ├─ labels\              학습용 표(labels.csv, 베이징은 labels_beijing.csv)와 API 원본 응답 — raw\airkorea\는 지우지 말 것
│  ├─ contact_sheets\      이미지 확인용 모아보기
│  ├─ model\               이미지 특징(features.csv)·베이스라인 사진별 예측(baseline_pred.csv)·CNN 결과(cnn\, 설계가 바뀌기 전 결과는 cnn_archive\)·384로 줄인 사진(cnn_cache\, 7GB)
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

### `build_labels.py` — 학습용 표 만들기
수집 로그(`capture_log.csv`)의 사진마다 **찍은 분의 기상값**(기상청 API허브 AWS 1분)과
**찍은 시간의 PM2.5**(에어코리아 1시간값)를 붙여 `data\labels\labels.csv`를 만듭니다.
사진 파일은 필요 없어 로그만 있으면 어디서든 돌릴 수 있습니다. 붙이는 규칙은 `docs/현황.md` §2·§6.

```
python Tools\build_labels.py --log 경로\capture_log.csv
python Tools\build_labels.py --log 경로\capture_log.csv --check-weather 경로\weather.csv   실황과 대조까지
```

`.env`에 `KMA_HUB_KEY`(API허브)와 `AIRKOREA_API_KEY`가 필요합니다.
받은 응답은 `data\labels\raw\`에 남아 다시 호출하지 않습니다.
**PM2.5는 3개월이 지나면 다시 받을 수 없으므로 수집 기간 중 3개월 안에 한 번은 돌려야 합니다.**

### `build_labels_beijing.py` — 베이징 외부 데이터를 같은 양식으로
베이징 데이터셋(Feng et al. 2021)을 `labels.csv`와 **같은 13칸**인 `data\labels\labels_beijing.csv`로 만듭니다.
낮 사진만 쓰고, 사진은 똑바로 돌려 540×960으로 줄인 사본을 만듭니다. PM2.5는 엑셀의 한 시간 뒤 행,
기온·습도·풍속·비는 베이징 관측소 54511의 30분 관측을 씁니다. 규칙과 근거는 `docs/현황.md` §5.

```
python Tools\build_labels_beijing.py      입력은 C:\26_2_AICapstone_external\beijing\ (저장소 밖), 돌린 사진은 data\images\bj\
```

입력(`Photo_Beijing\`, `PhotoRGB_AQI_Met.xlsx`, `noaa_isd_54511_full.csv`)은 저장소 밖에 둡니다. 재배포하지 않는 조건으로 쓰는 자료입니다.

### `contact_sheet.py` — 수집 이미지 확인
여러 장을 격자로 붙여 한 장으로 만듭니다. 서버에 쌓인 이미지를 한 장씩 내려받지 않고 훑어볼 때 씁니다.

```
python Tools\contact_sheet.py --date 2026-09-28 --time 1200     그 시각의 카메라 전체
python Tools\contact_sheet.py --date 2026-09-28 --camera 113    그 카메라의 하루
```

### `make_prototype_data.py` — 조원에게 넘길 샘플 생성
`model_output\`의 두 파일을 만듭니다. 카메라 번호·이름·좌표는 실제 값이고 농도만 가상입니다.
형식은 `docs/인터페이스_모델출력.md` 참고.

### `fetch_songui.py` / `analyze_songui.py` — PM2.5 라벨 분석
에어코리아에서 측정소 데이터를 받아 농도 등급 분포를 집계합니다.
고농도 구간 데이터가 부족하다는 구조적 리스크(`docs/현황.md` §5)의 근거입니다.

---

## `Model\`: 모델 학습

로컬 GPU에서 돌립니다. `data\labels\labels.csv`(build_labels 결과)와 `data\images\`가 있어야 합니다.
VM에서 사진을 받는 절차는 `docs/운영.md` §3.

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128      GPU용 PyTorch (먼저)
pip install -r Model\requirements.txt
python Model\features.py          사진마다 이미지 통계 계산 (이미 계산한 사진은 건너뜀, 1만 장에 약 40초)
python Model\view_check.py        화각 변경·깨진 화면 찾기 (1만 장에 약 4분) → 검수 안 한 후보가 있으면 data\review\view_review.html 생성
python Model\view_check.py --apply 내려받기\view_review.csv          검토 페이지에서 저장한 결정 반영
python Model\view_check.py --import-folders 검수폴더                  폴더로 검수한 결과 반영 (카메라\제외\사유\사진)
python Model\baseline.py          베이스라인 — 날짜 단위 교차검증 점수 + 사진별 예측값
python Model\baseline.py --cv3    같은 점수를 CNN과 같은 날짜 3묶음으로 (CNN과 나란히 비교)
python Model\baseline.py --test   시험 A 점수. 모델을 다 고른 뒤에만 볼 것
python Model\cnn.py --version incheon   CNN 인천 ver — 연습시험(3묶음×최대 15바퀴)으로 바퀴 수를 정하고 최종 모델 저장
python Model\cnn.py --version incheon_log   CNN 인천 ver을 로그로 학습 (로그 효과 확인용)
python Model\cnn.py --version mixed     CNN 인천+베이징 ver — 베이징 학습 날짜를 섞는다
python Model\cnn.py --version mixed --input top60 --weather humidity --final-only --seed 1   연습시험은 두고 최종 모델만 다시 (시험지 열기 전에만)
python Model\cnn.py --test              저장된 CNN 전부를 시험지 A(인천)·B(베이징)로 채점. 모든 작업이 끝난 뒤 한 번만
python Model\run_queue.py "--version incheon_log" "--version mixed"   여러 개를 이어서 (기록: data\model\cnn\run_log.txt)
python Model\text_mask.py               글자 자리 다시 찾기 (새 카메라·해상도가 생겼을 때) → data\review\text_mask_*.jpg로 확인
python Model\roi.py                     ROI 상자 그리기 페이지 → data\review\roi.html (브라우저로 열어 그린 뒤 [저장])
python Model\roi.py --apply 내려받기\roi.csv   그린 상자를 Model\roi.csv로 반영
python Model\roi_auto.py                자동 상자 계산 → data\review\roi_auto_*.jpg로 지도 확인 (몇 분, CPU)
python Model\screen.py                  입력 후보(ROI 6개·사진 전체·글자 가림) 예선 → data\model\screen.csv (cnn.py 학습과 동시에 돌리지 말 것)
python Model\screen.py --inputs sky --weather humidity --weather humidity sun_alt   기상 조합 예선
python Model\gradcam.py                 1단계 CNN이 본 곳 → data\review\gradcam_*.jpg
python Model\cnn.py --version mixed --input sky --weather humidity sun_alt --epochs 12   ROI·기상 결선 학습
```

**CNN을 돌리는 동안에는 게임 등 GPU를 쓰는 프로그램을 끕니다.** 같이 쓰면 속도가 몇 분의 1로 떨어지고,
GPU 메모리가 넘치면 Windows가 PC 메모리를 빌려 쓰며 수십 배 느려집니다(코드에서 상한을 걸어 그 전에 오류가 나게 해 둠).

- 특징은 **사진 전체**와 **위쪽 1/3 띠**(대개 하늘·원경)에서 따로 잽니다. 밝기·대비·채도·암채널·선명도·파랑/빨강 비, 위−가운데 밝기차
- 거르기: PM2.5 결측, 강수 감지, 습도 95% 이상, 화각 변경·깨진 화면
- **화각 검수**: `view_check.py`가 화면 위 글자를 가리고 평소 모습과 특징점 위치를 비교해 화각 변경을, 회색·정지·아래쪽 번짐으로 깨진 화면을 찾습니다.
  20분 간격 사진을 구간으로 묶어 판단하므로 역광·어둠에 덜 흔들립니다. 10/9까지 사진은 사람이 전부 검수했고(제외 168장) 자동 판단과 완전히 일치합니다.
  검수한 사진은 사람의 결정을, 새 사진은 자동 판단을 따릅니다. 새 후보는 검토 페이지에서 확인해 `--apply`로 반영합니다
- 특징 표준화와 카메라 구분은 **장면(카메라@해상도)** 단위입니다. 스트림 해상도가 바뀌면 화질 특성이 달라지기 때문입니다(10042번, 10/8)
- 시험 A는 **9/22부터 4일마다 하루**입니다. 규칙으로 정하므로 데이터가 늘어도 이미 정한 시험일은 그대로입니다
- 성적표에는 평균만·카메라별 평균·기상만·이미지만·이미지+기상이 나란히 나오고, 이미지+기상은 카메라 차이를 다루는 방식 셋으로 각각 냅니다
- 성적은 사진 단위와 **같은 카메라·같은 1시간 예측 평균** 단위로 내고, 늘 평균만 부르는 모델 대비 개선폭을 같이 봅니다
- 사진별 정답·예측은 `data\model\baseline_pred.csv`에 남습니다. 어디서 틀리는지 볼 때 씁니다
- **CNN**은 사진 전체(또는 ROI 상자)를 384×384로 맞춰 넣고, 카메라·도시 정보는 주지 않습니다. 화면 위 글자(시각·지명)는 평균색으로 가립니다(`--input`, 기본 masked). 증강은 살짝 잘라내기·좌우 뒤집기만(색·밝기는 정답 신호라 금지).
  인천+베이징 ver은 log(1+PM2.5)를 배웁니다. 시험지 B는 베이징 첫날(2019-05-19)부터 4일마다 하루로, 어느 버전도 학습에 쓰지 않습니다.
  연습시험은 인천·베이징을 각각 3묶음으로 나눠 같은 번호끼리 함께 빼고, 두 도시 점수를 모두 기록합니다.
  바퀴 수는 인천 ver은 인천 점수로, 인천+베이징 ver은 "인천이 최고와 0.1 이내인 바퀴 중 베이징이 가장 좋은 바퀴"로 고릅니다.
  결과는 `data\model\cnn\{버전}_full\`에 바퀴별 연습시험 점수(epochs.csv)·사진별 예측(pred_cv.csv)·최종 모델(model.pt)로 남습니다

---

## `.env`

```
UTIC_API_KEY=...          UTIC 오픈API (현재 수집에는 사용하지 않음)
AIRKOREA_API_KEY=...      에어코리아 대기오염정보
KMA_HUB_KEY=...           기상청 API허브 AWS 1분 자료 (build_labels.py)
DISCORD_WEBHOOK_URL=...   수집 장애 알림 (선택)
```

**인증키가 들어 있으므로 공유하거나 GitHub에 올리면 안 됩니다.**

---

## 출처 표기 의무

결과보고서·발표자료에 다음을 반영해야 합니다.

- CCTV 영상: 인천교통정보센터
- 대기오염 정보: 한국환경공단 에어코리아, 공공누리 제3유형(출처표시 + 변경금지)
- 에어코리아 자료는 **"인증을 받지 않은 실시간자료"**이므로 값이 다를 수 있음을 명시해야 합니다 (이용약관 제8조)
