# 근거 위치 표시 시안 (F8, 2026-11-08 확정)

지점을 고르면 그 값의 근거인 **가까운 측정소**와 **가까운 CCTV** 자리를 지도에 표시하는 시안 3.
방식과 결정 경위는 [docs/전시_준비.md](../../../docs/전시_준비.md) "고칠 것" F8.

| 파일 | 내용 |
|---|---|
| `proto-station.mjs` | 시안 모듈. 서비스 저장소 `web/`에 넣어 쓰도록 만들었다 |
| `app.mjs.diff` | 서비스 저장소 PR #13 시점의 `web/app.mjs`에 붙인 연결 부분 |

## 돌려 보는 법

1. 서비스 저장소 `web/`에 `proto-station.mjs`를 복사하고, `app.mjs`에 `app.mjs.diff`의 내용을 손으로 반영한다(줄 위치는 그 뒤 바뀌었을 수 있음)
2. 저장소 루트에서 `python -m http.server`로 띄우고 `/web/?api=https://incheon-air.duckdns.org&proto=station`을 연다
   - `&pick=경도,위도` 그 지점을 바로 선택, `&evopen=station|cctv` 몇 초 뒤 그 핀을 누른 상태(캡처용), `&line=1` 연결선 항상 표시(비교용)

## 실제로 넣을 때

- `?proto=station` 조건과 `pick`·`evopen`(캡처용)은 뺀다. 패널 등급 배지(`stationBadge`)는 F4로 따로
- **측정소·카메라 좌표**: 시안은 `../shared/stations.json`·`cameras.json`을 읽는데, 배포 서버의 Caddy는 `/web/`과 `/shared/boundary.geojson`만 파일로 내준다 → 둘을 내주게 하거나 `web/` 안에 복사
- CCTV는 백엔드 S3의 `camera`가 오면 그 카메라를 쓰고, 지금처럼 가장 가까운 카메라로 고르는 것은 임시
- 가리는 카드는 `position:fixed`라 `offsetParent`가 늘 null → 보이는지는 `getClientRects().length`로
- 고른 곳이 화면 위쪽에 오는 것은 F6(패널 여백)을 고치면 같이 풀린다
