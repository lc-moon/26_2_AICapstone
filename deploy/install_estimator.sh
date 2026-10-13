#!/usr/bin/env bash
# 추정기를 systemd에 등록한다. VM에서 한 번만 실행하면 된다. 수집기(cctv-collector)는 건드리지 않는다.
#   bash deploy/install_estimator.sh
#
# pm-estimator.timer  5분마다 Estimator/estimate.py를 한 번 돌린다(돌 때만 메모리를 쓰고 끝나면 내려놓는다)
# pm-web.service      data/estimates/index.html을 127.0.0.1:8080으로 보여 준다. 외부에 열지 않는다 — PC에서 SSH 터널로 본다
# 먼저 필요한 것: $VENV에 onnxruntime, .env의 KMA_HUB_KEY, data/model/onnx/{모델}.onnx·.json (PC에서 scp)
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"
USER_NAME="$(whoami)"

[ -x "$VENV/bin/python" ] || { echo "가상환경을 찾을 수 없습니다: $VENV/bin/python"; exit 1; }
"$VENV/bin/python" -c "import onnxruntime" || { echo "onnxruntime이 없습니다: $VENV/bin/pip install onnxruntime"; exit 1; }
grep -q "^KMA_HUB_KEY=" "$REPO/.env" || { echo ".env에 KMA_HUB_KEY가 없습니다"; exit 1; }
ls "$REPO"/data/model/onnx/*.onnx > /dev/null || { echo "data/model/onnx/에 모델이 없습니다 (PC에서 scp)"; exit 1; }
mkdir -p "$REPO/data/logs" "$REPO/data/estimates"

sudo tee /etc/systemd/system/pm-estimator.service > /dev/null <<UNIT_EOF
[Unit]
Description=PM2.5 추정기 한 번 실행 (26-2 AI 캡스톤)
After=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=$VENV/bin/python Estimator/estimate.py
Nice=10
# 수집기 메모리를 빼앗지 않게, 넘치면 수집기 대신 추정기가 멈춘다
MemoryMax=600M
TimeoutStartSec=15min
StandardOutput=append:$REPO/data/logs/estimator.log
StandardError=append:$REPO/data/logs/estimator.log
UNIT_EOF

sudo tee /etc/systemd/system/pm-estimator.timer > /dev/null <<UNIT_EOF
[Unit]
Description=PM2.5 추정기 5분마다

[Timer]
OnCalendar=*:0/5
AccuracySec=30s

[Install]
WantedBy=timers.target
UNIT_EOF

sudo tee /etc/systemd/system/pm-web.service > /dev/null <<UNIT_EOF
[Unit]
Description=PM2.5 추정값 페이지 (127.0.0.1:8080, SSH 터널로만 접속)
After=network.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=$VENV/bin/python -m http.server 8080 --bind 127.0.0.1 --directory $REPO/data/estimates
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT_EOF

sudo systemctl daemon-reload
sudo systemctl enable --now pm-estimator.timer pm-web.service
sleep 2
systemctl --no-pager list-timers pm-estimator.timer | head -3
systemctl --no-pager status pm-web | head -5

echo
echo "등록 완료. 자주 쓰는 명령:"
echo "  systemctl list-timers pm-estimator.timer      다음 실행 시각"
echo "  tail -f $REPO/data/logs/estimator.log         추정 기록"
echo "  sudo systemctl disable --now pm-estimator.timer pm-web   추정기 끄기 (수집기는 그대로)"
echo "  PC에서 보기: ssh -i ~/.ssh/google_compute_engine -L 8080:127.0.0.1:8080 $USER_NAME@<VM 주소>  →  http://localhost:8080"
