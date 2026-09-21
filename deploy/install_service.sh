#!/usr/bin/env bash
# 수집기를 systemd 서비스로 등록한다. VM에서 한 번만 실행하면 된다.
#   bash deploy/install_service.sh
#
# 사용자 이름과 경로를 현재 환경에서 읽어 채우므로 직접 고칠 필요가 없다.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"
USER_NAME="$(whoami)"
UNIT=/etc/systemd/system/cctv-collector.service

[ -x "$VENV/bin/python" ] || { echo "가상환경을 찾을 수 없습니다: $VENV/bin/python"; exit 1; }
[ -f "$REPO/.env" ] || { echo "경고: $REPO/.env 가 없습니다. 알림이 동작하지 않습니다."; }

mkdir -p "$REPO/data/logs"

sudo tee "$UNIT" > /dev/null <<UNIT_EOF
[Unit]
Description=CCTV 이미지·기상 수집기 (26-2 AI 캡스톤)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO/Collector
ExecStart=$VENV/bin/python collect_cctv.py
Restart=always
RestartSec=30
StandardOutput=append:$REPO/data/logs/service.log
StandardError=append:$REPO/data/logs/service.log

[Install]
WantedBy=multi-user.target
UNIT_EOF

sudo systemctl daemon-reload
sudo systemctl enable cctv-collector
sudo systemctl restart cctv-collector
sleep 3
sudo systemctl --no-pager status cctv-collector | head -15

echo
echo "등록 완료. 자주 쓰는 명령:"
echo "  sudo systemctl status cctv-collector     상태 확인"
echo "  sudo systemctl restart cctv-collector    재시작"
echo "  sudo systemctl stop cctv-collector       중지"
echo "  tail -f $REPO/data/logs/collector.log    수집 로그 보기"
