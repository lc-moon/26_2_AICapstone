# 학습 여러 개를 이어서 돌린다. 하나가 실패해도 다음은 돈다.
#
#   python Model/run_queue.py "--version incheon_log" "--version mixed"
#
# 따옴표 하나가 cnn.py에 넘길 인자 한 묶음이다. 출력은 화면과 data/model/cnn/run_log.txt에 함께 남는다(이어 쓴다).
# 도는 동안 Windows에 절전하지 말라고 요청한다(SetThreadExecutionState). 이 프로그램이 끝나면 저절로 풀리고
# 전원 설정은 바꾸지 않는다. 노트북 덮개를 닫거나 직접 절전을 누르면 멈춘다.
# 학습 중에는 게임처럼 GPU를 많이 쓰는 프로그램을 끈다 (README Model 절).
import ctypes
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/model/cnn/run_log.txt"
KEEP_AWAKE = 0x80000000 | 0x00000001      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    jobs = sys.argv[1:]
    if not jobs:
        sys.exit('사용법: python Model/run_queue.py "--version incheon_log" "--version mixed"')
    ctypes.windll.kernel32.SetThreadExecutionState(KEEP_AWAKE)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    results = []
    with open(LOG, "a", encoding="utf-8") as log:
        def out(line):
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        for job in jobs:
            out(f"\n===== {job} 시작 {datetime.now():%m-%d %H:%M}\n")
            p = subprocess.Popen([sys.executable, "-u", str(ROOT / "Model/cnn.py"), *shlex.split(job)], cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                if "HF_TOKEN" not in line:          # 로그인 없이 가중치를 받는다는 안내. 매번 나와서 뺀다
                    out(line)
            code = p.wait()
            results.append((job, code))
            out(f"===== {job} 끝 (종료 코드 {code}{', 실패' if code else ''}) {datetime.now():%m-%d %H:%M}\n")
        out("\n" + "\n".join(f"  {'성공' if c == 0 else '실패'}  {j}" for j, c in results) + "\n")


if __name__ == "__main__":
    main()
