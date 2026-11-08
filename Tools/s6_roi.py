# 원리 페이지(S6) 방법 절 그림: 같은 카메라의 맑은 날·탁한 날 원래 사진에 "모델이 보는 위쪽 60%" 점선.
# 사진 쌍은 Model/gradcam_pairs.py의 pick()과 같은 규칙(시험 기간 실측 최고 사진 + 같은 시각 ±1시간의 실측 최저 사진).
# 서비스 저장소 web/about/img/1_roi.png는 79번으로 그렸다(2026-11-08).
#   python Tools/s6_roi.py 79                 → data/review/s6/0_보는곳_79.png
#   python Tools/s6_roi.py --sheet            → data/review/s6/roi_cand/sheet*.png (카메라 고르기용, 위쪽 60%만 5대씩)
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cnn
from gradcam_pairs import pick, picture

STROKE = [pe.withStroke(linewidth=3, foreground="black")]
plt.rcParams["font.family"] = "Malgun Gothic"
OUT = ROOT / "data" / "review" / "s6"


def figure(pairs, cam, out):
    clear, hazy = pairs[cam]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), dpi=110, facecolor="#fcfcfb")
    for ax, r, t in zip(axes, (clear, hazy), ("맑은 날", "탁한 날")):
        im = picture(r, None, (0, 0, 1, 0.6))
        h = im.shape[0]
        ax.imshow(im)
        ax.axhline(0.6 * h, color="#ffd400", lw=2, ls="--")
        ax.text(12, 0.6 * h - 14, "모델이 보는 곳 (위쪽 60%)", color="#ffd400", fontsize=11, fontweight="bold", path_effects=STROKE)
        ax.text(12, 0.6 * h + 34, "보지 않는 곳 (도로·차, 화면에서는 가림)", color="white", fontsize=11, fontweight="bold", path_effects=STROKE)
        ax.set_title(f"{t} · 실측 {r.pm25:.0f} ㎍/㎥", fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")
    plt.close(fig)
    print(cam, clear.captured_at, hazy.captured_at, "→", out)


def sheets(pairs, out):
    out.mkdir(parents=True, exist_ok=True)
    cams = sorted(pairs)
    for k in range(0, len(cams), 5):
        grp = cams[k:k + 5]
        fig, axes = plt.subplots(len(grp), 2, figsize=(10, 2.9 * len(grp)), dpi=80, squeeze=False)
        for i, c in enumerate(grp):
            for j, r in enumerate(pairs[c]):
                im = picture(r, None, (0, 0, 1, 0.6))
                axes[i, j].imshow(im[: int(im.shape[0] * 0.6)])   # 위쪽 60%만(비교용)
                axes[i, j].set_title(f"{c}번 {'맑은' if j == 0 else '탁한'} {r.captured_at[5:16]} 실측 {r.pm25:.0f} 습도 {r.humidity:.0f}%", fontsize=9)
                axes[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"sheet{k // 5}.png")
        plt.close(fig)
    print("카메라:", cams, "→", out)


ap = argparse.ArgumentParser()
ap.add_argument("camera", nargs="?", help="카메라 번호 (예: 79)")
ap.add_argument("--out", help="출력 파일 (기본 data/review/s6/0_보는곳_<카메라>.png)")
ap.add_argument("--sheet", action="store_true", help="모든 카메라의 사진 쌍을 5대씩 모아 보기")
args = ap.parse_args()
if not args.camera and not args.sheet:
    ap.error("카메라 번호나 --sheet 중 하나")
d = cnn.load()
pairs = pick(d[d.part == "A"])
if args.sheet:
    sheets(pairs, OUT / "roi_cand")
if args.camera:
    OUT.mkdir(parents=True, exist_ok=True)
    figure(pairs, args.camera, Path(args.out) if args.out else OUT / f"0_보는곳_{args.camera}.png")
