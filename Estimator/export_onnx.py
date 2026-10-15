# 학습한 최종 모델(model.pt)을 VM에서 가볍게 돌릴 ONNX 파일로 바꾸고, 같은 입력에서 PyTorch와 결과가 같은지 확인한다. 로컬(PC)에서 실행.
#
#   python Estimator/export_onnx.py                       mixed_top60_w-hum (인천+베이징 ver 최종) → data/model/onnx/
#   (추정기는 10/15부터 인천 ver incheon_log_top60을 쓴다)
#   python Estimator/export_onnx.py --run incheon_log_top60
#
# 결과: data/model/onnx/{run}.onnx  입력 image (N,3,384,384 표준화한 RGB), humidity (N,1, %) → 출력 log(1+PM2.5)
#       data/model/onnx/{run}.json  입력 종류·기상 칸·확인 결과
# 저장소에 올리지 않는다(data/). 베이징 데이터로 학습했고 가중치가 비상업 라이선스라 VM에는 scp로만 옮긴다
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Model"))
import cnn  # noqa: E402  (GRN 계산 순서 교체도 여기서 적용된다)

OUT = ROOT / "data/model/onnx"


def main():
    ap = argparse.ArgumentParser(description="최종 모델 → ONNX")
    ap.add_argument("--run", default="mixed_top60_w-hum", help="data/model/cnn/ 아래 결과 폴더 이름")
    ap.add_argument("--check", type=int, default=256, help="PyTorch와 비교할 사진 수")
    args = ap.parse_args()

    info = json.loads((cnn.OUT / args.run / "info.json").read_text(encoding="utf-8"))
    weather = info["weather"]
    model = cnn.make_model(pretrained=False, n_weather=len(weather)).to("cpu").float().eval()
    model.load_state_dict(torch.load(cnn.OUT / args.run / "model.pt", map_location="cpu"))

    x = torch.zeros(1, 3, cnn.SIZE, cnn.SIZE)
    names, inputs, axes = ["image"], (x,), {"image": {0: "n"}, "pm_log": {0: "n"}}
    if weather:
        w = torch.full((1, len(weather)), 60.0)
        names, inputs = ["image", "weather"], (x, w)
        axes["weather"] = {0: "n"}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.run}.onnx"
    torch.onnx.export(model, inputs, path, input_names=names, output_names=["pm_log"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)

    # 같은 사진·같은 기상으로 PyTorch(fp32, CPU)와 비교. 연습시험에 쓴 인천 사진 사본을 쓴다
    import onnxruntime as ort
    d = cnn.load()
    arr = cnn.cache(d, info["input"])
    rows = d.index[(d.city == "inc") & (d.part != "A")].values[:: max(1, (d.city == "inc").sum() // args.check)][: args.check]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    diff, t_onnx = [], 0.0
    for i in range(0, len(rows), 16):
        r = rows[i:i + 16]
        xb = ((torch.from_numpy(np.ascontiguousarray(arr[r])).permute(0, 3, 1, 2).float() - cnn.MEAN) / cnn.STD)
        feeds = {"image": xb.numpy()}
        if weather:
            wb = torch.from_numpy(d.loc[r, weather].values.astype(np.float32))
            feeds["weather"] = wb.numpy()
            with torch.no_grad():
                pt = model(xb, wb).squeeze(1).numpy()
        else:
            with torch.no_grad():
                pt = model(xb).squeeze(1).numpy()
        t0 = time.time()
        ox = sess.run(None, feeds)[0].squeeze(1)
        t_onnx += time.time() - t0
        diff.append(np.abs(np.expm1(ox) - np.expm1(pt)) if info["log_target"] else np.abs(ox - pt))
    diff = np.concatenate(diff)
    res = {"run": args.run, "input": info["input"], "weather": weather, "log_target": info["log_target"],
           "best_epoch": info["best_epoch"], "cv": info["cv"], "size": cnn.SIZE,
           "mean": (cnn.MEAN.view(-1) / 255).tolist(), "std": (cnn.STD.view(-1) / 255).tolist(), "fill_bgr": list(cnn.FILL),
           "check": {"photos": int(len(diff)), "max_abs_diff_ugm3": float(diff.max()), "mean_abs_diff_ugm3": float(diff.mean()),
                     "onnx_sec_per_photo_pc": round(t_onnx / len(diff), 3)}}
    (OUT / f"{args.run}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {path} ({path.stat().st_size / 1e6:.0f}MB)")
    print(f"PyTorch와 차이 (사진 {len(diff)}장): 최대 {diff.max():.5f}, 평균 {diff.mean():.6f} ㎍/㎥, ONNX PC 속도 {res['check']['onnx_sec_per_photo_pc']}초/장")


if __name__ == "__main__":
    main()
