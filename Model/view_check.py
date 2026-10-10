# 화각이 바뀌었거나 깨진 사진을 찾아, 눈으로 검토할 페이지를 만든다.
#
#   python Model/view_check.py                     잰 뒤 아직 검토 안 한 후보로 data/review/view_review.html 생성
#   python Model/view_check.py --apply 경로.csv     검토 페이지에서 저장한 CSV를 Model/view_review.csv에 반영
#
# 결과: data/model/view_check.csv (사진마다 잰 값, 이미 잰 사진은 건너뜀)
#       Model/view_ref.csv     카메라마다 "평소 모습" 기준 사진. 한 번 정하면 바꾸지 않는다 (바뀌면 옛 값과 비교가 안 된다)
#       Model/view_review.csv  사람이 내린 결정. 사진 경로와 결정만 있어 저장소에 올린다
#
# 재는 값 (docs/현황.md §7 거르기)
#   shift    기준 사진과 특징점(ORB)을 맞춰 구한 화면 이동량(픽셀, 640×360 기준). 맞출 수 없으면 999
#            윤곽선이 닮았는지로 재면 뿌연 사진이 낮게 나와 탁한 날 사진을 버리게 된다.
#            특징점 위치는 흐려져도 그대로라 PM2.5와 무관하다 (카메라 안 상관 중앙값 −0.02, 2026-10-10)
#   broken   회색·검은 화면(밝기 표준편차 5 미만) 또는 직전 회차와 완전히 같은 화면(평균 차 1 미만)
#
# 후보 = shift 10 초과 또는 broken. 검토 페이지의 기본값은 shift 20 초과와 broken을 제외로 둔다
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/labels/labels.csv"
IMAGES = ROOT / "data/images"
OUT = ROOT / "data/model/view_check.csv"
REF = ROOT / "Model/view_ref.csv"
REVIEW = ROOT / "Model/view_review.csv"
PAGE = ROOT / "data/review/view_review.html"
SIZE = (640, 360)
CANDIDATE, DEFAULT_EXCLUDE = 10, 20      # 픽셀


def gray(image, size=SIZE):
    return cv2.resize(cv2.imread(str(IMAGES / image), cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_AREA)


def mask():
    m = np.full(SIZE[::-1], 255, np.uint8)
    m[int(SIZE[1] * 0.80):] = 0          # 아래 자막
    m[: int(SIZE[1] * 0.07)] = 0         # 위 시각 표시
    return m


def top_edges(image):
    """위쪽 45% 윤곽선. 기준 사진을 고를 때만 쓴다"""
    g = gray(image, (320, 180)).astype(np.float32)[: int(180 * 0.45)]
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(cv2.resize(e, (160, 40), interpolation=cv2.INTER_AREA), (5, 5), 0)


def pick_ref(images):
    """카메라의 윤곽선 중앙값과 가장 닮은 사진 = 가장 전형적인 화각"""
    E = np.stack([top_edges(i) for i in images])
    ref = np.median(E, axis=0)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    return images[int(np.argmax([(z(e) * z(ref)).mean() for e in E]))]


def measure(args):
    image, ref, prev = args
    b = gray(image)
    out = {"image": image, "gray_std": round(float(b.std()), 2),
           "diff_prev": round(float(np.abs(b.astype(np.float32) - gray(prev)).mean()), 3) if prev else np.nan}
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(gray(ref), mask())
    kb, db = orb.detectAndCompute(b, mask())
    shift = 999.0
    if da is not None and db is not None and len(kb) >= 10:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        if len(m) >= 8:
            pa = np.float32([ka[x.queryIdx].pt for x in m])
            pb = np.float32([kb[x.trainIdx].pt for x in m])
            M, inl = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=3)
            if M is not None and inl.sum() >= 8:
                # 이동량 + 확대·축소 정도(배율 1%마다 3픽셀)
                shift = float(np.hypot(M[0, 2], M[1, 2]) + abs(np.log(np.hypot(M[0, 0], M[0, 1]))) * 300)
    out["shift"] = round(shift, 2)
    return out


def load_review():
    if REVIEW.exists():
        return pd.read_csv(REVIEW, encoding="utf-8-sig").set_index("image").decision
    return pd.Series(dtype=str)


def flags(v):
    """사진마다 broken·candidate·기본 제외 여부"""
    broken = (v.gray_std < 5) | (v.diff_prev < 1)
    return broken, (v["shift"] > CANDIDATE) | broken, (v["shift"] > DEFAULT_EXCLUDE) | broken


def build_page(cand, refs):
    items = [{"image": r.image, "cam": r.camera_id, "time": r.captured_at[5:16], "shift": round(float(r.shift), 1),
              "broken": bool(r.broken), "exclude": bool(r.default)} for r in cand.itertuples()]
    groups = {c: refs[c] for c in cand.camera_id.unique()}
    page = PAGE_TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace(
        "__REF__", json.dumps(groups, ensure_ascii=False))
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="화각 변경·깨진 화면 찾기와 검토")
    ap.add_argument("--apply", help="검토 페이지에서 저장한 view_review.csv")
    args = ap.parse_args()

    if args.apply:
        new = pd.read_csv(args.apply, encoding="utf-8-sig", dtype=str)[["image", "decision"]]
        old = pd.read_csv(REVIEW, encoding="utf-8-sig", dtype=str) if REVIEW.exists() else new.iloc[:0]
        merged = pd.concat([old[~old.image.isin(new.image)], new]).sort_values("image")
        merged.to_csv(REVIEW, index=False, encoding="utf-8-sig")
        print(f"{REVIEW}  {len(merged)}장 (이번에 반영 {len(new)}장, 제외 {int((new.decision == 'exclude').sum())}장)")
        return

    L = pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"camera_id": str})
    L = L.sort_values(["camera_id", "captured_at"]).reset_index(drop=True)
    refs = pd.read_csv(REF, encoding="utf-8-sig", dtype=str).set_index("camera_id").image.to_dict() if REF.exists() else {}
    missing = [c for c in L.camera_id.unique() if c not in refs]
    for c in missing:
        refs[c] = pick_ref(L[L.camera_id == c].image.tolist())
    if missing:
        pd.DataFrame(sorted(refs.items()), columns=["camera_id", "image"]).to_csv(REF, index=False, encoding="utf-8-sig")
        print(f"기준 사진을 새로 정한 카메라 {len(missing)}대 → {REF}")

    done = pd.read_csv(OUT, encoding="utf-8-sig") if OUT.exists() else pd.DataFrame(columns=["image"])
    L["prev"] = L.groupby("camera_id").image.shift()
    todo = L[~L.image.isin(done.image)]
    print(f"사진 {len(L)}장 중 새로 잴 사진 {len(todo)}장")
    if len(todo):
        jobs = [(r.image, refs[r.camera_id], r.prev if isinstance(r.prev, str) else None) for r in todo.itertuples()]
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(measure, jobs, chunksize=32))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(OUT, index=False, encoding="utf-8-sig")

    v = L[["image", "camera_id", "captured_at"]].merge(done, on="image")
    v["broken"], v["candidate"], v["default"] = flags(v)
    review = load_review()
    cand = v[v.candidate & ~v.image.isin(review.index)]
    print(f"후보 {int(v.candidate.sum())}장 (깨진 화면 {int(v.broken.sum())}장), 검토 완료 {int(v.image.isin(review.index).sum())}장, "
          f"검토할 것 {len(cand)}장")
    if len(cand):
        build_page(cand, refs)
        print(f"검토 페이지: {PAGE}  — 브라우저로 열어 결정한 뒤 저장한 CSV를 --apply로 반영")


PAGE_TEMPLATE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>화각 검토</title>
<style>
:root{--bg:#fafafa;--fg:#222;--card:#fff;--line:#ddd;--ex:#d33;--ok:#2a7}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1b;--fg:#eee;--card:#262626;--line:#444}}
body{font-family:system-ui,'Malgun Gothic',sans-serif;background:var(--bg);color:var(--fg);margin:16px}
h1{font-size:20px;margin:0 0 4px} .help{font-size:14px;line-height:1.6;margin-bottom:12px}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{font-size:14px;padding:6px 12px;cursor:pointer}
.cam{margin:20px 0;padding:12px;background:var(--card);border:1px solid var(--line);border-radius:8px}
.cam h2{font-size:16px;margin:0 0 8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.t{border:4px solid var(--ok);border-radius:6px;cursor:pointer;position:relative;background:#000}
.t.ex{border-color:var(--ex)} .t img{width:100%;display:block}
.t .cap{position:absolute;left:0;top:0;background:rgba(0,0,0,.65);color:#fff;font-size:12px;padding:2px 6px}
.t .st{position:absolute;right:0;top:0;font-size:12px;padding:2px 6px;color:#fff;background:var(--ok)} .t.ex .st{background:var(--ex)}
.ref{border:4px solid #888;border-radius:6px;position:relative} .ref img{width:100%;display:block}
.ref .cap{position:absolute;left:0;top:0;background:#555;color:#fff;font-size:12px;padding:2px 6px}
#count{font-weight:bold}
</style></head><body>
<h1>화각 검토</h1>
<div class="help">카메라마다 <b>회색 테두리가 평소 모습</b>입니다. 그 옆 사진을 보고 <b>다른 곳을 비췄거나 깨진 사진</b>이면 클릭해 <span style="color:var(--ex)"><b>제외</b></span>로,
같은 장면이면 <span style="color:var(--ok)"><b>정상</b></span>으로 두세요. 기본값은 20픽셀 넘게 옮겨 간 사진과 깨진 화면이 제외입니다.
<b>뿌연 사진·역광·차 많은 사진은 정상</b>입니다(뿌연 사진은 꼭 필요한 데이터). 사진 위 <i>sh</i>는 평소 사진에서 화면이 옮겨 간 정도(픽셀)입니다.
다 보신 뒤 <b>[결정 저장]</b>을 누르면 CSV가 내려받기 폴더에 저장됩니다.</div>
<div class="bar"><button onclick="save()">결정 저장 (CSV)</button><span id="count"></span></div>
<div id="root"></div>
<script>
const ITEMS = __ITEMS__; const REF = __REF__;
const root = document.getElementById('root');
function upd(){ const n = ITEMS.filter(x=>x.exclude).length; document.getElementById('count').textContent = `후보 ${ITEMS.length}장 중 제외 ${n}장`; }
for (const cam of Object.keys(REF)) {
  const sec = document.createElement('div'); sec.className = 'cam';
  const its = ITEMS.filter(x=>x.cam===cam);
  sec.innerHTML = `<h2>카메라 ${cam} <small>(${its.length}장)</small> <button data-a="ex">모두 제외</button><button data-a="ok">모두 정상</button></h2>`;
  const g = document.createElement('div'); g.className = 'grid';
  g.innerHTML = `<div class="ref"><img loading="lazy" src="../images/${REF[cam]}"><span class="cap">평소 모습 ${REF[cam].split('/')[1]}</span></div>`;
  const tiles = [];
  for (const it of its) {
    const t = document.createElement('div'); t.className = 't' + (it.exclude ? ' ex' : '');
    t.innerHTML = `<img loading="lazy" src="../images/${it.image}"><span class="cap">${it.time} sh${it.shift}${it.broken?' 깨짐':''}</span><span class="st">${it.exclude?'제외':'정상'}</span>`;
    t.onclick = () => { it.exclude = !it.exclude; t.classList.toggle('ex', it.exclude); t.querySelector('.st').textContent = it.exclude?'제외':'정상'; upd(); };
    tiles.push([t, it]); g.appendChild(t);
  }
  sec.querySelectorAll('button').forEach(b => b.onclick = () => { const ex = b.dataset.a==='ex'; for (const [t,it] of tiles){ it.exclude=ex; t.classList.toggle('ex',ex); t.querySelector('.st').textContent=ex?'제외':'정상'; } upd(); });
  sec.appendChild(g); root.appendChild(sec);
}
function save(){
  const rows = ['image,camera_id,captured_at,shift,decision'].concat(ITEMS.map(x=>`${x.image},${x.cam},${x.time},${x.shift},${x.exclude?'exclude':'keep'}`));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([String.fromCharCode(0xFEFF)+rows.join(String.fromCharCode(10))], {type:'text/csv'}));
  a.download = 'view_review.csv'; a.click();
}
upd();
</script></body></html>"""


if __name__ == "__main__":
    main()
