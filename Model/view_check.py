# 화각이 바뀌었거나 깨진 사진을 찾아, 사람이 검수한 결과와 함께 거르기 목록을 만든다.
#
#   python Model/view_check.py                          새 사진을 재고, 검수 안 한 후보가 있으면 data/review/view_review.html 생성
#   python Model/view_check.py --apply 경로.csv          검토 페이지에서 저장한 CSV(또는 image,decision[,reason] 표)를 반영
#   python Model/view_check.py --import-folders 경로     폴더로 검수한 결과를 반영 (아래 "폴더 검수")
#
# 결과: data/model/view_check.csv  사진마다 잰 값 (이미 잰 사진은 건너뜀)
#       Model/view_ref.csv         장면마다 "평소 모습" 기준 사진. 한 번 정하면 바꾸지 않는다
#       Model/view_mask/{장면}.png  장면마다 화면 위 글자(지명 표시) 자리. 한 번 정하면 바꾸지 않는다
#       Model/view_review.csv      사람이 내린 결정(image, decision, reason). 저장소에 올린다
#
# 장면 = "카메라@해상도". 같은 카메라도 스트림 해상도가 바뀌면 화질 특성이 달라 따로 본다
#   (10042번이 2026-10-08 08:41부터 1280×720 → 640×360. 바뀐 뒤 선명도가 28% 올랐다 — 먼지가 아니라 스트림 탓)
#
# 판단 (2026-10-11, 사람이 11,420장을 전부 본 결과와 165장 중 164장 일치)
#   이동량  기준 사진과 ORB 특징점을 맞춘 화면 이동(픽셀, 640×360 기준). 화면 위 글자는 가린다 —
#           카메라가 돌아가도 글자는 제자리라, 가리지 않으면 화각이 바뀐 사진 절반이 "안 움직였다"로 나왔다
#           윤곽선 유사도로 재지 않는 이유: 뿌연 사진이 화각 변경처럼 나와 탁한 날 사진을 버리게 된다
#   구간    20분 간격으로 이어진 사진끼리 잘 맞으면(맞춘 점 20개 이상, 이동 10픽셀 미만) 한 구간으로 묶고,
#           구간 안에서 기준 사진과 확실히 맞춰진 사진(맞춘 점 20개 이상)의 다수결로 구간 전체를 판단한다.
#           역광·어둠·연무로 점이 적은 사진이 혼자 "옮겨 갔다"로 나오는 것을 막는다 (131번 아침, 212·13번 해 질 녘)
#   깨짐    회색·검은 화면(밝기 표준편차 5 미만), 직전 회차와 똑같은 화면(평균 차 1 미만),
#           아래쪽이 세로로 번진 화면(아래 60% 안에서 위아래 줄이 같은 구간이 화면 높이의 2% 이상 — 전송 중 영상 손실)
#
# 폴더 검수: 카메라 폴더({카메라}/YYYYMMDD_HHMM.jpg)를 넘겨보며 이상한 사진을 {카메라}/제외/{사유}/ 로 복사해 둔 구조.
#   폴더에 남은 사진은 keep, 제외/ 아래는 exclude(사유 = 하위 폴더 이름), "애매…" 폴더는 exclude로 넣고 목록을 보여 준다.
#   카메라 폴더 없이 사진이 바로 있는 구조(베이징)는 --camera bj 를 같이 준다
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/labels/labels.csv"
IMAGES = ROOT / "data/images"
OUT = ROOT / "data/model/view_check.csv"
REF = ROOT / "Model/view_ref.csv"
MASKS = ROOT / "Model/view_mask"
REVIEW = ROOT / "Model/view_review.csv"
PAGE = ROOT / "data/review/view_review.html"
SIZE = (640, 360)
MOVED = 20          # 픽셀. 기준 사진에서 이만큼 넘게 옮겨 가면 화각 변경
RELIABLE = 20       # 맞춘 특징점 수. 이 이상이어야 판단 근거로 쓴다
LINK = 10           # 픽셀. 직전 사진과 이보다 덜 움직였으면 같은 구간
LINK_GAP = 45 * 60  # 초. 직전 사진과 이보다 멀면 잇지 않는다
STREAK = 0.02       # 아래쪽 세로 번짐 길이 / 화면 높이
MASK_SAMPLE, MASK_STD = 150, 6


def gray(image):
    return cv2.resize(cv2.imread(str(IMAGES / image), cv2.IMREAD_GRAYSCALE), SIZE, interpolation=cv2.INTER_AREA)


def scene_of(image):
    with Image.open(IMAGES / image) as im:
        w, h = im.size
    return f"{image.split('/')[0]}@{w}x{h}"


def top_edges(image):
    """위쪽 45% 윤곽선. 기준 사진을 고를 때만 쓴다"""
    g = cv2.resize(cv2.imread(str(IMAGES / image), cv2.IMREAD_GRAYSCALE), (320, 180), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)[: int(180 * 0.45)]
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(cv2.resize(e, (160, 40), interpolation=cv2.INTER_AREA), (5, 5), 0)


def pick_ref(images):
    """장면의 윤곽선 중앙값과 가장 닮은 사진 = 가장 전형적인 화각"""
    E = np.stack([top_edges(i) for i in images])
    ref = np.median(E, axis=0)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    return images[int(np.argmax([(z(e) * z(ref)).mean() for e in E]))]


def make_mask(images):
    """모든 사진에서 거의 같은 픽셀 = 화면 위 글자. 특징점을 찾을 때 이 자리는 뺀다"""
    sd = np.stack([gray(i) for i in images]).astype(np.float32).std(axis=0)
    return cv2.dilate((sd < MASK_STD).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))


def match(a, b, keep):
    """두 사진의 화면 이동량(픽셀)과 맞춘 특징점 수. 맞출 수 없으면 (999, 점 수)"""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, keep)
    kb, db = orb.detectAndCompute(b, keep)
    if da is None or db is None or len(kb) < 10:
        return 999.0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 8:
        return 999.0, len(m)
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=3)
    if M is None or inl.sum() < 8:
        return 999.0, 0 if inl is None else int(inl.sum())
    # 이동량 + 확대·축소 정도(배율 1%마다 3픽셀)
    return float(np.hypot(M[0, 2], M[1, 2]) + abs(np.log(np.hypot(M[0, 0], M[0, 1]))) * 300), int(inl.sum())


def streak(g):
    """아래쪽 60% 안에서 위아래 줄이 같은(세로로 번진) 가장 긴 구간 / 화면 높이"""
    g = cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)[int(179 * 0.4):]
    best = run = 0
    for v in dy:
        run = run + 1 if v < 0.5 else 0
        best = max(best, run)
    return best / 179


def measure(args):
    image, ref, mask_path, prev = args
    keep = cv2.bitwise_not(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
    b = gray(image)
    shift, inl = match(gray(ref), b, keep)
    out = {"image": image, "gray_std": round(float(b.std()), 2), "streak": round(streak(b), 4),
           "shift": round(shift, 2), "inliers": inl, "diff_prev": np.nan, "pshift": np.nan, "pinl": np.nan}
    if prev:
        p = gray(prev)
        out["diff_prev"] = round(float(np.abs(b.astype(np.float32) - p).mean()), 3)
        ps, pi = match(p, b, keep)
        out["pshift"], out["pinl"] = round(ps, 2), pi
    return out


def flags(v):
    """v: image, scene, captured_at와 잰 값. (깨짐, 화각 변경) 두 열을 v의 순서대로 돌려준다"""
    v = v.reset_index(drop=True)
    broken = (v.gray_std < 5) | (v.diff_prev < 1) | (v.streak > STREAK)
    o = v.sort_values(["scene", "captured_at"])
    same_day = o.captured_at.str[:10] == o.captured_at.shift().str[:10]
    link = (o.scene == o.scene.shift()) & same_day & (o.pinl >= RELIABLE) & (o.pshift < LINK)
    seg = (~link).cumsum()
    moved = o["shift"] > MOVED
    rel = o.inliers >= RELIABLE
    vote = moved[rel].groupby(seg[rel]).mean()
    st = seg.map(vote)
    moved = (st > 0.5).where(st.notna(), moved).astype(bool)
    return broken, moved.reindex(v.index)


def load_review():
    if REVIEW.exists():
        return pd.read_csv(REVIEW, encoding="utf-8-sig", dtype=str).set_index("image").decision
    return pd.Series(dtype=str)


def save_review(new):
    """image, decision[, reason] 표를 기존 결정 위에 덮어쓴다 (같은 사진은 새 결정이 이긴다)"""
    new = new.copy()
    if "reason" not in new:
        new["reason"] = ""
    new = new[["image", "decision", "reason"]].fillna("")
    old = pd.read_csv(REVIEW, encoding="utf-8-sig", dtype=str).fillna("") if REVIEW.exists() else new.iloc[:0]
    if "reason" not in old:
        old["reason"] = ""
    merged = pd.concat([old[~old.image.isin(new.image)], new]).sort_values("image")
    merged.to_csv(REVIEW, index=False, encoding="utf-8-sig")
    print(f"{REVIEW}  {len(merged)}장 (이번에 반영 {len(new)}장, 그중 제외 {int((new.decision == 'exclude').sum())}장)")


def import_folders(root, camera=None):
    """폴더 검수 결과 → image, decision, reason"""
    root = Path(root)
    rows = []
    for cam_dir in ([root] if camera else sorted(p for p in root.iterdir() if p.is_dir())):
        cam = camera or cam_dir.name
        for f in cam_dir.rglob("*.jpg"):
            d = datetime.strptime(f.stem, "%Y%m%d_%H%M")
            rel = f.relative_to(cam_dir).parts
            if len(rel) == 1:
                decision, reason = "keep", ""
            elif rel[0] == "제외":
                decision, reason = "exclude", rel[1] if len(rel) > 2 else "제외"
            else:
                decision, reason = "exclude", rel[0]
            rows.append({"image": f"{cam}/{d:%Y-%m-%d}/{d:%H%M}.jpg", "decision": decision, "reason": reason})
    df = pd.DataFrame(rows)
    # 같은 사진이 폴더와 제외 둘 다에 있으면(복사해 둔 경우) 제외가 이긴다
    df = df.sort_values("decision").drop_duplicates("image", keep="first")
    amb = df[df.reason.str.startswith("애매")]
    if len(amb):
        print(f"애매 폴더 {len(amb)}장은 제외로 넣었다. 다시 볼 것:", ", ".join(amb.image))
    return df


def build_page(cand, refs):
    items = [{"image": r.image, "cam": r.scene, "time": r.captured_at[5:16], "shift": round(float(r.shift), 1),
              "broken": bool(r.broken), "exclude": True} for r in cand.itertuples()]
    groups = {s: refs[s] for s in cand.scene.unique()}
    page = PAGE_TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace(
        "__REF__", json.dumps(groups, ensure_ascii=False))
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="화각 변경·깨진 화면 찾기와 검수 결과 관리")
    ap.add_argument("--apply", help="image,decision[,reason] 표 (검토 페이지에서 저장한 CSV 등)")
    ap.add_argument("--import-folders", help="폴더 검수 결과가 있는 폴더")
    ap.add_argument("--camera", help="--import-folders에서 카메라 폴더 없이 사진이 바로 있을 때 그 카메라 이름")
    args = ap.parse_args()
    if args.apply:
        save_review(pd.read_csv(args.apply, encoding="utf-8-sig", dtype=str))
        return
    if args.import_folders:
        save_review(import_folders(args.import_folders, args.camera))
        return

    L = pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"camera_id": str})[["image", "camera_id", "captured_at"]]
    done = pd.read_csv(OUT, encoding="utf-8-sig") if OUT.exists() else pd.DataFrame(columns=["image", "scene"])
    known = dict(zip(done.image, done.scene))
    L["scene"] = [known.get(i) or scene_of(i) for i in L.image]
    L = L.sort_values(["scene", "captured_at"]).reset_index(drop=True)

    refs = pd.read_csv(REF, encoding="utf-8-sig", dtype=str).set_index("scene").image.to_dict() if REF.exists() else {}
    new_refs = [s for s in L.scene.unique() if s not in refs]
    for s in new_refs:
        refs[s] = pick_ref(L[L.scene == s].image.tolist())
    if new_refs:
        pd.DataFrame(sorted(refs.items()), columns=["scene", "image"]).to_csv(REF, index=False, encoding="utf-8-sig")
        print(f"기준 사진을 새로 정한 장면 {len(new_refs)}개: {', '.join(new_refs)}")
    MASKS.mkdir(parents=True, exist_ok=True)
    for s in L.scene.unique():
        path = MASKS / f"{s}.png"
        if not path.exists():
            imgs = L[L.scene == s].image
            cv2.imwrite(str(path), make_mask(imgs.sample(min(MASK_SAMPLE, len(imgs)), random_state=0).tolist()))
            print(f"글자 가림 영역을 새로 만든 장면: {s}")

    t = pd.to_datetime(L.captured_at)
    gap = t.diff().dt.total_seconds()
    prev = L.image.shift().where((L.scene == L.scene.shift()) & (gap <= LINK_GAP))
    todo = L[~L.image.isin(done.image)]
    print(f"사진 {len(L)}장 중 새로 잴 사진 {len(todo)}장")
    if len(todo):
        jobs = [(r.image, refs[r.scene], MASKS / f"{r.scene}.png", prev[r.Index] if isinstance(prev[r.Index], str) else None)
                for r in todo.itertuples()]
        with ProcessPoolExecutor() as ex:
            rows = pd.DataFrame(ex.map(measure, jobs, chunksize=32))
        rows["scene"] = todo.scene.values
        done = pd.concat([done, rows], ignore_index=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(OUT, index=False, encoding="utf-8-sig")

    v = L.merge(done.drop(columns="scene"), on="image")
    v["broken"], v["moved"] = flags(v)
    review = load_review()
    cand = v[(v.broken | v.moved) & ~v.image.isin(review.index)]
    print(f"자동 판단: 깨짐 {int(v.broken.sum())}장, 화각 변경 {int(v.moved.sum())}장 / "
          f"검수 완료 {int(v.image.isin(review.index).sum())}장, 검수 안 한 후보 {len(cand)}장")
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
같은 장면이면 <span style="color:var(--ok)"><b>정상</b></span>으로 두세요. 자동 판단이 화각 변경·깨짐으로 본 사진만 모았고, 기본값은 모두 제외입니다.
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
  sec.innerHTML = `<h2>장면 ${cam} <small>(${its.length}장)</small> <button data-a="ex">모두 제외</button><button data-a="ok">모두 정상</button></h2>`;
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
