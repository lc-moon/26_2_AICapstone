# 화면 위 글자(시각 표시·지명) 자리 — CNN 입력에서 평균색으로 칠해 가린다.
#
#   python Model/text_mask.py      장면마다 글자 자리를 찾아 Model/text_mask/{장면}.png로 저장하고,
#                                  기준 사진 위에 겹친 확인용 모음을 data/review/text_mask_*.jpg로 만든다 (눈으로 확인할 것)
#
# 왜 가리나 (2026-10-11): 시각 표시는 몇 시인지 알려 줘서, 모델이 뿌연 정도 대신 "시계를 읽는" 지름길을 배울 수 있다.
# 지명 글자는 늘 같아 정보가 없고 화면을 차지한다. 베이징 사진에는 글자가 없어 가리지 않는다.
#
# 찾는 법: 글자는 조명과 상관없이 늘 "흰 글자 + 검은 테두리"다. 장면마다 사진 80장에서 그런 픽셀이 85% 넘게 반복되는 곳을 글자로 본다.
# 건물·차선은 아침·저녁·역광 사진에서 조건이 깨져 빠진다. 시각 줄은 숫자가 바뀌어 일부만 잡히므로, 위쪽 12% 안에서 잡힌 글자를
# 감싸는 직사각형으로 통째로 가린다. 시각 표시는 기간에 따라 있다 없다 해서(find 참고) 3일 구간마다 찾아 합친다.
# 화각 검사용 view_mask(모든 사진에서 같은 픽셀)는 시각 줄을 못 잡아 따로 만들었다.
# 21개 장면을 겹쳐 보고 시각 줄·지명 글자만 잡힌 것을 확인했다.
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from view_check import OUT as VIEW_CHECK, REF, load_review

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/images"
MASKS = Path(__file__).with_name("text_mask")
SHEETS = ROOT / "data/review"
SAMPLE, REPEAT = 80, 0.85
WINDOW, WINDOW_SAMPLE = 3, 60     # 시각 줄은 3일 구간마다 따로 찾는다 (기간에 따라 있다 없다 하므로)
# 가끔만 나오는 다른 지명 글자 — 사람이 검수로 찾은 본보기 사진. 같은 글자가 있는 사진을 모두 찾아 그 자리도 가린다.
# 62번은 592장 중 7장(날짜·시각 불규칙)에 평소와 다른 더 긴 글자(도화IC(인천항)·서인천IC 등)가 나와 가림 밖으로 삐져나왔다(2026-10-11).
# 드물어서 반복 기준으로는 안 잡히고, 자동으로 찾는 규칙은 저녁 햇빛 반사를 글자로 잘못 잡아 쓰지 않았다
EXAMPLES = {"62@960x540": ["62/2026-09-29/1220.jpg"]}
MATCH = 0.3                       # 본보기의 가림 밖 글자 픽셀 중 이 비율 넘게 겹치면 같은 글자로 본다


def components(images):
    """사진들에서 흰 글자 + 검은 테두리가 REPEAT 넘게 반복되는 덩어리들 → (덩어리 그림, 위쪽 12% 안 덩어리들의 테두리 상자)"""
    frac = None
    for p in images:
        g = cv2.imread(str(IMAGES / p), cv2.IMREAD_GRAYSCALE)
        t = ((g > 185) & (cv2.erode(g, np.ones((5, 5), np.uint8)) < 90)).astype(np.float32)
        frac = t if frac is None else frac + t
    frac /= len(images)
    h, w = frac.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats((frac > REPEAT).astype(np.uint8), 8)
    text, top = np.zeros((h, w), np.uint8), []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 6:
            continue
        text[lab == i] = 1
        if cen[i][1] < 0.12 * h:
            top.append(st[i])
    return text, top


def time_rect(m, top, side):
    """위쪽 덩어리 중 화면 한쪽(왼쪽·오른쪽 절반)에 있는 것들을 감싸는 직사각형을 칠한다 = 시각 줄.
    글자 크기 이상(960×540에서 10픽셀) 덩어리가 5개 이상일 때만 — 10041번에서 가는 선 하나(1×11픽셀)가 시각 줄로 잡힌 적이 있다.
    진짜 시각 줄은 글자 덩어리가 20개 넘는다"""
    h, w = m.shape
    top = [s for s in top if (s[0] + s[2] / 2 < w / 2) == (side == "left") and s[cv2.CC_STAT_AREA] >= max(15, w * h * 2e-5)]
    if len(top) < 5:
        return
    x0, y0 = min(s[0] for s in top), min(s[1] for s in top)
    x1, y1 = max(s[0] + s[2] for s in top), max(s[1] + s[3] for s in top)
    # 바깥쪽은 화면 끝까지. 맨 끝 숫자(초)는 매번 바뀌어 잘 안 잡혀서, 여백만 두면 작은 화면(10041·10042 640×360)에서 삐져나왔다
    x0, x1 = (0, x1 + int(0.02 * w)) if side == "left" else (x0 - int(0.02 * w), w - 1)
    cv2.rectangle(m, (max(0, x0), max(0, y0 - int(0.025 * h))), (min(w - 1, x1), min(h - 1, y1 + int(0.025 * h))), 1, -1)


def find(g):
    """g: 한 장면의 사진 표 (image, day). 지명 글자는 전체 사진에서, 시각 줄은 3일 구간마다 찾아 모두 합친다.
    시각 표시가 일부 기간에만 있었어도(대부분 9/23부터, 143번은 9/28 오전까지 왼쪽 위) 그 자리는 늘 가린다 —
    시각이 있는 사진과 없는 사진이 같은 모양으로 들어가게. 하루치만 보면 흰 건물·구름이 가끔 잡혀 3일씩 모은다"""
    text, top = components(g.image.sample(min(SAMPLE, len(g)), random_state=0).tolist())
    h, w = text.shape
    # 글자 사이 틈을 메우고 테두리까지 덮게 넓힌다. 크기는 화면 폭에 비례 (960 폭에서 메우기 20, 넓히기 9픽셀)
    text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, np.ones((w // 48, w // 48), np.uint8))
    m = cv2.dilate(text, np.ones((w // 100 | 1, w // 100 | 1), np.uint8))
    tops = [top]
    days = sorted(g.day.unique())
    for i in range(max(1, len(days) - WINDOW + 1)):
        part = g[g.day.isin(days[i:i + WINDOW])]
        tops.append(components(part.image.sample(min(WINDOW_SAMPLE, len(part)), random_state=0).tolist())[1])
    for side in ("left", "right"):
        for t in tops:
            time_rect(m, t, side)
    m = m > 0
    band = np.zeros_like(m)
    band[int(0.65 * h):] = True       # 지명 글자 줄이 있는 아래쪽만 본다. 위까지 보면 차선·건물이 본보기에 섞인다
    for ex in EXAMPLES.get(g.scene.iloc[0], []):
        strict = lambda p: (lambda x: (x > 230) & (cv2.erode(x, np.ones((5, 5), np.uint8)) < 40))(
            cv2.imread(str(IMAGES / p), cv2.IMREAD_GRAYSCALE))
        tmpl = strict(ex) & ~m & band
        same = [p for p in g.image if (strict(p) & tmpl).sum() > MATCH * tmpl.sum()]
        alt, _ = components(same)
        alt[~band] = 0
        alt = cv2.morphologyEx(alt, cv2.MORPH_CLOSE, np.ones((w // 48, w // 48), np.uint8))
        m |= cv2.dilate(alt, np.ones((w // 100 | 1, w // 100 | 1), np.uint8)) > 0
        print(f"  {g.scene.iloc[0]}: 본보기 {ex}와 같은 글자 사진 {len(same)}장 → 그 자리도 가림")
    return m


def load(scene):
    """장면의 글자 자리 (True = 가림). 베이징·글자 자리를 만들지 않은 장면은 None"""
    p = MASKS / f"{scene}.png"
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 if p.exists() else None


def main():
    v = pd.read_csv(VIEW_CHECK, encoding="utf-8-sig")
    v = v[v.image.map(load_review()) == "keep"]
    v["day"] = v.image.str.split("/").str[1]
    ref = pd.read_csv(REF).set_index("scene").image
    MASKS.mkdir(exist_ok=True)
    tiles = []
    for scene, g in v.groupby("scene"):
        m = find(g)
        cv2.imwrite(str(MASKS / f"{scene}.png"), m.astype(np.uint8) * 255)
        over = cv2.imread(str(IMAGES / ref[scene]))
        over[m] = (0.4 * over[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        over = cv2.resize(over, (640, 360))
        cv2.putText(over, f"{scene} {m.mean():.1%}", (5, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(over)
        print(f"  {scene:16} 가림 {m.mean():.1%}")
    for k in range(0, len(tiles), 9):
        part = tiles[k:k + 9]
        part += [np.zeros_like(tiles[0])] * (-len(part) % 3)
        cv2.imwrite(str(SHEETS / f"text_mask_{k // 9}.jpg"), np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"글자 자리: {MASKS}\n확인용 모음: {SHEETS}\\text_mask_*.jpg (빨간 곳이 가리는 자리)")


if __name__ == "__main__":
    main()
