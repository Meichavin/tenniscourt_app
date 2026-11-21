"""
ひび割れ検出コア（Streamlit 非依存）

旧版からの主な改善点
  1. 解像度の正規化   : 画像を一定サイズ (STD_SIZE) に揃えてから解析
                       → 1280px と 3024px の写真を同じ尺度で比較できる
  2. 計測枠の除外     : 緑色の枠ひも・四隅の白いピン・枠外の地面を自動でマスク
                       → 枠ひもを「ひび」と誤検出しない
  3. 照明ムラの補正   : 大きなぼかしで推定した背景との比（対数）をとる
                       → 木の影・濡れムラ・日なた/日かげの影響を軽減
  4. 線状構造の強調   : ヘッセ行列による「暗い谷線」フィルタ（多重スケール）
                       ＋ 16 方向の線積分で「長く続く線」だけを強調
                       → 砂粒・小石・点状の汚れを抑制
  5. 雑音基準の閾値   : 画像ごとの地面のザラつき（雑音レベル）を基準にした
                       ヒステリシス閾値 → Otsu のように「必ず何かを検出する」ことがない
  6. 形状フィルタ     : 長さ・細長さで連結成分をふるい、足跡や影の塊を除去
  7. 長さの計測       : 斜め方向を √2 で数える骨格長 → 総延長の過小評価を解消
  8. 高速化           : OpenCV の畳み込みのみで実装（1 枚あたり数秒）
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage import morphology

# ── パラメータ（画素単位はすべて STD_SIZE 基準）──────────────
STD_SIZE = 1600  # 解析時の長辺サイズ [px]


@dataclass
class Params:
    bg_sigma: float = 20.0                 # 背景（照明）推定のぼかし幅
    sigmas: tuple = (1.5, 2.0, 3.0)        # 谷線フィルタのスケール（ひび幅 ≒ 3〜8px）
    local_sigma: float = 40.0              # 局所的なザラつきを測る範囲
    line_len: int = 31                     # 線積分の長さ
    n_dir: int = 16                        # 線積分の方向数
    th_high: float = 3.5                   # ヒステリシス上側（雑音レベルの何倍か）
    th_low: float = 2.4                    # ヒステリシス下側（合成ひび試験で F1 最大）
    prune: int = 12                        # これより短いヒゲ状の枝は削除 [px]
    min_length: float = 60.0               # これより短い線は除外 [px]
    min_extent: float = 50.0               # 端から端までの広がりの最小値 [px]
    max_width: float = 8.0                 # ひび幅の上限（面積計算用）[px]
    dark_th: float = -0.03                 # ひび領域とみなす暗さ（背景比の対数）
    border_margin: float = 0.015           # 枠付近の除外幅（画像サイズ比）

    def to_dict(self):
        return asdict(self)


DEFAULT = Params()


# ── 前処理 ────────────────────────────────────────────
def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # EXIF の向きも適用される
    if img is None:
        raise ValueError("画像を読み込めませんでした")
    return img


def normalize_size(img: np.ndarray, size: int = STD_SIZE) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp), scale


def _green_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    g = ((h >= 38) & (h <= 88) & (s >= 65) & (v >= 35)).astype(np.uint8)
    # 小さな点（青っぽい小石など）は除き、ひもらしい大きさの成分だけ残す
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    big = np.zeros(n, np.uint8)
    big[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 120) | (np.maximum(stats[1:, 2], stats[1:, 3]) >= 40)
    return big[lab]


def _frame_bounds(green: np.ndarray, search: float = 0.12, min_frac: float = 0.2):
    """外周付近で緑の画素が多い行・列＝枠ひもとみなし、その内側の矩形を返す"""
    h, w = green.shape
    col, row = green.mean(axis=0), green.mean(axis=1)
    sw, sh = int(w * search), int(h * search)

    def find(profile, lo, hi, take_last):
        idx = np.where(profile[lo:hi] >= min_frac)[0]
        if len(idx) == 0:
            return None
        return lo + (idx.max() if take_last else idx.min())

    left, right = find(col, 0, sw, True), find(col, w - sw, w, False)
    top, bottom = find(row, 0, sh, True), find(row, h - sh, h, False)
    return (0 if top is None else top, h - 1 if bottom is None else bottom,
            0 if left is None else left, w - 1 if right is None else right)


def court_mask(img: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """解析対象（枠の内側の地面）= 1 のマスク"""
    h, w = img.shape[:2]
    green = _green_mask(img)
    top, bottom, left, right = _frame_bounds(green)
    m = int(max(h, w) * p.border_margin)
    mask = np.zeros((h, w), np.uint8)
    mask[top + m:bottom - m + 1, left + m:right - m + 1] = 1

    # 枠内に入り込んだ緑のひもを除外
    bad = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # 四隅の白いピン（角から 8% 以内の明るい無彩色）を除外
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 215) & (hsv[..., 1] < 40)).astype(np.uint8)
    c = int(max(h, w) * 0.08)
    corner = np.zeros_like(white)
    corner[:c, :c] = corner[:c, -c:] = corner[-c:, :c] = corner[-c:, -c:] = 1
    bad |= cv2.dilate(white & corner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    mask[bad > 0] = 0
    return mask


def flatten_illumination(img: np.ndarray, mask: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """輝度と背景の対数比（0 付近が地面、負＝周囲より暗い）"""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) + 1.0
    fill = L.copy()
    fill[mask == 0] = np.median(L[mask > 0]) if mask.any() else np.median(L)
    bg = cv2.GaussianBlur(fill, (0, 0), p.bg_sigma)
    return np.log(L) - np.log(bg)


# ── 線状構造の強調 ──────────────────────────────────────
def _valley_filter(f: np.ndarray, sigmas) -> np.ndarray:
    """ヘッセ行列の固有値から「暗い谷線」らしさを計算（多重スケールの最大値）"""
    out = np.zeros_like(f)
    for s in sigmas:
        g = cv2.GaussianBlur(f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + tmp)   # 大きい方：谷を横切る方向の曲率（谷なら正）
        l2 = 0.5 * (dxx + dyy - tmp)   # 小さい方：谷に沿う方向の曲率
        # 線：l1 大・|l2| 小 ／ 点（小石）：l1 と l2 がともに大 → 抑制
        r = (s ** 2) * np.maximum(l1 - np.abs(l2), 0)
        out = np.maximum(out, r)
    return out


def _line_kernels(length: int, n_dir: int):
    ks = []
    c = length // 2
    for i in range(n_dir):
        th = np.pi * i / n_dir
        k = np.zeros((length, length), np.float32)
        dx, dy = np.cos(th) * c, np.sin(th) * c
        cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1, cv2.LINE_AA)
        ks.append(k / k.sum())
    return ks


def crack_response(img_std: np.ndarray, mask: np.ndarray, p: Params = DEFAULT):
    """ひびらしさマップ（雑音レベル = 1 に正規化）と照明補正画像を返す"""
    flat = flatten_illumination(img_std, mask, p)
    v = _valley_filter(flat, p.sigmas)
    v[mask == 0] = 0
    # 局所的なザラつきで割る（足跡で荒れた所・粒の粗い所で誤検出しにくくする）
    m = mask.astype(np.float32)
    local = cv2.GaussianBlur(v, (0, 0), p.local_sigma) / (cv2.GaussianBlur(m, (0, 0), p.local_sigma) + 1e-6)
    v = v / (local + np.median(local[mask > 0]) * 0.5 + 1e-9)
    v[mask == 0] = 0
    # 方向ごとに線に沿って平均 → 長く続く線ほど値が残る
    line = np.zeros_like(v)
    for k in _line_kernels(p.line_len, p.n_dir):
        line = np.maximum(line, cv2.filter2D(v, -1, k, borderType=cv2.BORDER_REFLECT))
    # 雑音レベル（地面のザラつき）で正規化：中央値 + 1.4826·MAD
    vals = line[mask > 0]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    z = (line - med) / mad
    z[mask == 0] = 0
    return z.astype(np.float32), flat


# ── 2 値化と形状フィルタ ─────────────────────────────────
def _skeleton_length(skel: np.ndarray) -> float:
    """8 近傍のつながりを考慮した骨格の長さ（縦横 = 1、斜め = √2）"""
    s = skel.astype(np.uint8)
    straight = int((s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum())
    a, b, c, d = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    diag = int((a & d).sum() + (b & c).sum())
    tri = int((a & d & (b | c)).sum() + (b & c & (a | d)).sum())  # L 字の角で二重計上しない
    return float(straight + np.sqrt(2) * (diag - tri))


def _hysteresis(z: np.ndarray, low: float, high: float) -> np.ndarray:
    weak = (z >= low).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(lab[z >= high])
    keep = np.zeros(n, bool)
    keep[strong_labels] = True
    keep[0] = False
    return keep[lab]


_NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


def _endpoints(skel: np.ndarray) -> np.ndarray:
    nb = cv2.filter2D(skel.astype(np.float32), -1, _NB, borderType=cv2.BORDER_CONSTANT)
    return skel & (nb <= 1.5)


def _prune(skel: np.ndarray, k: int) -> np.ndarray:
    """長さ k 以下のヒゲ（枝分かれした短い枝）を取り除き、本線の先端は元に戻す"""
    s = skel.copy()
    for _ in range(k):
        s &= ~_endpoints(s)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(k):
        grow = cv2.dilate(_endpoints(s).astype(np.uint8), k3) > 0
        new = s | (grow & skel)
        if (new == s).all():
            break
        s = new
    return s


def segment_cracks(z: np.ndarray, flat: np.ndarray, p: Params = DEFAULT):
    """ひびの中心線（骨格）とひび領域を返す"""
    band = _hysteresis(z, p.th_low, p.th_high)
    # 帯の輪郭をなめらかにしてから細線化（ヒゲの発生を抑える）
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.0) > 0.5
    skel =_prune(morphology.skeletonize(band), p.prune)

    # ひび領域：中心線から max_width/2 以内で、周囲より暗い画素
    r = int(np.ceil(p.max_width / 2))
    near = cv2.dilate(skel.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    region = (near & (flat < p.dark_th)) | skel

    n, lab, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    comps = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if max(w, h) < p.min_extent:               # 端から端までの広がりが小さい＝砂粒の模様
            continue
        sub = lab[y:y + h, x:x + w] == i
        length = _skeleton_length(sub)
        if length < p.min_length:
            continue
        keep[i] = True
        comps.append(dict(length=length, bbox=(x, y, w, h)))
    skel_keep = keep[lab]
    near_keep = cv2.dilate(skel_keep.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    region_keep = region & near_keep
    return skel_keep, region_keep, comps


# ── 公開 API ──────────────────────────────────────────
def detect_cracks(image_bytes: bytes, p: Params = DEFAULT, frame_size_cm: float | None = None) -> dict:
    return detect_cracks_img(decode_image(image_bytes), p, frame_size_cm)


def detect_cracks_img(img_orig: np.ndarray, p: Params = DEFAULT, frame_size_cm: float | None = None) -> dict:
    img, _ = normalize_size(img_orig)
    h, w = img.shape[:2]
    mask = court_mask(img, p)
    z, flat = crack_response(img, mask, p)
    skeleton, closed, comps = segment_cracks(z, flat, p)

    n, labels = cv2.connectedComponents(closed.astype(np.uint8), connectivity=8)
    total_len = _skeleton_length(skeleton)
    total_area = int(closed.sum())
    roi_area = int(mask.sum()) or h * w

    res = dict(
        img=img, mask=mask, response=z, closed=closed.astype(np.uint8) * 255,
        labels=labels, skeleton=skeleton, comps=comps,
        w=w, h=h, orig_w=img_orig.shape[1], orig_h=img_orig.shape[0], roi_area=roi_area,
        count=len(comps),
        total_px=int(round(total_len)),
        total_area=total_area,
        mean_width=total_area / total_len if total_len > 0 else 0.0,
        density_len=total_len / roi_area * 1_000_000,   # px / 100万px²（STD_SIZE 基準）
        density_area=total_area / roi_area * 100,       # %
    )
    if frame_size_cm:
        # 枠ひもの内側の一辺 [px] を実寸 frame_size_cm に対応させる
        top, bottom, left, right = _frame_bounds(_green_mask(img))
        side_px = max(bottom - top, right - left)
        cm_per_px = frame_size_cm / side_px
        res["cm_per_px"] = cm_per_px
        res["length_cm"] = total_len * cm_per_px
        res["mean_width_mm"] = res["mean_width"] * cm_per_px * 10
        res["density_cm_per_m2"] = res["length_cm"] / (roi_area * cm_per_px ** 2 / 10_000)
    return res
