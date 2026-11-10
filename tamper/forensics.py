"""Classical image forensics for scanned documents.

Three independent signals, each reduced to a block-level heat map of robust
z-scores (how unusual a block is relative to the rest of the same page):

* ELA  - error level analysis: recompress as JPEG and measure what changes.
* noise - local sensor/scan noise level, estimated on flat paper pixels.
* copy-move - pixel-identical patches found at two places on one page.

Every map is expressed on the same BLOCK-pixel grid so they can be fused and
turned into rectangular region proposals for the LLM stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

BLOCK = 16          # px per heat-map cell
MAX_SIDE = 1600     # analysis resolution cap (longest side)
SIGNALS = ("ela", "noise", "copymove")


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    score: float
    signals: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForensicResult:
    scale: float                      # analysis px / original px
    shape: tuple[int, int]            # analysis (h, w)
    maps: dict[str, np.ndarray]       # block-grid z-score maps
    features: dict[str, float]        # image-level scalars
    regions: list[Region] = field(default_factory=list)  # original-pixel coords
    copy_pairs: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # native px clusters


def load_for_analysis(path: str) -> tuple[np.ndarray, float]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _block_reduce(a: np.ndarray, fn=np.mean) -> np.ndarray:
    h, w = a.shape[:2]
    gh, gw = h // BLOCK, w // BLOCK
    a = a[: gh * BLOCK, : gw * BLOCK]
    return fn(a.reshape(gh, BLOCK, gw, BLOCK), axis=(1, 3))


def _robust_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z-score against the median/MAD of the masked blocks; unmasked blocks -> 0."""
    out = np.zeros_like(values, dtype=np.float32)
    if mask.sum() < 20:
        return out
    ref = values[mask]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-6
    out[mask] = (values[mask] - med) / mad
    return out


def content_mask(gray: np.ndarray) -> np.ndarray:
    """Blocks that contain ink (text, logos, pen). Blank paper carries no ELA evidence."""
    dark = (gray < np.percentile(gray, 50) - 40).astype(np.float32)
    return _block_reduce(dark) > 0.03


# --------------------------------------------------------------------------- ELA
def ela_map(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """Per-block error level, z-scored against other inked blocks.

    A region pasted in after the last JPEG save has a different compression
    history, so it responds differently to one more round of compression.
    Ink density drives ELA too, so the error is normalised by local edge energy
    before scoring; otherwise every bold headline looks 'tampered'.
    """
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img, rec).astype(np.float32).mean(axis=2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge = _block_reduce(np.abs(gx) + np.abs(gy))
    err = _block_reduce(diff)
    mask = content_mask(gray)
    ratio = err / (edge + 8.0)
    return _robust_z(ratio, mask)


# ------------------------------------------------------------------------- noise
def noise_map(img: np.ndarray) -> np.ndarray:
    """Per-block noise level on flat pixels, z-scored against the page.

    Scanner noise is roughly uniform over a page. Digitally drawn or pasted
    patches are often too clean (flat fill, synthetic font) or too noisy
    (taken from another scan). Both directions are suspicious, so |z| is used.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resid = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    flat = ((np.abs(gx) + np.abs(gy)) < 40).astype(np.float32)
    num = _block_reduce(np.abs(resid) * flat)
    den = _block_reduce(flat)
    sigma = num / (den + 1e-3)
    # only judge blocks near ink: an edit to a receipt changes text, not margins
    near_ink = cv2.dilate(content_mask(gray.astype(np.uint8)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = (den > 0.3) & near_ink
    return np.abs(_robust_z(sigma, mask))


# --------------------------------------------------------------------- copy-move
CM_B = 8            # duplicate-block size (px, native resolution)
_rng = np.random.default_rng(7)
_K1 = _rng.integers(1, 1 << 20, (CM_B, CM_B)).astype(np.float64)
_K2 = _rng.integers(1, 1 << 20, (CM_B, CM_B)).astype(np.float64)


def copy_move(gray: np.ndarray, min_shift: int = 6, min_blocks: int = 6) -> list[tuple[int, int, int, int, int]]:
    """Find areas duplicated within the page, at native resolution.

    Printed receipts repeat glyphs constantly, so feature matching fires
    everywhere. The discriminating fact for *scans* is that two prints of the
    same digit differ by scan noise, while a copy-pasted digit is identical
    pixel for pixel. So this looks for exact duplicates: every CM_B x CM_B
    block is hashed, equal hashes at different positions are paired, and pairs
    only count when several neighbouring blocks share the same displacement
    (one stray identical block proves nothing on a near-binary scan).

    Returns clusters as (x, y, w, h, n_blocks) covering both copies.
    """
    g = gray.astype(np.float64)
    h, w = gray.shape
    # exact integer hashes via correlation with two random integer kernels
    h1 = cv2.filter2D(g, cv2.CV_64F, _K1, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    h2 = cv2.filter2D(g, cv2.CV_64F, _K2, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    mean = cv2.blur(g, (CM_B, CM_B), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    sq = cv2.blur(g * g, (CM_B, CM_B), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    # a block must carry real texture: ink edges with anti-aliased greys, not
    # flat paper or pure black/white runs, which repeat exactly by chance
    mid = ((gray > 40) & (gray < 215)).astype(np.float32)
    n_mid = cv2.blur(mid, (CM_B, CM_B), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT) * CM_B * CM_B
    ok = (std > 25) & (n_mid >= 10)
    ok[h - CM_B + 1:, :] = False
    ok[:, w - CM_B + 1:] = False
    ys, xs = np.nonzero(ok)
    if len(ys) < 2:
        return []
    k1, k2 = h1[ys, xs], h2[ys, xs]
    order = np.lexsort((k2, k1))
    ys, xs, k1, k2 = ys[order], xs[order], k1[order], k2[order]
    same = (k1[1:] == k1[:-1]) & (k2[1:] == k2[:-1])
    i = np.nonzero(same)[0]
    if len(i) == 0:
        return []
    dy, dx = ys[i + 1] - ys[i], xs[i + 1] - xs[i]
    far = (np.abs(dy) + np.abs(dx)) >= min_shift
    i, dy, dx = i[far], dy[far], dx[far]
    flip = (dy < 0) | ((dy == 0) & (dx < 0))           # canonical direction
    ay = np.where(flip, ys[i + 1], ys[i]); ax = np.where(flip, xs[i + 1], xs[i])
    dy, dx = np.where(flip, -dy, dy), np.where(flip, -dx, dx)
    # group pairs by displacement; a real paste moves many blocks by one vector
    shifts, inv, counts = np.unique(np.stack([dy, dx], 1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    clusters = []
    for si in np.nonzero(counts >= min_blocks)[0]:
        sel = inv == si
        py, px = ay[sel], ax[sel]
        sdy, sdx = int(shifts[si][0]), int(shifts[si][1])
        mask = np.zeros((h, w), np.uint8)
        mask[py, px] = 1
        mask = cv2.dilate(mask, np.ones((CM_B, CM_B), np.uint8), anchor=(0, 0))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for c in range(1, n):
            x, y, cw, ch, area = stats[c]
            nb = int(((py >= y) & (py < y + ch) & (px >= x) & (px < x + cw)).sum())
            if nb < min_blocks or cw < CM_B + 2 or ch < CM_B + 2:
                continue
            clusters.append((int(x), int(y), int(cw), int(ch), nb))
            clusters.append((int(x + sdx), int(y + sdy), int(cw), int(ch), nb))
    return clusters


def copy_move_map(gray_native: np.ndarray, scale: float, grid_shape: tuple[int, int]) -> tuple[np.ndarray, list]:
    """Rasterise duplicate clusters onto the block grid (value = log2 of block count)."""
    grid = np.zeros(grid_shape, np.float32)
    clusters = copy_move(gray_native)
    for x, y, w, h, nb in clusters:
        x0, y0 = int(x * scale) // BLOCK, int(y * scale) // BLOCK
        x1, y1 = int((x + w) * scale) // BLOCK, int((y + h) * scale) // BLOCK
        v = float(np.log2(nb))
        grid[y0:y1 + 1, x0:x1 + 1] = np.maximum(grid[y0:y1 + 1, x0:x1 + 1], v)
    return grid, clusters


# ------------------------------------------------------------------------ fusion
THRESH = {"ela": 6.0, "noise": 6.0, "copymove": 2.5}   # copymove: log2(6 blocks)


def _smooth(m: np.ndarray) -> np.ndarray:
    return cv2.blur(m, (3, 3))


def propose_regions(maps: dict[str, np.ndarray], scale: float, top_k: int = 5) -> list[Region]:
    """Threshold each signal, merge into connected components, rank by evidence."""
    norm = {k: np.clip((maps[k] if k == "copymove" else _smooth(maps[k])) / THRESH[k], 0, 4) for k in SIGNALS}
    fused = np.maximum.reduce([norm[k] for k in SIGNALS])
    binary = (fused >= 1.0).astype(np.uint8)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = labels == i
        per_signal = {k: float(norm[k][comp].max()) for k in SIGNALS}
        fired = [k for k, v in per_signal.items() if v >= 1.0]
        # evidence = strongest signal, with a bonus when independent signals agree
        score = max(per_signal.values()) + 0.5 * (len(fired) - 1)
        s = BLOCK / scale
        regions.append(Region(int(x * s), int(y * s), int(w * s), int(h * s), round(score, 3), fired))
    regions.sort(key=lambda r: -r.score)
    return regions[:top_k]


def analyze(path: str, top_k: int = 5) -> ForensicResult:
    img, scale = load_for_analysis(path)
    native = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ela = ela_map(img)
    cm_grid, pairs = copy_move_map(native, scale, ela.shape)
    maps = {"ela": ela, "noise": noise_map(img), "copymove": cm_grid}
    feats: dict[str, float] = {}
    for k in ("ela", "noise"):
        sm = _smooth(maps[k])
        feats[f"{k}_max"] = float(sm.max())
        feats[f"{k}_p99"] = float(np.percentile(sm, 99))
        feats[f"{k}_frac"] = float((sm >= THRESH[k]).mean())
    feats["copymove_clusters"] = float(len(pairs) // 2)
    feats["copymove_blocks"] = float(sum(p[4] for p in pairs) // 2)
    feats["copymove_max"] = float(cm_grid.max())
    regions = propose_regions(maps, scale, top_k)
    feats["top_region_score"] = regions[0].score if regions else 0.0
    feats["n_regions"] = float(len(regions))
    return ForensicResult(scale, img.shape[:2], maps, feats, regions, pairs)


# ----------------------------------------------------------------- visualisation
def heat_overlay(img: np.ndarray, m: np.ndarray, vmax: float) -> np.ndarray:
    """Blend a block map over the image as a heat map (for humans and the LLM)."""
    h, w = img.shape[:2]
    norm = np.clip(m / vmax, 0, 1)
    big = cv2.resize(norm, (m.shape[1] * BLOCK, m.shape[0] * BLOCK), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w), np.float32)
    canvas[: big.shape[0], : big.shape[1]] = big[:h, :w]
    color = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.15 + 0.6 * canvas)[..., None]
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


def draw_regions(img: np.ndarray, regions: list[Region], scale: float) -> np.ndarray:
    out = img.copy()
    for i, r in enumerate(regions, 1):
        x, y, w, h = (int(v * scale) for v in (r.x, r.y, r.w, r.h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, f"R{i}", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
