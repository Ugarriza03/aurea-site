"""Procedural backgrounds matching the reference: sun-lit garden bokeh above a dark
wooden table (locked-off shots) and a darker garden with warm/red accents and
dappled light for the hero shots."""
import numpy as np
import cv2
from gfx import fbm, smoothstep

OX, OY = 330, 420          # design (0,0) -> texture (OX,OY)
TW, TH = 1740, 2760
TABLE_Y = 1310             # far edge of the table (design y)


def _palette(v, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    out = np.zeros(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, xs, cs[:, c])
    return out


def _bokeh(img, rng, n, ymax, rmin, rmax, cols, alpha=0.55, weight=None):
    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    acc = np.zeros((h, w), np.float32)
    for _ in range(n):
        for _try in range(8):
            x, y = rng.integers(0, w), rng.integers(0, ymax)
            if weight is None or weight[y, x] > rng.random():
                break
        r = rng.uniform(rmin, rmax)
        c = np.array(cols[rng.integers(len(cols))], np.float32) * rng.uniform(0.7, 1.1)
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (int(x) * 4, int(y) * 4), int(r * 4), 255, -1, cv2.LINE_AA, shift=2)
        sub = m > 0
        ring = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.2)
        layer[sub] = np.maximum(layer[sub], c * alpha)
        acc = np.maximum(acc, ring * alpha)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.5)
    return img * (1 - acc[..., None] * 0.6) + layer


def make_table(tex, seed, y_top_tex, base=(0.20, 0.085, 0.04), sheen=0.10):
    h, w = tex.shape[:2]
    rng = np.random.default_rng(seed)
    ys = np.arange(h, dtype=np.float32)
    rows = slice(y_top_tex, h)
    n = h - y_top_tex
    # perspective: grain stretched, compressed toward the far edge
    depth = 1.0 / (np.arange(n, dtype=np.float32) + 40.0)
    zz = (depth - depth.min()) / (depth.max() - depth.min())
    g = fbm((512, 2048), seed + 3, octaves=6, base=3)
    gx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    v = np.cumsum(np.full(n, 1.0) * (0.25 + 1.6 * (1 - zz)) ) ; v = (v / v.max())[:, None]
    mapx = (gx * 2047).repeat(n, 0).astype(np.float32)
    mapy = (v * 511 * np.ones((1, w))).astype(np.float32)
    grain = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR)
    streak = fbm((n, w), seed + 5, octaves=4, base=2)
    lines = 0.5 + 0.5 * np.sin(2 * np.pi * (v * 26.0 + 3.5 * grain + 0.8 * streak))
    lines = lines ** 3
    wood = np.array(base, np.float32)[None, None] * (0.6 + 0.6 * grain[..., None] + 0.25 * lines[..., None]) * (0.7 + 0.6 * streak[..., None])
    wood *= (0.85 + 1.1 * zz[:, None, None] ** 1.5)
    # sheen: reflection of bright garden onto glossy wood, stronger far away (fresnel)
    refl = tex[max(0, y_top_tex - n):y_top_tex][::-1]
    if refl.shape[0] < n:
        refl = np.pad(refl, ((0, n - refl.shape[0]), (0, 0), (0, 0)), mode='edge')
    fres = (sheen * (0.3 + 0.7 * zz))[:, None, None]
    wood = wood * (1 - fres) + cv2.GaussianBlur(refl, (0, 0), 6) * fres
    # depth of field: far edge blurred, near (bottom) mildly blurred
    blur_far = cv2.GaussianBlur(wood, (0, 0), 16)
    blur_near = cv2.GaussianBlur(wood, (0, 0), 2.2)
    dist_focus = np.abs(np.arange(n) - (1530 + OY - y_top_tex)).astype(np.float32)
    k = smoothstep(10, 90, dist_focus)[:, None, None]
    kf = (np.arange(n) < (1530 + OY - y_top_tex)).astype(np.float32)[:, None, None]
    wood = wood * (1 - k) + (blur_far * kf + blur_near * (1 - kf)) * k
    tex[rows] = wood
    # soft far edge
    edge = slice(y_top_tex - 30, y_top_tex + 30)
    tex[edge] = cv2.GaussianBlur(tex[y_top_tex - 60:y_top_tex + 60], (0, 0), 10)[30:90]
    return tex


def garden(seed=11, variant='garden'):
    rng = np.random.default_rng(seed)
    h, w = TH, TW
    f = fbm((h // 4, w // 4), seed, octaves=6, base=5, persistence=0.58)
    f2 = fbm((h // 4, w // 4), seed + 1, octaves=4, base=3)
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_CUBIC)
    sun = smoothstep(0.45, 0.85, f2)
    if variant == 'garden':
        stops = [(0.0, (0.03, 0.07, 0.015)), (0.35, (0.10, 0.24, 0.04)), (0.55, (0.30, 0.52, 0.07)),
                 (0.72, (0.62, 0.78, 0.14)), (0.88, (0.93, 0.90, 0.40)), (1.0, (1.0, 0.98, 0.78))]
        v = np.clip(f * 0.8 + sun * 0.45 - 0.02, 0, 1)
    else:  # darker, warmer garden for the hero shots
        stops = [(0.0, (0.02, 0.035, 0.012)), (0.40, (0.07, 0.15, 0.03)), (0.62, (0.22, 0.38, 0.05)),
                 (0.80, (0.55, 0.62, 0.12)), (1.0, (0.98, 0.88, 0.50))]
        v = np.clip(f * 0.7 + sun * 0.3 - 0.08, 0, 1)
    img = _palette(v, stops)
    # defocused foliage: thousands of leaf shapes lit by the low sun, then blurred
    leaves = np.zeros_like(img)
    lacc = np.zeros((h, w), np.float32)
    nleaf = 3600 if variant == 'garden' else 2600
    for _ in range(nleaf):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, TABLE_Y + OY))
        lit = sun[y, x] * 0.8 + f[y, x] * 0.5 + rng.uniform(-0.25, 0.25)
        vv = float(np.clip(lit, 0, 1))
        col = _palette(np.array([vv], np.float32), stops)[0] * rng.uniform(0.75, 1.25)
        ax_, ay_ = int(rng.uniform(16, 52)), int(rng.uniform(9, 26))
        cv2.ellipse(leaves, (x, y), (ax_, ay_), float(rng.uniform(0, 180)), 0, 360, col.tolist(), -1, cv2.LINE_AA)
        cv2.ellipse(lacc, (x, y), (ax_, ay_), 0, 0, 360, 1.0, -1)
    lacc = np.clip(lacc, 0, 1)
    img = img * (1 - 0.85 * lacc[..., None]) + leaves * 0.85
    img = cv2.GaussianBlur(img, (0, 0), 7.5)
    ymax = TABLE_Y + OY
    cols = [(1.0, 0.97, 0.72), (0.85, 0.95, 0.40), (0.60, 0.85, 0.25), (1.0, 0.85, 0.45)]
    wgt = cv2.resize(sun, (w, h))
    img = _bokeh(img, rng, 260, ymax, 6, 22, cols, 0.45, wgt)
    img = cv2.GaussianBlur(img, (0, 0), 2.5)
    img = _bokeh(img, rng, 70, ymax, 18, 46, cols, 0.22, wgt)
    if variant == 'dapple':
        # warm terracotta / red pots and dark planters on the far edge (as in the reference hero shot)
        for _ in range(6):
            x = rng.integers(0, w); y = TABLE_Y + OY - rng.integers(10, 170)
            col = np.array([(0.42, 0.05, 0.035), (0.05, 0.025, 0.02), (0.55, 0.16, 0.06)][rng.integers(3)], np.float32)
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (int(x), int(y)), (int(rng.integers(40, 90)), int(rng.integers(30, 80))), 0, 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 22) * 0.85
            img = img * (1 - m[..., None]) + col * m[..., None]
        img = cv2.GaussianBlur(img, (0, 0), 2.0)
    # gentle vertical light falloff (sky brighter on top)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img *= (1.08 - 0.25 * yy)
    tex = make_table(img.astype(np.float32), seed + 20, TABLE_Y + OY,
                     base=(0.22, 0.09, 0.045) if variant == 'garden' else (0.26, 0.12, 0.06))
    return np.clip(tex, 0, 2).astype(np.float32)


def dapple_pattern(t, seed=5, shape=(TH // 4, TW // 4)):
    """Leaf-shadow light pattern (animated sway) in texture coordinates."""
    base = fbm(shape, seed, octaves=4, base=9, persistence=0.5)
    sway = np.float32(np.sin(t * 1.3) * 3.0)
    M = np.float32([[1, 0, sway], [0, 1, sway * 0.4]])
    b = cv2.warpAffine(base, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT)
    pat = smoothstep(0.50, 0.62, b)
    pat = cv2.GaussianBlur(pat, (0, 0), 1.2)
    return cv2.resize(pat, (TW, TH), interpolation=cv2.INTER_CUBIC)


def sample(tex, U, V, blur_tex=None, dx=None, dy=None):
    mx = U + OX
    my = V + OY
    if dx is not None:
        mx = mx + dx
        my = my + dy
    return cv2.remap(tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
