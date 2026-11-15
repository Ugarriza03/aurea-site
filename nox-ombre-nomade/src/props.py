"""Props seen in the reference reel, rebuilt procedurally: glass pipettes, droppers with
black rubber bulbs, glass funnels, syringe, amber jar, tweezers, beaker, crimping tool,
black nitrile gloves, liquid streams, incense smoke and the final atomizer mist.
Also builds the ingredient sprites (oud wood, chips, botanicals, cardamom, saffron)."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from gfx import (Mask, smoothstep, over, add, mul, thick_path, rot_pts, rrect_pts, ellipse_pts,
                 blob_normals, shade_material, cylinder_metal, fbm)

CX = 540.0
NITRILE = dict(base=(0.018, 0.018, 0.022), spec=0.30, shin=14, rim=0.42, rim_col=(1.0, 0.86, 0.62))


# ======================================================================= sprites
def _rgba(img):
    return np.asarray(img).astype(np.float32) / 255.0


def make_sprites(seed=7):
    rng = np.random.default_rng(seed)
    sp = {}
    # --- agarwood (oud) piece: long, tapered, dark resinous wood
    L, Wd = 420, 70
    im = Image.new('RGBA', (L, Wd), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    xs = np.linspace(0, L - 1, 60)
    half = 4 + 25 * (1 - xs / L) ** 0.7
    bend = 6 * np.sin(xs / L * np.pi * 1.2)
    top = [(x, Wd / 2 - h + b) for x, h, b in zip(xs, half, bend)]
    bot = [(x, Wd / 2 + h * 0.9 + b) for x, h, b in zip(xs[::-1], half[::-1], bend[::-1])]
    d.polygon(top + bot, fill=(255, 255, 255, 255))
    a = np.asarray(im)[..., 3].astype(np.float32) / 255
    grain = fbm((Wd, L), seed, octaves=5, base=6)
    streak = cv2.resize(fbm((8, L // 3), seed + 1, octaves=3, base=3), (L, Wd))
    yy = np.linspace(-1, 1, Wd)[:, None]
    shade = 0.75 + 0.45 * np.cos(yy * 1.4)
    base = np.array([0.11, 0.055, 0.025], np.float32)
    col = base * (0.5 + 0.9 * grain[..., None]) * shade[..., None]
    col = col * (0.7 + 0.6 * (streak[..., None] > 0.55)) + np.array([0.20, 0.10, 0.04]) * (grain[..., None] > 0.72) * 0.6
    sp['stick'] = np.dstack([np.clip(col, 0, 1), cv2.GaussianBlur(a, (0, 0), 0.7)]).astype(np.float32)

    def blobby(w, h, colfn, nverts=9, jitter=0.35, blur=0.6):
        out = []
        for k in range(6):
            im = Image.new('L', (w, h), 0)
            dd = ImageDraw.Draw(im)
            ang = np.sort(rng.uniform(0, 2 * np.pi, nverts))
            r = 1 - jitter * rng.random(nverts)
            pts = [(w / 2 + np.cos(t) * rr * (w / 2 - 2), h / 2 + np.sin(t) * rr * (h / 2 - 2)) for t, rr in zip(ang, r)]
            dd.polygon(pts, fill=255)
            a = cv2.GaussianBlur(np.asarray(im).astype(np.float32) / 255, (0, 0), blur)
            out.append(np.dstack([colfn(w, h, k), a]).astype(np.float32))
        return out

    def chipcol(w, h, k):
        g = fbm((h, w), seed + 10 + k, octaves=3, base=3)
        c = np.array([0.16, 0.08, 0.035]) * (0.5 + g[..., None])
        return np.clip(c, 0, 1)

    def flakecol(w, h, k):
        g = fbm((h, w), seed + 20 + k, octaves=3, base=3)
        c = np.array([0.55, 0.52, 0.30]) * (0.55 + 0.6 * g[..., None])
        return np.clip(c, 0, 1)

    sp['chip'] = blobby(30, 20, chipcol)
    sp['flake'] = blobby(24, 16, flakecol, jitter=0.5)
    # --- green cardamom pods
    pods = []
    for k in range(4):
        w, h = 58, 30
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ex = (xx - w / 2) / (w / 2 - 2)
        ey = (yy - h / 2) / (h / 2 - 2)
        r = ex ** 2 + ey ** 2 * (1 + 0.6 * ex ** 2)
        a = smoothstep(1.0, 0.85, r)
        tip = smoothstep(0.75, 1.0, np.abs(ex))
        ridges = 0.5 + 0.5 * np.sin(ey * 9 + ex * 1.5 + k)
        n = np.clip(1 - r, 0, 1) ** 0.5
        c = np.array([0.42, 0.52, 0.18]) * (0.45 + 0.6 * n[..., None]) * (0.85 + 0.25 * ridges[..., None])
        c = c * (1 - 0.35 * tip[..., None]) + np.array([0.25, 0.2, 0.08]) * 0.35 * tip[..., None]
        pods.append(np.dstack([np.clip(c, 0, 1), a]).astype(np.float32))
    sp['pod'] = pods
    # --- saffron threads
    th = []
    for k in range(8):
        w, h = 56, 16
        im = Image.new('RGBA', (w * 4, h * 4), (0, 0, 0, 0))
        dd = ImageDraw.Draw(im)
        xs = np.linspace(6, w * 4 - 6, 24)
        ys = h * 2 + np.sin(xs / (w * 4) * np.pi * rng.uniform(0.6, 1.6) + rng.uniform(0, 3)) * rng.uniform(5, 18)
        for i in range(len(xs) - 1):
            t = i / len(xs)
            wd = int(3 + 9 * t ** 1.5)
            colr = (int(150 + 60 * t), int(20 + 30 * t ** 3), 10, 255)
            dd.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=colr, width=wd)
        dd.line([(xs[0], ys[0]), (xs[3], ys[3])], fill=(230, 160, 40, 255), width=3)
        im = im.resize((w, h), Image.LANCZOS)
        th.append(_rgba(im))
    sp['saffron'] = th
    return sp


def draw_sprite(frame, cam, spr, x, y, ang, scale=1.0, alpha=1.0, tint=None):
    h, w = spr.shape[:2]
    s = scale * cam.s
    p = cam.fwd([(x, y)])[0]
    tot = ang + cam.rot
    R = cv2.getRotationMatrix2D((w / 2, h / 2), -np.degrees(tot), s)
    R[0, 2] += p[0] - w / 2
    R[1, 2] += p[1] - h / 2
    r = int(np.ceil(max(w, h) * s * 0.75)) + 3
    x0, y0 = int(max(0, p[0] - r)), int(max(0, p[1] - r))
    x1, y1 = int(min(frame.shape[1], p[0] + r)), int(min(frame.shape[0], p[1] + r))
    if x1 <= x0 or y1 <= y0:
        return
    R[0, 2] -= x0
    R[1, 2] -= y0
    out = cv2.warpAffine(spr, R, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rgb = out[..., :3]
    if tint is not None:
        rgb = rgb * np.asarray(tint, np.float32)
    over(frame, (slice(y0, y1), slice(x0, x1)), rgb, out[..., 3] * alpha)


# ======================================================================= glass helpers
def glass_overlay(frame, m, cam, tint=(0.92, 0.97, 0.95), strength=1.0, stripes=((-0.55, 0.07, 0.45), (0.6, 0.04, 0.3)), axis_ang=0.0):
    a = m.f()
    if a is None:
        return
    d = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    s = cam.s
    mul(frame, m.sl, tint, a * 0.55 * strength)
    over(frame, m.sl, (0.02, 0.025, 0.02), a * np.exp(-((d - 1.3) / (1.0 + 0.4 * s)) ** 2) * 0.6 * strength)
    add(frame, m.sl, (1, 1, 0.95), a * np.exp(-((d - 4.0 * s) / (1.2 * s)) ** 2) * 0.28 * strength)
    add(frame, m.sl, (0.12, 0.13, 0.12), a * 0.25 * strength)
    return a, d


def tube_shade(frame, m, U, cx, r, cam, stripes=True, strength=1.0):
    """Vertical glass tube look (two edge lines + specular stripes) for a mask in a tube."""
    a = m.f()
    xr = (U - cx) / r
    mul(frame, m.sl, (0.9, 0.96, 0.93), a * 0.45 * strength)
    over(frame, m.sl, (0.02, 0.025, 0.02), a * np.exp(-((np.abs(xr) - 0.9) / 0.09) ** 2) * 0.7 * strength)
    if stripes:
        add(frame, m.sl, (1, 1, 0.96), a * (np.exp(-((xr + 0.5) / 0.12) ** 2) * 0.55 + np.exp(-((xr - 0.62) / 0.06) ** 2) * 0.35) * strength)


def liquid_fill(frame, m, color, glow=0.08):
    a = m.f()
    if a is None:
        return
    col = np.asarray(color, np.float32)
    mul(frame, m.sl, col, a * 0.92)
    add(frame, m.sl, col * glow, a)


# ======================================================================= gloves
def _rr(c, hx, hy, r, ang):
    return rot_pts(rrect_pts(c[0] - hx, c[1] - hy, c[0] + hx, c[1] + hy, r), ang, c)


def glove(frame, cam, anchor, ang=0.0, kind='pinch', scale=1.0, alpha=1.0):
    """Black nitrile glove; parts in hand-local coords (anchor = fingertip contact)."""
    if kind == 'pinch':          # thumb + index pinching something, hand above-right
        arm = ([(250, -400), (720, -1150)], 230)
        palm = _rr((165, -290), 118, 132, 62, -0.55)
        fingers = [([(40, -190), (-22, -112), (-14, -34), (-4, 0)], 50),
                   ([(250, -170), (150, -62), (62, -12), (16, 6)], 58),
                   ([(80, -215), (12, -152), (2, -100)], 50),
                   ([(122, -232), (66, -176), (56, -134)], 46),
                   ([(165, -244), (124, -200), (116, -170)], 40)]
    elif kind == 'press':        # index finger pointing down, hand above
        arm = ([(120, -420), (380, -1150)], 230)
        palm = _rr((60, -300), 116, 120, 60, 0.2)
        fingers = [([(0, -210), (0, -100), (0, 0)], 52),
                   ([(62, -205), (78, -135), (56, -102)], 52),
                   ([(114, -212), (122, -152), (100, -126)], 47),
                   ([(-62, -226), (-86, -152), (-52, -112)], 58)]
    elif kind == 'grip':         # fingers wrapped horizontally around an object, hand from the right
        arm = ([(380, 30), (1100, 110)], 250)
        palm = _rr((255, 30), 112, 150, 70, 0.05)
        fingers = [([(200, -110), (90, -106), (8, -92)], 56),
                   ([(200, -38), (80, -34), (-6, -22)], 59),
                   ([(200, 34), (84, 40), (0, 54)], 57),
                   ([(200, 102), (100, 110), (24, 124)], 50),
                   ([(220, 175), (130, 205), (50, 214)], 60)]
    else:
        raise ValueError(kind)
    c = np.array(anchor, np.float64)
    T = lambda pts: rot_pts(np.array(pts, np.float64) * scale + c, ang, c)
    allpts = np.vstack([T(palm)] + [T(f) for f, _ in fingers] + [T(arm[0])])
    m = Mask(cam, allpts, pad=int(140 * scale))
    if not m.ok:
        return
    comps = []
    for kind_, pts, wdt in [('poly', palm, 0)] + [('line', f, w_) for f, w_ in fingers] + [('line', arm[0], arm[1])]:
        fm = Mask(cam, allpts, pad=int(140 * scale))
        if kind_ == 'poly':
            fm.poly(T(pts))
        else:
            fm.line(T(pts), wdt * scale)
        comps.append(fm.f())
    a = np.clip(np.maximum.reduce(comps), 0, 1)
    rad = 24 * scale * cam.s
    n = blob_normals(a, rad, comps)
    col = shade_material(n, **NITRILE)
    # fine nitrile texture + darker creases where components meet
    over(frame, m.sl, col, a * alpha)


# ======================================================================= tools
def pipette(frame, cam, tip, top_y=-200, r=11.0, fill_col=None, fill_from=None, fill_to=None):
    """Straight thin glass pipette ending at `tip` (design), extending up off frame."""
    pts = np.array([(tip[0] - r, top_y), (tip[0] + r, top_y), (tip[0] + r, tip[1] - 30), (tip[0] + 4, tip[1]), (tip[0] - 4, tip[1]), (tip[0] - r, tip[1] - 30)])
    m = Mask(cam, pts).poly(pts)
    if not m.ok:
        return
    U, V = m.design_grid()
    if fill_col is not None:
        f0 = fill_from if fill_from is not None else tip[1]
        f1 = fill_to if fill_to is not None else top_y
        lm = Mask(cam, pts).poly(pts)
        a = lm.f() * smoothstep(f1 - 1, f1 + 1, V) * smoothstep(f0 + 1, f0 - 1, V)
        col = np.asarray(fill_col, np.float32)
        mul(frame, lm.sl, col, a * 0.9)
        add(frame, lm.sl, col * 0.10, a)
        over(frame, lm.sl, (1, 1, 0.9), np.exp(-((V - f1) / 1.5) ** 2) * lm.f() * 0.5 * (f1 > top_y + 2))
    tube_shade(frame, m, U, tip[0], r, cam)


def dropper(frame, cam, tip, color=None, fill=0.6, squeeze=0.0, glove_on=True, length=330):
    x, y = tip
    r = 13.0
    top = y - length
    pts = np.array([(x - r, top), (x + r, top), (x + r, y - 48), (x + 4, y), (x - 4, y), (x - r, y - 48)])
    m = Mask(cam, pts).poly(pts)
    if m.ok:
        U, V = m.design_grid()
        if color is not None and fill > 0:
            f1 = y - fill * (length - 10)
            a = m.f() * smoothstep(f1 - 1, f1 + 1, V)
            col = np.asarray(color, np.float32)
            mul(frame, m.sl, col, a * 0.95)
            add(frame, m.sl, col * 0.12, a)
            over(frame, m.sl, (1, 1, 0.92), np.exp(-((V - f1) / 1.6) ** 2) * m.f() * 0.55)
        tube_shade(frame, m, U, x, r, cam)
    # threaded black cap / collar + rubber bulb
    sq = 1.0 - 0.18 * squeeze
    cap = rrect_pts(x - 30, top - 34, x + 30, top + 6, 5)
    bulb = ellipse_pts((x, top - 122), 27 * sq, 96)
    for pts_, rad, base in [(bulb, 20, (0.02, 0.02, 0.022)), (cap, 12, (0.025, 0.025, 0.028))]:
        mm = Mask(cam, pts_, pad=10).poly(pts_)
        if mm.ok:
            a = mm.f()
            n = blob_normals(a, rad * cam.s)
            col = shade_material(n, base=base, spec=0.45, shin=30, rim=0.35)
            over(frame, mm.sl, col, a)
    if glove_on:
        glove(frame, cam, (x + 10, top - 120), ang=0.15, kind='pinch', scale=0.9)


def funnel(frame, cam, content_col=None, fill=0.5, texture=None, t=0.0, y_top=300.0, half_top=235.0):
    y_neck_top, y_cone_end = 862.0, 745.0
    cone = np.array([(CX - half_top, y_top), (CX + half_top, y_top), (CX + 16, y_cone_end), (CX + 10, y_cone_end + 20),
                     (CX + 9, 955), (CX - 9, 955), (CX - 10, y_cone_end + 20), (CX - 16, y_cone_end)])
    m = Mask(cam, cone).poly(cone)
    if not m.ok:
        return
    U, V = m.design_grid()
    # contents
    if content_col is not None and fill > 0:
        yl = y_cone_end - fill * (y_cone_end - y_top - 20)
        a = m.f() * smoothstep(yl - 1, yl + 1, V) * (V < y_cone_end + 190)
        col = np.asarray(content_col, np.float32)
        mul(frame, m.sl, col, a * 0.95)
        add(frame, m.sl, col * 0.10, a)
        if texture is not None:
            tx = cv2.resize(texture, (a.shape[1], a.shape[0]))
            over(frame, m.sl, tx[..., :3], tx[..., 3] * a * smoothstep(yl, yl + 25, V))
        # surface ellipse inside the cone
        half = 16 + (half_top - 16) * (y_cone_end - yl) / (y_cone_end - y_top)
        sm = Mask(cam, ellipse_pts((CX, yl), half, 8)).ellipse((CX, yl), half * 0.97, 6)
        if sm.ok:
            over(frame, sm.sl, col * 0.7 + 0.3, sm.f() * 0.6)
    glass_overlay(frame, m, cam, strength=0.9)
    # rolled rim at the top
    rim = Mask(cam, ellipse_pts((CX, y_top), half_top + 6, 16)).ellipse((CX, y_top), half_top + 4, 13)
    if rim.ok:
        a = rim.f()
        Ur, Vr = rim.design_grid()
        rr = np.sqrt(((Ur - CX) / (half_top + 4)) ** 2 + ((Vr - y_top) / 13) ** 2)
        add(frame, rim.sl, (1, 1, 0.95), a * np.exp(-((rr - 0.9) / 0.06) ** 2) * 0.6)
        over(frame, rim.sl, (0.02, 0.03, 0.02), a * np.exp(-((rr - 0.75) / 0.06) ** 2) * 0.35)
    # cone specular stripes
    xr = (U - CX) / np.maximum(16 + (half_top - 16) * np.clip((y_cone_end - V) / (y_cone_end - y_top), 0, 1), 9)
    add(frame, m.sl, (1, 1, 0.95), m.f() * (np.exp(-((xr + 0.6) / 0.06) ** 2) * 0.45 + np.exp(-((xr - 0.7) / 0.04) ** 2) * 0.3) * (V < y_cone_end))


def syringe(frame, cam, plunger=0.0, tint=(0.97, 0.98, 1.0)):
    bx0, bx1, by0, by1 = CX - 62, CX + 62, 250.0, 780.0
    barrel = rrect_pts(bx0, by0, bx1, by1, 8)
    m = Mask(cam, barrel).poly(barrel)
    yp = by0 + 30 + plunger * (by1 - by0 - 70)
    if m.ok:
        U, V = m.design_grid()
        a = m.f()
        liq = a * smoothstep(yp + 30, yp + 32, V)
        mul(frame, m.sl, tint, liq * 0.8)
        add(frame, m.sl, (0.08, 0.08, 0.08), a * 0.35)
        tube_shade(frame, m, U, CX, 62, cam)
        # graduations
        for k in range(14):
            y = by1 - 30 - k * 30
            g = np.exp(-((V - y) / 1.0) ** 2) * smoothstep(CX - 54, CX - 52, U) * smoothstep(CX - (18 if k % 2 else -8), CX - (20 if k % 2 else -6), U)
            over(frame, m.sl, (0.08, 0.08, 0.08), g * a * 0.75)
    # luer tip
    tip = np.array([(CX - 12, by1), (CX + 12, by1), (CX + 7, by1 + 40), (CX + 5, 905), (CX - 5, 905), (CX - 7, by1 + 40)])
    tm = Mask(cam, tip).poly(tip)
    if tm.ok:
        U, V = tm.design_grid()
        tube_shade(frame, tm, U, CX, 12, cam)
    # flange
    fl = rrect_pts(CX - 105, by0 - 16, CX + 105, by0 + 6, 7)
    fm = Mask(cam, fl).poly(fl)
    if fm.ok:
        add(frame, fm.sl, (0.6, 0.6, 0.6), fm.f() * 0.5)
        glass_overlay(frame, fm, cam, strength=0.7)
    # black rubber stopper + rod
    rod = np.array([(CX - 16, -300), (CX + 16, -300), (CX + 16, yp), (CX - 16, yp)])
    rm = Mask(cam, rod).poly(rod)
    if rm.ok:
        add(frame, rm.sl, (0.5, 0.5, 0.5), rm.f() * 0.35)
        glass_overlay(frame, rm, cam, strength=0.6)
    st = rrect_pts(CX - 60, yp, CX + 60, yp + 36, 8)
    sm = Mask(cam, st).poly(st)
    if sm.ok:
        a = sm.f()
        n = blob_normals(a, 8 * cam.s)
        over(frame, sm.sl, shade_material(n, base=(0.03, 0.03, 0.03), spec=0.5, shin=40, rim=0.3), a)


def jar(frame, cam, mouth, ang, fill_col=(0.55, 0.52, 0.3)):
    """Small amber glass jar, mouth at `mouth`, body extending up-right, rotated by ang."""
    c = np.array(mouth, np.float64)
    body = np.array([(0, -40), (36, -40), (52, -78), (240, -78), (240, 78), (52, 78), (36, 40), (0, 40)], np.float64)
    body = rot_pts(body + c, ang, c)
    m = Mask(cam, body).poly(body)
    if not m.ok:
        return
    a = m.f()
    mul(frame, m.sl, (0.55, 0.30, 0.10), a * 0.85)
    glass_overlay(frame, m, cam, tint=(0.8, 0.6, 0.4), strength=0.9)
    glove(frame, cam, c + rot_pts([(150, -20)], ang, (0, 0))[0], ang=ang, kind='grip', scale=0.62)


def tweezers(frame, cam, tip, open_=0.0, pod=None, sprites=None):
    x, y = tip
    gap = 4 + open_ * 22
    base = np.array([x + 320, y - 300])
    for sgn in (-1, 1):
        p = [(x + sgn * gap * 0.5, y + sgn * gap * 0.3), (x + 40 + sgn * gap * 0.8, y - 40), base + [sgn * 16, sgn * 6]]
        poly = thick_path(np.array(p), [5, 9, 14])
        m = Mask(cam, poly).poly(poly)
        if m.ok:
            U, V = m.design_grid()
            col = cylinder_metal(U * 0.7 + V * 0.7, (x + y) * 0.7, 30)
            over(frame, m.sl, col * 0.9, m.f())
    if pod is not None and sprites is not None and open_ < 0.3:
        draw_sprite(frame, cam, sprites['pod'][pod], x - 6, y + 10, 0.9)
    glove(frame, cam, base + [20, -40], ang=-0.6, kind='pinch', scale=0.8)


def beaker(frame, cam, lip, ang, fill=0.5):
    """Glass beaker tilted around its pouring lip (lip at design `lip`)."""
    c = np.array(lip, np.float64)
    w, h = 260, 330
    body = np.array([(-10, 0), (w, 0), (w, h), (0, h)], np.float64)
    body = rot_pts(body + c, ang, c)
    m = Mask(cam, body).poly(body)
    if not m.ok:
        return
    U, V = m.design_grid()
    # liquid inside: horizontal surface in world space
    lv = c[1] - 30 - fill * 170
    a = m.f() * smoothstep(lv - 1, lv + 1, V)
    mul(frame, m.sl, (0.96, 0.98, 0.97), a * 0.6)
    add(frame, m.sl, (0.10, 0.10, 0.09), a * 0.35)
    over(frame, m.sl, (1, 1, 0.95), m.f() * np.exp(-((V - lv) / 1.5) ** 2) * 0.5)
    glass_overlay(frame, m, cam, strength=1.0)
    glove(frame, cam, c + rot_pts([(w * 0.62, h * 0.55)], ang, (0, 0))[0] + [30, 0], ang=0.15, kind='grip', scale=0.75)


def crimper(frame, cam, bottom_y):
    body = rrect_pts(CX - 100, bottom_y - 360, CX + 100, bottom_y, 26)
    m = Mask(cam, body).poly(body)
    if not m.ok:
        return
    U, V = m.design_grid()
    a = m.f()
    nx = np.clip((U - CX) / 100, -1, 1)
    lum = 0.62 + 0.30 * np.cos(nx * 1.4 - 0.5) - 0.18 * smoothstep(0.85, 1.0, np.abs(nx))
    col = np.stack([lum * 0.97, lum * 0.97, lum * 0.95], -1)
    col *= (1 - 0.2 * np.exp(-((V - bottom_y + 40) / 2) ** 2))[..., None]
    over(frame, m.sl, col, a)
    glove(frame, cam, (CX + 120, bottom_y - 210), ang=0.05, kind='grip', scale=0.75)


# ======================================================================= fluids in air
def stream(frame, cam, pts, width, color=(0.97, 0.98, 1.0), alpha=1.0, t=0.0):
    """Falling liquid stream: refractive core + bright edges, coloured by its liquid."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or width <= 0:
        return
    poly = thick_path(pts, width)
    m = Mask(cam, poly).poly(poly)
    if not m.ok:
        return
    a = m.f() * alpha
    col = np.asarray(color, np.float32)
    mul(frame, m.sl, col, a * 0.85)
    add(frame, m.sl, col * 0.12 + 0.03, a)
    d = cv2.distanceTransform((a > 0.3).astype(np.uint8), cv2.DIST_L2, 3)
    over(frame, m.sl, (0.05, 0.04, 0.02), a * np.exp(-((d - 1.0) / 0.8) ** 2) * 0.35)
    add(frame, m.sl, (1, 1, 0.96), a * np.exp(-((d - max(1.6, width * cam.s * 0.28)) / 0.9) ** 2) * 0.55)


def stream_path(x0, y0, y1, t, wobble=2.0, n=40, curve=0.0, seed=0):
    ys = np.linspace(y0, y1, n)
    k = (ys - y0) / max(1.0, (y1 - y0))
    xs = x0 + curve * k ** 2 + wobble * k * np.sin(ys * 0.045 - t * 22 + seed) + wobble * 0.5 * k * np.sin(ys * 0.11 + t * 31 + seed * 2)
    return np.stack([xs, ys], 1)


def smoke_wisps(frame, cam, t, origin=(CX, 860), strength=1.0, seed=3):
    rng = np.random.default_rng(seed)
    layer_pts = []
    m = Mask(cam, [(origin[0] - 300, origin[1] - 700), (origin[0] + 300, origin[1] + 10)])
    if not m.ok:
        return
    for k in range(7):
        ph = rng.uniform(0, 6)
        ys = np.linspace(origin[1], origin[1] - rng.uniform(380, 640), 70)
        kk = (origin[1] - ys) / 600.0
        xs = origin[0] + rng.uniform(-14, 14) + kk * 140 * np.sin(kk * 3.2 + t * 1.3 + ph) + kk ** 2 * rng.uniform(-150, 150) \
             + 18 * kk * np.sin(ys * 0.04 - t * 2.4 + ph)
        layer_pts.append((np.stack([xs, ys], 1), rng.uniform(4, 10)))
    for pts, w in layer_pts:
        m.line(pts, w, val=90)
    a = cv2.GaussianBlur(m.f(), (0, 0), 7 * cam.s) * 0.6 + cv2.GaussianBlur(m.f(), (0, 0), 2.0 * cam.s) * 0.25
    add(frame, m.sl, (0.85, 0.85, 0.82), a * 0.9 * strength)


def mist(frame, cam, origin, age, seed=11, n=2600, sun=1.0):
    """Atomizer spray: fine droplets bursting toward the camera from the nozzle."""
    if age <= 0:
        return
    rng = np.random.default_rng(seed)
    birth = rng.uniform(0, 0.35, n)
    a_ = age - birth
    ok = a_ > 0
    a_ = a_[ok]
    ang = rng.normal(0.0, 0.9, n)[ok] + np.pi * 0.0
    dirx = np.cos(rng.uniform(0, 2 * np.pi, n))[ok]
    diry = np.sin(rng.uniform(0, 2 * np.pi, n))[ok]
    spd = rng.uniform(200, 900, n)[ok]
    dist = spd * (1 - np.exp(-a_ * 2.2)) / 2.2
    x = origin[0] + dirx * dist * 0.9 + 90 * a_
    y = origin[1] + diry * dist * 0.55 - 60 * a_ ** 2
    r = (1.0 + 4.5 * a_ * rng.uniform(0.5, 1.5, n)[ok])
    life = np.clip(1 - a_ / 1.6, 0, 1)
    p = cam.fwd(np.stack([x, y], 1))
    layer = np.zeros(frame.shape[:2], np.float32)
    for (px, py), rr, l in zip(p, r * cam.s, life):
        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(layer, (int(px * 4), int(py * 4)), max(1, int(rr * 4)), float(l * 0.35), -1, cv2.LINE_AA, shift=2)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.2) + cv2.GaussianBlur(layer, (0, 0), 14) * 0.5
    frame += layer[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * sun
