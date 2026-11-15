"""Finishing: bloom, filmic shoulder, warm golden-hour grade, vignette, grain, motion blur
and NOX typography (olfactory-note captions + closing title)."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from gfx import W, H, smoothstep

_rng = np.random.default_rng(99)
_GRAIN = _rng.standard_normal((H // 2, W // 2)).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_VIG = 1.0 - 0.30 * smoothstep(0.45, 1.25, np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2))
del yy, xx


def tracked_text(text, font, tracking, fill=(255, 255, 255, 255)):
    widths = [font.getlength(c) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = font.getmetrics()
    im = Image.new('RGBA', (int(total) + 8, asc + desc + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 4
    for c, w in zip(text, widths):
        d.text((x, 4), c, font=font, fill=fill)
        x += w + tracking
    return np.asarray(im).astype(np.float32) / 255.0


class TextLayers:
    def __init__(self, fontdir):
        f = lambda n, s: ImageFont.truetype(os.path.join(fontdir, n), s)
        self.f_cap = f('montserrat-latin-500-normal.woff', 36)
        self.f_nox = f('cinzel-latin-600-normal.woff', 172)
        self.f_parf = f('montserrat-latin-300-normal.woff', 34)
        self.f_name = f('cormorant-garamond-latin-300-italic.woff', 78)
        self.cache = {}

    def caption(self, text):
        if text not in self.cache:
            self.cache[text] = tracked_text(text, self.f_cap, 13)
        return self.cache[text]

    def title_parts(self):
        if 'title' not in self.cache:
            self.cache['title'] = (tracked_text('NOX', self.f_nox, 40), tracked_text('PARFUMS', self.f_parf, 22),
                                   tracked_text('Ombre Nomade', self.f_name, 3))
        return self.cache['title']


def paste(frame, layer, cx, cy, alpha, color=(1.0, 0.96, 0.88), blur=0.0):
    h, w = layer.shape[:2]
    a = layer[..., 3] * alpha
    if blur > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    a = a[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
    reg = frame[ya:yb, xa:xb]
    reg *= (1 - a)
    reg += np.asarray(color, np.float32) * a


def directional_blur(frame, st):
    k = np.zeros((1, 31), np.float32)
    k[0, :] = 1 / 31
    b = cv2.filter2D(frame, -1, k)
    u = (st['tl']) / 0.9
    amt = float(np.sin(np.pi * np.clip(u, 0, 1)))
    return frame * (1 - amt) + b * amt


def finish(frame, st, text):
    f = np.maximum(frame, 0)
    # bloom
    hi = np.maximum(f - 0.78, 0)
    small = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (W, H))
    b2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 14), (W, H))
    f = f + (b1 * 0.35 + b2 * 0.30) * np.array([1.0, 0.92, 0.75], np.float32)
    # filmic shoulder
    f = np.where(f < 0.78, f, 0.78 + 0.22 * np.tanh((f - 0.78) / 0.22))
    # golden-hour grade
    f *= np.array([1.035, 1.0, 0.93], np.float32)
    lum = (f @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    f = lum + (f - lum) * 1.10
    f = f + 0.06 * (f - 0.5) * (1 - np.abs(2 * f - 1))
    f = 0.012 + f * 0.985
    f *= _VIG[..., None]
    # typography
    if st.get('caption'):
        txt, a = st['caption']
        if a > 0:
            lay = text.caption(txt)
            paste(f, lay, W / 2, 1772, a * 0.92)
            ln = np.zeros((3, 60, 4), np.float32); ln[..., 3] = 1
            paste(f, ln, W / 2, 1810, a * 0.55)
    if st.get('title', 0) > 0:
        tt = st['title']
        nox, parf, name = text.title_parts()
        top = smoothstep(560, 0, np.arange(H, dtype=np.float32))[:, None, None] * 0.5 * min(1, tt * 1.5)
        f *= (1 - top)
        a1 = smoothstep(0.0, 0.55, tt)
        a2 = smoothstep(0.25, 0.8, tt)
        a3 = smoothstep(0.45, 1.0, tt)
        paste(f, nox, W / 2, 250 + 18 * (1 - a1), a1, blur=(1 - a1) * 6)
        ln = np.zeros((2, int(320 * a2) + 1, 4), np.float32); ln[..., 3] = 1
        paste(f, ln, W / 2, 352, a2 * 0.7)
        paste(f, parf, W / 2, 392 + 10 * (1 - a2), a2)
        paste(f, name, W / 2, 486 + 10 * (1 - a3), a3, color=(1.0, 0.90, 0.70), blur=(1 - a3) * 4)
    # grain
    g = cv2.resize(np.roll(_GRAIN, (st['fi'] * 37) % (H // 2), 0), (W, H), interpolation=cv2.INTER_LINEAR)
    f = f + g[..., None] * 0.011
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
