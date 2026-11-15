"""Low-level 2D graphics helpers: camera, anti-aliased vector masks, compositing, shading.

All geometry is authored in *design space* (pixels of the locked 1080x1920 camera).
A Cam maps design space to screen space (zoom / roll / translation), so the same
scene description renders the locked-off shots and the hand-held shots.
"""
import numpy as np
import cv2

W, H = 1080, 1920


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_io(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def lerp(a, b, t):
    return a + (b - a) * t


class Cam:
    """screen = s * R(rot) * (p - pivot) + center + shake"""

    def __init__(self, s=1.0, rot=0.0, pivot=(540.0, 960.0), center=(540.0, 960.0)):
        self.s, self.rot = float(s), float(rot)
        self.pivot = np.array(pivot, np.float64)
        self.center = np.array(center, np.float64)

    @property
    def M(self):
        c, s_ = np.cos(self.rot), np.sin(self.rot)
        A = self.s * np.array([[c, -s_], [s_, c]])
        t = self.center - A @ self.pivot
        return np.hstack([A, t[:, None]])

    @property
    def Minv(self):
        return cv2.invertAffineTransform(self.M)

    def fwd(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        M = self.M
        return pts @ M[:, :2].T + M[:, 2]

    def inv_grid(self, x0=0, y0=0, w=W, h=H):
        """design-space coords (U,V) for every screen pixel of a sub-rectangle."""
        Mi = self.Minv
        xs = np.arange(x0, x0 + w, dtype=np.float32)
        ys = np.arange(y0, y0 + h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        U = Mi[0, 0] * X + Mi[0, 1] * Y + Mi[0, 2]
        V = Mi[1, 0] * X + Mi[1, 1] * Y + Mi[1, 2]
        return U.astype(np.float32), V.astype(np.float32)

    def copy(self):
        return Cam(self.s, self.rot, tuple(self.pivot), tuple(self.center))


# ---------------------------------------------------------------- vector masks
class Mask:
    """Anti-aliased mask rasterised in a screen-space bounding box."""

    def __init__(self, cam, design_pts_for_bbox, pad=6, shape=(H, W)):
        p = cam.fwd(design_pts_for_bbox)
        self.cam = cam
        self.x0 = int(max(0, np.floor(p[:, 0].min()) - pad))
        self.y0 = int(max(0, np.floor(p[:, 1].min()) - pad))
        self.x1 = int(min(shape[1], np.ceil(p[:, 0].max()) + pad))
        self.y1 = int(min(shape[0], np.ceil(p[:, 1].max()) + pad))
        self.ok = self.x1 > self.x0 + 1 and self.y1 > self.y0 + 1
        if self.ok:
            self.m = np.zeros((self.y1 - self.y0, self.x1 - self.x0), np.uint8)

    def _sp(self, pts):
        p = self.cam.fwd(pts) - [self.x0, self.y0]
        return np.round(p * 16).astype(np.int32)

    def poly(self, pts, val=255):
        if self.ok:
            cv2.fillPoly(self.m, [self._sp(pts)], val, cv2.LINE_AA, shift=4)
        return self

    def line(self, pts, thick, val=255):
        if self.ok:
            t = max(1, int(round(thick * self.cam.s)))
            cv2.polylines(self.m, [self._sp(pts)], False, val, t, cv2.LINE_AA, shift=4)
        return self

    def circle(self, c, r, val=255):
        if self.ok:
            p = self._sp([c])[0]
            cv2.circle(self.m, (int(p[0]), int(p[1])), int(round(r * self.cam.s * 16)), val, -1, cv2.LINE_AA, shift=4)
        return self

    def ellipse(self, c, ax, ay, ang=0.0, val=255):
        pts = ellipse_pts(c, ax, ay, ang)
        return self.poly(pts, val)

    def f(self):
        return self.m.astype(np.float32) / 255.0 if self.ok else None

    @property
    def sl(self):
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def design_grid(self):
        return self.cam.inv_grid(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)


def ellipse_pts(c, ax, ay, ang=0.0, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = ax * np.cos(a), ay * np.sin(a)
    ca, sa = np.cos(ang), np.sin(ang)
    return np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], 1)


def rrect_pts(x0, y0, x1, y1, r, n=10):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]:
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))))
    return np.array(pts)


def thick_path(pts, widths):
    """Polygon outline of a variable-width stroke along a polyline (no caps)."""
    pts = np.asarray(pts, np.float64)
    widths = np.broadcast_to(np.asarray(widths, np.float64), (len(pts),))
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    left = pts + nrm * widths[:, None] / 2
    right = pts - nrm * widths[:, None] / 2
    return np.vstack([left, right[::-1]])


def rot_pts(pts, ang, c):
    pts = np.asarray(pts, np.float64) - c
    ca, sa = np.cos(ang), np.sin(ang)
    return np.stack([pts[:, 0] * ca - pts[:, 1] * sa, pts[:, 0] * sa + pts[:, 1] * ca], 1) + c


# ---------------------------------------------------------------- compositing
def over(frame, sl, rgb, a):
    """Alpha-over onto frame[sl]. rgb: (h,w,3) or (3,), a: (h,w)."""
    if a is None:
        return
    a3 = a[..., None]
    region = frame[sl]
    region *= (1.0 - a3)
    region += np.asarray(rgb, np.float32) * a3


def add(frame, sl, rgb, a):
    frame[sl] += np.asarray(rgb, np.float32) * a[..., None]


def mul(frame, sl, rgb, a):
    a3 = a[..., None]
    frame[sl] *= (1 - a3) + np.asarray(rgb, np.float32) * a3


# ---------------------------------------------------------------- shading
KEY = np.array([-0.45, -0.65, 0.62]); KEY /= np.linalg.norm(KEY)
SUN = np.array([0.25, -0.55, -0.80]); SUN /= np.linalg.norm(SUN)


def blob_normals(mask, radius_px, comps=None):
    """Pseudo-3D normals for a soft 2D shape via its distance field ("pillow" shading)."""
    hs = []
    for m in (comps if comps is not None else [mask]):
        d = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(d / max(radius_px, 1.0), 0, 1)
        hs.append(np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1)) * radius_px)
    h = np.maximum.reduce(hs) if len(hs) > 1 else hs[0]
    h = cv2.GaussianBlur(h, (0, 0), max(0.8, radius_px * 0.06))
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def shade_material(n, base, spec=0.4, shin=24, rim=0.3, rim_col=(1.0, 0.85, 0.6), amb=1.0, key=KEY):
    nl = np.clip((n @ key), 0, 1)
    rz = 2 * nl * n[..., 2] - key[2]
    sp = np.clip(rz, 0, 1) ** shin
    rimf = np.clip(1 - n[..., 2], 0, 1) ** 1.6 * np.clip(0.4 - n[..., 1], 0, 1.4)
    col = np.asarray(base, np.float32) * (0.35 * amb + 0.75 * nl[..., None])
    col = col + spec * sp[..., None] + rim * rimf[..., None] * np.asarray(rim_col, np.float32)
    return col.astype(np.float32)


def chrome_env(rx, ry, env_mid=(0.30, 0.38, 0.16)):
    """Environment colour seen by a chrome reflection direction (garden at golden hour)."""
    sky = np.array([1.0, 0.97, 0.86], np.float32)
    mid = np.array(env_mid, np.float32)
    ground = np.array([0.10, 0.06, 0.035], np.float32)
    up = smoothstep(-0.05, -0.55, ry)[..., None]
    dn = smoothstep(0.0, 0.5, ry)[..., None]
    c = mid * (1 - up - dn) + sky * up + ground * dn
    # studio-like strip reflections
    strip = np.exp(-((rx - 0.55) / 0.10) ** 2) * 0.9 + np.exp(-((rx + 0.35) / 0.05) ** 2) * 0.6
    c = c + strip[..., None] * np.array([1.0, 0.98, 0.92], np.float32)
    dark = np.exp(-((rx + 0.05) / 0.12) ** 2) * 0.55
    return c * (1 - dark[..., None])


def cylinder_metal(U, cx, r, base_tint=(0.95, 0.95, 0.97), env_mid=(0.30, 0.38, 0.16), ny=0.0):
    nx = np.clip((U - cx) / r, -0.999, 0.999)
    nz = np.sqrt(1 - nx ** 2)
    rx = 2 * nz * nx
    ry = np.full_like(rx, ny)
    c = chrome_env(rx, ry, env_mid) * np.asarray(base_tint, np.float32)
    edge = smoothstep(0.75, 1.0, np.abs(nx))[..., None]
    return c * (1 - 0.55 * edge)


def fbm(shape, seed, octaves=5, base=4, persistence=0.55):
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        f = base * 2 ** o
        g = rng.random((max(2, int(h * f / max(h, w)) + 2), max(2, int(w * f / max(h, w)) + 2))).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out
