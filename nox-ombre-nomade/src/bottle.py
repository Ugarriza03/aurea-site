"""NOX Ombre Nomade bottle, reconstructed from the product photographs.

Measured from assets/product/ombre_nomade_front.jpg (front view, 630 px glass width
-> 960 design px, scale 1.524):
  * flat rectangular flacon, width : front height ~ 1.7 : 1, small corner radius,
    thick side walls and a heavy glass base;
  * low circular glass boss around the neck;
  * silver atomizer: flared ferrule collar (221 px), short stem (160 px),
    cylindrical actuator (183 px, 168 px tall) with the white nozzle dot on the front;
  * clear dip tube curving to the lower-left corner;
  * matte black presentation tray (990 x 297 px) hiding the lower ~27 % of the glass.
"""
import numpy as np
import cv2
from gfx import (Cam, Mask, smoothstep, rrect_pts, ellipse_pts, cylinder_metal, over, add, mul, thick_path,
                 blob_normals, shade_material, W, H)
import background as bgm

CX = 540.0
X0, X1 = 60.0, 1020.0
YT, YB = 964.0, 1530.0
TOPF = 26.0
RW, RB, RT = 30.0, 72.0, 30.0
IX0, IX1 = X0 + RW, X1 - RW
IY0, IY1 = YT + RT, YB - RB
R_OUT, R_IN = 30.0, 16.0
SURF_BAND = 20.0
# fluid grid
CELL = 4.0
GX, GY = int((IX1 - IX0) / CELL), int((IY1 - IY0) / CELL)

# atomizer (design px)
PUMP = dict(collar_y0=940.0, collar_y1=812.0, collar_w=200.0, flare_w=222.0,
            stem_y1=774.0, stem_w=158.0, act_y1=606.0, act_w=184.0,
            dot=(CX + 24.0, 676.0), dot_r=14.0)
TRAY = dict(x0=45.0, x1=1035.0, y0=1376.0, y1=1673.0, top=16.0)
DIP = np.array([(CX, 990), (CX + 2, 1080), (CX - 18, 1200), (CX - 90, 1320), (CX - 210, 1420), (CX - 330, 1452)], np.float64)


def level_y(level):
    return IY1 - level * (IY1 - IY0)


def sd_rrect(U, V, x0, y0, x1, y1, r):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx = np.abs(U - cx) - hx
    qy = np.abs(V - cy) - hy
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r


class BottleRenderer:
    def __init__(self, textures, sprites):
        self.tex = textures          # dict name -> (sharp, soft)
        self.sprites = sprites
        n = np.random.default_rng(3).random((64, 64)).astype(np.float32)
        n = cv2.resize(n, (512, 512), interpolation=cv2.INTER_CUBIC)
        n = cv2.GaussianBlur(n, (0, 0), 6)
        n = (n - n.min()) / (n.max() - n.min())
        self.caustic = np.exp(-np.abs(np.sin(n * 18.0)) * 6.0).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def _sample(self, texname, cam, cam_bg, Su, Sv, soft=False):
        """Sample background texture at design coords (Su,Sv) seen through `cam`, bg parallax cam_bg."""
        M = cam.M
        Mb = cam_bg.Minv
        sx = M[0, 0] * Su + M[0, 1] * Sv + M[0, 2]
        sy = M[1, 0] * Su + M[1, 1] * Sv + M[1, 2]
        bu = Mb[0, 0] * sx + Mb[0, 1] * sy + Mb[0, 2] + bgm.OX
        bv = Mb[1, 0] * sx + Mb[1, 1] * sy + Mb[1, 2] + bgm.OY
        t = self.tex[texname][1 if soft else 0]
        return cv2.remap(t, bu.astype(np.float32), bv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------ table shadow / caustic
    def table_light(self, frame, st):
        cam = st['cam']
        floor_y = TRAY['y1'] if st.get('tray') else YB
        lvl = st['level']
        T = np.exp(-np.asarray(st['A'], np.float32))
        m = Mask(cam, [(X0 - 140, floor_y - 30), (X1 + 140, floor_y - 30), (X1 + 260, floor_y + 360), (X0 - 260, floor_y + 360)])
        if not m.ok:
            return
        U, V = m.design_grid()
        dy = V - floor_y
        spread = 1 + np.clip(dy, 0, None) / 900.0
        xr = (U - CX) / (spread * (X1 - X0) / 2)
        inside = smoothstep(1.08, 0.92, np.abs(xr)) * smoothstep(-6, 4, dy) * smoothstep(340, 120, dy)
        contact = np.exp(-np.clip(dy, 0, None) / 10.0) * smoothstep(1.03, 0.95, np.abs((U - CX) / ((X1 - X0) / 2))) * smoothstep(-4, 1, dy)
        shadow = np.clip(inside * 0.55 + contact * 0.5, 0, 0.85)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 6 * cam.s)
        mul(frame, m.sl, (0.0, 0.0, 0.0), shadow)
        if lvl > 0.02 and not st.get('tray'):
            # caustic: sun focused through the liquid column, coloured by the liquid
            liq = smoothstep(lvl * 300 + 20, lvl * 300 - 20, np.clip(dy, 0, None))
            ct = st['t']
            c1 = cv2.remap(self.caustic, ((U * 0.9 + ct * 23) % 511).astype(np.float32), ((V * 1.6 - ct * 11) % 511).astype(np.float32), cv2.INTER_LINEAR)
            c2 = cv2.remap(self.caustic, ((U * 1.1 - ct * 17 + 200) % 511).astype(np.float32), ((V * 1.4 + ct * 9 + 90) % 511).astype(np.float32), cv2.INTER_LINEAR)
            cz = np.minimum(c1, c2) * 1.6
            caust = (0.35 + cz) * liq * inside * smoothstep(1.0, 0.55, np.abs(xr))
            caust = cv2.GaussianBlur(caust, (0, 0), 3)
            col = np.clip(T * 1.25, 0, 1.2) * np.array([1.0, 0.85, 0.55], np.float32)
            add(frame, m.sl, col * 0.40 * st.get('sun', 1.0), caust)

    # ------------------------------------------------------------------ main body
    def render(self, frame, st, draw_inside):
        cam, cam_bg = st['cam'], st['cam_bg']
        texname = st['bg']
        bbox = [(X0 - 8, YT - TOPF - 70), (X1 + 8, YT - TOPF - 70), (X1 + 8, YB + 6), (X0 - 8, YB + 6)]
        m = Mask(cam, bbox, pad=4)
        if not m.ok:
            return
        sl = m.sl
        U, V = m.design_grid()
        d_out = sd_rrect(U, V, X0, YT, X1, YB, R_OUT)
        d_in = sd_rrect(U, V, IX0, IY0, IX1, IY1, R_IN)
        aa = 1.0 / cam.s
        body = smoothstep(aa, -aa, d_out)
        inner = smoothstep(aa, -aa, d_in)
        # top face sliver (bottle seen slightly from above)
        topface = smoothstep(aa, -aa, sd_rrect(U, V, X0 + 14, YT - TOPF, X1 - 14, YT + 12, 22)) * (V < YT)
        lvl = st['level']
        yl0 = st.get('yl_override') or level_y(lvl)
        tilt = st.get('tilt', 0.0)
        ripple = st.get('ripple', 0.0)
        ph = st['t'] * 7.0
        yl = yl0 + tilt * (U - CX) + ripple * (np.sin(U * 0.035 + ph) * 0.6 + np.sin(U * 0.083 - ph * 1.3) * 0.4)
        has_liq = lvl > 0.003
        liq = (smoothstep(yl - aa, yl + aa, V) * inner) if has_liq else np.zeros_like(U)
        # -------- refraction of the background
        # air part of the interior: slight magnification
        Su_air = CX + (U - CX) * 0.965
        Sv_air = 1000 + (V - 1000) * 0.975
        # side walls: compressed, mirrored view from outside the silhouette
        wall_u = np.clip(-d_out, 0, RW)
        side = (np.abs(U - CX) > (IX1 - CX)) & (body > 0)
        sgn = np.sign(U - CX)
        Su_wall = np.where(side, CX + sgn * ((X1 - CX) + (RW - wall_u) * 4.0), Su_air)
        img = self._sample(texname, cam, cam_bg, Su_wall, Sv_air)
        if has_liq:
            sch = st.get('schl_grid')
            dxs = dys = 0.0
            if sch is not None:
                sg = self._grid_sample(sch, U, V)
                dxs, dys = sg[..., 0], sg[..., 1]
            Su_l = CX + (U - CX) * 0.90 + dxs
            Sv_l = yl0 + (V - yl0) * 0.90 + 30 + dys
            img_l = self._sample(texname, cam, cam_bg, Su_l, Sv_l, soft=True)
            img = img * (1 - liq[..., None]) + img_l * liq[..., None]
        # top region: looking through the shoulder glass -> sky from above
        topw = smoothstep(IY0 + 2, IY0 - 2, V) * body
        img_top = self._sample(texname, cam, cam_bg, CX + (U - CX) * 0.9, V - 160, soft=True)
        tw = np.clip(topw + topface, 0, 1)
        img = img * (1 - tw[..., None]) + img_top * tw[..., None]
        glass_a = np.clip(body + topface, 0, 1)
        region = frame[sl]
        region[:] = region * (1 - glass_a[..., None]) + img * glass_a[..., None]
        # glass is very slightly green and looses some light
        mul(frame, sl, (0.93, 0.97, 0.94), glass_a * 0.9)

        # -------- interior objects (wood, particles, tools that enter the neck, streams)
        draw_inside(frame)

        # -------- liquid
        if has_liq:
            A = np.asarray(st['A'], np.float32)
            Aimg = st.get('dye_A')
            depth = np.clip((V - yl0) / (IY1 - IY0), 0, 1)[..., None]
            Atot = A[None, None, :] * (0.85 + 0.35 * depth)
            if Aimg is not None:
                Atot = Atot + self._grid_sample(Aimg, U, V)
            T = np.exp(-Atot)
            glow = T * (1 - T) * np.array([0.55, 0.42, 0.20], np.float32) * st.get('sun', 1.0)
            liq_rgb = frame[sl] * T * 1.12 + glow
            cloud = st.get('cloud_img')
            if cloud is not None:
                c = np.clip(self._grid_sample(cloud[..., None], U, V)[..., 0], 0, 1)[..., None]
                warm = np.exp(-A * 0.4) * 0.78
                liq_rgb = liq_rgb * (1 - c * 0.75) + warm * c * 0.75
            uc = st.get('cloud_u', 0.0)
            if uc > 0:
                liq_rgb = liq_rgb * (1 - uc) + np.exp(-A * 0.5) * 0.7 * uc
            region = frame[sl]
            region[:] = region * (1 - liq[..., None]) + liq_rgb * liq[..., None]
            # surface seen from slightly above + meniscus lines
            band = smoothstep(yl - SURF_BAND - aa, yl - SURF_BAND + 1.5, V) * smoothstep(yl + aa, yl - aa, V) * inner
            band_col = frame[sl] * (np.exp(-A) * 0.7)[None, None, :] + 0.22
            over(frame, sl, band_col, band * 0.55)
            over(frame, sl, (1.0, 0.98, 0.9), np.exp(-((V - yl + 0.5) / (1.4 * aa + 0.6)) ** 2) * inner * 0.75)
            over(frame, sl, (0.08, 0.05, 0.02), np.exp(-((V - yl - 3.5) / (1.6 * aa + 0.8)) ** 2) * inner * 0.35)
            over(frame, sl, (1.0, 0.98, 0.9), np.exp(-((V - yl + SURF_BAND) / (1.2 * aa + 0.6)) ** 2) * inner * 0.35)
        # smoke inside empty bottle
        smoke = st.get('smoke_img')
        if smoke is not None:
            s_ = np.clip(self._grid_sample(smoke[..., None], U, V)[..., 0], 0, 3)
            s_ = cv2.GaussianBlur(s_, (0, 0), 6 * cam.s)
            sa = (1 - np.exp(-s_ * 1.1)) * inner * (1 - liq)
            over(frame, sl, (0.92, 0.90, 0.86), sa * 0.6)

        # -------- thick glass: base and walls show mirrored (TIR) interior
        self._mirror_pass(frame, m, U, V, cam, body, d_out)

        # -------- glass overlays
        self._glass_overlays(frame, sl, U, V, d_out, d_in, body, inner, topface, aa, st)

    def _grid_sample(self, grid, U, V):
        gx = ((U - IX0) / CELL - 0.5).astype(np.float32)
        gy = ((V - IY0) / CELL - 0.5).astype(np.float32)
        g = grid if grid.ndim == 3 else grid[..., None]
        out = cv2.remap(np.ascontiguousarray(g.astype(np.float32)), gx, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if out.ndim == 2:
            out = out[..., None]
        return out

    def _mirror_pass(self, frame, m, U, V, cam, body, d_out):
        sl = m.sl
        # base: reflection of the interior floor compressed upwards
        base = smoothstep(IY1 - 2, IY1 + 2, V) * body
        Sv = IY1 - (V - IY1) * 2.6 - 6
        Su = CX + (U - CX) * 0.97
        side = (np.abs(U - CX) > (IX1 - CX - 1)) * body * (1 - base)
        # side walls: mirrored interior near the walls
        sgn = np.sign(U - CX)
        Su = np.where(side > 0, CX + sgn * ((IX1 - CX) - (np.abs(U - CX) - (IX1 - CX)) * 2.2 - 4), Su)
        Sv = np.where(side > 0, V, Sv)
        M = cam.M
        sx = (M[0, 0] * Su + M[0, 1] * Sv + M[0, 2]).astype(np.float32)
        sy = (M[1, 0] * Su + M[1, 1] * Sv + M[1, 2]).astype(np.float32)
        src = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wbase = base * 0.85
        wside = side * 0.6
        wt = np.clip(wbase + wside, 0, 1)
        region = frame[sl]
        region[:] = region * (1 - wt[..., None]) + src * wt[..., None]
        # glass body colour in the thick parts
        mul(frame, sl, (0.86, 0.95, 0.90), base * 0.5 + side * 0.25)
        mul(frame, sl, (0.55, 0.6, 0.55), side * smoothstep(0, RW, np.clip(-d_out, 0, RW)) * 0.5)

    def _glass_overlays(self, frame, sl, U, V, d_out, d_in, body, inner, topface, aa, st):
        a = aa * 1.2
        # silhouette: dark rim + bright inner rim (fresnel)
        over(frame, sl, (0.02, 0.025, 0.02), np.exp(-((d_out + 1.2) / (a + 0.7)) ** 2) * 0.65)
        over(frame, sl, (1.0, 1.0, 0.97), np.exp(-((d_out + 5.0) / (a + 1.4)) ** 2) * body * 0.45)
        add(frame, sl, (0.10, 0.11, 0.10), np.exp(np.clip(d_out, -60, 0) / 9.0) * body * 0.9)
        # inner boundary of the thick walls / base
        over(frame, sl, (0.02, 0.03, 0.02), np.exp(-((d_in - 1.0) / (a + 1.6)) ** 2) * body * 0.30)
        over(frame, sl, (1.0, 1.0, 0.95), np.exp(-((d_in + 2.5) / (a + 0.8)) ** 2) * 0.16)
        # vertical back edges seen through the front wall
        for x, al in [(IX0 + 20, 0.07), (IX1 - 20, 0.07)]:
            g = np.exp(-((U - x) / (1.0 + a)) ** 2) * smoothstep(IY0, IY0 + 30, V) * smoothstep(IY1, IY1 - 20, V)
            over(frame, sl, (0.95, 1.0, 0.95), g * al)
        g = np.exp(-((V - (IY1 - 16)) / (1.2 + a)) ** 2) * smoothstep(IX0 + 18, IX0 + 30, U) * smoothstep(IX1 - 18, IX1 - 30, U)
        over(frame, sl, (0.95, 1.0, 0.95), g * 0.07)
        # base highlights
        for y, al, col in [(IY1 + 3, 0.5, (1, 1, 0.95)), (YB - 4, 0.45, (1, 1, 0.95)), (IY1 + RB * 0.55, 0.22, (0.0, 0.03, 0.02))]:
            g = np.exp(-((V - y) / (1.5 + a)) ** 2) * body * smoothstep(X0 + 8, X0 + 40, U) * smoothstep(X1 - 8, X1 - 40, U)
            over(frame, sl, col, g * al)
        # vertical specular strips (window reflections) and soft sky reflection on the upper front
        shine = st.get('shine', 1.0)
        for x, w_, al in [(X0 + 62, 9.0, 0.22), (X0 + 86, 3.0, 0.30), (X1 - 74, 5.0, 0.32), (X1 - 52, 2.0, 0.25)]:
            g = np.exp(-((U - x) / (w_ + a)) ** 2) * smoothstep(YT, YT + 60, V) * smoothstep(YB - 20, YB - 140, V)
            add(frame, sl, (1.0, 0.99, 0.93), g * al * body * shine)
        sky = smoothstep(IY0 + 330, IY0, V) * inner * 0.10 * shine
        add(frame, sl, (0.95, 0.97, 1.0), sky)
        # shoulder: strong top front edge highlight + top face
        over(frame, sl, (1.0, 1.0, 0.96), np.exp(-((V - YT) / (1.3 + a)) ** 2) * smoothstep(X0, X0 + 20, U) * smoothstep(X1, X1 - 20, U) * 0.8)
        over(frame, sl, (0.02, 0.03, 0.02), np.exp(-((V - (YT + 5)) / (1.5 + a)) ** 2) * body * 0.3)
        over(frame, sl, (1.0, 1.0, 0.96), np.exp(-((V - (YT - TOPF + 1)) / (1.2 + a)) ** 2) * smoothstep(X0 + 8, X0 + 30, U) * smoothstep(X1 - 8, X1 - 30, U) * 0.55)
        over(frame, sl, (1.0, 1.0, 1.0), np.exp(-((V - IY0) / (1.5 + a)) ** 2) * inner * 0.12)
        add(frame, sl, (0.10, 0.11, 0.10), topface * 0.35)
        dap = st.get('dapple_img')
        if dap is not None:
            dp = dap[sl] if dap.shape[:2] == frame.shape[:2] else None
            if dp is not None:
                g = np.clip(body + topface, 0, 1) * dp
                add(frame, sl, (1.0, 0.9, 0.65), g * 0.22)
                mul(frame, sl, (0.55, 0.52, 0.48), np.clip(body + topface, 0, 1) * (1 - dp) * 0.6)

    # ------------------------------------------------------------------ neck / boss
    def neck(self, frame, st):
        cam = st['cam']
        # circular glass boss around the neck (seen almost edge-on)
        m = Mask(cam, ellipse_pts((CX, YT - TOPF + 2), 212, 22)).ellipse((CX, YT - TOPF + 2), 205, 18)
        if m.ok:
            a = m.f()
            U, V = m.design_grid()
            # ring highlights
            r = np.sqrt(((U - CX) / 205) ** 2 + ((V - (YT - TOPF + 2)) / 18) ** 2)
            over(frame, m.sl, (1, 1, 0.96), a * np.exp(-((r - 0.93) / 0.05) ** 2) * 0.55)
            over(frame, m.sl, (0.02, 0.03, 0.02), a * np.exp(-((r - 0.80) / 0.06) ** 2) * 0.3)
            add(frame, m.sl, (0.12, 0.13, 0.12), a * 0.5)
        if st.get('pump') is None:
            # bare crimp neck with bead
            pts = np.array([(CX - 46, YT - TOPF), (CX - 46, 884), (CX - 56, 880), (CX - 56, 862), (CX + 56, 862), (CX + 56, 880), (CX + 46, 884), (CX + 46, YT - TOPF)])
            m = Mask(cam, pts).poly(pts)
            if m.ok:
                a = m.f()
                U, V = m.design_grid()
                xr = (U - CX) / np.where(V < 884, 56, 46)
                mul(frame, m.sl, (0.88, 0.95, 0.9), a * 0.6)
                over(frame, m.sl, (0.02, 0.03, 0.02), a * np.exp(-((np.abs(xr) - 0.93) / 0.07) ** 2) * 0.65)
                add(frame, m.sl, (1, 1, 0.96), a * (np.exp(-((xr + 0.55) / 0.08) ** 2) * 0.45 + np.exp(-((xr - 0.62) / 0.05) ** 2) * 0.35))
                add(frame, m.sl, (1, 1, 0.96), a * np.exp(-((V - 863) / 2.0) ** 2) * 0.5)
                over(frame, m.sl, (0.05, 0.05, 0.04), a * np.exp(-((V - 884) / 1.8) ** 2) * 0.35)
                # opening
            m = Mask(cam, ellipse_pts((CX, 862), 40, 7)).ellipse((CX, 862), 34, 5)
            if m.ok:
                over(frame, m.sl, (0.05, 0.05, 0.04), m.f() * 0.35)

    def dip_tube(self, frame, st, offset_y=0.0):
        cam = st['cam']
        pts = DIP + [0, offset_y]
        poly = thick_path(pts, 10.0)
        m = Mask(cam, poly).poly(poly)
        if not m.ok:
            return
        a = m.f()
        U, V = m.design_grid()
        d = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        edge = np.exp(-((d - 1.2) / 1.0) ** 2)
        over(frame, m.sl, (0.05, 0.05, 0.04), a * edge * 0.45)
        add(frame, m.sl, (1, 1, 0.95), a * np.exp(-((d - 3.8 * cam.s) / 1.0) ** 2) * 0.22)

    def pump(self, frame, st, dy=0.0, act_dy=0.0, env_mid=(0.42, 0.45, 0.34)):
        cam = st['cam']
        P = PUMP
        parts = [
            # (x half width bottom, top, y0 (bottom), y1(top), radius for shading)
            ('flare', P['flare_w'] / 2, P['collar_w'] / 2, P['collar_y0'] + dy, P['collar_y0'] - 16 + dy),
            ('collar', P['collar_w'] / 2, P['collar_w'] / 2, P['collar_y0'] - 16 + dy, P['collar_y1'] + dy),
            ('ring', P['collar_w'] / 2 - 10, P['collar_w'] / 2 - 10, P['collar_y1'] + dy, P['collar_y1'] - 8 + dy),
            ('stem', P['stem_w'] / 2, P['stem_w'] / 2, P['collar_y1'] - 8 + dy, P['stem_y1'] + dy),
            ('act', P['act_w'] / 2, P['act_w'] / 2, P['stem_y1'] + dy + act_dy, P['act_y1'] + dy + act_dy),
        ]
        for name, hb, ht, y0, y1 in parts:
            pts = np.array([(CX - hb, y0), (CX + hb, y0), (CX + ht, y1), (CX - ht, y1)])
            if name == 'act':
                pts = rrect_pts(CX - hb, y1, CX + hb, y0, 7)
            m = Mask(cam, pts).poly(pts)
            if not m.ok:
                continue
            a = m.f()
            U, V = m.design_grid()
            r = np.maximum(hb, ht)
            col = cylinder_metal(U, CX, r, env_mid=env_mid)
            if name == 'stem':
                col *= 0.7
            if name in ('flare', 'ring'):
                col *= 0.8
            # horizontal grooves / rims
            col *= (1 - 0.45 * np.exp(-((V - y0 + 1.5) / 1.6) ** 2))[..., None]
            col += 0.35 * np.exp(-((V - y1 - 1.5) / 1.4) ** 2)[..., None]
            if name == 'act':
                col *= (1 - 0.35 * smoothstep(y0 - 30, y0, V))[..., None]
            if name == 'collar':
                # brushed micro texture
                col *= (0.95 + 0.05 * np.sin(V * 2.1))[..., None]
            if st.get('dapple_img') is not None:
                dp = st['dapple_img'][m.sl]
                col = col * (0.55 + 0.6 * dp[..., None])
            over(frame, m.sl, col, a)
        # nozzle dot (white plastic insert facing the camera)
        c = (P['dot'][0], P['dot'][1] + dy + act_dy)
        m = Mask(cam, ellipse_pts(c, 20, 20)).circle(c, P['dot_r'])
        if m.ok:
            a = m.f()
            U, V = m.design_grid()
            g = np.clip(1 - ((U - c[0]) ** 2 + (V - c[1]) ** 2) / P['dot_r'] ** 2, 0, 1)
            over(frame, m.sl, np.stack([0.80 + 0.15 * g, 0.80 + 0.15 * g, 0.78 + 0.14 * g], -1), a)
            m2 = Mask(cam, ellipse_pts(c, 6, 6)).circle((c[0] + 1, c[1] + 1), 3.0)
            if m2.ok:
                over(frame, m2.sl, (0.25, 0.25, 0.25), m2.f() * 0.8)

    def tray(self, frame, st, dy=0.0):
        cam = st['cam']
        T = TRAY
        # top rim (seen from above) and front face
        rim = np.array([(T['x0'] + 6, T['y0'] - T['top'] + dy), (T['x1'] - 6, T['y0'] - T['top'] + dy), (T['x1'], T['y0'] + dy), (T['x0'], T['y0'] + dy)])
        m = Mask(cam, rim).poly(rim)
        if m.ok:
            U, V = m.design_grid()
            g = smoothstep(T['y0'] - T['top'] + dy, T['y0'] + dy, V)
            col = np.stack([0.05 + 0.05 * g] * 3, -1) * np.array([1.0, 0.98, 0.95], np.float32)
            over(frame, m.sl, col, m.f())
            over(frame, m.sl, (0.35, 0.34, 0.32), m.f() * np.exp(-((V - (T['y0'] - T['top'] + dy + 1)) / 1.2) ** 2) * 0.8)
        front = rrect_pts(T['x0'], T['y0'] + dy, T['x1'], T['y1'] + dy, 3)
        m = Mask(cam, front).poly(front)
        if m.ok:
            a = m.f()
            U, V = m.design_grid()
            yy = (V - T['y0'] - dy) / (T['y1'] - T['y0'])
            base = 0.028 + 0.03 * np.exp(-yy * 5) + 0.012 * smoothstep(0.2, 1.0, yy)
            sheen = 0.03 * np.exp(-((U - (CX - 220)) / 260) ** 2)
            col = np.stack([base + sheen] * 3, -1)
            if st.get('dapple_img') is not None:
                col = col * (0.6 + 1.2 * st['dapple_img'][m.sl][..., None])
            over(frame, m.sl, col, a)
            over(frame, m.sl, (0.30, 0.29, 0.27), a * np.exp(-((V - (T['y0'] + dy + 1.5)) / 1.3) ** 2) * 0.7)
            over(frame, m.sl, (0.16, 0.16, 0.15), a * (np.exp(-((U - T['x0'] - 2) / 1.5) ** 2) + np.exp(-((U - T['x1'] + 2) / 1.5) ** 2)) * 0.5)
