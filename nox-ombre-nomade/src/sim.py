"""Physics for the filling sequence.

* Fluid: 2D stable-fluids solver (semi-Lagrangian advection, Jacobi pressure projection,
  vorticity confinement, per-channel buoyancy) on a 4-px grid covering the bottle interior.
  Channels carry incense smoke, coloured tinctures (ink-in-water plumes), turbidity and a
  'schlieren' index used to distort the refraction when alcohol mixes with oils.
* Particles: oud chips, botanicals, cardamom pods, saffron threads (gravity, drag, fluid
  coupling, floating, settling), plus rising bubbles and surface foam.
"""
import numpy as np
import cv2
from bottle import IX0, IX1, IY0, IY1, CELL, GX, GY, CX, level_y

CH = ['smoke', 'amber', 'teal', 'orange', 'crimson', 'brown', 'yellow', 'cloud', 'schl']
CI = {c: i for i, c in enumerate(CH)}
# absorbance (R,G,B) per unit density of each tincture
ABS = {
    'amber': (0.10, 0.55, 1.60),
    'teal': (2.40, 0.35, 0.55),
    'orange': (0.10, 1.40, 2.40),
    'crimson': (0.35, 2.60, 1.90),
    'brown': (0.90, 1.40, 1.90),
    'yellow': (0.02, 0.18, 2.60),
}
BUOY = {'smoke': -22.0, 'amber': 6.0, 'teal': 14.0, 'orange': 13.0, 'crimson': 16.0, 'brown': 12.0,
        'yellow': 9.0, 'cloud': -4.0, 'schl': 0.0}


class Fluid:
    def __init__(self):
        self.u = np.zeros((GY, GX), np.float32)
        self.v = np.zeros((GY, GX), np.float32)
        self.d = np.zeros((len(CH), GY, GX), np.float32)
        gx, gy = np.meshgrid(np.arange(GX, dtype=np.float32), np.arange(GY, dtype=np.float32))
        self.gx, self.gy = gx, gy
        self.decay = {c: 0.05 for c in CH}
        self.decay.update(smoke=0.25, cloud=0.7, schl=0.9)

    def cell(self, x, y):
        return (x - IX0) / CELL - 0.5, (y - IY0) / CELL - 0.5

    def splat(self, ch, x, y, r, amount, vx=0.0, vy=0.0):
        cx, cy = self.cell(x, y)
        rc = max(r / CELL, 0.7)
        g = np.exp(-(((self.gx - cx) ** 2 + (self.gy - cy) ** 2) / (rc * rc)))
        if ch is not None:
            self.d[CI[ch]] += g * amount
        if vx or vy:
            self.u += g * vx / CELL
            self.v += g * vy / CELL

    def stir(self, amp, rng):
        n = rng.standard_normal((4, 8)).astype(np.float32)
        self.u += cv2.resize(n, (GX, GY), interpolation=cv2.INTER_CUBIC) * amp
        n = rng.standard_normal((4, 8)).astype(np.float32)
        self.v += cv2.resize(n, (GX, GY), interpolation=cv2.INTER_CUBIC) * amp

    def _adv(self, f, u, v, dt):
        return cv2.remap(f, self.gx - dt * u, self.gy - dt * v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def step(self, dt, level, air_mode=False):
        surf = GY if air_mode else int(np.clip(((level_y(level) - IY0) / CELL), 0, GY))
        mask = np.ones((GY, GX), np.float32)
        if not air_mode:
            mask[:surf] = 0.0
        # forces: buoyancy of each channel + vorticity confinement
        for c, b in BUOY.items():
            if b:
                self.v += dt * b * self.d[CI[c]]
        w = (np.roll(self.v, -1, 1) - np.roll(self.v, 1, 1) - np.roll(self.u, -1, 0) + np.roll(self.u, 1, 0)) * 0.5
        aw = np.abs(w)
        nx = (np.roll(aw, -1, 1) - np.roll(aw, 1, 1)) * 0.5
        ny = (np.roll(aw, -1, 0) - np.roll(aw, 1, 0)) * 0.5
        nl = np.sqrt(nx * nx + ny * ny) + 1e-5
        eps = 2.2
        self.u += dt * eps * (ny / nl) * w
        self.v -= dt * eps * (nx / nl) * w
        self.u *= mask
        self.v *= mask
        # advect velocity
        u2 = self._adv(self.u, self.u, self.v, dt)
        v2 = self._adv(self.v, self.u, self.v, dt)
        self.u, self.v = u2 * mask, v2 * mask
        self._project(mask, surf)
        # advect dyes
        for i, c in enumerate(CH):
            if self.d[i].max() > 1e-4:
                self.d[i] = self._adv(self.d[i], self.u, self.v, dt) * np.exp(-dt * self.decay[c])
                if c != 'smoke' and not air_mode:
                    self.d[i] *= mask
        damp = np.exp(-dt * 0.6)
        self.u *= damp
        self.v *= damp

    def _project(self, mask, surf, iters=36):
        u, v = self.u, self.v
        u[:, 0] = u[:, -1] = 0
        v[-1, :] = 0
        if surf < GY:
            v[max(surf - 1, 0), :] = np.minimum(v[max(surf - 1, 0), :], 0) * 0
        div = 0.5 * (np.roll(u, -1, 1) - np.roll(u, 1, 1) + np.roll(v, -1, 0) - np.roll(v, 1, 0)) * mask
        p = np.zeros_like(div)
        for _ in range(iters):
            pp = np.pad(p, 1, mode='edge')
            p = (pp[1:-1, :-2] + pp[1:-1, 2:] + pp[:-2, 1:-1] + pp[2:, 1:-1] - div) * 0.25
            p *= mask
        pp = np.pad(p, 1, mode='edge')
        self.u = (u - 0.5 * (pp[1:-1, 2:] - pp[1:-1, :-2])) * mask
        self.v = (v - 0.5 * (pp[2:, 1:-1] - pp[:-2, 1:-1])) * mask

    def vel_at(self, x, y):
        cx, cy = self.cell(x, y)
        i = int(np.clip(round(cy), 0, GY - 1)); j = int(np.clip(round(cx), 0, GX - 1))
        return self.u[i, j] * CELL, self.v[i, j] * CELL

    # ---------------------------------------------------- outputs for the renderer
    def absorb_grid(self):
        A = np.zeros((GY, GX, 3), np.float32)
        for c, a in ABS.items():
            A += self.d[CI[c]][..., None] * np.asarray(a, np.float32)
        return A

    def schlieren(self, k=26.0):
        s = cv2.GaussianBlur(self.d[CI['schl']], (0, 0), 1.2)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3) * k
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3) * k
        return gx, gy


class Particles:
    SINK = {'chip': 75.0, 'flake': 22.0, 'pod': 150.0, 'saffron': 40.0}
    SIZE = {'chip': 14.0, 'flake': 9.0, 'pod': 13.0, 'saffron': 6.0}

    def __init__(self, rng):
        self.p = []
        self.rng = rng

    def spawn(self, kind, x, y, vx=0.0, vy=0.0, n=1, spread=4.0, float_t=0.0):
        r = self.rng
        for _ in range(n):
            self.p.append(dict(kind=kind, x=x + r.uniform(-spread, spread), y=y + r.uniform(-spread, spread),
                               vx=vx + r.uniform(-20, 20), vy=vy + r.uniform(-10, 30),
                               ang=r.uniform(0, 2 * np.pi), av=r.uniform(-6, 6), spr=int(r.integers(0, 8)),
                               scale=r.uniform(0.8, 1.2), settled=False, float_t=float_t,
                               rest=r.uniform(0, 12)))

    def step(self, dt, yl, fluid, level, kick=0.0):
        r = self.rng
        for q in self.p:
            if q['settled'] and kick <= 0:
                # flatten slowly on the floor
                q['ang'] += (np.round(q['ang'] / np.pi) * np.pi - q['ang']) * min(1, dt * 3)
                continue
            if kick > 0 and q['settled'] and r.random() < kick * dt * 3:
                q['settled'] = False
                q['vy'] = -r.uniform(40, 160) * kick
                q['vx'] = r.uniform(-120, 120) * kick
            inside = IX0 < q['x'] < IX1
            in_liq = inside and level > 0.003 and q['y'] > yl
            if not in_liq:
                q['vy'] += 2300 * dt
                q['vx'] *= np.exp(-dt * 0.5)
                if abs(q['x'] - CX) < 60 and q['y'] < IY0 + 10:
                    q['x'] += (CX - q['x']) * min(1, dt * 4)
            else:
                if q['float_t'] > 0:
                    q['float_t'] -= dt
                    q['vy'] = (yl + 3 - q['y']) * 6
                    fu, fv = fluid.vel_at(q['x'], yl + 10)
                    q['vx'] = q['vx'] * np.exp(-dt * 4) + fu * 0.6
                else:
                    fu, fv = fluid.vel_at(q['x'], q['y'])
                    sink = self.SINK[q['kind']] * (1 - 0.5 * kick)
                    a = np.exp(-dt * 7)
                    q['vx'] = q['vx'] * a + (1 - a) * (fu + 25 * np.sin(q['rest'] + q['y'] * 0.03))
                    q['vy'] = q['vy'] * a + (1 - a) * (fv + sink)
                    q['av'] *= np.exp(-dt * 2)
            q['x'] += q['vx'] * dt
            q['y'] += q['vy'] * dt
            q['ang'] += q['av'] * dt
            q['x'] = float(np.clip(q['x'], IX0 + 12, IX1 - 12)) if q['y'] > IY0 else q['x']
            floor = IY1 - self.SIZE[q['kind']] * 0.6 - q['rest'] * 0.5
            if q['y'] >= floor:
                q['y'] = floor
                q['vy'] = 0
                q['vx'] = 0
                q['settled'] = True

    def snapshot(self):
        return [(q['kind'], q['spr'], q['x'], q['y'], q['ang'], q['scale']) for q in self.p]


class Bubbles:
    def __init__(self, rng):
        self.b = np.zeros((0, 5), np.float32)   # x, y, r, life, foam(0/1)
        self.rng = rng

    def spawn(self, x, y, n, spread=20.0, rmin=1.5, rmax=6.0):
        r = self.rng
        if n <= 0:
            return
        nb = np.stack([x + r.normal(0, spread, n), y + np.abs(r.normal(0, spread * 0.8, n)),
                       r.uniform(rmin, rmax, n), np.zeros(n), np.zeros(n)], 1).astype(np.float32)
        self.b = np.vstack([self.b, nb])[-1400:]

    def step(self, dt, yl, t, fluid):
        b = self.b
        if len(b) == 0:
            return
        rise = b[:, 4] == 0
        b[rise, 1] -= (45 + 22 * b[rise, 2]) * dt
        b[rise, 0] += np.sin(t * 9 + b[rise, 1] * 0.05 + b[rise, 2]) * 22 * dt
        b[:, 3] += dt
        hit = rise & (b[:, 1] <= yl + b[:, 2])
        b[hit, 4] = 1
        b[hit, 3] = 0
        foam = b[:, 4] == 1
        b[foam, 1] = yl + b[foam, 2] * 0.6
        b[foam, 0] += np.sin(b[foam, 3] * 2 + b[foam, 2]) * 8 * dt
        keep = ~(foam & (b[:, 3] > 0.25 + b[:, 2] * 0.12)) & (b[:, 0] > IX0 + 4) & (b[:, 0] < IX1 - 4)
        self.b = b[keep]
