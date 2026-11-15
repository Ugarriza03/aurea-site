"""Shot list (cut points measured on the reference reel) and the director that turns it
into a per-frame scene state, stepping the physics as it goes.

Reference cut points were measured by frame differencing at 60 fps (analysis/scene_cuts.json).
"""
import numpy as np
from gfx import ease, ease_io, lerp
from bottle import CX, IX0, IX1, IY0, IY1, level_y, PUMP
import sim as S

FPS = 30
DURATION = 43.933
NFRAMES = int(round(DURATION * FPS))

# name, t0, t1, caption (olfactory note printed on the NOX box)
SHOTS = [
    ('A_humo',        0.000,  0.400, 'AHUMADO'),
    ('B_humo_disipa', 0.400,  0.917, 'AHUMADO'),
    ('C_pipeta',      0.917,  2.150, None),
    ('D_embudo_ambar', 2.150, 4.150, 'ÁMBAR'),
    ('E_jeringa',     4.150,  6.150, 'BALSÁMICO'),
    ('F_pipeta',      6.150,  6.683, None),
    ('G_embudo_oud',  6.683,  9.083, 'OUD'),
    ('H_frasco_botanicos', 9.083, 11.083, 'LEÑOSO'),
    ('I_gotero_claro', 11.083, 13.250, None),
    ('J_gotero_turquesa', 13.250, 16.117, None),
    ('K_gotero_naranja', 16.117, 18.883, 'AFRUTADO'),
    ('L_gotero_carmesi', 18.883, 21.633, 'ROSA'),
    ('M_gotero_marron', 21.633, 24.217, 'CUERO'),
    ('N_pipeta_amarilla', 24.217, 26.950, 'DULCE'),
    ('O_pinzas_cardamomo', 26.950, 29.250, 'PICANTE CÁLIDO'),
    ('P1_gotero',     29.250, 29.783, None),
    ('P2_embudo_azafran', 29.783, 32.783, None),
    ('Q_vaso_alcohol', 32.783, 35.050, None),
    ('R_pipeta',      35.050, 35.550, None),
    ('S_atomizador_crimpado', 35.550, 36.583, None),
    ('T_reposo',      36.583, 37.117, None),
    ('U_mano_agitado', 37.117, 38.583, None),
    ('V_luz_moteada', 38.583, 40.650, None),
    ('W_barrido',     40.650, 41.550, None),
    ('X_bandeja',     41.550, 42.417, None),
    ('Y_hero',        42.417, 43.017, None),
    ('Z_spray_titulo', 43.017, 43.933, None),
]

LEVEL_KEYS = [(0, 0), (0.917, 0), (2.15, 0.012), (2.15, 0.014), (4.15, 0.13), (4.15, 0.15), (6.15, 0.28),
              (6.15, 0.31), (6.683, 0.33), (6.683, 0.34), (9.083, 0.46), (9.083, 0.47), (13.25, 0.475),
              (16.117, 0.48), (24.217, 0.49), (26.95, 0.51), (29.25, 0.515), (29.783, 0.53), (32.783, 0.59),
              (32.783, 0.60), (32.95, 0.60), (34.95, 0.925), (35.55, 0.93), (99, 0.93)]
A_KEYS = [(0, (0, 0, 0)), (2.15, (0.0, 0.02, 0.05)), (4.15, (0.04, 0.22, 0.62)), (6.15, (0.03, 0.18, 0.52)),
          (6.683, (0.03, 0.18, 0.52)), (9.083, (0.07, 0.33, 0.88)), (13.25, (0.08, 0.36, 0.95)),
          (21.63, (0.10, 0.42, 1.05)), (29.25, (0.11, 0.46, 1.15)), (32.783, (0.12, 0.54, 1.32)),
          (33.4, (0.12, 0.52, 1.28)), (35.0, (0.073, 0.174, 0.598)), (99, (0.073, 0.174, 0.598))]
DROPPERS = {  # colour seen in the pipette, tincture channel, tip position
    'I_gotero_claro': ((0.96, 0.98, 0.96), 'schl', (575, None)),
    'J_gotero_turquesa': ((0.22, 0.78, 0.80), 'teal', (520, IY1 - 105)),
    'K_gotero_naranja': ((0.98, 0.40, 0.10), 'orange', (600, IY1 - 135)),
    'L_gotero_carmesi': ((0.72, 0.04, 0.12), 'crimson', (505, IY1 - 60)),
    'M_gotero_marron': ((0.50, 0.27, 0.12), 'brown', (575, IY1 - 115)),
    'P1_gotero': ((0.96, 0.98, 0.96), 'schl', (545, None)),
}
STICK_LYING = (600.0, IY1 - 24.0, -0.10, 1.05)
STICK_UP = (420.0, IY1 - 125.0, -1.25, 0.70)


def keyed(keys, t):
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            k = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return np.asarray(v0, np.float64) * (1 - k) + np.asarray(v1, np.float64) * k
    return np.asarray(keys[-1][1], np.float64)


def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s[1] <= t < s[2]:
            return i, s
    return len(SHOTS) - 1, SHOTS[-1]


def cam_t(s=1.0, rot=0.0, pivot=(540, 960), center=(540, 960)):
    return (float(s), float(rot), (float(pivot[0]), float(pivot[1])), (float(center[0]), float(center[1])))


class Director:
    def __init__(self, seed=2026):
        self.rng = np.random.default_rng(seed)
        self.fluid = S.Fluid()
        self.parts = S.Particles(self.rng)
        self.bub = S.Bubbles(self.rng)
        self.tilt = 0.0
        self.tilt_v = 0.0
        self.ripple = 0.0
        self.yl_shift = 0.0
        self._accum = {}
        # pre-roll: the bottle has been filled with incense smoke before the reel starts
        for k in range(75):
            tt = k / 30.0
            self.fluid.splat('smoke', CX + 30 * np.sin(tt * 3), IY0 + 20, 26, 0.35, vx=60 * np.sin(tt * 2.3), vy=260)
            self.fluid.splat(None, CX + 200 * np.sin(tt * 1.7), IY1 - 120, 60, 0, vx=-120 * np.cos(tt * 2), vy=-60)
            self.fluid.step(1 / 30, 0, air_mode=True)

    # ------------------------------------------------------------------ helpers
    def every(self, key, t, period):
        last = self._accum.get(key, -1e9)
        if t - last >= period:
            self._accum[key] = t
            return True
        return False

    def surface(self, level):
        return level_y(level)

    def pour(self, x, yl, level, ch=None, amount=0.0, strength=200.0, bubbles=3, width=10, cloud=0.0, schl=0.0):
        if level <= 0.004:
            yimp = IY1 - 4
        else:
            yimp = yl + 6
        self.fluid.splat(ch, x, yimp + 14, width * 1.6, amount, vy=strength)
        if cloud:
            self.fluid.splat('cloud', x, yimp + 30, width * 2.5, cloud, vy=strength * 0.5)
        if schl:
            self.fluid.splat('schl', x, yimp + 30, width * 3, schl, vy=strength * 0.3)
        if level > 0.004:
            self.bub.spawn(x, yimp + 25, int(bubbles), spread=width * 1.2 + 8)
        self.ripple = max(self.ripple, min(3.0, strength / 120))

    # ------------------------------------------------------------------ main
    def state(self, fi):
        t = fi / FPS
        dt = 1 / FPS
        si, (name, t0, t1, caption) = shot_at(t)
        tl = t - t0
        dur = t1 - t0
        u = tl / dur
        level = float(keyed(LEVEL_KEYS, t))
        A = keyed(A_KEYS, t)
        yl = self.surface(level)
        st = dict(t=t, fi=fi, shot=name, shot_idx=si, tl=tl, level=level, A=tuple(A), bg='garden', sun=1.0,
                  cam=cam_t(), cam_bg=cam_t(), pump=None, dip_dy=None, tray=None, props=[], front=[], streams=[],
                  smoke=0.0, stick=STICK_LYING if t < 9.083 else STICK_UP, caption=None, title=0.0, mist=0.0,
                  dapple=False, subcams=None, cloud_u=0.0)
        kick = 0.0
        air_mode = level <= 0.003
        f = self.fluid
        # caption timing (fade inside the shot)
        if caption:
            sh = [s for s in SHOTS if s[3] == caption]
            c0, c1 = sh[0][1], sh[-1][2]
            a = min(1, (t - c0) / 0.25, (c1 - t) / 0.25)
            st['caption'] = (caption, float(np.clip(a, 0, 1)))

        # ------------------------------------------------ shots
        if name == 'A_humo':
            st['smoke'] = 1.0
            f.splat('smoke', CX, IY0 + 10, 20, 0.12, vy=120)
        elif name == 'B_humo_disipa':
            st['smoke'] = 1 - u
            f.decay['smoke'] = 0.9
        elif name in ('C_pipeta', 'F_pipeta', 'R_pipeta'):
            f.decay['smoke'] = 3.0
            tip = (CX + 6, 900)
            st['props'].append(('pipette', dict(tip=tip, r=10.0, fill_col=(0.97, 0.98, 0.98), fill_to=-100)))
            on = tl > 0.12 and tl < dur - 0.05
            if on:
                yend = yl if level > 0.004 else IY1 - 2
                st['streams'].append((tip[0], tip[1], yend, 3.5, (0.97, 0.98, 1.0)))
                self.pour(tip[0], yl, level, 'schl', 0.08, 140, bubbles=1.5, width=4)
        elif name in ('D_embudo_ambar', 'G_embudo_oud', 'P2_embudo_azafran'):
            kind = {'D_embudo_ambar': 'amber', 'G_embudo_oud': 'oud', 'P2_embudo_azafran': 'saffron'}[name]
            col = {'amber': (0.98, 0.66, 0.18), 'oud': (0.80, 0.52, 0.26), 'saffron': (0.98, 0.52, 0.12)}[kind]
            scol = {'amber': (0.99, 0.80, 0.38), 'oud': (0.90, 0.68, 0.40), 'saffron': (0.99, 0.72, 0.30)}[kind]
            st['props'].append(('funnel', dict(content_col=col, fill=lerp(0.85, 0.30, u), texture=kind)))
            wob = 1.5
            yend = yl if level > 0.004 else IY1 - 2
            st['streams'].append((CX, 952, yend, 9.0, scol))
            ch = {'amber': 'amber', 'oud': 'brown', 'saffron': 'yellow'}[kind]
            self.pour(CX, yl, level, ch, 0.05 if kind == 'amber' else 0.025, 240, bubbles=4, width=9, schl=0.05)
            if kind == 'oud' and self.every('chip', t, 0.11):
                self.parts.spawn('chip', CX, 960, vx=self.rng.uniform(-90, 90), vy=250, n=1)
            if kind == 'saffron' and self.every('saf', t, 0.07):
                self.parts.spawn('saffron', CX, 960, vx=self.rng.uniform(-110, 110), vy=200, n=1)
        elif name == 'E_jeringa':
            p = ease_io(np.clip((tl - 0.15) / (dur - 0.3), 0, 1))
            st['props'].append(('syringe', dict(plunger=p)))
            if 0.15 < tl < dur - 0.15:
                st['streams'].append((CX, 905, yl, 5.0, (0.98, 0.98, 1.0)))
                self.pour(CX, yl, level, None, 0, 520, bubbles=14, width=7, cloud=0.22, schl=0.25)
                f.splat(None, CX, yl + 120, 80, 0, vx=float(self.rng.normal(0, 160)))
        elif name == 'H_frasco_botanicos':
            ang = lerp(-0.85, -0.55, ease_io(u))
            st['props'].append(('jar', dict(mouth=(568, 846), ang=ang)))
            if tl > 0.25 and self.every('flk', t, 0.035):
                self.parts.spawn('flake', 560, 850, vx=self.rng.uniform(-40, 10), vy=60, n=2, spread=6,
                                 float_t=self.rng.uniform(0.3, 2.5))
        elif name in DROPPERS:
            col, ch, (tx, ty) = DROPPERS[name]
            if ty is None:
                ty = yl + 55
            descend = ease(min(1, tl / 0.35))
            tip = (tx, lerp(820, ty, descend))
            squeeze = ease(np.clip((tl - 0.45) / 0.5, 0, 1)) if ch != 'schl' else ease(np.clip((tl - 0.3) / 0.4, 0, 1))
            fill = lerp(0.75, 0.12, np.clip((tl - 0.45) / (dur - 0.7), 0, 1))
            st['props'].append(('dropper', dict(tip=tip, color=col, fill=fill, squeeze=squeeze, length=ty - 470)))
            if tl > 0.45 and fill > 0.13:
                if ch == 'schl':
                    if self.every('dropc', t, 0.18):
                        f.splat('schl', tip[0], tip[1] + 10, 14, 0.6, vy=160)
                        self.bub.spawn(tip[0], tip[1] + 6, 2, spread=4, rmax=3.5)
                else:
                    f.splat(ch, tip[0], tip[1] + 10, 16, 0.42, vy=90 + 50 * np.sin(t * 5), vx=40 * np.sin(t * 3.1))
                    f.splat('schl', tip[0], tip[1] + 8, 10, 0.05)
        elif name == 'N_pipeta_amarilla':
            tip = (CX + 8, lerp(820, IY1 - 90, ease(min(1, tl / 0.4))))
            st['props'].append(('pipette', dict(tip=tip, r=11.0, fill_col=(0.98, 0.84, 0.12),
                                                fill_from=tip[1], fill_to=lerp(-120, 560, np.clip((tl - 0.5) / (dur - 0.6), 0, 1)))))
            if tl > 0.5:
                f.splat('yellow', tip[0], tip[1] + 8, 12, 0.22, vy=40)
        elif name == 'O_pinzas_cardamomo':
            subs = [26.95, 27.55, 28.283, 28.717, 29.25]
            k = max(i for i in range(4) if t >= subs[i])
            sl = t - subs[k]
            sd = subs[k + 1] - subs[k]
            open_ = ease(np.clip((sl - sd * 0.35) / 0.12, 0, 1))
            tip = (CX - 8 + 12 * k, 842 - 20 * (1 - ease(min(1, sl / 0.2))))
            st['props'].append(('tweezers', dict(tip=tip, open_=open_, pod=k % 4)))
            if open_ > 0.2 and not self._accum.get(('pod', k)):
                self._accum[('pod', k)] = 1
                self.parts.spawn('pod', tip[0] - 6, tip[1] + 10, vy=120, n=1, spread=1)
                self.parts.p[-1]['spr'] = k % 4
                self.parts.p[-1]['ang'] = 0.9
        elif name == 'Q_vaso_alcohol':
            ang = lerp(-1.62, -1.92, ease_io(u))
            st['props'].append(('beaker', dict(lip=(600, 838), ang=ang, fill=lerp(0.9, 0.25, u))))
            on = tl > 0.1 and tl < dur - 0.1
            if on:
                st['streams'].append(('arc', (600, 838), yl, 14.0, (0.98, 0.99, 1.0)))
                self.pour(CX + 4, yl, level, None, 0, 650, bubbles=26, width=14, cloud=0.35, schl=0.6)
                f.splat(None, CX + self.rng.normal(0, 200), yl + 150, 120, 0, vx=float(self.rng.normal(0, 300)))
            for c in ('teal', 'orange', 'crimson', 'brown', 'yellow', 'amber'):
                f.decay[c] = lerp(0.3, 2.2, u)
        elif name == 'S_atomizador_crimpado':
            if tl < 0.40:
                dy = -320 * (1 - ease_io(tl / 0.40))
                st['pump'] = dict(dy=dy)
                st['dip_dy'] = dy
                st['front'].append(('glove', dict(anchor=(CX + 70, 700 + dy), ang=0.1, kind='pinch', scale=0.95)))
            else:
                st['pump'] = dict(dy=0.0)
                st['dip_dy'] = 0.0
                c = tl - 0.40
                press = ease(np.clip((c - 0.25) / 0.12, 0, 1)) - ease(np.clip((c - 0.45) / 0.12, 0, 1))
                by = lerp(300, PUMP['act_y1'] + 4, ease_io(min(1, c / 0.28))) + 10 * press
                st['props_after_pump'] = [('crimper', dict(bottom_y=by))]
                if c > 0.25 and not self._accum.get('crimp'):
                    self._accum['crimp'] = 1
        elif name == 'T_reposo':
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            st['cam'] = cam_t(1.05, 0, (540, 1120), (540, 1030))
            st['cam_bg'] = cam_t(1.02, 0, (540, 1120), (540, 1040))
        elif name == 'U_mano_agitado':
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            rot = -0.40 + 0.16 * np.sin(2 * np.pi * 1.4 * tl) + 0.25 * ease_io(u)
            st['cam'] = cam_t(1.62, rot, (700, 1180), (540 + 30 * np.sin(tl * 5), 1000 + 20 * np.cos(tl * 4)))
            st['cam_bg'] = cam_t(1.25, 0.0, (540, 1000), (540 + 60 * np.sin(tl * 2), 960 - 80 * u))
            st['bg'] = 'garden_soft'
            st['front'].append(('glove', dict(anchor=(1005, 1210), ang=0.0, kind='grip', scale=1.0)))
            target = float(np.clip(-np.tan(rot), -0.6, 0.6))
            self.tilt_v += (target - self.tilt) * 140 * dt - self.tilt_v * 5 * dt
            self.tilt += self.tilt_v * dt
            kick = 1.0
            f.stir(25.0, self.rng)
        elif name == 'V_luz_moteada':
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            st['bg'] = 'dapple'
            st['dapple'] = True
            s = lerp(1.13, 1.22, ease_io(u))
            st['cam'] = cam_t(s, -0.02, (540, 1140), (540, 1010))
            st['cam_bg'] = cam_t(1.0 + 0.4 * (s - 1), 0, (540, 1100), (540 + 30 * u, 1060))
            st['front'].append(('glove', dict(anchor=(CX + 30, PUMP['act_y1'] + 4), ang=-0.05, kind='press', scale=1.0)))
            self.tilt *= 0.85
        elif name == 'W_barrido':
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            st['bg'] = 'dapple' if u < 0.5 else 'garden_soft'
            st['dapple'] = u < 0.5
            st['tray'] = dict(dy=200.0)
            subs = []
            for k in range(9):
                uu = np.clip(u + (k - 4) * 0.012, 0, 1)
                e = ease_io(uu)
                rot = lerp(-0.02, 0.42, e) + 0.25 * np.sin(np.pi * e)
                s = lerp(1.22, 1.6, e)
                cen = (540 - 520 * np.sin(np.pi * e), lerp(1010, 980, e))
                pv = (lerp(540, 560, e), lerp(1140, 1300, e))
                subs.append((cam_t(s, rot, pv, cen), cam_t(1.2, 0.3 * np.sin(np.pi * e), (540, 1000), (540 - 400 * np.sin(np.pi * e), 960))))
            st['subcams'] = subs
            st['cam'], st['cam_bg'] = subs[4]
        elif name == 'X_bandeja':
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            st['bg'] = 'garden_soft'
            e = ease_io(u)
            st['tray'] = dict(dy=lerp(200, 0, e))
            st['cam'] = cam_t(1.6, lerp(0.42, 0.34, e), (560, 1300), (560, 980))
            st['cam_bg'] = cam_t(1.2, 0.3, (540, 1000), (540, lerp(900, 1050, e)))
            st['front'].append(('glove', dict(anchor=(CX + 60, 660), ang=0.1, kind='pinch', scale=0.95)))
        elif name in ('Y_hero', 'Z_spray_titulo'):
            st['pump'] = dict(dy=0.0)
            st['dip_dy'] = 0.0
            st['tray'] = dict(dy=0.0)
            T0 = 42.417
            uu = (t - T0) / (43.933 - T0)
            s = lerp(0.93, 0.97, ease_io(uu))
            st['cam'] = cam_t(s, 0.0, (540, 1150), (540, 1105))
            st['cam_bg'] = cam_t(1.0, 0.0, (540, 1100), (540, 1050))
            if name == 'Y_hero':
                st['front'].append(('glove', dict(anchor=(lerp(1060, 1500, ease(u)), 1180), ang=0.0, kind='grip', scale=1.0)))
            else:
                press = ease(np.clip((tl - 0.12) / 0.16, 0, 1))
                rel = ease(np.clip((tl - 0.55) / 0.2, 0, 1))
                fy = lerp(520, PUMP['act_y1'] + 6, press) - 40 * rel
                fx = lerp(CX + 420, CX + 30, press) + 650 * rel
                st['pump'] = dict(dy=0.0, act_dy=8.0 * press * (1 - rel))
                st['front'].append(('glove', dict(anchor=(fx, fy), ang=0.75, kind='press', scale=1.0)))
                if press > 0.95:
                    st['mist'] = max(0.0, tl - 0.28)
            st['title'] = float(np.clip((t - 42.55) / 0.9, 0, 1))
        if 'pipette' not in [p[0] for p in st['props']] and name in ('C_pipeta',):
            pass

        # ------------------------------------------------ physics
        steps = 2
        for _ in range(steps):
            f.step(dt / steps, level, air_mode=air_mode)
        if not air_mode:
            f.d[S.CI['smoke']] *= 0.7
        if name not in ('U_mano_agitado',):
            self.tilt_v += (-self.tilt) * 140 * dt - self.tilt_v * 6 * dt
            self.tilt += self.tilt_v * dt
        # volume conservation when tilted: shift the mean level so the air area stays constant
        yl0 = yl
        if abs(self.tilt) > 1e-3 and level > 0.5:
            xs = np.linspace(IX0, IX1, 64)
            air_target = (yl - IY0)
            lo, hi = IY0 - 400, IY1
            for _ in range(30):
                mid = (lo + hi) / 2
                air = np.mean(np.clip(mid + self.tilt * (xs - CX) - IY0, 0, None))
                lo, hi = (mid, hi) if air < air_target else (lo, mid)
            yl0 = (lo + hi) / 2
        st['tilt'] = self.tilt
        st['yl_override'] = yl0
        self.parts.step(dt, yl0, f, level, kick=kick)
        self.bub.step(dt, yl0, t, f)
        self.ripple *= np.exp(-dt * 2.5)
        st['ripple'] = self.ripple + 0.6 * abs(self.tilt_v) * 10 if level > 0.004 else 0
        # ------------------------------------------------ export fields
        A_grid = f.absorb_grid()
        st['dye_A'] = A_grid.astype(np.float16) if A_grid.max() > 0.01 else None
        cl = f.d[S.CI['cloud']]
        st['cloud'] = cl.astype(np.float16) if cl.max() > 0.01 else None
        sm = f.d[S.CI['smoke']]
        st['smoke_grid'] = sm.astype(np.float16) if sm.max() > 0.01 else None
        gx, gy = f.schlieren()
        st['schl'] = np.dstack([gx, gy]).astype(np.float16) if np.abs(gx).max() > 0.05 else None
        st['particles'] = self.parts.snapshot()
        st['bubbles'] = self.bub.b.copy()
        return st
