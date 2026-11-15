"""NOX Parfums — Ombre Nomade. Frame renderer + pipeline.

    python3 src/render.py                 # full render: scenes + master + final MP4
    python3 src/render.py --frames 300,900 --preview   # stills for checking

Pipeline: Director (timeline + physics, sequential) -> pool of workers composing each
frame (background, glass, liquid, props, post) -> one ProRes-quality H.264 file per
shot -> concatenation + sound design -> /exports/NOX_OMBRE_NOMADE.mp4
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfx import Cam, W, H, smoothstep, over, add, mul
import background as bgm
import bottle as B
import props as P
import timeline as TL
import post

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build')
EXPORTS = os.path.join(ROOT, 'exports')

G = {}


def load_textures():
    os.makedirs(BUILD, exist_ok=True)
    tex = {}
    for name, seed, var in [('garden', 11, 'garden'), ('dapple', 29, 'dapple')]:
        fn = os.path.join(BUILD, f'bg_{name}.npy')
        if not os.path.exists(fn):
            np.save(fn, bgm.garden(seed, var))
        t = np.load(fn)
        tex[name] = (t, cv2.GaussianBlur(t, (0, 0), 5))
    g = tex['garden'][0]
    soft = cv2.GaussianBlur(g, (0, 0), 9)
    tex['garden_soft'] = (soft, cv2.GaussianBlur(soft, (0, 0), 5))
    fn = os.path.join(BUILD, 'dapple_pat.npy')
    if not os.path.exists(fn):
        np.save(fn, bgm.dapple_pattern(0.0))
    tex['_dapple'] = np.load(fn)
    return tex


def funnel_textures(sprites, rng):
    out = {}
    for kind, key, n in [('oud', 'chip', 220), ('saffron', 'saffron', 260)]:
        w, h = 340, 360
        layer = np.zeros((h, w, 4), np.float32)
        for _ in range(n):
            spr = sprites[key][rng.integers(len(sprites[key]))]
            sh, sw = spr.shape[:2]
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), rng.uniform(0, 360), rng.uniform(0.8, 1.3))
            M[0, 2] += rng.uniform(0, w) - sw / 2
            M[1, 2] += rng.uniform(0, h) - sh / 2
            o = cv2.warpAffine(spr, M, (w, h))
            a = o[..., 3:4]
            layer[..., :3] = layer[..., :3] * (1 - a) + o[..., :3] * a
            layer[..., 3:4] = layer[..., 3:4] * (1 - a) + a
        out[kind] = layer
    out['amber'] = None
    return out


def init_worker():
    G['tex'] = load_textures()
    G['spr'] = P.make_sprites()
    G['ftex'] = funnel_textures(G['spr'], np.random.default_rng(4))
    G['br'] = B.BottleRenderer(G['tex'], G['spr'])
    G['text'] = post.TextLayers(os.path.join(ROOT, 'assets', 'fonts'))


def mk_cam(c):
    return Cam(c[0], c[1], c[2], c[3])


def stream_points(spec, t):
    if spec[0] == 'arc':
        _, lip, yend, w, col = spec
        k = np.linspace(0, 1, 18)
        p0 = np.array(lip, np.float64)
        p1 = np.array((B.CX + 6, 870.0))
        ctrl = np.array((B.CX + 22, 800.0))
        arc = ((1 - k) ** 2)[:, None] * p0 + (2 * k * (1 - k))[:, None] * ctrl + (k ** 2)[:, None] * p1
        down = P.stream_path(B.CX + 6, 870, yend, t, wobble=3.0, n=30)
        return np.vstack([arc, down[1:]]), w, col
    x, y0, y1, w, col = spec
    return P.stream_path(x, y0, y1, t, wobble=1.6 if w < 6 else 2.4, n=40), w, col


def draw_tool(frame, st, cam, name, kw):
    if name == 'pipette':
        P.pipette(frame, cam, **kw)
    elif name == 'dropper':
        P.dropper(frame, cam, **kw)
    elif name == 'funnel':
        kw = dict(kw)
        kw['texture'] = G['ftex'][kw['texture']]
        P.funnel(frame, cam, t=st['t'], **kw)
    elif name == 'syringe':
        P.syringe(frame, cam, **kw)
    elif name == 'jar':
        P.jar(frame, cam, **kw)
    elif name == 'tweezers':
        P.tweezers(frame, cam, sprites=G['spr'], **kw)
    elif name == 'beaker':
        P.beaker(frame, cam, **kw)
    elif name == 'crimper':
        P.crimper(frame, cam, **kw)
    elif name == 'glove':
        P.glove(frame, cam, **kw)


def draw_bubbles(frame, cam, bubbles, yl):
    if bubbles is None or len(bubbles) == 0:
        return
    pts = cam.fwd(bubbles[:, :2])
    r = bubbles[:, 2] * cam.s
    x0, y0 = int(max(0, pts[:, 0].min() - 20)), int(max(0, pts[:, 1].min() - 20))
    x1, y1 = int(min(W, pts[:, 0].max() + 20)), int(min(H, pts[:, 1].max() + 20))
    if x1 <= x0 or y1 <= y0:
        return
    dark = np.zeros((y1 - y0, x1 - x0), np.float32)
    lite = np.zeros_like(dark)
    for (px, py), rr in zip(pts, r):
        c = (int((px - x0) * 4), int((py - y0) * 4))
        cv2.circle(dark, c, int(rr * 4), 0.45, max(1, int(rr * 0.9)), cv2.LINE_AA, shift=2)
        cv2.circle(lite, (c[0] - int(rr * 1.4), c[1] - int(rr * 1.4)), max(2, int(rr * 1.3)), 0.8, -1, cv2.LINE_AA, shift=2)
        cv2.circle(lite, c, int(rr * 4), 0.25, -1, cv2.LINE_AA, shift=2)
    sl = (slice(y0, y1), slice(x0, x1))
    mul(frame, sl, (0.2, 0.15, 0.08), dark)
    add(frame, sl, (1.0, 0.98, 0.9), lite * 0.8)


def render_one(st, cam_override=None):
    tex, br = G['tex'], G['br']
    st = dict(st)
    cam = mk_cam(cam_override[0] if cam_override else st['cam'])
    cam_bg = mk_cam(cam_override[1] if cam_override else st['cam_bg'])
    st['cam'], st['cam_bg'] = cam, cam_bg
    Ub, Vb = cam_bg.inv_grid()
    frame = bgm.sample(tex[st['bg']][0], Ub, Vb).copy()
    if st['dapple']:
        pat = cv2.remap(tex['_dapple'], (Ub + bgm.OX + 12 * np.sin(st['t'] * 1.3)).astype(np.float32),
                        (Vb + bgm.OY).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tbl = smoothstep(bgm.TABLE_Y + 120, bgm.TABLE_Y + 260, Vb)
        frame *= (1 - tbl[..., None] * (0.55 - 0.9 * pat[..., None]))
        U, V = cam.inv_grid()
        st['dapple_img'] = cv2.remap(tex['_dapple'], (U * 0.9 + bgm.OX + 300).astype(np.float32), (V * 0.9 + bgm.OY).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # schlieren grid -> design-pixel offsets
    if st.get('schl') is not None:
        st['schl_grid'] = st['schl'].astype(np.float32)
    if st.get('dye_A') is not None:
        st['dye_A'] = st['dye_A'].astype(np.float32)
    if st.get('cloud') is not None:
        st['cloud_img'] = st['cloud'].astype(np.float32)
    if st.get('smoke_grid') is not None:
        st['smoke_img'] = st['smoke_grid'].astype(np.float32)
    br.table_light(frame, st)
    yl = st.get('yl_override') or B.level_y(st['level'])

    def inside(fr):
        if st['stick'] is not None:
            x, y, a, s = st['stick']
            P.draw_sprite(fr, cam, G['spr']['stick'], x, y, a, s)
        for kind, spr, x, y, ang, sc in st['particles']:
            lst = G['spr'][kind]
            P.draw_sprite(fr, cam, lst[spr % len(lst)], x, y, ang, sc)
        if st['dip_dy'] is not None:
            br.dip_tube(fr, st, st['dip_dy'])
        draw_bubbles(fr, cam, st['bubbles'], yl)
        for spec in st['streams']:
            pts, w, col = stream_points(spec, st['t'])
            P.stream(fr, cam, pts, w, col, t=st['t'])
        for name, kw in st['props']:
            draw_tool(fr, st, cam, name, kw)

    br.render(frame, st, inside)
    br.neck(frame, st)
    if st['smoke'] > 0:
        P.smoke_wisps(frame, cam, st['t'], strength=st['smoke'])
    if st['pump'] is not None:
        br.pump(frame, st, st['pump'].get('dy', 0.0), st['pump'].get('act_dy', 0.0))
    for name, kw in st.get('props_after_pump', []):
        draw_tool(frame, st, cam, name, kw)
    if st['tray'] is not None:
        br.tray(frame, st, st['tray']['dy'])
    for name, kw in st['front']:
        draw_tool(frame, st, cam, name, kw)
    if st['mist'] > 0:
        d = B.PUMP['dot']
        P.mist(frame, cam, (d[0], d[1] + 8), st['mist'])
    return frame


def render_frame(st):
    if st.get('subcams'):
        acc = None
        for sc in st['subcams']:
            f = render_one(st, sc)
            acc = f if acc is None else acc + f
        frame = acc / len(st['subcams'])
        frame = post.directional_blur(frame, st)
    else:
        frame = render_one(st)
    frame = post.finish(frame, st, G['text'])
    return st['fi'], frame


# ======================================================================= pipeline
def ffmpeg_writer(path, crf=12):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(TL.FPS),
           '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def states(frames):
    d = TL.Director()
    want = set(frames)
    for fi in range(max(frames) + 1):
        st = d.state(fi)
        if fi in want:
            yield st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=None, help='comma list of frame numbers to preview')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=TL.NFRAMES)
    args = ap.parse_args()
    load_textures()
    if args.frames:
        frames = [int(x) for x in args.frames.split(',')]
        outdir = os.path.join(BUILD, 'preview')
        os.makedirs(outdir, exist_ok=True)
        with Pool(args.workers, initializer=init_worker) as pool:
            for fi, fr in pool.imap(render_frame, states(frames)):
                cv2.imwrite(os.path.join(outdir, f'f{fi:04d}.jpg'), fr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                print('frame', fi, flush=True)
        return
    scenes_dir = os.path.join(EXPORTS, 'scenes')
    os.makedirs(scenes_dir, exist_ok=True)
    frames = list(range(args.start, args.end))
    t0 = time.time()
    writers = {}
    cur = None
    with Pool(args.workers, initializer=init_worker) as pool:
        for fi, fr in pool.imap(render_frame, states(frames), chunksize=1):
            si, shot = TL.shot_at(fi / TL.FPS)
            if cur != si:
                if cur is not None:
                    writers[cur].stdin.close(); writers[cur].wait()
                path = os.path.join(scenes_dir, f'{si + 1:02d}_{shot[0]}.mp4')
                writers[si] = ffmpeg_writer(path)
                cur = si
            writers[si].stdin.write(fr.tobytes())
            if fi % 15 == 0:
                el = time.time() - t0
                done = fi - args.start + 1
                print(f'frame {fi}/{args.end} {shot[0]}  {el:.0f}s elapsed, eta {el / done * (len(frames) - done):.0f}s', flush=True)
    writers[cur].stdin.close(); writers[cur].wait()
    print('render done', time.time() - t0)


if __name__ == '__main__':
    main()
