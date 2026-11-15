"""Original sound design, synthesised and synced to the shot list (no third-party audio).

Layers: warm ambient pad (A minor add9 voicing, slow filter swell), soft garden air,
liquid pours (band-passed noise with gurgle modulation), drops/plinks, glass taps on
every jump cut, botanicals rain, crimp click, whooshes on the hand-held cuts, the
atomizer 'psst' and a low resolve under the NOX title.
"""
import os
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL

SR = 48000
DUR = TL.DURATION
N = int(SR * DUR)
rng = np.random.default_rng(5)
t = np.arange(N) / SR


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)


def bandpass(x, lo, hi, order=2):
    from scipy.signal import butter, sosfilt
    sos = butter(order, [lo, hi], btype='band', fs=SR, output='sos')
    return sosfilt(sos, x)


def env(t0, t1, a=0.02, r=0.08):
    e = np.zeros(N)
    i0, i1 = int(t0 * SR), int(t1 * SR)
    i1 = min(i1, N)
    n = max(1, i1 - i0)
    seg = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    seg[:min(na, n)] = np.linspace(0, 1, min(na, n))
    if nr and nr < n:
        seg[-nr:] *= np.linspace(1, 0, nr)
    e[i0:i0 + n] = seg[:max(0, min(n, N - i0))]
    return e


def place(buf, sig, t0, gain=1.0, pan=0.0):
    i0 = int(t0 * SR)
    n = min(len(sig), N - i0)
    if n <= 0:
        return
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[0, i0:i0 + n] += sig[:n] * gain * l
    buf[1, i0:i0 + n] += sig[:n] * gain * r


def plink(f=1400, d=0.18):
    tt = np.arange(int(d * SR)) / SR
    fr = f * (1 + 0.6 * np.exp(-tt * 60))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    return np.sin(ph) * np.exp(-tt * 28) * (1 - np.exp(-tt * 3000))


def glass_tap(f=2300, d=0.6):
    tt = np.arange(int(d * SR)) / SR
    s = sum(a * np.sin(2 * np.pi * f * k * tt) * np.exp(-tt * (9 + 6 * k)) for k, a in [(1, 1), (2.76, 0.5), (5.4, 0.25), (8.9, 0.12)])
    return s * (1 - np.exp(-tt * 4000))


def whoosh(d=0.45, up=True):
    n = int(d * SR)
    x = rng.standard_normal(n)
    tt = np.linspace(0, 1, n)
    e = np.sin(np.pi * tt) ** 2
    lo = bandpass(x, 300, 2500)
    hi = bandpass(x, 2500, 9000)
    mix = tt if up else 1 - tt
    return (lo * (1 - mix) + hi * mix) * e


def pour(t0, t1, lo=500, hi=3500, gurgle=12.0, gain=0.25, pan=0.0, buf=None):
    n = int((t1 - t0) * SR)
    if n <= 0:
        return
    x = rng.standard_normal(n)
    y = bandpass(x, lo, hi)
    tt = np.arange(n) / SR
    am = 0.6 + 0.4 * np.sin(2 * np.pi * gurgle * tt + 2 * np.sin(2 * np.pi * 3.1 * tt)) * (0.5 + 0.5 * rng.random())
    e = np.minimum(1, np.minimum(tt / 0.05, (t1 - t0 - tt) / 0.12))
    sig = y * am * np.clip(e, 0, 1)
    # bubbly plinks on top
    for k in range(int((t1 - t0) * 9)):
        tk = rng.uniform(0, max(0.01, t1 - t0 - 0.2))
        p = plink(rng.uniform(600, 1800), 0.12) * rng.uniform(0.1, 0.35)
        i = int(tk * SR)
        m = min(len(p), n - i)
        sig[i:i + m] += p[:m]
    place(buf, sig, t0, gain, pan)


def pad(buf):
    # A minor add9 / F maj7 / C / G6 — slow, warm, filtered saws
    chords = [(0, [110.0, 164.8, 220.0, 246.9, 329.6]), (11.08, [87.3, 130.8, 174.6, 220.0, 329.6]),
              (21.63, [130.8, 196.0, 261.6, 329.6, 392.0]), (32.78, [98.0, 146.8, 196.0, 246.9, 329.6]),
              (37.12, [110.0, 164.8, 220.0, 261.6, 329.6, 493.9]), (42.42, [87.3, 130.8, 220.0, 261.6, 329.6, 440.0])]
    out = np.zeros(N)
    for i, (c0, freqs) in enumerate(chords):
        c1 = chords[i + 1][0] if i + 1 < len(chords) else DUR
        e = env(c0 - 0.6 if i else 0, c1 + 0.6, a=1.2 if i else 2.0, r=1.2)
        sig = np.zeros(N)
        for f in freqs:
            for det in (-0.12, 0.0, 0.11):
                ph = rng.uniform(0, 1)
                saw = 2 * ((t * f * (1 + det / 100) + ph) % 1) - 1
                sig += saw
        out += sig * e / (len(freqs) * 3)
    cutoff_sweep = 600 + 900 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 22))
    y = onepole_lp(out, 900)
    y = onepole_lp(y, 1400)
    lvl = 0.11 + 0.05 * np.clip((t - 32.8) / 4, 0, 1) + 0.04 * np.clip((t - 42.4) / 1, 0, 1)
    fade = np.minimum(1, t / 1.5) * np.minimum(1, (DUR - t) / 0.35)
    y = y * lvl * fade
    # stereo widen with a short delay
    d = int(0.013 * SR)
    buf[0] += y
    buf[1, d:] += y[:-d]
    # low resolve + shimmer at the title
    tt = np.clip(t - 42.42, 0, None)
    sub = np.sin(2 * np.pi * 55 * t) * np.exp(-tt * 0.8) * (t > 42.42) * 0.12
    shimmer = sum(np.sin(2 * np.pi * f * t) for f in (1318.5, 1760.0, 2637.0)) * (t > 42.5) * np.exp(-np.clip(t - 42.5, 0, None) * 1.2) * 0.012
    buf += (sub + shimmer)[None, :]


def main(out_wav):
    buf = np.zeros((2, N))
    pad(buf)
    # garden air bed
    air = onepole_lp(rng.standard_normal(N), 700) * 0.05
    buf += air[None, :] * np.minimum(1, (DUR - t) / 2)[None, :]
    # glass tap on every jump cut of the locked-off part (like tools touching the neck)
    for name, t0, t1, _ in TL.SHOTS:
        if 0.5 < t0 < 36.6:
            place(buf, glass_tap(rng.uniform(1900, 2700)), t0 + 0.06, 0.08, rng.uniform(-0.2, 0.2))
    # incense hiss
    place(buf, bandpass(rng.standard_normal(int(0.9 * SR)), 2000, 7000) * np.linspace(1, 0, int(0.9 * SR)), 0.0, 0.05)
    # pours
    pour(1.05, 2.1, 1500, 5000, 20, 0.12, 0.0, buf)
    pour(2.27, 4.1, 500, 3000, 11, 0.22, -0.05, buf)
    pour(4.3, 6.0, 900, 6000, 25, 0.30, 0.05, buf)
    pour(6.27, 6.65, 1500, 5000, 20, 0.12, 0.0, buf)
    pour(6.8, 9.05, 450, 2800, 9, 0.22, 0.0, buf)
    pour(29.9, 32.75, 500, 3000, 10, 0.20, 0.0, buf)
    pour(32.9, 34.95, 300, 2600, 7, 0.38, 0.1, buf)
    pour(35.15, 35.5, 1500, 5000, 20, 0.10, 0.0, buf)
    # botanicals rain
    for k in range(70):
        place(buf, plink(rng.uniform(2500, 5000), 0.05), rng.uniform(9.35, 11.0), rng.uniform(0.03, 0.08), rng.uniform(-0.4, 0.4))
    # dropper squeezes + drops
    for name, (col, ch, _) in TL.DROPPERS.items():
        s = [x for x in TL.SHOTS if x[0] == name][0]
        place(buf, bandpass(rng.standard_normal(int(0.25 * SR)), 200, 900) * np.hanning(int(0.25 * SR)), s[1] + 0.45, 0.10)
        tk = s[1] + 0.55
        while tk < s[2] - 0.2:
            place(buf, plink(rng.uniform(700, 1300), 0.2), tk, 0.18, rng.uniform(-0.2, 0.2))
            tk += rng.uniform(0.35, 0.7)
    # yellow pipette trickle
    pour(24.75, 26.8, 1800, 6000, 30, 0.06, 0.0, buf)
    # cardamom pods plop
    subs = [26.95, 27.55, 28.283, 28.717, 29.25]
    for a, b in zip(subs, subs[1:]):
        tp = a + (b - a) * 0.35 + 0.18
        place(buf, plink(520, 0.3), tp, 0.35)
        place(buf, glass_tap(3100, 0.25), tp - 0.15, 0.04)
    # pump placement, crimp clicks
    place(buf, glass_tap(1500, 0.4), 35.93, 0.12)
    for tc in (36.22, 36.27):
        click = bandpass(rng.standard_normal(int(0.04 * SR)), 2000, 9000) * np.exp(-np.arange(int(0.04 * SR)) / SR * 120)
        place(buf, click, tc, 0.5)
    # whooshes on hand-held cuts
    for tw, d, g in [(36.45, 0.3, 0.18), (37.0, 0.35, 0.22), (38.45, 0.35, 0.18), (40.62, 0.9, 0.35), (42.30, 0.3, 0.2)]:
        place(buf, whoosh(d), tw, g)
    # liquid slosh during the shake
    pour(37.2, 38.5, 250, 1500, 5, 0.18, 0.0, buf)
    # atomizer
    tsp = 43.017 + 0.28
    n = int(0.42 * SR)
    psst = bandpass(rng.standard_normal(n), 3500, 12000) * np.minimum(1, np.arange(n) / (0.01 * SR)) * np.exp(-np.arange(n) / SR * 5)
    place(buf, psst, tsp, 0.32)
    place(buf, bandpass(rng.standard_normal(int(0.03 * SR)), 1500, 6000), tsp - 0.04, 0.3)
    # master: gentle compression via tanh, normalise to -1 dBFS
    buf = np.tanh(buf * 1.6) / 1.6
    buf *= 0.89 / np.abs(buf).max()
    pcm = (buf.T * 32767).astype(np.int16)
    import wave
    with wave.open(out_wav, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('wrote', out_wav)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports', 'NOX_OMBRE_NOMADE_audio.wav'))
