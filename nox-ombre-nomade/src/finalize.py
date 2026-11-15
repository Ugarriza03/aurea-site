"""Concatenate the per-shot renders, mux the sound design and export the Instagram master.

Output: 1080x1920, 30 fps CFR, H.264 High@4.2 yuv420p BT.709, AAC 48 kHz stereo,
loudness -14 LUFS / -1.5 dBTP, +faststart.
"""
import os
import glob
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'exports')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio


def run(cmd):
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    scenes = sorted(glob.glob(os.path.join(EXP, 'scenes', '*.mp4')))
    lst = os.path.join(ROOT, 'build', 'concat.txt')
    with open(lst, 'w') as f:
        for s in scenes:
            f.write(f"file '{s}'\n")
    wav = os.path.join(EXP, 'NOX_OMBRE_NOMADE_audio.wav')
    audio.main(wav)
    out = os.path.join(EXP, 'NOX_OMBRE_NOMADE.mp4')
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wav,
         '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-maxrate', '24M', '-bufsize', '48M',
         '-profile:v', 'high', '-level', '4.2', '-pix_fmt', 'yuv420p', '-r', '30', '-g', '30',
         '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
         '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
         '-shortest', '-movflags', '+faststart', out])
    os.makedirs('/exports', exist_ok=True)
    shutil.copy(out, '/exports/NOX_OMBRE_NOMADE.mp4')
    print('final:', out, 'and /exports/NOX_OMBRE_NOMADE.mp4')


if __name__ == '__main__':
    main()
