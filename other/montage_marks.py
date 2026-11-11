"""Склеивает dataset/diagnostics/Mark_*.png в одну картинку montage.png (для отправки).
Запуск из корня проекта:  python other/montage_marks.py"""
import os, re, glob, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = os.path.join(ROOT, 'dataset', 'diagnostics')
files = sorted(glob.glob(os.path.join(d, 'Mark_*.png')),
               key=lambda p: int(re.search(r'Mark_(\d+)', p).group(1)))
if not files:
    sys.exit(f'нет Mark_*.png в {d}')
cols = 6; rows = (len(files) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
for ax in axs.ravel(): ax.axis('off')
for ax, f in zip(axs.ravel(), files):
    ax.imshow(mpimg.imread(f))
plt.tight_layout(pad=0.2)
out = os.path.join(d, 'montage.png')
fig.savefig(out, dpi=110); print('готово:', out, f'({len(files)} меток)')
