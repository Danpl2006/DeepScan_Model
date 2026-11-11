"""
Проверка: попадают ли метки HyScan на реальные объекты в наших данных.
Запуск из корня проекта (пути по умолчанию подходят под структуру DeepScan_Model):
    python other/check_marks.py
Для каждой метки сохраняет PNG-вырез вокруг неё с нарисованной рамкой в dataset/diagnostics.
Объект должен быть внутри красной рамки.
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sonar_io.reader import open_channel, parse_marks, C_WATER, FS_SS

M_PER_SAMPLE = C_WATER / (2 * FS_SS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default=os.path.join(ROOT, 'dataset', '001'))
    ap.add_argument('--marks', default=os.path.join(ROOT, 'dataset', 'project.prm', 'waterfall-mark.prm'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'dataset', 'diagnostics'))
    ap.add_argument('--ping_m', type=float, default=0.18, help='шаг между пингами вдоль курса, м (оценка)')
    a = ap.parse_args()
    marks = parse_marks(a.marks)
    os.makedirs(a.out, exist_ok=True)
    chans = {}
    for m in marks:
        if m.source not in chans:
            try:
                chans[m.source] = open_channel(a.track, m.source)
            except FileNotFoundError as e:
                print(f'ПРОПУСК {m.name}: {e}')
                chans[m.source] = None
        ch = chans[m.source]
        if ch is None:
            continue
        if m.ping >= ch.n_pings:
            print(f'ПРОПУСК {m.name}: пинг {m.ping}, а в {m.source} найдено только {ch.n_pings} пингов '
                  f'(частей: {len(ch.parts)}). Не хватает файлов _00000N.d/.i или номер пинга не сквозной.')
            continue
        hw = int(m.half_w_m / M_PER_SAMPLE); hh = int(m.half_h_m / a.ping_m)
        pad_s, pad_p = hw * 3, hh * 3
        p0, p1 = max(0, m.ping - hh - pad_p), min(ch.n_pings, m.ping + hh + pad_p)
        s0, s1 = max(0, m.sample - hw - pad_s), m.sample + hw + pad_s
        img = np.zeros((p1 - p0, s1 - s0), np.float32)
        for k, p in enumerate(range(p0, p1)):
            amp = ch.ping_amp(p)[s0:s1]; img[k, :len(amp)] = amp
        img = np.log1p(img)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, aspect='auto', cmap='copper', origin='lower',
                  extent=[s0, s1, p0, p1], vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
        ax.add_patch(plt.Rectangle((m.sample - hw, m.ping - hh), 2 * hw, 2 * hh, fill=False, ec='red'))
        ax.set_title(f'{m.name} {m.source} ping={m.ping}'); ax.set_xlabel('sample'); ax.set_ylabel('ping')
        fig.savefig(os.path.join(a.out, f"{m.name.replace(' #','_')}.png"), dpi=90); plt.close(fig)
        print('ok', m.name)

if __name__ == '__main__':
    main()
