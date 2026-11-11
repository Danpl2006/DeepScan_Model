"""
Оценка сдвига между рамками HyScan и реальными яркими объектами.
Запуск из корня проекта:  python other/estimate_offset.py
Для каждой метки: нормализует каждый пинг (убирает скачки усиления), ищет яркое пятно
в расширенной рамке и печатает смещение центра пятна от центра метки.
Результат также сохраняется в dataset/diagnostics/offsets.csv
"""
import argparse, os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from sonar_io.reader import open_channel, parse_marks, C_WATER, FS_SS

M_PER_SAMPLE = C_WATER / (2 * FS_SS)


def box_filter(a, ky, kx):
    """Скользящее среднее ky x kx без scipy."""
    def one(x, k, ax):
        c = np.cumsum(np.pad(x, [(k // 2 + 1, k // 2) if i == ax else (0, 0) for i in range(2)], mode='edge'), axis=ax)
        n = x.shape[ax]
        hi = np.take(c, np.arange(k, k + n), axis=ax); lo = np.take(c, np.arange(0, n), axis=ax)
        return (hi - lo) / k
    return one(one(a, ky, 0), kx, 1)


def load_tile(ch, p0, p1, s0, s1):
    img = np.zeros((p1 - p0, s1 - s0), np.float32); valid = np.zeros(img.shape, bool)
    for k, p in enumerate(range(p0, p1)):
        a = ch.ping_amp(p)[s0:s1]; img[k, :len(a)] = a; valid[k, :len(a)] = True
    return img, valid


def estimate(ch, m, ping_m=0.18, grow=2.5):
    hw = max(1, int(m.half_w_m / M_PER_SAMPLE)); hh = max(1, int(m.half_h_m / ping_m))
    p0, p1 = max(0, m.ping - int(hh * grow * 2)), min(ch.n_pings, m.ping + int(hh * grow * 2))
    s0, s1 = max(0, m.sample - int(hw * grow * 2)), m.sample + int(hw * grow * 2)
    img, valid = load_tile(ch, p0, p1, s0, s1)
    img = np.log1p(img)
    # нормализация каждого пинга: вычесть медиану по валидным сэмплам, поделить на робастный разброс
    for k in range(img.shape[0]):
        v = img[k][valid[k]]
        if len(v) > 50:
            med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6
            img[k] = (img[k] - med) / mad
    img[~valid] = 0
    sm = box_filter(img, 3, 31)
    thr = np.percentile(sm[valid], 90) + 3 * np.std(sm[valid])
    # поиск только в расширенной рамке (x grow)
    cy, cx = m.ping - p0, m.sample - s0
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    zone = (np.abs(ys - cy) <= hh * grow) & (np.abs(xs - cx) <= hw * grow) & valid
    sel = zone & (sm > thr)
    if sel.sum() < 30:
        return None
    w = (sm[sel] - thr)
    my, mx = (ys[sel] * w).sum() / w.sum(), (xs[sel] * w).sum() / w.sum()
    return dict(name=m.name, source=m.source, sample=m.sample, ping=m.ping,
                dx=round(mx - cx), dy=round(my - cy), px=int(sel.sum()),
                dx_m=round((mx - cx) * M_PER_SAMPLE, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default=os.path.join(ROOT, 'dataset', '001'))
    ap.add_argument('--marks', default=os.path.join(ROOT, 'dataset', 'project.prm', 'waterfall-mark.prm'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'dataset', 'diagnostics', 'offsets.csv'))
    ap.add_argument('--ping_m', type=float, default=0.18)
    a = ap.parse_args()
    chans, rows = {}, []
    for m in parse_marks(a.marks):
        if m.source not in chans:
            chans[m.source] = open_channel(a.track, m.source)
        ch = chans[m.source]
        if m.ping >= ch.n_pings:
            print('ПРОПУСК', m.name); continue
        r = estimate(ch, m, a.ping_m)
        if r is None:
            print(f'{m.name:8s} {m.source:13s} sample={m.sample:6d}  пятно не найдено'); continue
        rows.append(r)
        print(f"{r['name']:8s} {r['source']:13s} sample={r['sample']:6d}  dx={r['dx']:+5d} сэмпл ({r['dx_m']:+.2f} м)  dy={r['dy']:+4d} пингов  пикс={r['px']}")
    if rows:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        dx = np.array([r['dx'] for r in rows]); dy = np.array([r['dy'] for r in rows]); s = np.array([r['sample'] for r in rows])
        print(f'\nМедиана dx = {np.median(dx):+.0f} сэмплов, dy = {np.median(dy):+.0f} пингов (n={len(rows)})')
        if len(rows) > 3:
            k = np.polyfit(s, dx, 1)
            print(f'Тренд dx от дальности: dx = {k[0]:+.4f}*sample {k[1]:+.0f}   (наклон ~0 -> постоянная задержка; >0 -> масштаб)')
        print('CSV:', a.out)

if __name__ == '__main__':
    main()
