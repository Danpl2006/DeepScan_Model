"""
Нарезка патчей для автоэнкодера. Запуск из корня проекта:
    python data_generation/generate_dataset.py
Шаги:
 1. читаем ВСЕ пинги каналов ss-port / ss-starboard, усредняем по --pool сэмплов (кэшируем в dataset/patches/cache)
 2. нормализация: log, поправка каждого пинга (скачки усиления TVG), вычитание медленного фона по дальности
 3. области меток (ROI): рамка HyScan + запас + сдвиг --shift сэмплов вправо (измеренный сдвиг ~+480)
 4. патчи PATCH x PATCH; разбиение по БЛОКАМ пингов:
      train = блоки без меток, val/test = блоки с метками (соседние блоки = один кластер, кластеры не смешиваются)
 5. сохраняем dataset/patches/{train,val,test}.npz  и split_info.json
Порт и правый борт НЕ отражаем: тень лежит в сторону большей дальности на обоих бортах.
"""
import argparse, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from sonar_io.reader import open_channel, parse_marks, C_WATER, FS_SS

M_PER_SAMPLE = C_WATER / (2 * FS_SS)
SIDES = ['ss-port', 'ss-starboard']


def build_raw(ch, pool):
    sizes = np.concatenate([p.index[:, 5] for p in ch.parts]) // 4
    cols = int(sizes.max()) // pool
    amp = np.zeros((ch.n_pings, cols), np.float32); valid = np.zeros(amp.shape, bool)
    for p in range(ch.n_pings):
        a = ch.ping_amp(p); n = len(a) // pool
        if n:
            amp[p, :n] = a[:n * pool].reshape(n, pool).mean(1); valid[p, :n] = True
        if p % 2000 == 0:
            print(f'   пинг {p}/{ch.n_pings}', flush=True)
    return amp, valid


def roll_sum(a, win):
    h = win // 2
    c = np.cumsum(np.pad(a, ((h + 1, h), (0, 0))), axis=0)
    return c[win:] - c[:a.shape[0]]


def normalize(amp, valid, col_min, win=301):
    L = np.log1p(amp); L[~valid] = np.nan
    med = np.nanmedian(L[:, col_min:], axis=1)
    mad = np.nanmedian(np.abs(L[:, col_min:] - med[:, None]), axis=1) + 1e-3
    L = (L - med[:, None]) / mad[:, None]                      # убрали скачки усиления между пингами
    ok = ~np.isnan(L); x = np.nan_to_num(L)
    cnt = np.maximum(roll_sum(ok.astype(np.float32), win), 1)
    bg = roll_sum(x, win) / cnt
    sd = np.sqrt(np.maximum(roll_sum(x * x, win) / cnt - bg ** 2, 0)) + 1e-2   # медленный фон по дальности
    Z = np.where(ok, (x - bg) / sd, 0.0)
    return np.clip(Z, -8, 8).astype(np.float32) / 4.0


def roi_map(shape, marks, side, pool, shift, ping_m, margin=0.2):
    roi = np.zeros(shape, np.int16)
    for k, m in enumerate(marks, 1):
        if m.source != side:
            continue
        hw = m.half_w_m / M_PER_SAMPLE; hh = m.half_h_m / ping_m
        s0 = m.sample - hw * (1 + margin); s1 = m.sample + hw * (1 + margin) + shift
        p0 = m.ping - hh * (1 + margin) - 3; p1 = m.ping + hh * (1 + margin) + 3
        roi[max(0, int(p0)):max(0, int(p1)) + 1, max(0, int(s0 / pool)):max(0, int(s1 / pool)) + 1] = k
    return roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default=os.path.join(ROOT, 'dataset', '001'))
    ap.add_argument('--marks', default=os.path.join(ROOT, 'dataset', 'project.prm', 'waterfall-mark.prm'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'dataset', 'patches'))
    ap.add_argument('--pool', type=int, default=48, help='сэмплов в одном пикселе (48 ~ 0.16 м ~ шаг пингов)')
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--block', type=int, default=1000, help='размер блока пингов для разбиения')
    ap.add_argument('--col_min_samples', type=int, default=1100, help='отрезаем зону под катером (дно/надир)')
    ap.add_argument('--shift', type=int, default=480, help='сдвиг рамок вправо, сэмплов (измерено estimate_offset)')
    ap.add_argument('--ping_m', type=float, default=0.18)
    ap.add_argument('--test_clusters', default='0,1,3', help='номера кластеров меток для теста, остальные -> val')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, 'cache'), exist_ok=True)
    marks = parse_marks(a.marks)
    col_min = int(np.ceil(a.col_min_samples / a.pool))
    norm, rois = {}, {}
    for side in SIDES:
        cache = os.path.join(a.out, 'cache', f'{side}_pool{a.pool}.npz')
        if os.path.exists(cache) and not a.rebuild:
            z = np.load(cache); amp, valid = z['amp'], z['valid']; print('кэш:', cache)
        else:
            print('читаю', side); amp, valid = build_raw(open_channel(a.track, side), a.pool)
            np.savez_compressed(cache, amp=amp, valid=valid)
        norm[side] = (normalize(amp, valid, col_min), valid)
        rois[side] = roi_map(amp.shape, marks, side, a.pool, a.shift, a.ping_m)
        np.save(os.path.join(a.out, 'cache', f'{side}_norm.npy'), norm[side][0].astype(np.float16))

    # --- разбиение блоков по кластерам меток ---
    nb = max(n[0].shape[0] for n in norm.values()) // a.block + 1
    marked = sorted({b for s in SIDES for b in np.unique(np.nonzero(rois[s])[0] // a.block)})
    clusters, cur = [], []
    for b in marked:
        if cur and b - cur[-1] > 1:
            clusters.append(cur); cur = []
        cur.append(b)
    if cur: clusters.append(cur)
    test_ids = {int(x) for x in a.test_clusters.split(',') if x.strip()}
    split_of = {}
    for i, cl in enumerate(clusters):
        for b in cl: split_of[b] = 'test' if i in test_ids else 'val'
    unmarked = [b for b in range(nb) if b not in split_of]
    for j, b in enumerate(unmarked):
        split_of[b] = 'val' if j % 6 == 5 else 'train'

    out = {s: dict(x=[], y=[], meta=[]) for s in ['train', 'val', 'test']}
    P = a.patch
    for si, side in enumerate(SIDES):
        Z, valid = norm[side]; roi = rois[side]
        for p0 in range(0, Z.shape[0] - P + 1, a.stride):
            b = p0 // a.block
            if (p0 + P - 1) // a.block != b:
                continue                                     # патч не пересекает границу блоков
            sp = split_of[b]
            for c0 in range(col_min, Z.shape[1] - P + 1, a.stride):
                if valid[p0:p0 + P, c0:c0 + P].mean() < 0.95:
                    continue
                r = roi[p0:p0 + P, c0:c0 + P]
                if sp == 'train' and r.any():
                    continue
                ids = r[r > 0]
                mid = int(np.bincount(ids).argmax()) if ids.size else 0
                out[sp]['x'].append(Z[p0:p0 + P, c0:c0 + P].astype(np.float16))
                out[sp]['y'].append((r > 0).astype(np.uint8))
                out[sp]['meta'].append([si, p0, c0, mid])
    info = dict(pool=a.pool, patch=P, stride=a.stride, block=a.block, shift=a.shift,
                clusters=[[int(b) for b in cl] for cl in clusters], split_blocks={int(k): v for k, v in split_of.items()})
    for sp, d in out.items():
        if not d['x']:
            print(sp, ': пусто'); continue
        np.savez_compressed(os.path.join(a.out, f'{sp}.npz'), x=np.stack(d['x']), y=np.stack(d['y']),
                            meta=np.array(d['meta'], np.int32))
        npos = sum(1 for m in d['meta'] if m[3] > 0)
        mk = sorted({m[3] for m in d['meta'] if m[3] > 0})
        info[sp] = dict(patches=len(d['x']), positive=npos, marks=[marks[k - 1].name for k in mk])
        print(f'{sp:5s}: {len(d["x"]):6d} патчей, с метками {npos:4d}, меток {len(mk)}')
    json.dump(info, open(os.path.join(a.out, 'split_info.json'), 'w'), ensure_ascii=False, indent=1)
    print('кластеры блоков с метками:', info['clusters'])
    print('метки по разбиению:', {k: v.get('marks') for k, v in info.items() if isinstance(v, dict) and 'marks' in v})

if __name__ == '__main__':
    main()
