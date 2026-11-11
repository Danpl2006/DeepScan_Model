"""
Предсказание и оценка. Запуск из корня:   python model/predict.py            (берёт последний run)
                                          python model/predict.py --run runs/20261110-120000 --q 99.7
Шаги:
 1. по всему заезду скользящим окном 64x64 считаем ошибку восстановления -> карта аномальности
 2. порог берём из ФОНА валидации (перцентиль q ошибки вне областей меток)
 3. области выше порога = подозрительные места (пинги от-до, сэмплы от-до, оценка)
 4. сверяем с метками: recall по меткам (все / "ясные"), ложные срабатывания на 1000 пингов
Результаты: runs/<run>/predict/ (detections.csv, results.txt, marks/Mark_N.png)
Важно: неразмеченные настоящие объекты считаются ложными срабатываниями (разметка неполная).
"""
import argparse, os, sys, json, csv, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy import ndimage
from sonar_io.reader import parse_marks
from data_generation.generate_dataset import roi_map, SIDES

CLEAR = {1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 29}     # метки с визуально ясным объектом (мой разбор картинок)


# ---------- карта ошибки (не зависит от torch: recon = любая функция (N,64,64)->(N,64,64)) ----------
def error_map(Z, valid, recon, patch=64, stride=16, col_min=23, batch=256):
    P, C = Z.shape
    esum = np.zeros((P, C), np.float32); cnt = np.zeros((P, C), np.float32)
    coords = [(p, c) for p in range(0, P - patch + 1, stride) for c in range(col_min, C - patch + 1, stride)
              if valid[p:p + patch, c:c + patch].mean() >= 0.95]
    for i in range(0, len(coords), batch):
        ch = coords[i:i + batch]
        xb = np.stack([Z[p:p + patch, c:c + patch] for p, c in ch]).astype(np.float32)
        e = (recon(xb) - xb) ** 2
        for (p, c), ee in zip(ch, e):
            esum[p:p + patch, c:c + patch] += ee; cnt[p:p + patch, c:c + patch] += 1
    err = esum / np.maximum(cnt, 1)
    return err, cnt > 0


def detect(err_s, thr, roi, min_area=20):
    lab, n = ndimage.label(err_s > thr)
    dets = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i; area = int(m.sum())
        if area < min_area:
            continue
        ids = {int(k) for k in np.unique(roi[sl][m]) if k > 0}
        dets.append(dict(p0=sl[0].start, p1=sl[0].stop, c0=sl[1].start, c1=sl[1].stop, area=area,
                         score=float(err_s[sl][m].max() / thr), marks=ids))
    return dets


def evaluate(dets_by_side, marks, info, n_pings):
    """recall по меткам (по выборкам) и ложные срабатывания на 1000 пингов (test+val)."""
    block = info['block']; split_of = {int(k): v for k, v in info['split_blocks'].items()}
    hit = set(); fp = {'val': 0, 'test': 0}
    for side, dets in dets_by_side.items():
        for d in dets:
            if d['marks']:
                hit |= d['marks']
            else:
                sp = split_of.get(((d['p0'] + d['p1']) // 2) // block, 'train')
                if sp in fp: fp[sp] += 1
    res = {}
    for sp in ['test', 'val']:
        ks = [k for k, m in enumerate(marks, 1) if split_of.get(m.ping // block) == sp]
        cl = [k for k in ks if k in CLEAR]
        npg = sum(1 for b, s in split_of.items() if s == sp) * block
        res[sp] = dict(marks=len(ks), hit=len([k for k in ks if k in hit]), clear=len(cl),
                       clear_hit=len([k for k in cl if k in hit]), fp=fp[sp],
                       fp_per_1000=fp[sp] / max(npg * 2, 1) * 1000)   # на 1000 пингов одного борта
    return res, hit


def load_recon(run_dir):
    import torch
    from model.autoencoder import ConvAE
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    m = ConvAE(cfg['latent']); m.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu')); m.eval()
    def f(xb):
        with torch.no_grad():
            return m(torch.from_numpy(xb)[:, None])[:, 0].numpy()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    ap.add_argument('--patches', default=os.path.join(ROOT, 'dataset', 'patches'))
    ap.add_argument('--marks', default=os.path.join(ROOT, 'dataset', 'project.prm', 'waterfall-mark.prm'))
    ap.add_argument('--q', type=float, default=99.7, help='перцентиль ошибки на фоне val для порога')
    ap.add_argument('--min_area', type=int, default=20, help='минимум пикселей в области')
    ap.add_argument('--no_figs', action='store_true')
    ap.add_argument('--fake', action='store_true', help='тест без torch: простая имитация автоэнкодера')
    a = ap.parse_args()
    if a.fake:
        run = a.run or os.path.join(ROOT, 'runs', 'fake')
        recon = lambda xb: ndimage.uniform_filter(xb, size=(1, 9, 9))
    else:
        run = a.run or sorted(glob.glob(os.path.join(ROOT, 'runs', '2*')))[-1]
        recon = load_recon(run)
    out = os.path.join(run, 'predict'); os.makedirs(os.path.join(out, 'marks'), exist_ok=True)
    info = json.load(open(os.path.join(a.patches, 'split_info.json')))
    pool, block, shift = info['pool'], info['block'], info['shift']
    col_min = int(np.ceil(1100 / pool))
    marks = parse_marks(a.marks)
    split_of = {int(k): v for k, v in info['split_blocks'].items()}

    smooth, rois, valids, Zs = {}, {}, {}, {}
    for side in SIDES:
        Z = np.load(os.path.join(a.patches, 'cache', f'{side}_norm.npy')).astype(np.float32)
        valid = np.load(os.path.join(a.patches, 'cache', f'{side}_pool{pool}.npz'))['valid']
        print('карта ошибки:', side, Z.shape, flush=True)
        err, cov = error_map(Z, valid, recon, col_min=col_min)
        smooth[side] = ndimage.uniform_filter(err, size=(5, 5)) * cov
        rois[side] = roi_map(Z.shape, marks, side, pool, shift, 0.18)
        valids[side] = cov; Zs[side] = Z
    n_pings = {s: smooth[s].shape[0] for s in SIDES}

    # ---- порог по фону валидации (вне областей меток) ----
    def calib(sp):
        vals = []
        for s in SIDES:
            blk = np.arange(n_pings[s]) // block
            rows = np.array([split_of.get(int(b)) == sp for b in blk])
            mask = valids[s] & (rois[s] == 0) & rows[:, None]
            vals.append(smooth[s][mask])
        return np.concatenate(vals)
    cal = calib('val')
    if cal.size < 1000:
        cal = calib('train'); print('ВНИМАНИЕ: фона в val мало, порог взят по train (завышает качество)')
    print('пикселей фона для порога:', cal.size)

    lines = []
    def say(t=''):
        print(t); lines.append(t)
    say(f'run: {run}')
    say(f'{"q":>6} {"порог":>8} | test: меток/найдено  ясных/найдено  FP/1000пингов | val: меток/найдено  FP/1000')
    best = None
    for q in [99.0, 99.5, 99.7, 99.9, 99.95]:
        thr = float(np.percentile(cal, q))
        dets = {s: detect(smooth[s], thr, rois[s], a.min_area) for s in SIDES}
        r, hit = evaluate(dets, marks, info, n_pings)
        t, v = r['test'], r['val']
        say(f'{q:6.2f} {thr:8.4f} | test: {t["marks"]:3d}/{t["hit"]:<3d}   {t["clear"]:3d}/{t["clear_hit"]:<3d}   {t["fp_per_1000"]:7.2f}       | val: {v["marks"]:3d}/{v["hit"]:<3d}   {v["fp_per_1000"]:7.2f}')
        if abs(q - a.q) < 1e-9:
            best = (thr, dets, r, hit)
    if best is None:
        thr = float(np.percentile(cal, a.q)); dets = {s: detect(smooth[s], thr, rois[s], a.min_area) for s in SIDES}
        r, hit = evaluate(dets, marks, info, n_pings); best = (thr, dets, r, hit)
    thr, dets, r, hit = best
    say(f'\nВыбран q={a.q}: порог {thr:.4f}; найденные метки: {sorted(hit)}')
    say('Не найдены: ' + str([k for k in range(1, len(marks) + 1) if k not in hit]))

    with open(os.path.join(out, 'detections.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['side', 'ping0', 'ping1', 'sample0', 'sample1', 'score', 'area_px', 'marks'])
        for s in SIDES:
            for d in sorted(dets[s], key=lambda d: -d['score']):
                w.writerow([s, d['p0'], d['p1'], d['c0'] * pool, d['c1'] * pool, round(d['score'], 2), d['area'],
                            ' '.join(f'#{k}' for k in sorted(d['marks']))])
    open(os.path.join(out, 'results.txt'), 'w', encoding='utf-8').write('\n'.join(lines))

    if not a.no_figs:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        for k, m in enumerate(marks, 1):
            s = m.source; Z = Zs[s]; E = smooth[s]
            pc = m.ping; cc = int(m.sample / pool)
            p0, p1 = max(0, pc - 120), min(Z.shape[0], pc + 120); c0, c1 = max(0, cc - 90), min(Z.shape[1], cc + 90)
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            ax[0].imshow(Z[p0:p1, c0:c1], aspect='auto', origin='lower', cmap='copper', vmin=-0.5, vmax=1.5,
                         extent=[c0 * pool, c1 * pool, p0, p1]); ax[0].set_title(f'{m.name} {s}')
            ax[1].imshow(E[p0:p1, c0:c1], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=thr * 3,
                         extent=[c0 * pool, c1 * pool, p0, p1]); ax[1].set_title(f'ошибка (порог {thr:.3f})')
            rr = rois[s][p0:p1, c0:c1] == k
            for A in ax:
                if rr.any():
                    ys, xs = np.nonzero(rr)
                    A.add_patch(plt.Rectangle(((c0 + xs.min()) * pool, p0 + ys.min()), (xs.max() - xs.min()) * pool,
                                              ys.max() - ys.min(), fill=False, ec='red'))
                for d in dets[s]:
                    if d['p1'] > p0 and d['p0'] < p1 and d['c1'] > c0 and d['c0'] < c1:
                        A.add_patch(plt.Rectangle((d['c0'] * pool, d['p0']), (d['c1'] - d['c0']) * pool, d['p1'] - d['p0'],
                                                  fill=False, ec='cyan', ls='--'))
            fig.savefig(os.path.join(out, 'marks', f'Mark_{k}.png'), dpi=80); plt.close(fig)
    print('готово:', out)


if __name__ == '__main__':
    main()
