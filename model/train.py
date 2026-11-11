"""Обучение автоэнкодера на ФОНЕ. Запуск из корня:  python model/train.py
Результат: runs/<дата-время>/{model.pt, config.json, losses.csv}"""
import argparse, os, sys, json, time, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.dataloader import PatchDataset
from model.autoencoder import ConvAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'dataset', 'patches'))
    ap.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--latent', type=int, default=8)
    ap.add_argument('--patience', type=int, default=8, help='останов, если val не улучшается столько эпох')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    tr = PatchDataset(os.path.join(a.data, 'train.npz'), augment=True)
    va = PatchDataset(os.path.join(a.data, 'val.npz'), background_only=True)
    if len(va) == 0:
        sys.exit('в val нет фоновых патчей — проверь generate_dataset.py')
    print(f'train: {len(tr)} патчей, val-фон: {len(va)} патчей')
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=256)

    model = ConvAE(a.latent)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = torch.nn.MSELoss()
    print('параметров:', sum(p.numel() for p in model.parameters()))

    # для ориентира: ошибка "модели-пустышки", которая всегда выдаёт 0 (среднее нормализованного фона)
    with torch.no_grad():
        base = float(np.mean([(x ** 2).mean().item() for x in dl_va]))
    print(f'ошибка пустышки (все нули) на val: {base:.4f}  <- модель должна быть заметно ниже')

    run = os.path.join(a.runs, time.strftime('%Y%m%d-%H%M%S')); os.makedirs(run)
    json.dump(vars(a) | {'val_baseline': base}, open(os.path.join(run, 'config.json'), 'w'), indent=1)
    log = open(os.path.join(run, 'losses.csv'), 'w', newline=''); w = csv.writer(log); w.writerow(['epoch', 'train', 'val'])
    best, bad = 1e9, 0
    for ep in range(1, a.epochs + 1):
        model.train(); s = 0; n = 0
        for x in dl_tr:
            opt.zero_grad(); loss = lossf(model(x), x); loss.backward(); opt.step()
            s += loss.item() * len(x); n += len(x)
        model.eval(); v = 0; m = 0
        with torch.no_grad():
            for x in dl_va:
                v += lossf(model(x), x).item() * len(x); m += len(x)
        tr_l, va_l = s / n, v / m
        w.writerow([ep, tr_l, va_l]); log.flush()
        print(f'эпоха {ep:3d}  train {tr_l:.4f}  val {va_l:.4f}')
        if va_l < best - 1e-5:
            best, bad = va_l, 0; torch.save(model.state_dict(), os.path.join(run, 'model.pt'))
        else:
            bad += 1
            if bad >= a.patience:
                print('раннее выключение'); break
    print('лучший val:', round(best, 4), '| папка:', run)


if __name__ == '__main__':
    main()
