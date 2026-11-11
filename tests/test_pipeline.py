"""Быстрая проверка размеров и прохода модели. Запуск из корня:  python tests/test_pipeline.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from model.autoencoder import ConvAE
from model.dataloader import PatchDataset

m = ConvAE(8)
x = torch.randn(4, 1, 64, 64)
y = m(x)
assert y.shape == x.shape, y.shape
print('форма выхода ок:', tuple(y.shape), '| параметров:', sum(p.numel() for p in m.parameters()))
loss = torch.nn.functional.mse_loss(y, x); loss.backward()
assert torch.isfinite(loss); print('обратный проход ок, loss =', round(loss.item(), 3))
p = os.path.join(ROOT, 'dataset', 'patches', 'train.npz')
if os.path.exists(p):
    ds = PatchDataset(p, augment=True)
    assert ds[0].shape == (1, 64, 64); print('датасет ок:', len(ds), 'патчей')
else:
    print('train.npz нет — датасет пропущен')
print('ВСЁ ОК')
