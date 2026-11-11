"""Загрузка патчей (dataset/patches/*.npz) для PyTorch.
Каждый патч: (1, 64, 64) float32. Ось 0 патча = пинги (вдоль курса), ось 1 = дальность."""
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, npz_path, background_only=False, augment=False):
        z = np.load(npz_path)
        self.x, self.y, self.meta = z['x'], z['y'], z['meta']
        if background_only:                      # оставляем только патчи БЕЗ областей меток
            keep = self.y.reshape(len(self.y), -1).sum(1) == 0
            self.x, self.y, self.meta = self.x[keep], self.y[keep], self.meta[keep]
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i].astype(np.float32))[None]       # (1,64,64)
        if self.augment:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])      # зеркало по пингам (можно: направление курса не важно)
            # по дальности НЕ зеркалим: тень всегда лежит в сторону большей дальности
            x = x + 0.02 * torch.randn_like(x)   # лёгкий шум
        return x
