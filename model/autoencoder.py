"""Небольшой свёрточный автоэнкодер для патчей 64x64.
Идея: сеть учится восстанавливать ФОН. Узкое "горлышко" (latent) не даёт ей запомнить
всё подряд, поэтому незнакомые объекты восстанавливаются плохо -> большая ошибка."""
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.1))


def up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.1))


class ConvAE(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 ; горлышко = latent x 8 x 8 (при latent=8: 512 чисел вместо 4096)
        self.enc = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                 nn.Conv2d(64, latent, 1))
        self.dec = nn.Sequential(nn.Conv2d(latent, 64, 1), nn.LeakyReLU(0.1),
                                 up(64, 32), up(32, 16),
                                 nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):
        return self.dec(self.enc(x))
