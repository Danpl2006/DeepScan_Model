"""
Чтение сырых данных Hydra/HyScan (.d/.i/.prm) БЕЗ программы HyScan.

Что установлено по присланным файлам:
  * .i : заголовок 16 байт ('HSIX1701'), далее записи по 24 байта = 6 x uint32:
         [0]=?  [1]=время (1 ед. ~ 1 мкс)  [2]=0  [3]=смещение в .d  [4]=0  [5]=размер
  * ss-*  : комплексный сигнал, пары int16 (I,Q), центр около 8192 (14-битный АЦП)
  * длина записи (число сэмплов) МЕНЯЕТСЯ по ходу заезда (меняли дальность)
  * файл канала режется на части по ~1 ГиБ: ss-port_000000, ss-port_000001, ...
    номер пинга в метках (/index) -- сквозной по всем частям (проверяется скриптом check_marks.py)
  * метки: waterfall-mark.prm, /index = пинг, /count = номер комплексного сэмпла,
    /width,/height = ПОЛУразмеры рамки в метрах (поперёк / вдоль курса)
"""
import os, re, struct
from dataclasses import dataclass
import numpy as np

C_WATER = 1500.0           # м/с, как в мануале
FS_SS = 223214.28571428571  # Гц, из track.prm (/data/rate)
ADC_CENTER = 8192.0


def read_index(i_path):
    """-> массив (N,6) uint32: колонки offset=3, size=5, time=1."""
    with open(i_path, 'rb') as f:
        b = f.read()
    if b[:8] != b'HSIX1701':
        raise ValueError(f'{i_path}: неожиданный заголовок {b[:8]!r}')
    n = (len(b) - 16) // 24
    return np.frombuffer(b[16:16 + n * 24], dtype='<u4').reshape(n, 6)


@dataclass
class ChannelPart:
    d_path: str
    index: np.ndarray        # (n,6)
    @property
    def n(self):
        return len(self.index)


class Channel:
    """Один канал (например 'ss-starboard') со всеми частями _000000, _000001..."""
    def __init__(self, parts):
        self.parts = parts
        self.starts = np.cumsum([0] + [p.n for p in parts])   # глобальный номер первого пинга части

    @property
    def n_pings(self):
        return int(self.starts[-1])

    def _locate(self, ping):
        k = int(np.searchsorted(self.starts, ping, side='right') - 1)
        return self.parts[k], ping - int(self.starts[k])

    def ping_complex(self, ping):
        """Один пинг -> комплексный вектор (n_samples,) complex64, центрированный."""
        part, j = self._locate(ping)
        off, size = int(part.index[j, 3]), int(part.index[j, 5])
        with open(part.d_path, 'rb') as f:
            f.seek(off)
            raw = f.read(size)
        x = np.frombuffer(raw[:len(raw) // 4 * 4], dtype='<i2').astype(np.float32) - ADC_CENTER
        return (x[0::2] + 1j * x[1::2]).astype(np.complex64)

    def ping_amp(self, ping):
        return np.abs(self.ping_complex(ping))

    def ping_len(self, ping):
        part, j = self._locate(ping)
        return int(part.index[j, 5]) // 4


def open_channel(folder, name):
    """folder: папка заезда (ищем и во вложенных); name: 'ss-starboard' или 'ss-port'.
    Подходят имена и ss-port.000000.i, и ss-port_000000.i."""
    pat = re.compile(r'^' + re.escape(name) + r'[._](\d+)\.i$')
    found = []
    for root, _, files in os.walk(folder):
        for fn in files:
            m = pat.match(fn)
            if m:
                found.append((int(m.group(1)), os.path.join(root, fn)))
    found.sort()
    parts = []
    for _, ip in found:
        dp = ip[:-2] + '.d'
        if not os.path.exists(dp):
            raise FileNotFoundError(f'нет {dp} (есть индекс {ip})')
        parts.append(ChannelPart(dp, read_index(ip)))
    if not parts:
        have = sorted({f for _, _, fs in os.walk(folder) for f in fs if f.endswith('.i')})[:12]
        raise FileNotFoundError(f'нет файлов {name}.NNNNNN.i / {name}_NNNNNN.i в {folder}. '
                                f'Найденные .i: {have}')
    return Channel(parts)


@dataclass
class Mark:
    name: str
    source: str       # 'ss-starboard' | 'ss-port'
    ping: int         # /index  (центр)
    sample: int       # /count  (центр, номер комплексного сэмпла)
    half_w_m: float   # поперёк курса, полуразмер, м
    half_h_m: float   # вдоль курса, полуразмер, м

    def sample_halfwidth(self, c=C_WATER, fs=FS_SS):
        return self.half_w_m / (c / (2 * fs))   # метры -> сэмплы


def parse_marks(path):
    marks = []
    for block in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        d = dict(l.split('=', 1) for l in block.splitlines() if '=' in l)
        if d.get('schema-id') != 'waterfall-mark':
            continue
        marks.append(Mark(d['/name'], d['/source'], int(d['/index']), int(d['/count']),
                          float(d['/width']), float(d['/height'])))
    return marks


def build_waterfall(ch, p0, p1, s_max, mode='log'):
    """Матрица (p1-p0, s_max) амплитуд; короткие пинги дополняются нулями (NaN-маска отдельно)."""
    p1 = min(p1, ch.n_pings)
    out = np.zeros((p1 - p0, s_max), np.float32)
    valid = np.zeros_like(out, dtype=bool)
    for k, p in enumerate(range(p0, p1)):
        a = ch.ping_amp(p)[:s_max]
        out[k, :len(a)] = a
        valid[k, :len(a)] = True
    if mode == 'log':
        out = np.log1p(out)
    return out, valid
