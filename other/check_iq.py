from pathlib import Path  # Пути к данным и результатам.
import numpy as np  # Обработка числовых массивов.
import matplotlib.pyplot as plt  # Построение изображений.
from sonar_io.index_parser import read_index_file  # Индекс опорных сигналов.
from sonar_io.index_parser import read_track  # Индексы всех кусков.
from sonar_io.marks_parser import parse_waterfall_marks  # Метки HyScan.


ROOT = Path(__file__).resolve().parents[1]  # Корень проекта.
RAW = ROOT / "dataset" / "001"  # Папка существующей записи.
MARKS = Path(
    r"D:\HyScan\db\RealData\project.prm\waterfall-mark.prm"
)  # Оригинальный файл меток.
OUT = ROOT / "dataset" / "diagnostics"  # Папка результатов.
OUT.mkdir(parents=True, exist_ok=True)  # Создаём её, если нужно.

marks = parse_waterfall_marks(MARKS)  # Читаем метки.
mark = next(m for m in marks if m.name == "Mark #1")  # Берём бутылку.
records = read_track(RAW, mark.source)  # Читаем индекс нужного борта.

half_pings = 64  # По 64 пинга до и после метки.
half_samples = 512  # По 512 комплексных отсчётов в стороны.
lo = max(0, mark.ping - half_pings)  # Первый пинг окна.
hi = min(len(records), mark.ping + half_pings + 1)  # Конец окна.


signal_records = read_index_file(
    RAW / f"{mark.source}-signal.000000.i"
)  # Читаем индекс записанных опорных сигналов.

signal_cache = {}  # Не читаем один и тот же сигнал повторно.

def compress_ping(iq, timestamp_us):
    available = [
        rec for rec in signal_records
        if rec.timestamp_us <= timestamp_us
    ]  # Берём только сигналы, записанные не позже текущего пинга.

    if not available:
        raise ValueError("No reference signal before this ping")

    ref_rec = max(
        available, key=lambda rec: rec.timestamp_us
    )  # Находим последний доступный сигнал.

    if ref_rec.offset not in signal_cache:
        path = RAW / f"{mark.source}-signal.000000.d"
        with path.open("rb") as file:  # Исходный файл только читаем.
            file.seek(ref_rec.offset)
            payload = file.read(ref_rec.size)

        if len(payload) != ref_rec.size or ref_rec.size % 8:
            raise ValueError("Invalid complex-float32 reference")

        ref = np.frombuffer(
            payload, dtype="<c8"
        ).copy()  # Опорный сигнал уже записан как complex float32.

        energy = float(np.sum(np.abs(ref) ** 2))  # Энергия сигнала.
        if not np.isfinite(ref).all() or energy <= 0:
            raise ValueError("Invalid reference values")

        signal_cache[ref_rec.offset] = (ref, energy)
        print(f"Reference: {len(ref)} complex samples")

    ref, energy = signal_cache[ref_rec.offset]
    kernel = np.conj(ref[::-1])  # Разворот и сопряжение для корреляции.
    full_length = len(iq) + len(ref) - 1  # Длина полной свёртки.
    fft_length = 1 << (full_length - 1).bit_length()  # Размер FFT.

    convolution = np.fft.ifft(
        np.fft.fft(iq, fft_length)
        * np.fft.fft(kernel, fft_length)
    )  # Вычисляем свёртку через FFT.

    result = convolution[
        len(ref) - 1:len(ref) - 1 + len(iq)
    ] / energy  # Индекс результата соответствует началу сравниваемого эха.

    if len(ref) > 1:
        result[-(len(ref) - 1):] = np.nan  # Убираем неполные сравнения.

    return result


centers = [mark.sample, mark.sample // 2]  # Две трактовки координаты.
fig, axes = plt.subplots(2, 1, figsize=(13, 8))  # Два изображения.

for ax, center in zip(axes, centers):  # Проверяем оба варианта.
    start = max(0, center - half_samples)  # Левая граница окна.
    stop = center + half_samples  # Правая граница окна.
    image = np.full(
        (hi - lo, stop - start), np.nan, dtype=np.float32
    )  # Пустые области оставляем как NaN.

    for row, ping in enumerate(range(lo, hi)):  # Читаем пинги по очереди.
        rec = records[ping]  # Смещение и размер блока.
        path = RAW / f"{mark.source}.{rec.chunk:06d}.d"  # Нужный кусок.

        with path.open("rb") as file:  # Открываем только для чтения.
            file.seek(rec.offset)  # Переходим к началу блока.
            payload = file.read(rec.size)  # Читаем весь блок.

        if len(payload) != rec.size or rec.size % 4 != 0:
            raise ValueError(f"Unexpected block size at ping {ping}")

        words = np.frombuffer(payload, dtype="<u2")  # Читаем unsigned 16-bit.
        codes = words & 0x3FFF  # Оставляем только 14 бит, как делает HyScan.
        pairs = codes.reshape(-1, 2).astype(np.float32)  # Объединяем I и Q.
        pairs = pairs * (2.0 / 16383.0) - 1.0  # Декодируем обе составляющие.

        iq = pairs[:, 0] + 1j * pairs[
            :, 1]  # Собираем комплексный сигнал I + jQ.
        compressed = compress_ping(iq,
                                   rec.timestamp_us)  # Применяем опорный сигнал.
        amplitude = 20.0 * np.log10(
            np.maximum(np.abs(compressed), 1e-8)
        )  # Показываем амплитуду результата в децибелах.
        part = amplitude[start:stop]  # Вырезаем участок вокруг метки.
        image[row, :len(part)] = part  # Записываем его в строку картинки.


    valid = np.isfinite(image)  # Находим непустые пиксели.
    if not valid.any():  # Не строим полностью пустую картинку.
        ax.set_title(f"No data for center={center}")
        continue

    low, high = np.percentile(image[valid], [1, 99])  # Контраст окна.
    high = max(high, low + 1e-6)  # Защита от постоянных значений.

    ax.imshow(
        image, cmap="gray", aspect="auto",
        vmin=low, vmax=high, interpolation="nearest",
    )  # Показываем пробную амплитуду.

    ax.scatter(
        [center - start], [mark.ping - lo],
        s=120, facecolors="none", edgecolors="red",
    )  # Отмечаем проверяемую координату.

    ax.set_title(
        f"Mark #1 | ping={mark.ping} | complex sample={center} "
        "| HyScan ADC14 decode"
    )  # Подписываем вариант.
    ax.set_xlabel("Complex sample within crop")  # Горизонталь — отсчёты.
    ax.set_ylabel("Ping within crop")  # Вертикаль — пинги.

fig.tight_layout()  # Размещаем подписи без наложения.
result = OUT / "mark1_matched_filter.png"  # Отдельная картинка сравнения.
fig.savefig(result, dpi=150)  # Сохраняем картинку.
plt.close(fig)  # Закрываем рисунок.
print(f"SAVED {result}")  # Выводим путь.


