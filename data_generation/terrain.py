import numpy as np
from noise import pnoise2


def generate_seabed_heightmap(width, height, scale, octaves, base_depth,
                               lacunarity=2.0, persistence=0.5, seed=0):
    """
    width, height   — размер сетки в точках (например 200x200)
    scale           — масштаб шума: чем больше, тем более растянутые/плавные холмы
    octaves         — сколько слоёв шума разной частоты наложить друг на друга
    base_depth      — средняя глубина, вокруг которой всё "колышется"
    lacunarity      — во сколько раз растёт частота с каждой октавой (обычно 2.0)
    persistence     — во сколько раз падает амплитуда с каждой октавой (обычно 0.5)
    seed            — смещение шума, чтобы разные сцены не были идентичны

    Возвращает: 2D numpy-массив shape (height, width) со значениями Z (глубина)
    """
    heightmap = np.zeros((height, width), dtype=np.float32)

    frequency = 1.0
    amplitude = 1.0
    total_amplitude = 0.0  # сумма всех амплитуд — нужна, чтобы потом нормализовать результат

    for octave in range(octaves):
        for i in range(height):
            for j in range(width):
                # делим индексы на scale — это "растягивает" координаты в пространстве шума:
                # без этого соседние ячейки сетки (i=0, i=1) были бы слишком далеко друг
                # от друга в пространстве шума, и рельеф выглядел бы как случайный "снег",
                # а не как плавный ландшафт
                x = (j / scale) * frequency
                y = (i / scale) * frequency
                noise_value = pnoise2(x, y, base=seed)
                heightmap[i, j] += noise_value * amplitude

        total_amplitude += amplitude
        amplitude *= persistence   # следующий слой (мельче деталями) влияет слабее
        frequency *= lacunarity    # следующий слой более "частый" (мелькие детали)

    # без этого шага итоговые значения могли бы выйти далеко за [-1, 1],
    # особенно при большом количестве октав
    heightmap /= total_amplitude

    heightmap += base_depth
    return heightmap


def heightmap_to_pointcloud(heightmap, x_spacing, y_spacing):
    """
    heightmap  — 2D массив глубин, как из generate_seabed_heightmap
    x_spacing, y_spacing — расстояние между соседними точками сетки в реальных единицах
                           (например метрах), а не в индексах массива

    Возвращает: массив точек shape (N, 3), где N = height * width, каждая строка [X, Y, Z]
    """
    height, width = heightmap.shape

    # meshgrid с indexing='ij' даёт нам две сетки индексов той же формы, что heightmap:
    # row_idx[i, j] = i,  col_idx[i, j] = j — то есть просто "координатная сетка" индексов
    row_idx, col_idx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    X = col_idx * x_spacing   # столбец j -> координата X
    Y = row_idx * y_spacing   # строка i -> координата Y
    Z = heightmap             # глубина уже посчитана

    # stack по последней оси даёт форму (height, width, 3), reshape "разворачивает"
    # 2D-сетку в плоский список точек (N, 3), что и нужно для облака точек
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return points