from data_generation.terrain import generate_seabed_heightmap, heightmap_to_pointcloud

hm = generate_seabed_heightmap(width=100, height=100, scale=30, octaves=4, base_depth=-10.0)
points = heightmap_to_pointcloud(hm, x_spacing=1.0, y_spacing=1.0)

print(points.shape)   # ожидаем (10000, 3)
print(points[:5])     # первые 5 точек — глянуть, что значения Z осмысленные (около -10)