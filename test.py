import numpy as np

# 定义长方体参数
length = 4.0  # 长
width = 2.0   # 宽
height = 3.0  # 高
center = np.array([0.0, 0.0, 0.0])  # 长方体中心坐标

# 定义线段端点
segment_start = np.array([-2.0, 1.0, 0.0])
segment_end = np.array([2.0, 1.0, 0.0])

# 计算线段和长方体六个面的交点
intersections = []

# 计算线段与长方体各个面的交点
for i in range(3):
    # 计算长方体表面的一半
    half_extent = np.array([length, width, height]) / 2.0

    # 计算长方体表面的两个顶点
    min_corner = center - half_extent
    max_corner = center + half_extent

    # 计算线段与长方体某一面的交点
    t1 = (min_corner[i] - segment_start[i]) / (segment_end[i] - segment_start[i])
    t2 = (max_corner[i] - segment_start[i]) / (segment_end[i] - segment_start[i])

    # 确保 t1 是较小的参数值，t2 是较大的参数值
    if t1 > t2:
        t1, t2 = t2, t1

    # 检查交点是否在线段上
    if t1 <= 1.0 and t2 >= 0.0:
        intersection = segment_start + t1 * (segment_end - segment_start)
        intersections.append(intersection)

# 确定交点所在的长方体表面
surface_vertices = []

for intersection in intersections:
    # 确定交点在长方体的哪个面上
    for i in range(3):
        if intersection[i] == center[i] - (length / 2.0):
            surface_vertices.append(center - np.array([0.0, 0.0, height / 2.0]))
        elif intersection[i] == center[i] + (length / 2.0):
            surface_vertices.append(center + np.array([0.0, 0.0, height / 2.0]))
        elif intersection[i] == center[i] - (width / 2.0):
            surface_vertices.append(center - np.array([0.0, length / 2.0, 0.0]))
        elif intersection[i] == center[i] + (width / 2.0):
            surface_vertices.append(center + np.array([0.0, length / 2.0, 0.0]))
        elif intersection[i] == center[i] - (height / 2.0):
            surface_vertices.append(center - np.array([width / 2.0, 0.0, 0.0]))
        elif intersection[i] == center[i] + (height / 2.0):
            surface_vertices.append(center + np.array([width / 2.0, 0.0, 0.0]))

# 打印交点和所在的长方体表面顶点
print("线段与长方体的交点：")
for intersection in intersections:
    print(intersection)

print("\n交点所在的长方体表面顶点：")
for vertex in surface_vertices:
    print(vertex)
