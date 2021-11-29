import matplotlib.pyplot as plt
import math
import numpy as np


# Расстояние между двумя точками на плоскости
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# n -  количество точек
# Генерация случайных точек
def random_points(n):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return [x, y]


# Cluster - количество центроидов
# Генерация центроидов
def centroid(x, y, cluster):
    x_c = np.mean(x)
    y_c = np.mean(y)
    radius = dist(x_c, y_c, x[0], y[0])
    for i in range(n):
        if dist(x_c, y_c, x[i], y[i]) > radius:
            radius = dist(x_c, y_c, x[i], y[i])
    x_cntr, y_cntr = [], []
    for i in range(cluster):
        x_cntr.append(x_c + radius * np.cos(2 * i * np.pi / cluster))
        y_cntr.append(y_c + radius * np.sin(2 * i * np.pi / cluster))

    return [(x_cntr[i], y_cntr[i]) for i in range(len(x_cntr))]


# Построение словаря соответствия точек центроидам. Ключ - центроид, значене - массив точек
def build_centroids_dict(x, y, current_centroids):
    centroid_dict = {}
    for centroid in current_centroids:
        centroid_dict[(centroid[0], centroid[1])] = []

    for i in range(n):
        point = (x[i], y[i])
        nearest_centroid = current_centroids[
            np.argmin(
                [
                    dist(point[0], point[1], centroid[0], centroid[1])
                    for centroid in current_centroids
                ]
            )
        ]
        centroid_dict[(nearest_centroid[0], nearest_centroid[1])].append(point)

    return centroid_dict


# Шаг алгоритма - строим соответствие точек центроидам, генерируем новые центроиды, равные среднему по всем точкам
# в каждом из предыдущих, и строим новое соответствие
def expectation_maximization(x, y, current_centroids):
    centroid_dict = build_centroids_dict(x, y, current_centroids)
    new_centroids = np.array(
        [np.array(points).mean(axis=0) for points in centroid_dict.values()]
    )
    new_centroids_dict = build_centroids_dict(x, y, new_centroids)

    return new_centroids, new_centroids_dict


# Выполняем кластеризацию по cluster_count кластерам, запуская шаги алгоритма до тех пор, пока результаты не перестанут меняться
def clusterize(x, y, cluster_count, should_draw=False, colors=None):
    if colors is None:
        colors = []
    initial_centroids = centroid(x, y, cluster_count)
    step = expectation_maximization(x, y, initial_centroids)
    if should_draw:
        draw(step, colors)
    while True:
        next_step = expectation_maximization(x, y, step[0])
        if should_draw:
            draw(step, colors)
        if np.array_equal(step[0], next_step[0]):
            break
        else:
            step = next_step
    return step


# Считаем величину ошибки как сумму квадратов расстояний от точек до каждого из центроидов
def evaluate_error(centroid_dict: dict):
    error_metric = 0
    for centroid, points in centroid_dict.items():
        error_metric += np.sum(
            [dist(centroid[0], centroid[1], p[0], p[1]) ** 2 for p in points]
        )
    return error_metric


# Выполняем кластеризацию для каждого из значений boundaries, возвращая наиболее эффективный результат
def k_means_clusterization(x, y, boundaries=range(1, 11)):
    possible_clusterizations = {}
    for cluster_count in boundaries:
        possible_clusterizations[cluster_count] = clusterize(x, y, cluster_count)

    if len(boundaries) == 1:
        return possible_clusterizations[boundaries[0]][0]
    elif len(boundaries) == 2:
        first_error = evaluate_error(possible_clusterizations[boundaries[0]][1])
        second_error = evaluate_error(possible_clusterizations[boundaries[1]][1])
        if first_error > second_error:
            return possible_clusterizations[boundaries[1]][0]
        else:
            return possible_clusterizations[boundaries[0]][0]
    else:
        clusterizations = [
            possible_clusterizations[boundaries[0]],
            possible_clusterizations[boundaries[1]],
            possible_clusterizations[boundaries[2]],
        ]
        min_k_arg = -1
        min_k_value = -1
        for k in range(1, len(boundaries) - 1):
            if k > 1:
                clusterizations = clusterizations[1:] + [
                    possible_clusterizations[boundaries[k + 1]]
                ]

            mapped = [
                evaluate_error(clusterization[1]) for clusterization in clusterizations
            ]
            dk = abs(mapped[1] - mapped[2]) / abs(mapped[0] - mapped[1])
            if min_k_value == -1 or min_k_value > dk:
                min_k_value = dk
                min_k_arg = k

        return possible_clusterizations[boundaries[min_k_arg - 1]], min_k_arg


def draw(clusterization_result, colors):
    index = 0
    for centroid, points in clusterization_result[1].items():
        points_array = np.array(points)
        points_x = points_array[:, 0]
        points_y = points_array[:, 1]
        plt.scatter(points_x, points_y, color=colors[index])
        index += 1
    plt.scatter(
        clusterization_result[0][:, 0], clusterization_result[0][:, 1], color="r"
    )
    plt.show()


if __name__ == "__main__":
    n = 1000
    [x, y] = random_points(n)
    clusterized = k_means_clusterization(x, y)
    print(clusterized[1])
    colors = [np.random.rand(clusterized[1]) for i in range(clusterized[1])]
    clusterize(x, y, clusterized[1], True, colors)
