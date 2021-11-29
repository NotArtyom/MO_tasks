import math

import matplotlib.pyplot as plt
import numpy as np


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def random_points(n):
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    return [x, y]


def centroid(x, y, cluster, n):
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
    return [x_cntr, y_cntr]


def matrix(n, cluster, x, y, x_cntr, y_cntr, u, m):
    for i in range(n):
        sum = 0
        for j in range(cluster):
            u[i][j] = dist(x[i], y[i], x_cntr[j], y_cntr[j]) ** (2 / (1 - m))
            sum += u[i][j]
        for j in range(cluster):
            u[i][j] /= sum


def check(u_old, u_new, n, cluster, eps):
    for i in range(n):
        for j in range(cluster):
            if abs(u_old[i][j] - u_new[i][j]) > eps:
                return False
    return True


def recentr(
    cluster,
    n,
    m,
    x_cntr,
    y_cntr,
    u,
    x,
    y,
):
    for i in range(cluster):
        x_cntr[i] = 0
        y_cntr[i] = 0
        sum = 0
        for j in range(n):
            tmp = u[j][i] ** m
            x_cntr[i] += tmp * x[j]
            y_cntr[i] += tmp * y[j]
            sum += tmp
        x_cntr[i] /= sum
        y_cntr[i] /= sum


def show(cluster, x, y, u, n, x_cntr, y_cntr):
    colors_dots = {0: "b", 1: "g", 2: "y"}
    for i in range(n):
        color_ind = 0
        max = 0
        for j in range(cluster):
            if max < u[i][j]:
                color_ind = j
                max = u[i][j]
        plt.scatter(x[i], y[i], color=colors_dots[color_ind])
        plt.scatter(x_cntr[color_ind], y_cntr[color_ind], color="r")
    plt.show()


def c_means():
    # Задаем начальные значения
    m = 2
    eps = 0.01
    n = 100
    [x, y] = random_points(n)
    cluster = 3
    # подсчитываем начальные приближения центров
    [x_cntr, y_cntr] = centroid(x, y, cluster, n)

    plt.scatter(x, y, color="b")
    plt.scatter(x_cntr, y_cntr, color="r")
    plt.show()

    u_old = np.zeros((n, cluster))
    u_new = np.zeros((n, cluster))

    # подсчитаем матрицу вероятностей
    matrix(n, cluster, x, y, x_cntr, y_cntr, u_new, m)

    while not check(u_old, u_new, n, cluster, eps):
        u_old = u_new.copy()
        recentr(cluster, n, m, x_cntr, y_cntr, u_old, x, y)
        matrix(n, cluster, x, y, x_cntr, y_cntr, u_new, m)
        show(cluster, x, y, u_old, n, x_cntr, y_cntr)
    show(cluster, x, y, u_old, n, x_cntr, y_cntr)
