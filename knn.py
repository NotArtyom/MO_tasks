import math
import random
from collections import Counter
from dataclasses import dataclass

import pygame


@dataclass
class Point:
    x: int
    y: int
    cluster: int


cluster_obj = {0: "red", 1: "blue", 2: "green", -1: "orange"}
TRAINING_MODE_AMOUNT = 5
optimal_k = None
optimal_probability = 0


def text(surface, x, y, text, fontFace="Arial", size=18, colour=(123, 123, 123)):
    font = pygame.font.SysFont(fontFace, size)
    text = font.render(text, True, colour)
    surface.blit(text, (x, y))
    pygame.display.update()


def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        # Choose random center of 2-dimensional gaussian
        centerX, centerY = random.randint(20, 580), random.randint(60, 380)
        # Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(numberOfClassEl):
            data.append(
                Point(random.gauss(centerX, 20), random.gauss(centerY, 20), classNum)
            )
    return data


def draw_result(points, training_mode_amount: int):
    global optimal_probability, optimal_k
    pygame.font.init()
    screen = pygame.display.set_mode((800, 400), pygame.RESIZABLE)
    pygame.display.update()
    play = True
    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if training_mode_amount > 0:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if (
                            len(points_validate) == 0
                            or points_validate[-1].cluster != -1
                        ):
                            points_validate.append(
                                Point(event.pos[0], event.pos[1], -1)
                            )
                if event.type == pygame.KEYDOWN:
                    if points_validate[-1].cluster == -1:
                        prediction = kNN(points_validate[-1])
                        if event.key == pygame.K_1:
                            points_validate[-1].cluster = 0
                            if prediction[1] == 0:
                                if prediction[0] > optimal_probability:
                                    optimal_probability = prediction[0]
                                    optimal_k = prediction[2]
                            points.append(points_validate.pop(-1))
                            training_mode_amount -= 1
                        if event.key == pygame.K_2:
                            points_validate[-1].cluster = 1
                            if prediction[1] == 1:
                                if prediction[0] > optimal_probability:
                                    optimal_probability = prediction[0]
                                    optimal_k = prediction[2]
                            points.append(points_validate.pop(-1))
                            training_mode_amount -= 1
                        if event.key == pygame.K_3:
                            if prediction[1] == 2:
                                if prediction[0] > optimal_probability:
                                    optimal_probability = prediction[0]
                                    optimal_k = prediction[2]
                            points_validate[-1].cluster = 2
                            points.append(points_validate.pop(-1))
                            training_mode_amount -= 1
            else:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        predicted_cluster = kNN_fixed_k(
                            Point(event.pos[0], event.pos[1], -1), optimal_k
                        )
                        points.append(
                            Point(event.pos[0], event.pos[1], predicted_cluster)
                        )
            screen.fill("WHITE")
            for point in points:
                pygame.draw.circle(
                    screen, cluster_obj[point.cluster], (point.x, point.y), 3
                )
            for point in points_validate:
                pygame.draw.circle(
                    screen, cluster_obj[point.cluster], (point.x, point.y), 3
                )
            text(
                screen,
                0,
                0,
                f"TRAINING MODE - place {training_mode_amount} more dots and classify them manually. Current: (probability: {'%.2f' % optimal_probability}, k: {optimal_k})"
                if training_mode_amount > 0
                else f"PREDICTION MODE, optimal k selected with probability {optimal_probability}: {optimal_k}",
            )
            pygame.display.update()


def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def kNN(point) -> (float, int, int):
    highest_probability = 0
    highest_probability_cluster = -1
    opt_k = None
    for k in range(3, int(math.sqrt(len(points)))):
        k_nearest_clusters = map(
            lambda point: point.cluster,
            sorted(points, key=lambda p: dist(point, p))[:k],
        )
        counted = Counter(k_nearest_clusters)
        most_common = counted.most_common(1)
        probability = most_common[0][1] / k
        if probability > highest_probability:
            highest_probability = probability
            highest_probability_cluster = most_common[0][0]
            opt_k = k

    return highest_probability, highest_probability_cluster, opt_k


def kNN_fixed_k(point, k) -> int:
    k_nearest_clusters = map(
        lambda point: point.cluster, sorted(points, key=lambda p: dist(point, p))[:k]
    )
    counted = Counter(k_nearest_clusters)
    return counted.most_common(1)[0][0]


if __name__ == "__main__":
    n, cl = 100, 3
    points = generateData(n, cl)
    points_validate = []
    draw_result(points, TRAINING_MODE_AMOUNT)
