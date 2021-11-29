import random

import numpy as np
import pygame


class Point:
    def __init__(self, x, y, color="blue"):
        self.x = x
        self.y = y
        self.color = color


def dist(p1: Point, p2: Point):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def change_color():
    for i in range(len(points)):
        neighbors = -1
        for j in range(len(points)):
            if dist(points[i], points[j]) <= eps:
                neighbors += 1
        if neighbors >= min_pts:
            points[i].color = "green"
    for i in range(len(points)):
        if points[i].color != "green":
            for j in range(len(points)):
                if points[j].color == "green":
                    if dist(points[i], points[j]) > eps:
                        points[i].color = "yellow"
    for i in range(len(points)):
        if points[i].color != "green" and points[i].color != "yellow":
            points[i].color = "red"


def colors_default(color):
    for i in range(len(points)):
        points[i].color = color


r = 3
points = []
min_pts, eps = 3, 10


def random_near(point: Point):
    k = np.random.randint(2, 5)
    points.append(point)
    d = list(set(range(-5 * r, 5 * r)) - set(range(-2 * r, -2 * r)))
    for i in range(k):
        x = point.x + random.choice(d)
        y = point.y + random.choice(d)
        points.append(Point(x, y))


screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
pygame.display.update()
play = True
while play:
    for event in pygame.event.get():
        screen.fill("WHITE")
        if event.type == pygame.QUIT:
            play = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                random_near(Point(*event.pos))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                colors_default("blue")
                change_color()

        for point in points:
            pygame.draw.circle(screen, point.color, (point.x, point.y), r)
        pygame.display.update()

pygame.quit()
