import random
from dataclasses import dataclass
from sklearn.svm import SVC
import numpy as np
import pygame


@dataclass
class Point:
    x: int
    y: int
    cluster: int


@dataclass
class Line:
    start_point: (int, int)
    end_point: (int, int)
    color: str


@dataclass
class PredictionLineSystem:
    main: Line
    margin_above: Line
    margin_below: Line


cluster_map = {0: "red", 1: "blue"}  # Заданные кластеры
model = SVC(kernel="linear")


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


def features_and_labels(points: [Point]) -> (np.ndarray, np.ndarray):
    """
    Конвертируем точки в вид, который ожидается sklearn - массив точек и массив кластеров
    """
    features = []
    labels = []
    for point in points:
        features.append([point.x, point.y])
        labels.append(point.cluster)

    return np.array(features), np.array(labels)


def draw_pygame(points, prediction_line_system: PredictionLineSystem):
    pygame.font.init()
    screen = pygame.display.set_mode((800, 400), pygame.RESIZABLE)
    pygame.display.update()
    play = True
    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # предсказываем класс точки с помощью натренированной модели и добавляем новую точку в список
                    prediction = model.predict(np.array([[event.pos[0], event.pos[1]]]))
                    new_point = Point(event.pos[0], event.pos[1], prediction[0])
                    points.append(new_point)

            screen.fill("WHITE")
            for point in points:
                pygame.draw.circle(
                    screen, cluster_map[point.cluster], (point.x, point.y), 3
                )

            # Рисуем линии разделения классов
            pygame.draw.line(
                screen,
                prediction_line_system.main.color,
                prediction_line_system.main.start_point,
                prediction_line_system.main.end_point,
                2,
            )
            pygame.draw.line(
                screen,
                prediction_line_system.margin_above.color,
                prediction_line_system.margin_above.start_point,
                prediction_line_system.margin_above.end_point,
            )
            pygame.draw.line(
                screen,
                prediction_line_system.margin_below.color,
                prediction_line_system.margin_below.start_point,
                prediction_line_system.margin_below.end_point,
            )

            pygame.display.update()


if __name__ == "__main__":
    n, cl = 100, 2
    points = generateData(n, cl)

    X, y = features_and_labels(points)
    model.fit(X, y)  # обучаем модель
    # находим коэффициенты w и b
    w = model.coef_[0]
    b = model.intercept_[0]
    # вычисляем по ним 2 точки, через которые проведем прямую: по оси x берем 0 и 800 (по размеру экрана)
    x_points = np.linspace(0, 800, num=2)
    y_points = -(w[0] / w[1]) * x_points - b / w[1]

    # вычисляем единичный вектор, перпендикулярный вычисленной прямой
    unit_normal_vector = model.coef_[0] / (np.sqrt(np.sum(model.coef_[0] ** 2)))
    # вычисляем смещение границ "полосы" относительно главной прямой
    margin = 1 / np.sqrt(np.sum(model.coef_[0] ** 2))

    # вычисляем точки для граничных прямых, используя ранее вычисленное смещение и вектор
    main_points = np.array(list(zip(x_points, y_points)))
    points_of_line_above = main_points + unit_normal_vector * margin
    points_of_line_below = main_points - unit_normal_vector * margin

    line_system = PredictionLineSystem(
        main=Line(
            start_point=(main_points[0][0], main_points[0][1]),
            end_point=(main_points[1][0], main_points[1][1]),
            color="orange",
        ),
        margin_above=Line(
            start_point=(points_of_line_above[0][0], points_of_line_above[0][1]),
            end_point=(points_of_line_above[1][0], points_of_line_above[1][1]),
            color="gray",
        ),
        margin_below=Line(
            start_point=(points_of_line_below[0][0], points_of_line_below[0][1]),
            end_point=(points_of_line_below[1][0], points_of_line_below[1][1]),
            color="gray",
        ),
    )

    draw_pygame(points, line_system)
