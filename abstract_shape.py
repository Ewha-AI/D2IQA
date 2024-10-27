
"""
Reference : https://github.com/elkorchi/2DGeometricShapesGenerator
I did some modifications to existing code: change from turtle to cv2
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod

class AbstractShape(ABC):    

    def __init__(self, xoff, yoff, raidus):
    
        self.radius = raidus
        self.x = xoff
        self.y = yoff
        self.rotation = np.deg2rad(np.random.randint(-180, 180))
        
    @abstractmethod
    def get_shape_coordinates(self):
        raise NotImplementedError()

    @abstractmethod
    def get_shape_rotated_coordinates(self):
        raise NotImplementedError()

class AbstractPolygonShape(AbstractShape, ABC):

    number_of_vertices = None
    
    def get_shape_coordinates(self):

        if not self.number_of_vertices:
            raise NotImplementedError(
                "The number of vertices must be specified in sub classes."
            )

        coordinates = []
        for vertex in range(self.number_of_vertices):
            coordinates.append(
                (
                    self.radius * np.cos(
                        2 * np.pi * vertex / self.number_of_vertices
                    ) + self.x,
                    self.radius * np.sin(
                        2 * np.pi * vertex / self.number_of_vertices
                    ) + self.y
                )
            )
        return coordinates
    
    def get_shape_rotated_coordinates(self):

        shape_coordinates = self.get_shape_coordinates()
        
        r_coordinates = []

        for item in shape_coordinates:
            r_coordinates.append(
                (
                    (item[0] - self.x) * np.cos(self.rotation) -
                    (item[1] - self.y) * np.sin(self.rotation) + self.x,

                    (item[0] - self.x) * np.sin(self.rotation) +
                    (item[1] - self.y) * np.cos(self.rotation) + self.y
                )
            )

        return r_coordinates

class Triangle(AbstractPolygonShape):

    number_of_vertices = 3

class Square(AbstractPolygonShape):

    number_of_vertices = 4

class Star(AbstractPolygonShape):

    def get_shape_coordinates(self):
        pentagon_coordinates = []
        for vertex in range(6):
            pentagon_coordinates.append(
                (
                    self.radius * np.cos(2 * np.pi * vertex / 5) + self.x,
                    self.radius * np.sin(2 * np.pi * vertex / 5) + self.y
                )
            )

        pentagon_coordinates[5] = self.get_point(
            pentagon_coordinates[0], pentagon_coordinates[2],
            pentagon_coordinates[1], pentagon_coordinates[3]
        )

        # line1 = ((pentagon_coordinates[1][0], pentagon_coordinates[2][0]), (pentagon_coordinates[1][1], pentagon_coordinates[2][1]))
        # line2 = ((pentagon_coordinates[0][0], pentagon_coordinates[4][0]), (pentagon_coordinates[0][1], pentagon_coordinates[4][1]))
        # line3 = ((pentagon_coordinates[2][0], pentagon_coordinates[3][0]), (pentagon_coordinates[2][1], pentagon_coordinates[3][1]))
        # line4 = ((pentagon_coordinates[0][0], pentagon_coordinates[1][0]), (pentagon_coordinates[0][1], pentagon_coordinates[1][1]))
        # line5 = ((pentagon_coordinates[3][0], pentagon_coordinates[4][0]), (pentagon_coordinates[3][1], pentagon_coordinates[4][1]))

        # new_pentagon_coordinates = []
        # new_pentagon_coordinates.append(self.line_intersection(line1, line2))
        # new_pentagon_coordinates.append(self.line_intersection(line1, line5))
        # new_pentagon_coordinates.append(self.line_intersection(line4, line5))
        # new_pentagon_coordinates.append(self.line_intersection(line3, line4))

      
        coordinates = [
            pentagon_coordinates[2],
            pentagon_coordinates[4],
            pentagon_coordinates[1],
            pentagon_coordinates[3],
            pentagon_coordinates[0],
            pentagon_coordinates[5],
            pentagon_coordinates[2]
        ]

        return coordinates
    
    def line_params(self, point_1, point_2):
        x_1, y_1 = point_1[0], point_1[1]
        x_2, y_2 = point_2[0], point_2[1]
        k = (y_1 - y_2) / (x_1 - x_2)
        b = y_1 - k * x_1
        return k, b

    def get_point(self, point_0, point_2, point_1, point_3):
        k_1, b_1 = self.line_params(point_0, point_2)
        k_2, b_2 = self.line_params(point_1, point_3)
        x = (b_2 - b_1) / (k_1 - k_2)
        y = k_1 * x + b_1
        return (x, y)

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


class Circle(AbstractShape):

    def get_shape_coordinates(self):
        
        coordinates = [(self.x, self.y - self.radius),
                        (self.x + self.radius, self.y),
                        (self.x, self.y + self.radius),
                        (self.x - self.radius, self.y)]
        return coordinates

    def get_shape_rotated_coordinates(self):
        return self.get_shape_coordinates()
