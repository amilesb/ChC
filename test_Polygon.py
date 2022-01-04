''' Unit tests for ChC package. '''
# see calcresource.com/geom-pentagon.html for general polygon formula

import unittest
import random

import numpy as np
import Polygon

data = np.load("/home/amilesb/Desktop/ChC/Polygon_arrays.npz")
print(data['test_triangle'])

# with np.load("/home/amilesb/Desktop/ChC/Polygon_arrays.npz") as data:
#     print(data['test_triangle'])

class TestPoly(unittest.TestCase):

    Polygons = ['triangle', 'rectangle', 'pentagon', 'hexagon', 'complex']

    def setUp(self):
        self.triangle = Polygon(array_size=16, form='triangle', x = 7, y = 5,
                                width=8, height=12, angle = 0)
        self.rectangle = Polygon(array_size=12, form='rectangle', x = 5, y = 5,
                                 width=8, height=6, angle = 0)
        self.pentagon = Polygon(array_size=12, form='pentagon', x = 5, y = 5,
                                width=9, height=8, angle = 0)
        self.hexagon = Polygon(array_size=12, form='hexagon', x = 5, y = 5,
                               width=8, height=6, angle = 0)
        self.complex = Polygon(array_size=12, form='complex', x = 5, y = 5,
                               width=8, height=6, angle = 0)
        self.triangle30 = Polygon(array_size=16, form='triangle', x = 7, y = 5,
                                width=8, height=12, angle = 30)


    def test_insert_Polygon(self):
        for shape in Polygons:
            self.shape.insert_Polygon(display=True)




    def test_gauss(self):
        result = generateRandomNoise(self.fake_image)
        self.assertEqual
