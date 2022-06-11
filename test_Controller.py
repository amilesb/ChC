import numpy as np
import unittest
from unittest import mock
from pprint import pprint

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon
from Controller import Graph, Controller

class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.graph = [[1, 1, 2, 2, 2],
                      [0, 1, 3, 0, 1],
                      [1, 3, 2, 1, 1],
                      [3, 3, 3, 1, 0],
                      [1, 3, 1, 0, 1]]

    def test_countIslands(self):
        row = len(self.graph)
        col = len(self.graph)
        g = Graph(row, col, self.graph)
        centerRF = [[0, 0], [0, 4], [1, 0], [3, 2]]
        for value in range(4):
            print('hello', value)
            islandDict = g.countIslands(value, centerRF[value])
        pprint(g.islandDict)


######## Almost working need to fix centerRF tracking!

if __name__ == '__main__':
    unittest.main()
