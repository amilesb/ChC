import numpy as np
import unittest
from unittest import mock

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon
from Controller import Graph, Controller

class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.graph = [[1, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1],
                      [1, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1]]

    def test_countIslands(self):
        row = len(self.graph)
        col = len(self.graph)
        g = Graph(row, col, self.graph)
        islands = g.countIslands()
        assert islands == 6




if __name__ == '__main__':
    unittest.main()
