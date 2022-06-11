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
            g.countIslands(value)
        # pprint(g.islandDict)
        assert g.islandDict[0]['islandCenters'] == [[1, 0], [1, 3], [3, 4], [4, 3]]
        assert g.islandDict[0]['islandSize'] == [1, 1, 1, 1]

        assert g.islandDict[1]['islandCenters'] == [[0.3333333333333333, 0.6666666666666666],
                                                  [2.0, 3.5],
                                                  [2, 0],
                                                  [4, 0],
                                                  [4, 2],
                                                  [4, 4]]
        assert g.islandDict[1]['islandSize'] == [3, 4, 1, 1, 1, 1]

        assert g.islandDict[2]['islandCenters'] == [[0.0, 3.0], [2, 2]]
        assert g.islandDict[2]['islandSize'] == [3, 1]

        assert g.islandDict[3]['islandCenters'] == [[1, 2], [3.0, 1.0]]
        assert g.islandDict[3]['islandSize'] == [1, 5]


if __name__ == '__main__':
    unittest.main()
