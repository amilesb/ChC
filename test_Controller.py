import numpy as np
import unittest
from unittest import mock
from pprint import pprint

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon
from Controller import Graph, Controller


class TestController(unittest.TestCase):

    def setUp(self):
        self.controller = Controller()

    def test_processInput(self):
        # this represents an end to end test bc it is the master function call
        pass


    def test_buildPolygonAndAttachChC(self):
        pShape, attachedChC = self.controller.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)

        assert pShape.input_array.size == 100
        assert len(attachedChC.ChC_Coordinates) == 10

    def test_extractPieces(self):
        pass


    def test_findConnectedComponents(self):
        pass


    def test_applyReceptiveField(self):
        pass


    def test_calcCenterRF(self):
        pass


    def test_calcSalience(self):
        pass





class TestGraph(unittest.TestCase):

    def setUp(self):
        self.graph = [[1, 1, 2, 2, 2],
                      [0, 1, 3, 0, 1],
                      [1, 3, 2, 1, 1],
                      [3, 3, 3, 1, 0],
                      [1, 3, 1, 0, 1]]

        self.perim = [[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 2, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]]

    def test_countIslands(self):
        # this represents an end to end test bc the main utility is a recursive function call
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

        # p = Graph(row, col, self.perim)
        # for value in range(3):
        #     p.countIslands(value)
        # print('''note problem with centerRF bc with perimeter it puts center
        #         inside at location where actual value is something else''')
        # pprint(p.islandDict)

    def test_me(self):
        tester = {'a': 1, 'b': 2, 'c': 5}
        total = sum(tester.values(), 0)
        tester = {k: v/total for k, v in tester.items()}

        for i in range(20):
            val = np.random.choice(list(tester.keys()), replace=True, p=list(tester.values()))
            print(i, ':', val)





if __name__ == '__main__':
    unittest.main()
