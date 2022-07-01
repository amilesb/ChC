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
        self.pShape, self.attachedChC = self.controller.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)
        self.cornerStart = 0, 5
        self.intValArray = np.array([[1, 1, 2, 2, 2, 5, 5, 5, 5, 5],
                                     [0, 1, 3, 0, 1, 5 ,5, 5, 5, 5],
                                     [1, 3, 2, 1, 1, 5, 5, 5, 5, 5],
                                     [3, 3, 3, 1, 0, 5, 5, 5, 5, 5],
                                     [0, 1, 3, 0, 1, 9 ,9, 9, 9, 9],
                                     [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
                                     [3, 3, 3, 1, 0, 9, 9, 9, 9, 9],
                                     [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
                                     [3, 3, 3, 1, 0, 9, 9, 9, 9, 9],
                                     [1, 3, 1, 0, 1, 5, 5, 5, 5, 5]])
    def test_processInput(self):
        # this represents an end to end test bc it is the master function call
        pass


    def test_buildPolygonAndAttachChC(self):
        pShape, attachedChC = self.controller.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)

        assert pShape.input_array.size == 100
        assert len(attachedChC.ChC_Coordinates) == 10

    def test_extractPieces(self):
        binaryPieces, salience = self.controller.extractPieces(self.intValArray, self.attachedChC)
        print('bin',binaryPieces)


    def test_findConnectedComponents(self):

        g = self.controller.findConnectedComponents(self.intValArray)

        assert g.islandDict[9]['islandSize'] == [25]
        assert g.islandDict[5]['islandSize'] == [20, 5]


    def test_applyReceptiveField(self):
        filter = np.amax(self.intValArray)/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT

        binaryInputPiece = self.controller.applyReceptiveField(self.intValArray,
                                                               self.cornerStart,
                                                               filter,
                                                               self.attachedChC)
        assert binaryInputPiece.size == 16
        assert 0 or 1 in binaryInputPiece
        # print(binaryInputPiece)


    def test_calcCenterRF(self):
        # even case
        centerRF = self.controller.calcCenterRF(self.cornerStart)
        assert centerRF == (1, 6)

        # odd case
        self.controller.REC_FLD_LENGTH = 5
        centerRF = self.controller.calcCenterRF(self.cornerStart)
        assert centerRF == (2, 7)


    def test_calcSalience(self):
        # g = self.controller.findConnectedComponents(self.intValArray)
        #
        # self.controller.calcSalience(g, self.intValArray, binaryPieces)
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
            # print(i, ':', val)





if __name__ == '__main__':
    unittest.main()
