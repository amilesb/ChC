import numpy as np
import unittest
from unittest import mock
from pprint import pprint

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon
from Processor import Processor


class TestProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = Processor()
        self.pShape, self.attachedChC = self.processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)
        self.cornerStart = 0, 5
        self.intValArray = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 5, 9 ,0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 9 ,9, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # self.intValArray = np.array([[1, 1, 2, 2, 2, 5, 5, 5, 5, 5],
        #                              [0, 1, 3, 0, 1, 5 ,5, 5, 5, 5],
        #                              [1, 3, 2, 1, 1, 5, 5, 5, 5, 5],
        #                              [3, 3, 3, 1, 0, 5, 5, 5, 5, 5],
        #                              [0, 1, 3, 0, 1, 9 ,9, 9, 9, 9],
        #                              [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
        #                              [3, 3, 3, 1, 0, 9, 9, 9, 9, 9],
        #                              [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
        #                              [3, 3, 3, 1, 0, 9, 9, 9, 9, 9],
        #                              [1, 3, 1, 0, 1, 5, 5, 5, 5, 5]])
    def test_extractSDR(self):
        # this represents an end to end test bc it is the master function call
        pass


    def test_buildPolygonAndAttachChC(self):
        '''NOTE this test includes an test of calcSalience function.'''
        pShape, attachedChC = self.processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)

        assert pShape.input_array.size == 100
        assert len(attachedChC.ChC_Coordinates) == 10


    def test_applyReceptiveField(self):
        chcStep = np.amax(self.intValArray)/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT

        targetIndxs = self.processor.applyReceptiveField(self.intValArray,
                                                          self.attachedChC,
                                                          chcStep,
                                                          threshold=None,
                                                          sparseNum=10)

        print(targetIndxs)
        # assert targetIndxs.size == 10

    def test_calcInterference(self):
        pass




if __name__ == '__main__':
    unittest.main()
