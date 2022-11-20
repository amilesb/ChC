import numpy as np
import unittest
from unittest import mock
from pprint import pprint

from ChC import ChC, AIS
from Encoder import Encoder
from Polygon import Polygon
from Processor import Processor


class TestProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = Processor()
        self.pShape, self.attachedChC = self.processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)
        self.processor.AIS = AIS(self.pShape, self.attachedChC)
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
        trueTargs = np.where(self.intValArray > 0, 1, 0)
        row, col = self.processor.getNonzeroIndices(trueTargs)
        trueTargs = [(r, c) for r, c in zip(row, col)]
        self.processor.trueTargs = set(trueTargs)
        # self.intValArray = np.array([[1, 1, 2, 2, 2, 5, 5, 5, 5, 5],
        #                              [0, 1, 3, 0, 1, 5 ,5, 5, 5, 5],
        #                              [1, 3, 2, 1, 1, 5, 5, 5, 5, 5],
        #                              [3, 3, 3, 1, 0, 5, 5, 5, 5, 5],
        #                              [0, 1, 3, 0, 1, 9 ,9, 9, 9, 9],
        #                              [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
        #                              [3,sparseNum 3, 3, 1, 0, 9, 9, 9, 9, 9],
        #                              [1, 3, 2, 1, 1, 9, 9, 9, 9, 9],
        #                              [3, 3, 3, 1, 0, 9, 9, 9, 9, 9],
        #                              [1, 3, 1, 0, 1, 5, 5, 5, 5, 5]])
    def test_extractSDR(self):
        # this represents an end to end test bc it is the master function call
        flag, targetIndxs = self.processor.extractSDR('Exact', sparseHigh=10,
                                                      pShape=self.pShape,
                                                      attachedChC=self.attachedChC,
                                                      )

        assert flag == True
        for targ in targetIndxs:
            assert targ in self.processor.trueTargs
        assert len(targetIndxs) == 10


    def test_buildPolygonAndAttachChC(self):
        '''NOTE this test includes an test of calcSalience function.'''
        pShape, attachedChC = self.processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)

        assert pShape.input_array.size == 100
        assert len(attachedChC.ChC_Coordinates) == 10


    def test_applyReceptiveField(self):
        array_MAX=9
        self.pShape.input_array = self.intValArray
        threshold = np.ndarray((self.intValArray.shape[0], self.intValArray.shape[1]))
        threshold[0, 0] = -1
        self.processor.sparseNum = {'low': 10, 'high': 10}
        # chcStep = array_MAX/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
        # avgInputValInRF = np.mean(self.intValArray)
        # threshold[:] = avgInputValInRF/chcStep
        self.processor.pShape = self.pShape
        self.processor.attachedChC = self.attachedChC
        self.processor.threshold = threshold
        targetIndxs, confidenceFlag = self.processor.applyReceptiveField()
        assert len(targetIndxs) == 10

        self.intValArray[0, 0] = 18
        threshold[0, 0] = -1
        targetIndxs, confidenceFlag = self.processor.applyReceptiveField()
        # print('test_applyReceptiveField', targetIndxs)
        assert len(targetIndxs) == 10


    def test_moveAIS(self):
        binaryInputPiece = np.where(self.intValArray > -1, 1, 0)
        direction = 'decrease'
        for i in range(40):
            self.processor.moveAIS(binaryInputPiece, direction)

        assert self.processor.AIS.ais.any() == 0

        direction = 'increase'
        for i in range(50):
            self.processor.moveAIS(binaryInputPiece, direction)

        testArray = np.ones((self.intValArray.shape[0], self.intValArray.shape[1]))
        testArray *= 40

        assert self.processor.AIS.ais.all() == testArray.all()


    def test_adjustThreshold(self):
        # Test up
        binaryInputPiece = np.where(self.intValArray > -1, 1, 0)
        self.processor.threshold = np.ndarray((self.intValArray.shape[0],
                                               self.intValArray.shape[1]))
        self.processor.threshold[0:9] = 10
        self.processor.threshold[9:] = 5

        for direction in ['up', 'down']:
            self.processor.adjustThreshold(binaryInputPiece, direction)

            for i in range(10):
                for j in range(10):
                    if i < 9:
                        assert self.processor.threshold[i, j] == 11
                    else:
                        assert self.processor.threshold[i, j] == 6
            # Reset for 'down' 2nd case
            binaryInputPiece = np.where(self.intValArray > 10e10, 1, 0)
            self.processor.threshold[0:9] = 12
            self.processor.threshold[9:] = 7


    def test_getNonzeroIndices(self):
        array = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if i == j:
                    array[i, j] = 5

        row, col = self.processor.getNonzeroIndices(array)

        for i in range(10):
            assert row[i] == i
            assert col[i] == i


    def test_computeMinDist(self):
        miss = (1,1)
        hits = [(1,4), (1,7), (4,1), (4,4), (4,7), (8,0)]
        dist = self.processor.computeMinDist(miss, hits)

        assert dist == 3


    @mock.patch('Processor.Processor.applyReceptiveField')
    def test_internalMove(self, mockedApply):
        mockedApply.return_value = ([(i, i) for i in range(5, 10)], True)
        self.processor.pShape.input_array = np.ones((10, 10))
        self.processor.threshold = np.ones((10, 10))
        indxs = [(i, i) for i in range(5)]
        self.processor.sparseNum['low'] = 4
        self.processor.sparseNum['high'] = 20
        # TEST1 (initial if block for exit) start with indices (0,0)-(4,4) need to find (5,5)-(9,9)
        targetIndxs = self.processor.internalMove(indxs)

        assert targetIndxs == [(i, i) for i in range(5, 10)]
        assert self.processor.internalNoiseFlag == False

        for i in range(10):
            for j in range(10):
                if (i, j) not in mockedApply.return_value[0]:
                    mockedApply.return_value[0].append((i, j))
        # TEST2 force internal move to rely on safety exit of recursion
        self.processor.sparseNum['low'] = 21
        targetIndxs = self.processor.internalMove(indxs)

        assert len(targetIndxs) == self.processor.sparseNum['low']
        assert self.processor.internalNoiseFlag == True


    @mock.patch('Processor.Processor.internalMove')
    @mock.patch('Processor.Processor.applyReceptiveField')
    def test_externalMove(self, mockedApply, mockedInternal):
        trueTargs = np.where(self.intValArray > 0, 1, 0)
        row, col = self.processor.getNonzeroIndices(trueTargs)
        trueTargs = [(r, c) for r, c in zip(row, col)]
        self.processor.trueTargs = set(trueTargs)

        mockedApply.return_value = ([], True)
        mockedInternal.side_effect = [
                                       [(i, i) for i in range(5, 10)],
                                       self.processor.trueTargs,
                                       trueTargs[0:4],
                                       trueTargs[4:]
                                     ]
        self.processor.sparseNum['low'] = 10

        # Test1 first if conditional
        sdrTag, indxs = self.processor.externalMove([0, 0])
        assert sdrTag == True
        assert self.processor.countEXTERNAL_MOVE == 2

        # Test2 second if conditional
        self.processor.correctTargsFound.clear()
        sdrTag, indxs = self.processor.externalMove([0, 0])
        assert sdrTag == False
        # print('test external move: count external move', self.processor.countEXTERNAL_MOVE)
        assert self.processor.countEXTERNAL_MOVE == 4


    def applyThreshold(self):
        pass


    def test_calcInterference(self):
        pass




if __name__ == '__main__':
    unittest.main()
