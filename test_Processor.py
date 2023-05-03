import numpy as np
import unittest
from unittest import mock
from pprint import pprint
from collections import Counter
from itertools import chain
import os

from ChC import ChC, AIS
from Encoder import Encoder
from Polygon import Polygon
from Processor import Processor


class TestProcessor(unittest.TestCase):

    def setUp(self):
        self.intValArray = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 5, 9 ,0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 9 ,9, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]])

        self.pShape, self.attachedChC = Processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)
        self.pShape.input_array = self.intValArray

        self.processor = Processor('Exact', sparseHigh=10, pShape=self.pShape,
                                    attachedChC=self.attachedChC)
        self.processor.AIS = AIS(self.pShape, self.attachedChC)
        self.cornerStart = 0, 5

        trueTargs = np.where(self.intValArray > 0, 1, 0)
        row, col = trueTargs.nonzero()[0], trueTargs.nonzero()[1]
        trueTargs = [(r, c) for r, c in zip(row, col)]
        self.processor.trueTargs = set(trueTargs)
        self.processor.trueTargsList = trueTargs
        self.processor.pShape.activeElements = trueTargs
        self.processor.totTargVal = np.sum(self.intValArray)
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


    def test_buildPolygonAndAttachChC(self):
        '''NOTE this test includes an test of calcSalience function.'''
        pShape, attachedChC = self.processor.buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0)

        assert pShape.input_array.size == 100
        assert len(attachedChC.ChC_Coordinates) == 10


    def test_extractSDR(self):
        # this represents an end to end test bc it is the master function call
        self.pShape.input_array = self.intValArray
        self.processor.uncorruptedInput = self.intValArray.copy()
        self.processor.pShape.activeElements = [(1, 1), (1, 4), (1, 5), (3, 0), (4, 1), (4, 5), (4, 6), (5, 5), (5, 6), (7, 8)]
        flag, targetIndxs = self.processor.extractSDR()

        assert flag == True
        for targ in targetIndxs:
            assert targ in self.processor.trueTargs
        assert len(targetIndxs) == 10


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


    @mock.patch('ChC.ChC.total_Active_Weight')
    def test_calcWeights(self, mockedTotalActiveWeight):
        mockedTotalActiveWeight.return_value = 10
        weights = self.processor.calcWeights()

        assert weights[0, 0] == 5


    def test_selectNum(self):
        self.processor.countEXTERNAL_MOVE = 5
        num = self.processor.selectNum()
        assert num == 6 * self.processor.sparseNum['low']

        self.processor.num = 105
        num = self.processor.selectNum()
        assert num == 105 + self.processor.sparseNum['low']


    @mock.patch('Processor.Processor.applyReceptiveField')
    @mock.patch('Processor.Processor.selectFromMostCommon')
    def test_internalMove(self, mockedSelectFromMostCommon, mockedApply):
        mockedApply.return_value = ([(i, i) for i in range(5, 10)], True)
        mockedSelectFromMostCommon.return_value = [(i, i) for i in range(5, 10)]
        self.processor.pShape.input_array = np.ones((10, 10))
        indxs = [(i, i) for i in range(5)]

        # TEST1 - seek mode
        targetIndxs = self.processor.internalMove(indxs)

        assert targetIndxs == [(i, i) for i in range(5, 10)]
        assert self.processor.internalNoiseFlag == False

        # TEST2 force internal move to execute else (infer) block
        val0 = ([(i, i) for i in range(5, 10)], True)
        val2 = ([(i, i) for i in range(0, 25)], True)
        mockedApply.side_effect = [val0]*10+[val2]*10

        targetIndxs = self.processor.internalMove(indxs, mode='Inference')

        assert targetIndxs == [(i, i) for i in range(0, 10)]
        assert self.processor.internalNoiseFlag == True


    def test_noiseEstimate(self):
        targs = [(4, 5), (4, 6), (5, 5), (5, 6)]
        noiseEst = self.processor.noiseEstimate(targs)

        # 36 = sum value in intValArray for targs listed above; 64 = total of all vals in intValArray
        assert noiseEst == 36/64


    @mock.patch('numpy.random.randint', lambda x,y : 5)
    def test_selectFromMostCommon(self):
        self.processor.targsINTERNAL = Counter('abcde')
        targetIndxs = self.processor.selectFromMostCommon()

        assert len(targetIndxs) == 5


    @mock.patch('Processor.Processor.internalMove')
    @mock.patch('Processor.Processor.applyReceptiveField')
    def test_externalMoveSEEK(self, mockedApply, mockedInternal):
        # trueTargs = np.where(self.intValArray > 0, 1, 0)
        # row, col = self.processor.getNonzeroIndices(trueTargs)
        # trueTargs = [(r, c) for r, c in zip(row, col)]
        # self.processor.trueTargs = set(trueTargs)

        mockedApply.return_value = ([], True)
        mockedInternal.side_effect = [
                                       [(i, i) for i in range(5, 10)],
                                       self.processor.trueTargs,
                                       self.processor.trueTargsList[0:4],
                                       self.processor.trueTargsList[4:]
                                     ]
        self.processor.sparseNum['low'] = 10

        # Test1 first if conditional
        sdrTag, indxs = self.processor.externalMoveSEEK(self.processor.trueTargsList)
        assert sdrTag == True

        # Test2 Recursive Test
        sdrTag, indxs = self.processor.externalMoveSEEK([(0, 0)])
        assert sdrTag == False
        assert self.processor.countEXTERNAL_MOVE == 2


    def test_simulateExternalMove(self):
        initalTargVal = np.sum(self.intValArray)
        self.processor.simulateExternalMove(noiseLevel=0, blur=0, arrayNoise=0)
        finalTargVal = np.sum(self.processor.pShape.input_array)

        assert np.where(self.intValArray > 0, 1, 0).all() == np.where(self.processor.pShape.input_array > 0, 1, 0).all()
        assert initalTargVal == finalTargVal


    def test_refineSDR(self):
        targs = [(i, j) for i in range(10) for j in range(10)]
        refinedTargIndxs = self.processor.refineSDR(targs)
        assert set(refinedTargIndxs) == self.processor.trueTargs


    def test_splitTargs(self):
        targs = [i for i in range(100)]
        targsCut, refinedTargIndxs = self.processor.splitTargs(targs, 10)

        assert len(targsCut) == 10
        assert len(refinedTargIndxs) == 90


    def test_updateChCWeightsMatchedToSDR(self):
        targs = self.processor.trueTargsList
        self.processor.updateChCWeightsMatchedToSDR(targs)
        DIR1 = 'ChC_handles/Objects'
        DIR2 = 'ChC_handles/Targs'
        suffix = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])
        suffix -= 1
        old_file_C = os.path.join(DIR1, f'ChC_{suffix}')
        old_file_T = os.path.join(DIR2, f'targs_{suffix}')
        new_file_C = os.path.join(DIR1, f'ChC_TEST')
        new_file_T = os.path.join(DIR2, f'targs_TEST')
        os.rename(old_file_C, new_file_C)
        os.rename(old_file_T, new_file_T)

        assert os.path.isfile(new_file_C)

        self.processor.updateChCWeightsMatchedToSDR(targs, sdrName='TEST')

        assert os.path.isfile(new_file_C)


    def test_computeMinDist(self):
        miss = (1,1)
        hits = [(1,4), (1,7), (4,1), (4,4), (4,7), (8,0)]
        dist = self.processor.computeMinDist(miss, hits)

        assert dist == 3


    def test_calcInterference(self):
        targetIndxs = [(0,2), (2, 8), (3, 2), (5, 6), (8, 1), (8, 7), (9, 3)]
        self.processor.pShape.input_array = np.arange(100,200)
        self.processor.pShape.input_array = np.reshape(self.processor.pShape.input_array, (10,10))

        targetOutputStrengths = self.processor.calcInterference(targetIndxs)

        print('calcInterference', targetOutputStrengths)



    def test_findNamesForMatchingSDRs(self):
        indxs = [(i, i) for i in range(10)]
        knownSDRs = []
        for j in range(10):
            knownSDRs.append([(j+k, j+k) for k in range(10)])

        OL = self.processor.findNamesForMatchingSDRs(indxs, knownSDRs)

        assert OL[0] == 1
        assert OL[1] == 0.9
        assert OL[2] == 0.8


    def test_calcSDRMatchStrength(self):
        overlap = [0, 1, 2]
        matchLength = [10, 20, 15]
        OL = self.processor.calcSDRMatchStrength(overlap, matchLength)

        assert OL[0] == 0.5
        assert OL[1] == 1
        assert OL[2] == 0.75

    def test_setChCWeightsFromMatchedSDRs(self):
        '''sdrName "TEST{i}" created and saved to directory similar to
        test_updateChCWeightsMatchedToSDR(self)'''

        DIR1 = 'ChC_handles/Objects'
        DIR2 = 'ChC_handles/Targs'
        targ0 = [(0, 0)]
        targ1 = [(9, 9)]
        for i in range(2):
            if i == 0:
                targ = targ0
            else:
                targ = targ1
            self.processor.updateChCWeightsMatchedToSDR(targ)
            suffix = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])
            suffix -= 1
            old_file_C = os.path.join(DIR1, f'ChC_{suffix}')
            old_file_T = os.path.join(DIR2, f'targs_{suffix}')
            new_file_C = os.path.join(DIR1, f'ChC_TEST{i}')
            new_file_T = os.path.join(DIR2, f'targs_TEST{i}')
            os.rename(old_file_C, new_file_C)
            os.rename(old_file_T, new_file_T)

        # SIMPLE TEST
        for i in range(50):
            self.processor.updateChCWeightsMatchedToSDR(targ1, sdrName='TEST1')
        self.processor.setChCWeightsFromMatchedSDRs({'TEST1': 1})

        weights = {}
        for i in range(self.intValArray.shape[0]):
            for j in range(self.intValArray.shape[1]):
                tot = self.processor.attachedChC.total_Synapse_Weight((i, j))
                weights[(i, j)] = tot

        simple0 = [(7, 8), (7, 9), (8, 7), (8, 8), (8, 9), (9, 7), (9, 8), (9, 9)]
        for coords in simple0:
            assert weights[coords] == 0
        assert sum(weights.values()) == 3680 # 100 values all ChC set to 40 except 8 above in simple0

        # TEST with multiple SDR names
        for i in range(50):
            self.processor.updateChCWeightsMatchedToSDR(targ0, sdrName='TEST0')

        sdr_names = {'TEST0': 1, 'TEST1': 0.5}
        self.processor.setChCWeightsFromMatchedSDRs(sdr_names)

        weights = {}
        for i in range(self.intValArray.shape[0]):
            for j in range(self.intValArray.shape[1]):
                tot = self.processor.attachedChC.total_Synapse_Weight((i, j))
                weights[(i, j)] = tot

        for coords in simple0:
            assert weights[coords] == 20
        two_named = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        for coords in two_named:
            assert weights[coords] == 10
        assert sum(weights.values()) == 2760 # 100 values all ChC set to 30 except 8 in simple0 and 8 in two_named set to 30, 10 respectively



if __name__ == '__main__':
    unittest.main()


    # def test_adjustThreshold(self):
    #     # Test up
    #     binaryInputPiece = np.where(self.intValArray > -1, 1, 0)
    #     self.processor.threshold = np.ndarray((self.intValArray.shape[0],
    #                                            self.intValArray.shape[1]))
    #     self.processor.threshold[0:9] = 10
    #     self.processor.threshold[9:] = 5
    #
    #     for direction in ['up', 'down']:
    #         self.processor.adjustThreshold(binaryInputPiece, direction)
    #
    #         for i in range(10):
    #             for j in range(10):
    #                 if i < 9:
    #                     assert self.processor.threshold[i, j] == 11
    #                 else:
    #                     assert self.processor.threshold[i, j] == 6
    #         # Reset for 'down' 2nd case
    #         binaryInputPiece = np.where(self.intValArray > 10e10, 1, 0)
    #         self.processor.threshold[0:9] = 12
    #         self.processor.threshold[9:] = 7






    # def test_getNonzeroIndices(self):
    #     array = np.zeros((10, 10))
    #     for i in range(10):
    #         for j in range(10):
    #             if i == j:
    #                 array[i, j] = 5
    #
    #     row, col = self.processor.getNonzeroIndices(array)
    #
    #     for i in range(10):
    #         assert row[i] == i
    #         assert col[i] == i



    # def test_moveAIS(self):
    #     binaryInputPiece = np.where(self.intValArray > -1, 1, 0)
    #     direction = 'decrease'
    #     for i in range(40):
    #         self.processor.moveAIS(binaryInputPiece, direction)
    #
    #     assert self.processor.AIS.ais.any() == 0
    #
    #     direction = 'increase'
    #     for i in range(50):
    #         self.processor.moveAIS(binaryInputPiece, direction)
    #
    #     testArray = np.ones((self.intValArray.shape[0], self.intValArray.shape[1]))
    #     testArray *= 40
    #
    #     assert self.processor.AIS.ais.all() == testArray.all()
