''' Unit tests for Spatial Pooler module in ChC package. '''

import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng

from Encoder import Encoder
from Spatial_Pooler import Spatial_Pooler


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.lengthEncoding = 1200
        self.sp = Spatial_Pooler(lengthEncoding=self.lengthEncoding)
        self.sp2 = Spatial_Pooler(lengthEncoding=self.lengthEncoding)
        perms = np.ones(800)
        perms[0:400] = 0.15
        perms[401:800] = 0.25
        self.simpleSynapseDict = {'index': np.arange(800), 'permanence': perms,
                                  'boostScore': 1}
        self.currentInput = np.zeros(800)
        self.currentInput[500:700] = 1

        self.winningColumnsInd = [0, 1]

        self.currentInputA = np.ones(2048)
        self.currentInputB = np.zeros(2048)

        self.overlapScoreA = self.sp.computeOverlap(self.currentInputA)
        self.overlapScoreB = self.sp.computeOverlap(self.currentInputB)

    def testInitializeSynapses(self):
        synapseDict = self.sp.synapses[0]
        indices = synapseDict['index']
        perms = synapseDict['permanence']

        assert len(indices) == len(perms)
        assert ((0.1 <= perms) & (perms < 0.3)).all() # from self.connectedPerm = 0.2 +/- 0.1
        assert indices.all() < self.lengthEncoding

        uniqueConnections = np.unique(indices)
        assert len(uniqueConnections) == len(indices)

    def testComputeOverlap(self):
        for c in range(self.sp.columnCount):
            self.sp.synapses[c] = self.simpleSynapseDict
        overlapScore = self.sp.computeOverlap(self.currentInput)
        assert overlapScore == [200]*2048

    def testComputeConnectedSynapses(self):
        subset = Spatial_Pooler.computeConnectedSynapses(self.sp, self.simpleSynapseDict)
        assert len(subset) == 400

    def testComputeWinningCOlumns(self):
        overlapScore = [i+j for i in range(20) for j in range(20)]
        winners = [239, 278, 298, 297, 392, 393, 394, 279, 356, 355, 354, 353,
                   315, 373, 374, 375, 316, 317, 318, 395, 334, 259, 335, 336,
                   337, 299, 338, 379, 378, 377, 376, 339, 359, 358, 357, 319,
                   396, 397, 398, 399]

        smallOverlap = [0]* 50
        smallOverlap[45:50] = (0, 1, 2, 9, 10)

        winningColumnsInd = self.sp.computeWinningCOlumns(overlapScore)
        smallWin = self.sp.computeWinningCOlumns(smallOverlap)

        assert winners == winningColumnsInd
        assert len(smallWin) == 2
        assert 48 and 49 in smallWin

    def testUpdateSynapseParameters_first_for_loop(self):
        currentSynapsePermsA = self.sp.synapses[0]['permanence']
        currentSynapsePermsB = self.sp.synapses[1]['permanence']
        currentSynapsePermsC = self.sp.synapses[2]['permanence']

        self.sp.updateSynapseParameters(self.winningColumnsInd, self.overlapScoreA, self.currentInputA)
        self.sp.updateSynapseParameters(self.winningColumnsInd, self.overlapScoreB, self.currentInputA)

        # assertiions for first for loop
        assert (self.sp.synapses[0]['permanence'].all() ==
                (currentSynapsePermsA + self.sp.synPermActiveInc).all()
               )
        assert (self.sp.synapses[1]['permanence'].all() ==
                (currentSynapsePermsB - self.sp.synPermInactiveDec).all()
               )
        assert self.sp.synapses[2]['permanence'].all() == currentSynapsePermsC.all() # other columns not touched


    def testUpdateSynapseParameters_second_for_loop(self):

        # first 1024 columns are selected as winners 2nd 1024 not.  Overlap and Input are set to encompass every mini-column.
        winningColumnsIdx = [i for i in range(1024)]
        self.sp2.updateSynapseParameters(winningColumnsIdx, self.overlapScoreA, self.currentInputA)


        activeDC_c = self.sp2.calcActiveDutyCycle(0)
        overlapDC_c = self.sp2.updateOverlapDutyCycle(0, self.overlapScoreA[0])

        assert activeDC_c == 1  # first mini-column is winner so has active duty cycle == 1
        assert overlapDC_c == 1

        activeDC_c1025 = 0.5
        overlapDC_c1025 = 0
        for i in range(2048):
            if i < 1024:
                meanACTIVE_DC = self.sp2.updateAndCalcMeanDutyCycle(i, activeDC_c, type='active')
                meanOVERLAP_DC = self.sp2.updateAndCalcMeanDutyCycle(i, overlapDC_c, type='overlap')
            else:
                meanACTIVE_DC = self.sp2.updateAndCalcMeanDutyCycle(i, activeDC_c1025, type='active')
                meanOVERLAP_DC = self.sp2.updateAndCalcMeanDutyCycle(i, overlapDC_c1025, type='overlap')

            assert meanACTIVE_DC == 0.75 # mean value is same for all mini-columns.  Also, 1st half of the mini-columns selected are active 100% ([1, 1]) and half are active 50% ([1, 0]).
        assert meanOVERLAP_DC == 0.5 # similar test as for active DC but in this case done after all mini-columns updated 1 timestamp

        # Boost scores are updating correctly
        assert self.sp2.synapses[2047]['boostScore'] == np.exp(0.25)

        self.sp2.updateSynapseParameters(winningColumnsIdx, self.overlapScoreA, self.currentInputA)

        assert self.sp2.synapses[0]['boostScore'] == np.exp(-0.25)
        assert abs(self.sp2.synapses[2047]['boostScore'] - np.exp(1/3)*np.exp(0.25)) < 1e-5


        old = self.sp2.synapses[1]['permanence'].copy()
        self.sp2.updateSynapseParameters([], self.overlapScoreB, self.currentInputB)
        new = self.sp2.synapses[1]['permanence']
        test_old = list(self.sp2.permVar+old)
        for i in range(len(new)):
            assert new[i]-test_old[i] == 0


    def testUpdateActiveDutyCycle(self):

        self.sp.updateActiveDutyCycle(c=0, winningColumnsInd = [0])

        assert self.sp.synapses[0]['activeDutyCycle'] == [1, 1]

        self.sp.updateActiveDutyCycle(c=1, winningColumnsInd = [0])

        assert self.sp.synapses[1]['activeDutyCycle'] == [1, 0]

        testCycle = [i%2 for i in range(1000)]

        for i in range(1005):
            if i%2 == 0:
                self.sp.updateActiveDutyCycle(c=0, winningColumnsInd = [0])
            else:
                self.sp.updateActiveDutyCycle(c=0, winningColumnsInd = [1])

        assert self.sp.synapses[0]['activeDutyCycle'] == testCycle

    def testCalcActiveDutyCycle(self):
        activeDutyCycle = self.sp.calcActiveDutyCycle(0)

        assert activeDutyCycle == 1

        for i in range(1005):
            if i%2 == 0:
                self.sp.updateActiveDutyCycle(c=0, winningColumnsInd = [0])
            else:
                self.sp.updateActiveDutyCycle(c=0, winningColumnsInd = [1])

        activeDutyCycle = self.sp.calcActiveDutyCycle(0)

        assert activeDutyCycle == 0.5

    def testUpdateAndCalcMeanDutyCycle(self):
        meanADC = self.sp.updateAndCalcMeanDutyCycle(c=0, dutyCycle_c=1, type='active')

        assert meanADC == 1.0

        for i in range(1024):
            meanADC = self.sp.updateAndCalcMeanDutyCycle(c=i, dutyCycle_c=0, type='active')

        assert meanADC == 0.5

    def testUpdateOverlapDutyCycle(self):
        overlapDutyCycle = self.sp.updateOverlapDutyCycle(0, 10)

        assert overlapDutyCycle == 1

        for i in range(1005):
            if i%2 == 0:
                _ = self.sp.updateOverlapDutyCycle(c=0, overlap = 0)
            else:
                _ = self.sp.updateOverlapDutyCycle(c=0, overlap = 10)

        overlapDutyCycle = self.sp.updateOverlapDutyCycle(0, 10)

        assert overlapDutyCycle == 0.5

    def testCalcMinOverlapDutyCycle(self):
        minOverlapDC = self.sp.calcMinOverlapDutyCycle(meanOVERLAP_DC=0.5)
        poisson50 = (50-2*np.sqrt(50))/100

        assert minOverlapDC == poisson50

    def testVisualizeSP(self):
        c = 0
        timestamp = 0
        currentInput = np.random.randint(0,2,1200)
        overlapScore = self.sp.computeOverlap(currentInput)
        winningColumnsInd = self.sp.computeWinningCOlumns(overlapScore)
        self.sp.visualizeSP(c, currentInput, winningColumnsInd, timestamp)


if __name__ == '__main__':
    unittest.main()
