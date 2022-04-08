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
        perms = np.ones(800)
        perms[0:400] = 0.15
        perms[401:800] = 0.25
        self.simpleSynapseDict = {'index': np.arange(800), 'permanence': perms,
                                  'boostScore': 1}
        self.currentInput = np.zeros(800)
        self.currentInput[500:700] = 1

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

    def testUpdateSynapseParameters(self):
        pass

    def testUpdateActiveDutyCycle(self):
        pass


if __name__ == '__main__':
    unittest.main()
