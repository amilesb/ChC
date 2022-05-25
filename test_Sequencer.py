''' Unit tests for Sequencer module in ChC package. '''

import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng
import random

from Encoder import Encoder
from Spatial_Pooler import Spatial_Pooler
from Sequencer import SequenceMemory

class TestSequenceMemory(unittest.TestCase):

    def setUp(self):
        self.lengthEncoding = 1200
        self.sp = Spatial_Pooler(lengthEncoding=self.lengthEncoding)
        self.seq = SequenceMemory(self.sp)

    def test_processInputThroughSP(self):
        '''This function is a pass through of different spatial pooler
        functions.  See spatial pooler for more detailed tests.'''

        inputEncoding = np.random.randint(2, size=1200)
        winningColumnsIdx = self.seq.processInputThroughSP(inputEncoding)

        assert isinstance(winningColumnsIdx, list)

        columns = [i for i in range(self.seq.columnCount)]

        assert set(winningColumnsIdx).issubset(set(columns))

    def test_evalActiveColsVersusPreds(self):
        # self.seq.evalActiveColsVersusPreds([0, 1])
        pass


    def test_transferComponentToPrevAndReset(self):
        self.seq.activeCells = [0, 1, 100]
        self.seq.numActivePotentialSynapses = {0:10, 1:5}
        self.seq.transferComponentToPrevAndReset()

        assert self.seq.prevActiveCells == [0, 1, 100]
        assert self.seq.prevNumActivePotentialSynapses == {0:10, 1:5}

        self.seq.transferComponentToPrevAndReset()
        assert self.seq.prevNumActivePotentialSynapses[0] == self.seq.maxNewSynapseCount

    def test_countSegments(self):
        segsPerColumn = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell
        self.seq.matchingSegments = [0, 3, 765, 2039, 2040]

        colMatchingSegments, idxColSegments = self.seq.countSegments(
                                                  c=0,
                                                  prevSegments=self.seq.matchingSegments
                                                  )

        assert colMatchingSegments == [0, 3, 765, 2039]
        assert idxColSegments == [i for i in range(segsPerColumn)]

    @mock.patch('Sequencer.SequenceMemory.growSynapses')
    def test_activatePredictedCol(self, mockedUpdate):
        c = 0
        colActiveSegments = [0, 300]
        self.seq.prevActiveCells = [0]
        self.seq.prevWinnerCells = self.seq.prevActiveCells
        self.seq.activatePredictedCol(c, colActiveSegments)

        assert self.seq.activeCells == [0, 1]
        assert self.seq.winnerCells == [0, 1]

    @mock.patch('Sequencer.SequenceMemory.growSynapses')
    def test_updatePerms(self, mocked):
        self.seq.numActivePotentialSynapses = {}
        initPerm = 3
        idxColSegments = [i for i in range(2040)] # 2040 segments in column 1 (255*8)
        self.seq.prevActiveCells = [33, 65, 70, 1000] # pseudo random list of previous active cells
        self.seq.prevWinnerCells = [0]
        s = 0
        for segmentIdx in idxColSegments:
            if s > 254:
                s = 0
            self.seq.numActivePotentialSynapses[segmentIdx] = s
            s += 1
            idxCellSynapses = self.seq.indexHelper('segment', segmentIdx)
            for synapse in idxCellSynapses:
                self.seq.synapsePerm[synapse] = initPerm

        for idx in range(4):
            self.seq.upstreamCellIdx[idx] = self.seq.prevActiveCells[idx]
        self.seq.upstreamCellIdx[4] = 4

        numSynsColOne = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell*self.seq.maxSynapsePerSegment
        self.seq.updatePerms(idxColSegments)
        for idx in range(numSynsColOne):
            if idx < 4:
                assert np.allclose(self.seq.synapsePerm[idx], initPerm+self.seq.permIncrement)
            elif idx == 4:
                assert np.allclose(self.seq.synapsePerm[idx], initPerm-self.seq.permDecrement)
            else:
                assert np.allclose(self.seq.synapsePerm[idx], initPerm)

    @mock.patch('random.choice')
    def test_growSynapses(self, mockedChoice):
        segmentIdx = 0
        newSynapseCount = 2
        self.seq.prevWinnerCells = [0, 3, 5, 11, 50]
        mockedChoice.side_effect = self.seq.prevWinnerCells[:]
        self.seq.upstreamCellIdx[0:2] = self.seq.prevWinnerCells[0:2]
        self.seq.growSynapses(segmentIdx, newSynapseCount)

        upstreamCellIndices = []
        for i in range(2,4):
            assert np.allclose(self.seq.upstreamCellIdx[i], self.seq.prevWinnerCells[i-5])
            assert np.allclose(self.seq.synapsePerm[i], self.seq.initialPerm)
        for i in range(4,8):
            upstreamCellIndices.append(self.seq.upstreamCellIdx[i])
        assert 50 not in upstreamCellIndices


    def test_indexHelper(self):
        idxsCells = self.seq.indexHelper('column', 3)
        cellNum = self.seq.cellsPerColumn

        assert idxsCells == [i for i in range(3*cellNum, 4*cellNum)]

        idxsSegments = self.seq.indexHelper('cell', 8)
        segNum = self.seq.maxSegmentsPerCell

        assert idxsSegments == [i for i in range(8*segNum, 9*segNum)]

        lastCell = 2047*8*255

        idxsSynapses = self.seq.indexHelper('segment', lastCell)
        synNum = self.seq.maxSynapsePerSegment

        assert idxsSynapses == [i for i in range(lastCell*synNum, (lastCell+1)*synNum)]


    @mock.patch('Sequencer.SequenceMemory.leastUsedCell')
    @mock.patch('Sequencer.SequenceMemory.growSynapses')
    def test_burstColumn(self, mockedGrow, mocked_leastUsedCell):
        mocked_leastUsedCell.return_value = 8
        step = 96
        end = 384
        self.seq.prevActiveCells = [0]
        self.seq.prevWinnerCells = [0]
        self.seq.prevMatchingSegments = []

        self.seq.burstColumn(0)

        assert self.seq.winnerCells == [8]

        self.seq.winnerCells = [] # reset list
        self.seq.activeCells = []
        segsPerColumn = self.seq.cellsPerColumn * self.seq.maxSegmentsPerCell
        self.seq.prevMatchingSegments = [x for x in range(0, end*segsPerColumn, step*segsPerColumn)]
        for i in range(0, end, step):
            self.seq.burstColumn(i)

        assert len(self.seq.activeCells) == self.seq.cellsPerColumn*end/step
        assert self.seq.winnerCells == [i for i in range(0, end*self.seq.cellsPerColumn, step*self.seq.cellsPerColumn)]


    def test_bestMatchingSegment(self):
        c = 0
        matchSegsInCol = [i for i in range(0, 21, 3)]
        for i in range(21):
            if i < 13:
                self.seq.prevNumActivePotentialSynapses[i] = i
            else:
                self.seq.prevNumActivePotentialSynapses[i] = 0

        bestMatchSeg = self.seq.bestMatchingSegment(c, matchSegsInCol)

        assert bestMatchSeg == 12

    def test_leastUsedCell(self):
        c = 0
        self.seq.activeSegments = [i for i in range(0, 2040, 255)]
        self.seq.activeSegments[6] = 256 # make cell 7 least used and cell 2 max used so there is variability between cells.

        cellIdx = self.seq.leastUsedCell(c)

        assert cellIdx == 6

    def test_learnOnNewSegment(self):
        c = 0
        cell = 0

        for i in range(256):
            segmentIdx = self.seq.learnOnNewSegment(c, cell)

            if i < 255:
                assert segmentIdx == i
            if i > 254:
                assert segmentIdx == -1

    def test_punishPredictedColumn(self):
        c = 0
        initPerm = 0.5
        self.seq.matchingSegments = [i for i in range(0, 21, 3)]
        idxColSegments = [0, 5, 11]
        self.seq.activeCells = [0]
        numSynsColOne = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell*self.seq.maxSynapsePerSegment

        for synapse in range(numSynsColOne):
            self.seq.synapsePerm[synapse] = initPerm
            if synapse < 10:
                self.seq.upstreamCellIdx[synapse] = self.seq.activeCells[0]

        self.seq.punishPredictedColumn(c, idxColSegments)

        for idx in range(numSynsColOne):
            if idx < 10:
                assert np.allclose(self.seq.synapsePerm[idx], initPerm-self.seq.predictedDecrement)
            else:
                assert np.allclose(self.seq.synapsePerm[idx], initPerm)

    def test_activateDendriticSegments(self):
        pass
        # self.seq.activateDendriticSegments(self):

if __name__ == '__main__':
    unittest.main()
