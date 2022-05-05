''' Unit tests for Sequencer module in ChC package. '''

import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng

from Encoder import Encoder
from Spatial_Pooler import Spatial_Pooler
from Sequencer import SequenceMemory


class TestSequenceMemory(unittest.TestCase):

    def setUp(self):
        self.lengthEncoding = 1200
        self.sp = Spatial_Pooler(lengthEncoding=self.lengthEncoding)
        self.seq = SequenceMemory(self.sp)

    def test_processInputThroughSP(self):
        # This function is a pass through of different spatial pooler functions.
        # See spatial pooler for more detailed tests.
        inputEncoding = np.random.randint(2, size=1200)
        winningColumnsIdx = self.seq.processInputThroughSP(inputEncoding)

        assert isinstance(winningColumnsIdx, list)

        columns = [i for i in range(self.seq.columnCount)]

        assert set(winningColumnsIdx).issubset(set(columns))

    def test_evalActiveColsVersusPreds(self):
        pass

    def test_countActiveSegments(self):
        # # (32 cells/column) * (255 segments/cell) = 8160 segments/column
        # self.seq.activeSegments = [0, 3, 3825, 8159, 8160] # 3825 is segment for cell 15; 8159 is last segment of column 1
        # assert self.seq.activeCells == [0, 15, 31]

        segsPerColumn = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell
        self.seq.activeSegments = [0, 3, 765, 2039, 2040] # 3825 is segment for cell 15; 8159 is last segment of column 1

        numColActiveSegments, idxColSegments = self.seq.countActiveSegments(c=0)

        assert self.seq.activeCells == [0, 3, 7]
        assert numColActiveSegments == 4
        assert idxColSegments == [i for i in range(segsPerColumn)]

    def test_countMatchingSegments(self):
        segsPerColumn = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell
        self.seq.matchingSegments = [0, 3, 765, 2039, 2040]

        numColMatchingSegments, idxColSegments = self.seq.countMatchingSegments(c=0)

        assert numColMatchingSegments == 4
        assert idxColSegments == [i for i in range(segsPerColumn)]

    def test_activatePredictedCells(self):

        firstSegCell16 = 4080 # 16*255 == 1st segement on cell 64
        firstSegCell23 = 5865 # 23*255 == 1st segment on cell 73
        segments = self.seq.maxSegmentsPerCell
        synapses = self.seq.maxSynapsePerSegment
        for startIdx in [firstSegCell16, firstSegCell23]:
            for segmentIdx in range(startIdx, startIdx+segments):
                for synapseIdx in range(segmentIdx*synapses, segmentIdx*synapses+segments):
                    if segmentIdx % 2 == 0:
                        self.seq.synapsePerm[synapseIdx] = 1
                    else:
                        self.seq.synapsePerm[synapseIdx] = 0
                    self.seq.upstreamCellIdx[synapseIdx] = 16
            # Half above threshold; half below

        #Case 1
        self.seq.activeCells = []
        numActivePotentialSynapses_1 = self.seq.activatePredictedCells(0, prevActiveCells = [])

        assert numActivePotentialSynapses_1 == {}
        assert self.seq.activeSegments == []
        assert self.seq.matchingSegments == []

        #Case 2
        self.seq.activeCells = [16, 23] # 1st cell in column 2; 8th cell in column 2
        numActivePotentialSynapses_2 = self.seq.activatePredictedCells(2, [16, 23])

        for startIdx in [firstSegCell16, firstSegCell23]:
            for segmentIdx in range(startIdx, startIdx+segments):
                assert numActivePotentialSynapses_2[segmentIdx] == self.seq.maxSynapsePerSegment

        for a in [1,2]:
            segs = [i for i in range(firstSegCell16, firstSegCell16+segments, a)]
            segs2 = [i for i in range(firstSegCell23+1, firstSegCell23+segments, a)] # +1 is due to firstSegCell23 starting on odd integer
            segs.extend(segs2)
            if a % 2 == 0:
                assert self.seq.activeSegments == segs
            else:
                segs.insert(segments,firstSegCell23)
                assert self.seq.matchingSegments == segs

    def test_updatePerms(self):

        self.seq.numActivePotentialSynapses = {}
        initPerm = 3
        idxColSegments = [i for i in range(2040)] # 2040 segments in column 1 (255*8)
        prevActiveCells = [33, 65, 70, 1000] # pseudo random list of previous active cells
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
            self.seq.upstreamCellIdx[idx] = prevActiveCells[idx]

        self.seq.updatePerms(idxColSegments, prevActiveCells)
        for idx in range(128520): # 128520 synapses in column 1
            if idx < 4:
                assert self.seq.synapsePerm[idx] == initPerm+self.seq.permIncrement
            else:
                assert self.seq.synapsePerm[idx] == initPerm-self.seq.permDecrement

    def test_growSynapses(self):
        # growSynapses(self, segmentIdx, newSynapseCount):
        pass

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


    def test_burstColumn(self):
        for i in range(0, 384, 96):
            self.seq.burstColumn(i)

        assert len(self.seq.activeCells) == self.seq.cellsPerColumn*4 # 4 = 384/96


        print('need to finish burst test')

    def test_bestMatchingSegment(self):
        c = 0
        matchSegsInCol = [i for i in range(0, 21, 3)]
        for i in range(21):
            if i < 13:
                self.seq.numActivePotentialSynapses[i] = i
            else:
                self.seq.numActivePotentialSynapses[i] = 0

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
        pass


if __name__ == '__main__':
    unittest.main()
