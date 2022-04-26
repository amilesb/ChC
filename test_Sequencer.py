''' Unit tests for Sequencer module in ChC package. '''

import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng

from Encoder import Encoder
from Spatial_Pooler import Spatial_Pooler
from Sequencer import SequenceMemory


class TestEncoder(unittest.TestCase):

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
        # (32 cells/column) * (255 segments/cell) = 8160 segments/column
        self.seq.activeSegments = [0, 3, 500, 8159, 8160] # 8159 is last segment of column 1

        self.seq.countActiveSegments(c=0)

        assert self.seq.activeCells == [0, 15, 31]

    def test_activatePredictedCells(self):

        firstSegCell64 = 16320 # 16320 = 64*255 == 1st segement on cell 64
        firstSegCell73 = 18615 # 18615 = 64*73 == 1st segment on cell 73
        segments = self.seq.maxSegmentsPerCell
        synapses = self.seq.maxSynapsePerSegment
        for startIdx in [firstSegCell64, firstSegCell73]:
            for segmentIdx in range(startIdx, startIdx+segments):
                for synapseIdx in range(segmentIdx*synapses, segmentIdx*synapses+segments):
                    if segmentIdx % 2 == 0:
                        self.seq.synapses[synapseIdx] = {'permanence': 1}
                    else:
                        self.seq.synapses[synapseIdx] = {'permanence': 0}
                    self.seq.synapses[synapseIdx]['upstreamCellIdx'] = 64
            # Half above threshold; half below

        #Case 1
        self.seq.activeCells = []
        numActivePotentialSynapses_1 = self.seq.activatePredictedCells(0, prevActiveCells = [])

        assert numActivePotentialSynapses_1 == {}
        assert self.seq.activeSegments == []
        assert self.seq.matchingSegments == []

        #Case 2
        self.seq.activeCells = [64, 73] # 1st cell in column 2; 10th cell in column 2
        numActivePotentialSynapses_2 = self.seq.activatePredictedCells(2, [64, 73])

        for startIdx in [firstSegCell64, firstSegCell73]:
            for segmentIdx in range(startIdx, startIdx+segments):
                assert numActivePotentialSynapses_2[segmentIdx] == 255

        for a in [1,2]:
            segs = [i for i in range(firstSegCell64, firstSegCell64+segments, a)]
            segs2 = [i for i in range(firstSegCell73+1, firstSegCell73+segments, a)] # +1 is due to firstSegCell73 starting on odd integer
            segs.extend(segs2)
            if a % 2 == 0:
                assert self.seq.activeSegments == segs
            else:
                segs.insert(segments,18615)
                assert self.seq.matchingSegments == segs

    def test_updatePerms(self):
        # updatePerms(idxColSegments, prevActiveCells)
        # DO ME NEXT!!
        pass

    def test_growSynapses(self):
        # growSynapses(self, segmentIdx, newSynapseCount):
        pass

    def test_indexHelper(self):
        idxsCells = self.seq.indexHelper('column', 3)

        assert idxsCells == [i for i in range(3*32, 4*32)]

        idxsSegments = self.seq.indexHelper('cell', 8)

        assert idxsSegments == [i for i in range(8*255, 9*255)]

        lastCell = 2047*32*255

        idxsSynapses = self.seq.indexHelper('segment', lastCell)

        assert idxsSynapses == [i for i in range(lastCell*255, (lastCell+1)*255)]


    def test_burstColumn(self):
        for i in range(0, 384, 96):
            self.seq.burstColumn(i)

        assert len(self.seq.activeCells) == 128

    def test_punishPredictedColumn(self):
        pass


if __name__ == '__main__':
    unittest.main()
