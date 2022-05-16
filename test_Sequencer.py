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

    # def test_processInputThroughSP(self):
    #     '''This function is a pass through of different spatial pooler
    #     functions.  See spatial pooler for more detailed tests.'''
    #
    #     inputEncoding = np.random.randint(2, size=1200)
    #     winningColumnsIdx = self.seq.processInputThroughSP(inputEncoding)
    #
    #     assert isinstance(winningColumnsIdx, list)
    #
    #     columns = [i for i in range(self.seq.columnCount)]
    #
    #     assert set(winningColumnsIdx).issubset(set(columns))
    #
    # def test_evalActiveColsVersusPreds(self):
    #     pass
    #
    # def test_countSegments(self):
    #     segsPerColumn = self.seq.cellsPerColumn*self.seq.maxSegmentsPerCell
    #     self.seq.matchingSegments = [0, 3, 765, 2039, 2040]
    #
    #     colMatchingSegments, idxColSegments = self.seq.countSegments(
    #                                               c=0,
    #                                               prevSegments=self.seq.matchingSegments
    #                                               )
    #
    #     assert colMatchingSegments == [0, 3, 765, 2039]
    #     assert idxColSegments == [i for i in range(segsPerColumn)]
    #
    # def test_activatePredictedCol(self):
    #     c = 0
    #     colActiveSegments = [0, 300]
    #     prevActiveCells = [0]
    #     self.seq.activatePredictedCol(c, colActiveSegments, prevActiveCells)
    #
    #     assert self.seq.activeCells == [0, 1]
    #     assert self.seq.winnerCells == [0, 1]
    #
    # def test_updatePerms(self):
    #
    #     self.seq.numActivePotentialSynapses = {}
    #     initPerm = 3
    #     idxColSegments = [i for i in range(2040)] # 2040 segments in column 1 (255*8)
    #     prevActiveCells = [33, 65, 70, 1000] # pseudo random list of previous active cells
    #     s = 0
    #     for segmentIdx in idxColSegments:
    #         if s > 254:
    #             s = 0
    #         self.seq.numActivePotentialSynapses[segmentIdx] = s
    #         s += 1
    #         idxCellSynapses = self.seq.indexHelper('segment', segmentIdx)
    #         for synapse in idxCellSynapses:
    #             self.seq.synapsePerm[synapse] = initPerm
    #
    #     for idx in range(4):
    #         self.seq.upstreamCellIdx[idx] = prevActiveCells[idx]
    #
    #     self.seq.updatePerms(idxColSegments, prevActiveCells)
    #     for idx in range(128520): # 128520 synapses in column 1
    #         if idx < 4:
    #             assert self.seq.synapsePerm[idx] == initPerm+self.seq.permIncrement
    #         else:
    #             assert self.seq.synapsePerm[idx] == initPerm-self.seq.permDecrement
    # #
    def test_growSynapses(self):
        # growSynapses(self, segmentIdx, newSynapseCount):
        pass
    ###########  next do grow synapses and then activate segments + tests for last 2

    # def test_indexHelper(self):
    #     idxsCells = self.seq.indexHelper('column', 3)
    #     cellNum = self.seq.cellsPerColumn
    #
    #     assert idxsCells == [i for i in range(3*cellNum, 4*cellNum)]
    #
    #     idxsSegments = self.seq.indexHelper('cell', 8)
    #     segNum = self.seq.maxSegmentsPerCell
    #
    #     assert idxsSegments == [i for i in range(8*segNum, 9*segNum)]
    #
    #     lastCell = 2047*8*255
    #
    #     idxsSynapses = self.seq.indexHelper('segment', lastCell)
    #     synNum = self.seq.maxSynapsePerSegment
    #
    #     assert idxsSynapses == [i for i in range(lastCell*synNum, (lastCell+1)*synNum)]
    #
    # @mock.patch('Sequencer.SequenceMemory.leastUsedCell')
    # def test_burstColumn(self, mocked_leastUsedCell):
    #     mocked_leastUsedCell.return_value = 8
    #     step = 96
    #     end = 384
    #     prevActiveCells = [0]
    #     prevMatchingSegments = []
    #
    #     self.seq.burstColumn(0, prevMatchingSegments, prevActiveCells)
    #
    #     assert self.seq.winnerCells == [8]
    #
    #     self.seq.winnerCells = [] # reset list
    #     self.seq.activeCells = []
    #     segsPerColumn = self.seq.cellsPerColumn * self.seq.maxSegmentsPerCell
    #     prevMatchingSegments = [x for x in range(0, end*segsPerColumn, step*segsPerColumn)]
    #     for i in range(0, end, step):
    #         self.seq.burstColumn(i, prevMatchingSegments, prevActiveCells)
    #
    #     assert len(self.seq.activeCells) == self.seq.cellsPerColumn*end/step
    #     assert self.seq.winnerCells == [i for i in range(0, end*self.seq.cellsPerColumn, step*self.seq.cellsPerColumn)]
    #
    #
    #
    #
    # def test_bestMatchingSegment(self):
    #     c = 0
    #     matchSegsInCol = [i for i in range(0, 21, 3)]
    #     for i in range(21):
    #         if i < 13:
    #             self.seq.numActivePotentialSynapses[i] = i
    #         else:
    #             self.seq.numActivePotentialSynapses[i] = 0
    #
    #     bestMatchSeg = self.seq.bestMatchingSegment(c, matchSegsInCol)
    #
    #     assert bestMatchSeg == 12
    #
    # def test_leastUsedCell(self):
    #     c = 0
    #     self.seq.activeSegments = [i for i in range(0, 2040, 255)]
    #     self.seq.activeSegments[6] = 256 # make cell 7 least used and cell 2 max used so there is variability between cells.
    #
    #     cellIdx = self.seq.leastUsedCell(c)
    #
    #     assert cellIdx == 6
    #
    # def test_learnOnNewSegment(self):
    #     c = 0
    #     cell = 0
    #
    #     for i in range(256):
    #         segmentIdx = self.seq.learnOnNewSegment(c, cell)
    #
    #         if i < 255:
    #             assert segmentIdx == i
    #         if i > 254:
    #             assert segmentIdx == -1
    #
    # def test_punishPredictedColumn(self):
    #     c = 0
    #     initPerm = 0.5
    #     self.seq.matchingSegments = [i for i in range(0, 21, 3)]
    #     idxColSegments = [0, 5, 11]
    #     self.seq.activeCells = [0]
    #
    #     for synapse in range(128520): # 128520 synapses in column 1
    #         self.seq.synapsePerm[synapse] = initPerm
    #         if synapse < 10:
    #             self.seq.upstreamCellIdx[synapse] = self.seq.activeCells[0]
    #
    #     self.seq.punishPredictedColumn(c, idxColSegments)
    #
    #     for idx in range(128520):
    #         if idx < 10:
    #             assert self.seq.synapsePerm[idx] == initPerm-self.seq.predictedDecrement
    #         else:
    #             assert self.seq.synapsePerm[idx] == initPerm

##########################


    # def test_activatePredictedCells(self):
    #
    #     firstSegCell16 = 4080 # 16*255 == 1st segement on cell 64
    #     firstSegCell23 = 5865 # 23*255 == 1st segment on cell 73
    #     segments = self.seq.maxSegmentsPerCell
    #     synapses = self.seq.maxSynapsePerSegment
    #     for startIdx in [firstSegCell16, firstSegCell23]:
    #         for segmentIdx in range(startIdx, startIdx+segments):
    #             for synapseIdx in range(segmentIdx*synapses, segmentIdx*synapses+segments):
    #                 if segmentIdx % 2 == 0:
    #                     self.seq.synapsePerm[synapseIdx] = 1
    #                 else:
    #                     self.seq.synapsePerm[synapseIdx] = 0
    #                 self.seq.upstreamCellIdx[synapseIdx] = 16
    #         # Half above threshold; half below
    #
    #     #Case 1
    #     self.seq.activeCells = []
    #     numActivePotentialSynapses_1 = self.seq.activatePredictedCells(0, prevActiveCells = [])
    #
    #     assert numActivePotentialSynapses_1 == {}
    #     assert self.seq.activeSegments == []
    #     assert self.seq.matchingSegments == []
    #
    #     #Case 2
    #     self.seq.activeCells = [16, 23] # 1st cell in column 2; 8th cell in column 2
    #     numActivePotentialSynapses_2 = self.seq.activatePredictedCells(2, [16, 23])
    #
    #     for startIdx in [firstSegCell16, firstSegCell23]:
    #         for segmentIdx in range(startIdx, startIdx+segments):
    #             assert numActivePotentialSynapses_2[segmentIdx] == self.seq.maxSynapsePerSegment
    #
    #     for a in [1,2]:
    #         segs = [i for i in range(firstSegCell16, firstSegCell16+segments, a)]
    #         segs2 = [i for i in range(firstSegCell23+1, firstSegCell23+segments, a)] # +1 is due to firstSegCell23 starting on odd integer
    #         segs.extend(segs2)
    #         if a % 2 == 0:
    #             assert self.seq.activeSegments == segs
    #         else:
    #             segs.insert(segments,firstSegCell23)
    #             assert self.seq.matchingSegments == segs


if __name__ == '__main__':
    unittest.main()
