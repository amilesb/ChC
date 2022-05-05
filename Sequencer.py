import numpy as np
from numpy.random import default_rng
from collections import defaultdict
import random

from Spatial_Pooler import Spatial_Pooler
from Encoder import Encoder

class SequenceMemory:

    def __repr__(self):
        return (f'''This class implements the temporal memory algorithm (here
                renamed as SequenceMemory to point out its enhanced sequencing
                of SDRs mapped to a topographical input space) as described in
                HTM theory.  The SequenceMemory incorporates multiple cells per
                mini-column to enable different contextual representations of
                the same generic input (SDR of minicolumns).  IOW, if input is
                minicolumns A, C, and H then context 1 is A1, C9, H2 whereas
                context 2 could be A3, C5, H7 etc...  The property of sparsity
                reduces chance of noise overlap.  Additionally, each context
                makes prediction of next base input and corresponding context(s)
                Unpredicted input results in all cells in minicolumn firing (aka
                bursting).''')


    def __init__(self, spatialPooler):
        # '''Num mini-columns = 2048, 40 is roughly 2% of 2048, the percent of
        # inputs in radius initialized to be in column's potential synapses note
        # want ~ 50% or 15-20 on at beginning == (40*0.5*0.75 = 15).  The
        # connectedPerm value is arbitrary but the inc and dec are relative to
        # it.'''
        self.sp = spatialPooler
        self.columnCount = self.sp.columnCount # default is 2048
        self.cellsPerColumn = 8

        self.activationThreshold = 13
        self.initialPerm = 0.2
        self.connectedPerm = 0.5
        self.minThreshold = 0
        self.maxNewSynapseCount = 20
        self.permIncrement = 0.1
        self.permDecrement = 0.1
        self.predictedDecrement = 0.05
        self.maxSegmentsPerCell = 255
        self.maxSynapsePerSegment = 63

        self.activeCells = []
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []
        self.numActivePotentialSynapses = defaultdict(int)
        self.segmentsInUse = {}

        self.totalCells = self.columnCount*self.cellsPerColumn


        # '''NOTE ON IMPLEMENTATION - There are an extraordinary number of synapses
        # ~2048col*8cells/col*255segs/cell*63syns/seg.
        # This amount is too large for a dictionary so using numpy array to speed
        # up.  Note, each synapse consists of 3 parameters: an upstream cell, a
        # downstream cell, and a permanence (or connection strength).  In this
        # implementation, the downstream cell identity is implied by the index
        # into 2 numpy arrays while the other 2 parameters are the corresponding
        # value in each of the respective arrays.  Each initialized to -1 as this
        # represents a null value.  Using such a large numpy array is cheapest
        # with only 8 bits (dtype=int8) as a result permanence values and
        # corresponding initial values, increments, and decrements are scaled from
        # 0-1 to 1-10.'''

        self.upstreamCellIdx = np.ones(self.totalCells*self.maxSegmentsPerCell*
                                       self.maxSynapsePerSegment)
        self.upstreamCellIdx = self.upstreamCellIdx*(-1)

        self.synapsePerm = self.upstreamCellIdx.copy()

    def processInputThroughSP(self, currentInput):
        '''Starting interface for SequenceMemory.  Takes a spatial pooler
        attached to the sequence memory instance and uses the currentInput (a
        binary array) to pass through the spatial pooler functions.  Returns a
        list of indices for the active (winning) columns.'''

        overlapScore = self.sp.computeOverlap(currentInput)
        winningColumnsIdx = self.sp.computeWinningCOlumns(overlapScore)
        self.sp.updateSynapseParameters(winningColumnsIdx, overlapScore, currentInput) # learning in spatial pooler

        return winningColumnsIdx

    def evalActiveColsVersusPreds(self, winningColumnsIdx):

        prevActiveCells = self.activeCells
        self.activeCells = []
        prevWinnerCells = self.winnerCells
        self.winnerCells = []
        prevActiveSegments = self.activeSegments
        self.activeSegments = []
        prevMatchingSegments = self.matchingSegments
        self.matchingSegments = []
        prevNumActivePotentialSynapses = self.numActivePotentialSynapses
        self.numActivePotentialSynapses = defaultdict(int)

        for c in range(self.columnCount):
            if c in winningColumnsIdx:
                numColActiveSegments, idxColSegments = self.countActiveSegments(c)
                if columnActiveSegments > 0:
                    self.numActivePotentialSynapses = self.activatePredictedCells(c, prevActiveCells)
                    self.updatePerms(idxColSegments, prevActiveCells)
                else:
                    self.burstColumn(c)
            else:
                numColumnMatchingSegments, idxColSegments = self.countMatchingSegments(c)
                if columnActiveSegments > 0:
                    self.punishPredictedColumn(c, idxColSegments)


    def countActiveSegments(self, c):
        '''Input a column index, c, and return the number of active segments
        along with a list of indices for the dendritic segments for all cells in
        the column.'''

        idxColSegments = []
        columnActiveSegments = []

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            idxCellSegments = self.indexHelper('cell', cellIdx)
            cellActiveSegments = set(idxCellSegments) & set(self.activeSegments) # here activeSegments refers to previous iteration
            columnActiveSegments.extend(cellActiveSegments)
            if cellActiveSegments:
                self.activeCells.append(cellIdx)
                self.winnerCells.append(cellIdx)
            idxColSegments.extend(idxCellSegments)

        return len(columnActiveSegments), idxColSegments

    def countMatchingSegments(self, c):
        '''Input a column index, c, and return the number of matching segments
        along with a list of indices for the dendritic segments for all cells in
        the column.'''

        idxColSegments = []
        columnMatchingSegments = []

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            idxCellSegments = self.indexHelper('cell', cellIdx)
            cellMatchingSegments = set(idxCellSegments) & set(self.matchingSegments)
            columnMatchingSegments.extend(cellMatchingSegments)
            idxColSegments.extend(idxCellSegments)

        return len(columnMatchingSegments), idxColSegments

    def activatePredictedCells(self, c, prevActiveCells): # maybe this is activate segments?
        '''Input a column index, c, and activate any cells that were in a
        predicted state i.e. possess an active segment.  The function also
        records those synapses that were not active but could potentially
        become active at a later point (matching segments).  The function
        returns a dictionary consisting of the number of active potential
        synapses for each segment (to determine how many new connections to
        grow).'''

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            if cellIdx in self.activeCells:
                idxCellSegments = self.indexHelper('cell', cellIdx)
                for segmentIdx in idxCellSegments:
                    idxCellSynapses = self.indexHelper('segment', segmentIdx)
                    numActiveConnected = 0
                    numActivePotential = 0
                    for synapseIdx in idxCellSynapses:
                        if self.upstreamCellIdx[synapseIdx] in prevActiveCells:
                            if self.synapsePerm[synapseIdx] > self.connectedPerm:
                                numActiveConnected += 1
                            if self.synapsePerm[synapseIdx] >= 0:
                                numActivePotential += 1

                    if numActiveConnected>self.activationThreshold:
                        self.activeSegments.append(segmentIdx)

                    if numActivePotential>self.minThreshold:
                        self.matchingSegments.append(segmentIdx)

                    self.numActivePotentialSynapses[segmentIdx] = numActivePotential

        return self.numActivePotentialSynapses

    def updatePerms(self, idxColSegments, prevActiveCells):
        '''Input a list of dendritic segments for all cells in a column, a list
        of the active cells from the previous timestamp.  This function cycles
        through each segment's synapses to update the permanence value based on
        whether or not the upstream cell associated with that synapse was active
        during the previous iteration. It then computes the number of new
        synapses to add for each segment, if any and grows connections to new
        upstream cells.'''

        for segmentIdx in idxColSegments:
            idxCellSynapses = self.indexHelper('segment', segmentIdx)
            for synapseIdx in idxCellSynapses:
                if self.upstreamCellIdx[synapseIdx] in prevActiveCells:
                    self.synapsePerm[synapseIdx] += self.permIncrement
                else:
                    self.synapsePerm[synapseIdx] -= self.permDecrement

            newSynapseCount = (self.maxNewSynapseCount -
                               self.numActivePotentialSynapses[segmentIdx])

            self.growSynapses(segmentIdx, newSynapseCount)

    def indexHelper(self, type, metaIdx):
        '''Simple helper function select appropriate indices.  Inputs a type as
        a string variable set to: column, cell, or segment along with an index
        corresponding to the hierarchically defined structure above.  The
        function returns a list of indeices for the sub-hierarchical
        structure.'''

        if type == 'column':
            index = metaIdx*self.cellsPerColumn
            idxs = [i for i in range(index, index+self.cellsPerColumn)]
        elif type == 'cell':
            index = metaIdx*self.maxSegmentsPerCell
            idxs = [i for i in range(index, index+self.maxSegmentsPerCell)]
        elif type == 'segment':
            index = metaIdx*self.maxSynapsePerSegment
            idxs = [i for i in range(index, index+self.maxSynapsePerSegment)]
        else:
            raise TypeError('Incorrect meta-structure requested.')

        return idxs

    def growSynapses(self, segmentIdx, newSynapseCount):
        pass

    def burstColumn(self, c):
        '''Activate all cells within a column.'''

        idxColSegments = []
        for cellIdx in range(c, c+self.cellsPerColumn):
            self.activeCells.append(cellIdx)
            idxColSegments.extend(self.indexHelper('cell', cellIdx))
        matchSegsInCol = set(self.matchingSegments) & set(idxColSegments)
        if matchSegsInCol:
            learningSegmentIdx = self.bestMatchingSegment(c, matchSegsInCol)
            winnerCell = np.floor(learningSegmentIdx/self.maxSegmentsPerCell)
        else:
            winnerCell = self.leastUsedCell(c)
            learningSegmentIdx = self.learnOnNewSegment(c, winnerCell)

        self.winnerCells.append(winnerCell)

        idxCellSynapses = self.indexHelper('segment', learningSegmentIdx)

        for synapseIdx in idxCellSynapses:
            if self.upstreamCellIdx[synapseIdx] in self.activeCells:
                self.synapsePerm[synapseIdx] += self.permIncrement
            else:
                self.synapsePerm[synapseIdx] -= self.permDecrement

        newSynapseCount = (self.maxNewSynapseCount -
                           self.numActivePotentialSynapses[learningSegmentIdx])

        self.growSynapses(learningSegmentIdx, newSynapseCount)

    def bestMatchingSegment(self, c, matchSegsInCol):
        '''Input a column index, c and a list of indices for all the matching
        segments (i.e. segments in column with potential synapses).  Return the
        segment index with the largest number of active potential synapsess.'''

        bestMatchingSeg = None
        bestScore = -1
        for segmentIdx in matchSegsInCol:
            if self.numActivePotentialSynapses[segmentIdx] > bestScore:
                bestMatchingSeg = segmentIdx
                bestScore = self.numActivePotentialSynapses[segmentIdx]

        return bestMatchingSeg

    def leastUsedCell(self, c):
        '''Input a column index, c and return a cell index corresponding to the
        least used cell in that column.'''

        leastUsedCells = {}

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            idxCellSegments = self.indexHelper('cell', cellIdx)
            cellActiveSegments = set(idxCellSegments) & set(self.activeSegments)
            leastUsedCells[cellIdx] = len(cellActiveSegments)

        minval = min(leastUsedCells.values())
        leastUsedCellsList = [k for k,v in leastUsedCells.items() if v == minval]

        return random.choice(leastUsedCellsList)

    def learnOnNewSegment(self, c, cell):
        '''Input a column index, c, and cell index, cell.  Return segment index
        of first available segment or -1 if none available.'''

        key = (c, cell)
        try:
            lastUsedSeg = self.segmentsInUse[key]
            nextSeg = lastUsedSeg+1
            self.segmentsInUse[key] = nextSeg
        except KeyError:
            nextSeg = 0
            self.segmentsInUse[key] = nextSeg

        if nextSeg == self.maxSegmentsPerCell:
            self.segmentsInUse[key] = self.maxSegmentsPerCell-1
            nextSeg = -1

        return nextSeg

    def punishPredictedColumn(self, c, idxColSegments):
        '''Input a column index, c, that contains matching segments and punish
        the synapses that caused these false positive matches.'''

        falsePositiveSegments = set(idxColSegments) & set(self.matchingSegments)

        for segmentIdx in falsePositiveSegments:
            idxCellSynapses = self.indexHelper('segment', segmentIdx)
            for synapseIdx in idxCellSynapses:
                if self.upstreamCellIdx[synapseIdx] in self.activeCells:
                    self.synapsePerm[synapseIdx] -= self.predictedDecrement


'''Combinatorally speaking, the SequenceMemory algorithm needs to encompass an
enormous potential number of connections.  It is difficult to implement this
using data structures in software because of their massive size causing memory
issues and bogging down the program.  The strategy used here is to assign each
synapse to a unique integer and then to carefully select only those synapses
that are needed at each timestamp.  As a result, between each main iteration the
list(s) of active columns, cells, segments, and synapses need to be cleared to
allow the next pass through to work from a clean slate.  It is imperative that
careful consideration is made before changing this synapse labeling scheme to
ensure the algorithm still behaves as desired.'''
