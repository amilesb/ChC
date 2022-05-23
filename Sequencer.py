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
        '''Num mini-columns = 2048, 40 is roughly 2% of 2048, the percent of
        inputs in radius initialized to be in column's potential synapses note
        want ~ 50% or 15-20 on at beginning == (40*0.5*0.75 = 15).  The
        connectedPerm value is arbitrary but the inc and dec are relative to
        it.'''
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
        self.maxSynapsePerSegment = 127

        self.activeCells = []
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []
        self.numActivePotentialSynapses = defaultdict(int)
        self.segmentsInUse = {}

        self.totalCells = self.columnCount*self.cellsPerColumn

        self.upstreamCellIdx = np.ones(self.totalCells*self.maxSegmentsPerCell*
                                       self.maxSynapsePerSegment, dtype=np.float32)
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
        print('need to add string note describing function inputs and outputs!')

        self.transferComponentToPrevAndReset()

        for c in range(self.columnCount):
            if c in winningColumnsIdx:
                colActiveSegments, idxColSegments = self.countSegments(c, prevActiveSegments)
                if len(colActiveSegments) > 0:
                    self.activatePredictedCol(c, colActiveSegments,
                                              prevActiveCells, prevWinnerCells)
                else:
                    self.burstColumn(c, prevMatchingSegments, prevActiveCells)
            else:
                colMatchingSegments, idxColSegments = self.countSegments(c, prevMatchingSegments)
                if len(colMatchingSegments) > 0:
                    self.punishPredictedColumn(c, idxColSegments)

        self.activateDendriticSegments()

    def transferComponentToPrevAndReset(self):
        '''Input desired string to append to corresponding self variable.
        Function transfers stored component from previous iteration into
        previous and then resets new one for current iteration.'''

        prevActiveCells = self.activeCells.copy()
        self.activeCells = []
        prevWinnerCells = self.winnerCells.copy()
        self.winnerCells = []
        prevActiveSegments = self.activeSegments.copy()
        self.activeSegments = []
        prevMatchingSegments = self.matchingSegments.copy()
        self.matchingSegments = []
        if len(prevActiveCell) == 0: # don't want to grow synapses on first iteraton
            prevNumActivePotentialSynapses = {key: self.maxNewSynapseCount
                                              for key in
                                              self.numActivePotentialSynapses}
        else:
            prevNumActivePotentialSynapses = self.numActivePotentialSynapses.copy()
        self.numActivePotentialSynapses = defaultdict(int)

        return

    def countSegments(self, c, prevSegments):
        '''Input a column index, c, and return the indices of matching segments
        along with a list of indices for segments for all cells in the column.'''

        idxColSegments = []
        idxMatchOrActiveColSegments = []

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            idxCellSegments = self.indexHelper('cell', cellIdx)
            cellSegments = set(idxCellSegments) & set(prevSegments)
            idxMatchOrActiveColSegments.extend(cellSegments)
            idxColSegments.extend(idxCellSegments)

        return idxMatchOrActiveColSegments, idxColSegments

    def activatePredictedCol(self, c, colActiveSegments, prevActiveCells,
                             prevWinnerCells):
        '''Input a column index, c, a list of the active segments for all cells
        in that column, and a list of the previous active and winner cells.  The
        function generates a list of active cells (i.e. the corresponding cell
        indices to any active segment(s)).  These cells are also selected as
        winner cells marking them as presynaptic candidates for synapse growth
        in the next iteration.  Finally, those synapses which activated the
        segment are reinforced while those that did not contribute are punished
        - included in this update, new synapses are added to each segment up to
        maxNewSynapseCount from the pool of winner cells from the previous
        iteration in order to better recognize new and additional patterns.'''

        for segmentIdx in colActiveSegments:
            cellIdx = np.floor(segmentIdx/self.maxSegmentsPerCell)
            if cellIdx not in self.activeCells:
                self.activeCells.append(cellIdx)
            if cellIdx not in self.winnerCells:
                self.winnerCells.append(cellIdx)

        self.updatePerms(colActiveSegments, prevActiveCells, prevWinnerCells)


    def updatePerms(self, idxColSegments, prevActiveCells, prevWinnerCells):
        '''Input a list of dendritic segments for all cells in a column, a list
        of the active cells from the previous timestamp, and a list of winner
        cells from the previous timestamp (potential pool to connect to).  This
        function cycles through each segment's synapses to update the permanence
        value based on whether or not the upstream cell associated with that
        synapse was active during the previous iteration. It then computes the
        number of new synapses to add for each segment, if any and grows
        connections to new upstream cells.'''

        if isinstance(idxColSegments, int):
            idxColSegments = [idxColSegments]

        for segmentIdx in idxColSegments:
            idxCellSynapses = self.indexHelper('segment', segmentIdx)
            for synapseIdx in idxCellSynapses:
                if self.upstreamCellIdx[synapseIdx] == -1:
                    break
                if self.upstreamCellIdx[synapseIdx] in prevActiveCells:
                    self.synapsePerm[synapseIdx] += self.permIncrement
                else:
                    self.synapsePerm[synapseIdx] -= self.permDecrement

            newSynapseCount = (self.maxNewSynapseCount -
                               self.numActivePotentialSynapses[segmentIdx])

            self.growSynapses(segmentIdx, newSynapseCount, prevWinnerCells)

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


    def growSynapses(self, segmentIdx, newSynapseCount, prevWinnerCells):
        '''Input a segment index as an integer to learn on, the number of new
        synapses to add as integer, and a list of previous winning cells with
        which to connect patterns to.  Add synapses to those winning cells from
        the previous timestamp to the specified segment index.'''

        candidates = prevWinnerCells.copy()
        while len(candidates) > 0 and newSynapseCount > 0:
            preSynapticCell = random.choice(candidates)
            candidates.remove(preSynapticCell)

            alreadyConnected = False
            idxCellSynapses = self.indexHelper('segment', segmentIdx)
            for synapseIdx in idxCellSynapses:
                if self.upstreamCellIdx[synapseIdx] == preSynapticCell:
                    alreadyConnected = True
                    break
                elif self.upstreamCellIdx[synapseIdx] == -1:
                    self.upstreamCellIdx[synapseIdx] = preSynapticCell
                    self.synapsePerm[synapseIdx] = self.initialPerm
                    newSynapseCount -= 1
                    break



    def burstColumn(self, c, prevMatchingSegments, prevActiveCells,
                    prevWinnerCells):
        '''Inputs: column index, c, list of previous iteration matching segments
        and list of previous active and winner cells.  The function first
        activates all cells in the column.  It then searches for any segments
        from the last iteration that match this column's segments (i.e.
        synapses > 0 but not enough to make cells become active).  If any of
        these segments exist, then the winner cell index is selected from the
        best matching segment; otherwise the least used cell in the column is
        selected, in which case the first available segment (if any) is selected
        as the learning segment.  Finally, the learning segment's synapses are
        updated which includes adding new synapses up maxNewSynapseCount to
        learn new patterns.'''

        idxColSegments = []
        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            self.activeCells.append(cellIdx)
            idxColSegments.extend(self.indexHelper('cell', cellIdx))
        matchSegsInCol = set(prevMatchingSegments) & set(idxColSegments)
        if matchSegsInCol:
            learningSegmentIdx = self.bestMatchingSegment(c, matchSegsInCol)
            winnerCell = np.floor(learningSegmentIdx/self.maxSegmentsPerCell)
        else:
            winnerCell = self.leastUsedCell(c)
            learningSegmentIdx = self.learnOnNewSegment(c, winnerCell)

        if learningSegmentIdx == -1:
            return
        else:
            self.winnerCells.append(winnerCell)
            self.updatePerms(learningSegmentIdx, prevActiveCells,
                             prevWinnerCells)


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
                if self.upstreamCellIdx[synapseIdx] == -1:
                    break
                if self.upstreamCellIdx[synapseIdx] in self.activeCells:
                    self.synapsePerm[synapseIdx] -= self.predictedDecrement


    def activateDendriticSegments(self):
        print('need to add this function')
