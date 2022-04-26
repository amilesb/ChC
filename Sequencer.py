import numpy as np
from numpy.random import default_rng

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
        self.cellsPerColumn = 32

        self.activationThreshold = 13
        self.initialPerm = 0.21
        self.connectedPerm = 0.5
        self.minThreshold = 10
        self.maxNewSynapseCount = 20
        self.permIncrement = 0.1
        self.permDecrement = 0.1
        self.maxSegmentsPerCell = 255
        self.maxSynapsePerSegment = 255

        self.activeCells = [] # Note the purpose of this list is to get remade at each iteration
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []

        self.totalCells = self.columnCount*self.cellsPerColumn # default is 65536

        self.synapses = {'upstreamCellIdx': None, 'permanence': None}

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

        for c in range(self.columnCount):
            if c in winningColumnsIdx:
                columnActiveSegments, idxColSegments = self.countActiveSegments(c)
                if columnActiveSegments > 0:
                    numActivePotentialSynapses = self.activatePredictedCells(c, prevActiveCells)
                    self.updatePerms(idxColSegments, prevActiveCells,
                                     numActivePotentialSynapses)
                else:
                    self.burstColumn(c)
            else:
                columnActiveSegments = self.countActiveSegments(c)
                if columnActiveSegments > 0:
                    self.punishPredictedColumn(c)


    def countActiveSegments(self, c):
        '''Input a column index, c, and return the number of active segments
        along with a list of indices for the dendritic segments for all cells in
        the column.'''

        idxColSegments = []

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            idxCellSegments = self.indexHelper('cell', cellIdx)
            columnActiveSegments = set(idxCellSegments) & set(self.activeSegments) # note here activeSegments refers to previous iteration
            if columnActiveSegments:
                self.activeCells.append(cellIdx)
                self.winnerCells.append(cellIdx)
            idxColSegments.extend(idxCellSegments)

        return len(columnActiveSegments), idxColSegments

    def activatePredictedCells(self, c, prevActiveCells): # maybe this is activate segments?
        '''Input a column index, c, and activate any cells that were in a
        predicted state i.e. possess an active segment.  The function also
        records those synapses that were not active but could potentially
        become active at a later point (matching segments).  The function
        returns a dictionary consisting of the number of active potential
        synapses for each segment (to determine how many new connections to
        grow).'''

        numActivePotentialSynapses = {}

        idxColCells = self.indexHelper('column', c)
        for cellIdx in idxColCells:
            if cellIdx in self.activeCells:
                idxCellSegments = self.indexHelper('cell', cellIdx)
                for segmentIdx in idxCellSegments:
                    idxCellSynapses = self.indexHelper('segment', segmentIdx)
                    numActiveConnected = 0
                    numActivePotential = 0
                    for synapseIdx in idxCellSynapses:
                        if self.synapses[synapseIdx]['upstreamCellIdx'] in prevActiveCells:
                            if self.synapses[synapseIdx]['permanence'] > self.connectedPerm:
                                numActiveConnected += 1
                            if self.synapses[synapseIdx]['permanence'] >= 0:
                                numActivePotential += 1

                    if numActiveConnected>self.activationThreshold:
                        self.activeSegments.append(segmentIdx)

                    if numActivePotential>self.minThreshold:
                        self.matchingSegments.append(segmentIdx)

                    numActivePotentialSynapses[segmentIdx] = numActivePotential

        return numActivePotentialSynapses

    def updatePerms(idxColSegments, prevActiveCells, numActivePotentialSynapses):
        '''Input a list of dendritic segments for all cells in a column, a list
        of the active cells from the previous timestamp, and a dictionary
        consisting of segment index as the key and number of potential synapses
        (i.e. association with an upstream cell aka perm>=0) as the value.  This
        function cycles through each segment's synapses to update the permanence
        value based on whether or not the upstream cell associated with that
        synapse was active during the previous iteration.
        It then computes the number of new synapses to add for each segment, if
        any and grows connections to new upstream cells.'''

        for segmentIdx in idxColSegments:
            idxCellSynapses = self.indexHelper('segment', segmentIdx)
            for synapse in idxCellSynapses:
                if self.synapses[synapseIdx]['upstreamCellIdx'] in prevActiveCells:
                    self.synapses[synapseIdx]['permanence'] += self.permIncrement
                else:
                    self.synapses[synapseIdx]['permanence'] += self.permDecrement

            newSynapseCount = (self.maxNewSynapseCount -
                               numActivePotentialSynapses[segmentIdx])

            self.growSynapses(self, segmentIdx, newSynapseCount)

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
        self.activeCells.extend([c for c in range(c, c+32)])

    def punishPredictedColumn(self, c):
        pass



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
