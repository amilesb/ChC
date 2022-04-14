import numpy as np
from numpy.random import default_rng

from Encoder import Encoder


class Spatial_Pooler:

    def __repr__(self):
        return (f'''This class implements the spatial pooling algorithm as
                described in HTM theory.  The spatial pooler uses columns to
                represent mini-columns in the cortex.  Each column connects to a
                certain percent of the input space and is initialized such that
                only a sparse set of columns should be activated for any given
                input.  Through learning, the spatial pooler adjusts its
                permanence values as well as redistributes its connectivity
                wiring to better match the statistics of the input.''')

    def __init__(self, lengthEncoding):
        '''Num mini-columns = 2048, 40 is roughly 2% of 2048, the percent of
        inputs in radius initialized to be in column's potential synapses note
        want ~ 50% or 15-20 on at beginning == (40*0.5*0.75 = 15).  The
        connectedPerm value is arbitrary but the inc and dec are relative to
        it.'''
        self.columnCount = 2048
        self.numActiveColumnsPerInhArea = 40
        self.potentialPct = 0.75
        self.lengthEncoding = lengthEncoding
        self.initialNumConnected = int(np.round(0.75*lengthEncoding))
        self.connectedPerm = 0.2
        self.synPermActiveInc = 0.03
        self.synPermInactiveDec = 0.015
        self.stimulusThreshold = 8
        self.permVar = 0.1

        self.synapses = self.initializeSynapses()


    def initializeSynapses(self):
        '''This function works with the class initialization to connect each
        column to some of the input space and define a starting permanence value
        for each of those columns.'''
        synapses = {}
        for i in range(self.columnCount):
            rng = default_rng()
            input_indices = rng.choice(self.lengthEncoding,
                                       size=self.initialNumConnected,
                                       replace=False)
            permanences = np.random.uniform(size=self.initialNumConnected,
                                            low=self.connectedPerm-self.permVar,
                                            high=self.connectedPerm+self.permVar)
            synapses[i] = {'index': input_indices, 'permanence': permanences,
                           'boostScore': 1, 'activeDutyCycle': [1],
                           'overlapDutyCycle': [1]}

        synapses['meanACTIVE_DC'] = [1] * self.columnCount
        synapses['meanOVERLAP_DC'] = [1] * self.columnCount

        return synapses

    def computeOverlap(self, currentInput):
        '''Current input represents a binary encoding (array) that is obtained
        from applying a threshold over the input space.  This function takes
        each mini-column and returns a list with the scalar overlap score based
        on each mini-columns connectivity to the ON bits in the input.  Note,
        the index position of the overlapScore list is matched to the index
        identifying each mini-column.'''

        overlapScore = []

        for c in range(self.columnCount):
            synapseDict = self.synapses[c]
            connectedSynapses = self.computeConnectedSynapses(synapseDict)
            onBits = {i for i in range(len(currentInput)) if currentInput[i] == 1}
            overlapLength = len(connectedSynapses.intersection(onBits))
            boostScore = synapseDict['boostScore']
            overlapLength *= boostScore
            overlapScore.append(overlapLength)

        return overlapScore

    def computeConnectedSynapses(self, synapseDict):
        '''This function takes a dictionary consisting of an array of input
        indices which reference the input encoding along with a corresponding
        array of permanences and returns a set of indices that contain a
        permanence value greater than the threshold (self.connectedPerm).'''

        subset = set()
        for i, p in zip(synapseDict['index'], synapseDict['permanence']):
            if p > self.connectedPerm:
                subset.add(i)

        return subset

    def computeWinningCOlumns(self, overlapScore, stimulusT=8, numW=40):  # stimulusT = self.stimulusThreshold; numW = self.numActiveColumnsPerInhArea
        '''Inputs a list of each mini-column's overlap with the input space and
        ranks each mini-column.  Ties are broken pseudorandomly and a list with
        the mini-columns with greatest overlap is returned.'''

        overlapArray = np.array(overlapScore)
        topColumnsInd = np.argpartition(overlapArray, -numW)[-numW:]

        winningColumnsInd = [i for i in topColumnsInd if overlapArray[i]>stimulusT]

        return winningColumnsInd

    def updateSynapseParameters(self, winningColumnsInd, overlapScore, currentInput):
        '''Inputs a list of winning mini-column indices along with the current
        input as a binary array and updates the permancence values for the
        winning mini-columns only.  In the second loop, homeostatic boosting is
        implemented by reviewing how active a mini-column is relative to its
        neighbors and boosting up or down to normalize all mini-columns towards
        the same activity level.  Additionally, a list of overlap scores for all
        mini-columns is used to evaluate how often each mini-column has
        sufficient overlap with the input relative to its neighbors.  If any
        mini-columns fall below a minimum overlap duty cycle, all that
        mini-column's permanences are increased (i.e. search for new inputs).

        Note, the boost score is updated serially while ideally (in neuro it is
        almost certainly implemented in parallel via physical connections i.e.
        hardware).  As a result, each boost score incorporates the duty cycle
        from all minicolumns with index less than current from most recent time-
        stamp and all minicolumns with index greater from time-stamp minus 1.
        This translates to each minicolumn seeing a slightly different mean
        active duty cycle.  However, as the duty cycle represents a running
        average of previous 1000 timestamps, this slight difference is
        negligible and an implementation detail only.'''

        for c in winningColumnsInd:
            potentialSynapses = self.synapses[c]['index']
            for idx, s in enumerate(potentialSynapses):
                if currentInput[s] == 1:
                    self.synapses[c]['permanence'][idx] += self.synPermActiveInc
                    self.synapses[c]['permanence'][idx] = min(1.0, self.synapses[c]['permanence'][idx])
                else:
                    self.synapses[c]['permanence'][idx] += self.synPermInactiveDec
                    self.synapses[c]['permanence'][idx] = max(0.0, self.synapses[c]['permanence'][idx])

        for c in range(self.columnCount):
            self.updateActiveDutyCycle(c, winningColumnsInd)
            activeDC_c = self.calcActiveDutyCycle(c)
            meanACTIVE_DC = self.updateAndCalcMeanDutyCycle(c, activeDC_c, type='active')
            boostStrength = np.exp(-(activeDC_c-meanACTIVE_DC))
            self.synapses[c]['boostScore'] = (self.synapses[c]['boostScore']*boostStrength)
            overlapDC_c = self.updateOverlapDutyCycle(c, overlapScore[c])
            meanOVERLAP_DC = self.updateAndCalcMeanDutyCycle(c, overlapDC_c, type='overlap')
            minOverlapDC = self.calcMinOverlapDutyCycle(meanOVERLAP_DC)
            if overlapDC_c < minOverlapDC:
                self.synapses[c]['permanence'] += self.permVar

    def updateActiveDutyCycle(self, c, winningColumnsInd, n_iterations=1000):
        '''Input a mini-column index, a list of the winning columns, and update
        its active duty cycle based on a moving average of how often that
        column has been active after inhibition over the last n iterations.'''

        activeDutyCycleList = self.synapses[c]['activeDutyCycle']
        if c in winningColumnsInd:
            activeDutyCycleList.append(1)
        else:
            activeDutyCycleList.append(0)

        if len(activeDutyCycleList) > n_iterations:
            del activeDutyCycleList[0]

        self.synapses[c].update({'activeDutyCycle': activeDutyCycleList})

    def calcActiveDutyCycle(self, c):
        '''Input a mini-column index and compute the active duty cycle.  Return
        value as a scalar.'''

        activeDutyCycleList = self.synapses[c]['activeDutyCycle']

        return sum(activeDutyCycleList)/len(activeDutyCycleList)

    def updateAndCalcMeanDutyCycle(self, c, dutyCycle_c, type):
        '''The mean duty cycle is initialized to a list of length
        self.columnCount with '1' as each value because both the active and
        overlap duty cycles for each synapse are also initialized to 1.  This
        function takes the new duty cycle for mini-column c and replaces this
        value in the respective list.
        It then calculates the mean duty cycle and returns this value.  (Note,
        this function assumes global inhibition).'''

        if type == 'active':
            label = 'meanACTIVE_DC'
        else:
            label = 'meanOVERLAP_DC'

        self.synapses[label][c] = dutyCycle_c

        return sum(self.synapses[label])/len(self.synapses[label])

    def updateOverlapDutyCycle(self, c, overlap, n_iterations=1000):
        '''The overlap duty cycle measures how often a mini-column c has had
        overlap (active connected synapses) > stimulusThreshold over the last
        1000 iterations.  Function inputs a mini-column index c, and scalar
        overlap value and outputs a scalar overlap duty cycle along with
        updating the synapse dictionary to maintain a running average.'''

        overlapDutyCycleList = self.synapses[c]['overlapDutyCycle']
        if overlap > self.stimulusThreshold:
            overlapDutyCycleList.append(1)
        else:
            overlapDutyCycleList.append(0)

        if len(overlapDutyCycleList) > n_iterations:
            del overlapDutyCycleList[0]

        self.synapses[c].update({'overlapDutyCycle': overlapDutyCycleList})

        return sum(overlapDutyCycleList)/len(overlapDutyCycleList)

    def calcMinOverlapDutyCycle(self, meanOVERLAP_DC):
        '''Calculate the minimum overlap duty cycle based on assumed Poisson
        statistics and a goal for each column's overlap duty cycle to be within
        2 standard deviations of the mean.   meanOVERLAP_DC is a scalar value
        between 0 and 1 representing the percentage overlap for all
        mini-columns.  Returns a scalar value between 0 and 1.'''

        counts_avg = meanOVERLAP_DC*100

        return (counts_avg - (2*np.sqrt(counts_avg)) ) / 100
