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
                                            low=self.connectedPerm-0.1,
                                            high=self.connectedPerm+0.1)
            synapses[i] = {'index': input_indices, 'permanence': permanences,
                           'boostScore': 1, 'activeDutyCycle': [],
                           'overlapDutyCycle': []}

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

    def retrieveBoost(self, c):
        '''Input a mini-column index and return the homeostatic boost factor for
        that mini-column.'''
        pass

    def computeWinningCOlumns(self, overlapScore, stimulusT=8, numW=40):  # stimulusT = self.stimulusThreshold; numW = self.numActiveColumnsPerInhArea
        '''Inputs a list of each mini-column's overlap with the input space and
        ranks each mini-column.  Ties are broken pseudorandomly and a list with
        the mini-columns with greatest overlap is returned.'''

        overlapArray = np.array(overlapScore)
        topColumnsInd = np.argpartition(overlapArray, -numW)[-numW:]

        winningColumnsInd = [i for i in topColumnsInd if overlapArray[i]>stimulusT]

        return winningColumnsInd

    def updateSynapseParameters(self, winningColumnsInd, currentInput):
        '''Inputs a list of winning mini-column indices along with the current
        input as a binary array and updates the permancence values for the
        winning mini-columns only.  In addition, homeostatic boosting is
        implemented by reviewing how active a mini-column is relative to its
        neighbors and boosting up or down to normalize all mini-columns towards
        the same activity level.'''

        for c in winningColumnsInd:
            potentialSynapses = synapses[c]['index']
            for s in potentialSynapses:
                if currentInput[s] == 1:
                    synapses[c]['permanence'][s] += self.synPermActiveInc
                    synapses[c]['permanence'][s] = min(1.0, synapses[c]['permanence'][s])
                else:
                    synapses[c]['permanence'][s] += self.synPermInactiveDec
                    synapses[c]['permanence'][s] = max(0.0, synapses[c]['permanence'][s])

        for c in range(self.columnCount):
            self.updateActiveDutyCycle(c, winningColumnsInd)


            # overlapDutyCycle is sliding avg how oftec c has had overlap > stimulusThreshold over last 1000 iterations

    def updateActiveDutyCycle(self, c, winningColumnsInd, n_iterations=1000):
        '''Input a mini-column index, a list of the winning columns, and update
        its active duty cycle based on a moving average of how often that
        column has been active after inhibition over the last n iterations.'''

        activeDutyCycleList = synapses[c]['activeDutyCycle']
        if c in winningColumnsInd:
            activeDutyCycleList.append(1)
        else:
            activeDutyCycleList.append(0)

        if len(activeDutyCycleList) > n_iterations:
            del activeDutyCycleList[0]

        return sum(activeDutyCycleList)/len(activeDutyCycleList)

if __name__ == '__main__':
    pass

    # sp = Spatial_Pooler(1200)
    # print(sp.synapses[0])





# synapse == data structure represing permanence value and source input index


# potentialSynapses = []  list of potential synapses and their permanence
# connectedSynapses = []  subset of potentialSynapses where perm is greater than connectedPerm
