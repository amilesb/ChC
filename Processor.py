''' This script moves information from one type of encoding to the next to
form a regenerative hierarchically designed cycle.'''

import numpy as np
import os.path
import pickle
from scipy import stats, ndimage
from collections import Counter

from ChC import ChC, AIS
from Polygon import Polygon, Target

#### To do write tests test_extractSDR
#### Run experiment 1!!!!!
# story:
# 1) find targs
# 2) use sequence memory to find faster
#     a) sparse levels overlapping so multiple sdrs activated
# 3) representations need topography as if random then overlapping
#  for this fig. intentially set indexes to 1 over from true values internal move should find easily if topography but will struggle if random
#     will compete north south versus north and northeast so harder for sequence
#     memory to assist in narrowing search
# 4) generate firing rate output
#     a) discuss temporal coding as mainstream but segway into interference
#     b) note also if sdr is hit strong 18/20 synapses = strong output but if hit
#        weak say 12/20 then maybe low output firing rate
# 5) topography

'''
output dynamically shapes input receptive field so narrow triangle and wide triangle
even though different in lower level but output next layer input is constant!!!
note example using triangle in 2d but in reality is sdr in 3d and potentially
covering n dims feature space!

BIG IDEA
Use count of how many times target found as proxy for confidence so if target
found 5 times versus others 1 or 2 times then bias network to look for sdr with that 1
hit 5 times
This lends to topography bc circuit i am using is single column serially
touching but in reality array of columns so as col 1 moves away and col x
moves into its position, col 1 can only reliably relay its prediction to
col x if topoographical arrangement.

Also though imagine 2 sdr that are identical except all elements shifted over
1.  if input space underneath is random then they can represent totally different
things but topography then the overlap through the input space can be abstracted
away and used to teach other columns that have never seen that input!!!!!!!
Key idea is as explained above network biased with each movement towards the ones
that are more certain of being true i.e. 5 out of 5 hit right versus only once.
'''

class Processor:

    def __repr__(self):
        return (f'''This class processes information input in analog form to
        extract a binary sparse distributed representation carrying
        spatial information which is complementary to the raw rate encoded
        analog information.  It then pools the relevant information to
        regenerate an analog output which can then cycle through the process
        hierarchically.''')

    def __init__(self):
        '''B'''

        # self.input_array = input_array
        self.REC_FLD_LENGTH = 4
        self.maxFiringRateOutput = 255
        self.countAPPLY_RF = 0
        self.countINTERNAL_MOVE = 0
        self.countEXTERNAL_MOVE = 0
        self.internalNoiseFlag = False
        self.targsINTERNAL = Counter()
        self.suspectFalseTargsInternal = Counter()
        self.correctTargsFound = Counter() # keeps track of targets found at any point in search
        self.falseTargsFound = Counter() # keeps track of false positives at any point in search
        # stride = np.ceil(pShape.pShape[0]/REC_FLD_LENGTH) # 256/4 = 32
        # num_receptive_fields = stride**2 # 1024
        self.sparseNum = {}
        self.pShape = Polygon(array_size=10, x=3, y=3)
        self.pShape.insert_Polygon()

    def extractSDR(self, sparseType, sparseLow=0.02, sparseHigh=0.04, **kwargs):
        ''' Top level function to take an input and run through the network.

        Inputs:
        sparseType       - String: either 'Percent' or 'Exact' for target number
        sparseLow/High   - target sparsity level for extracted SDR
        *kwargs          - object to be defined as pShape (input_array),
                           attachedChC, self.seq, self.topo or parameters to
                           build polygon and attachedChC

        Returns:
        SDR              - extracted SDR
        sdrFoundWholeFlag- Boolean indicating that if true then indexes
                           corresponding to SDR match the lower bound of ground
                           truth indices.  If false, targetIndxs will be
                           incomplete i.e. less than sparse lower bound
                           indicating ambiguity.
        '''

        if kwargs:
            try:
                pShape = kwargs.get('pShape')
                attachedChC = kwargs.get('attachedChC')
            except NameError:
                pShape, attachedChC = self.buildPolygonAndAttachChC(**kwargs)

        self.pShape = pShape
        self.attachedChC = attachedChC
        self.trueTargs = set(pShape.activeElements)

        # build ais
        self.AIS = AIS(pShape, attachedChC)

        fldHEIGHT = self.pShape.input_array.shape[0]
        fldWIDTH = self.pShape.input_array.shape[1]
        intValArray = np.zeros((fldHEIGHT, fldWIDTH))
        self.threshold = np.zeros((fldHEIGHT, fldWIDTH))
        self.threshold[:] = -1

        if sparseType=='Percent':
            sparseLow = int(np.round(self.pShape.size*sparseLow))
            sparseHigh = int(np.round(self.pShape.size*sparseHigh))
        elif sparseType == 'Exact':
            sparseLow = int(sparseHigh)
        if sparseLow > len(self.trueTargs):
            sparseLow = int(len(self.trueTargs))
        self.sparseNum = {'low': sparseLow, 'high': sparseHigh}

        targetIndxs, confidenceFlag = self.applyReceptiveField()

        print('finished applyRF')
        # targetIndxs, indxCounter = self.internalMove(targetIndxs)
        targetIndxs = self.internalMove(targetIndxs)
        print('finished internalMove')
        if self.internalNoiseFlag:
            pass
            ###### Need to implement REWIRING!!

        sdrFoundWholeFlag, targetIndxs = self.externalMove(targetIndxs)

        # firingRateOutput = self.calcInterference(result, self.threshold)

        # if not self.seq:
        #     self.seq = Sequencer(self.sp)

        # if objectToTrain == 'seq':
        #     self.seq.evalActiveColsVersusPreds(winningColumnsInd)
        #
        # if objectToTrain == 'topo':
        #     self.trainTopo()


        return sdrFoundWholeFlag, targetIndxs


    def buildPolygonAndAttachChC(self, array_size=10, form='rectangle', x=4,
                                 y=4, wd=4, ht=3, angle=0,
                                 useTargetSubclass=False, numTargets=False,
                                 numClusters=0, maxInput=255,
                                 useVariableTargValue=False):
        '''Draw a polygon from the polygon class (numpy array) and then attach
        Chandelier Cells to that input.

        Inputs:
        array_size:         integer
        form:               string
        x, y:               integers representing center of polygon
        wd, ht:             integers corresponding to dimensions of 2D polygon
        angle:              float representing rotation angle of polygon
        useTargetSubclass:  Boolean
        numTargets:         integer
        numClusters:        integer

        Returns:
        pShape:             numpy array with specified shape perimeter defined within
        attachedChC:        dictionary which maps chandelier cells to the input_array
        '''

        if useTargetSubclass:
            if not numTargets:
                numTargets = np.floor(array_size*0.02)
            pShape = Target(array_size, numTargets, numClusters, maxInput)
            pShape.insert_Targets(useVariableTargValue)
        else:
            pShape = Polygon(array_size=array_size, form=form, x=x, y=y, width=wd,
                             height=ht, angle=angle)
            pShape.insert_Polygon(useVariableTargValue)

        if os.path.exists(f'ChC_handles/ChC_size_{array_size}'):
            with open(f'ChC_handles/ChC_size_{array_size}', 'rb') as ChC_handle:
                attachedChC = pickle.load(ChC_handle)
        else:
            attachedChC = ChC(pShape)
            ChC_dict, pyc_dict = attachedChC.attach_ChC_Coords(debug=False)
            attachedChC.sort_ChC()
            attachedChC.sort_PyC()
            with open(f'ChC_handles/ChC_size_{array_size}', 'wb') as ChC_handle:
                pickle.dump(attachedChC, ChC_handle)

        return pShape, attachedChC


    def applyReceptiveField(self, prevTargetsFound=0, oscFlag=0,
                            filterIndxs=None):
        ''' Recursive BIG function returns an array of the same size as the
        receptive field filtered by the self.threshold i.e. generates an SDR!

        Inputs:
        prevTargetsFound - float
        oscFlag          - Flag to prevent infinite recursion

        Returns:
        targetIndxs      - list of row, col indices for found targets
        confidenceFlag   - boolean based on if targets were found through 2
                           seperate filtrations or not
        '''

        self.countAPPLY_RF += 1

        weightsAIS = self.calcWeights()

        # Custom filter; This filter is designed to combat high noise
        binaryInputPiece, targetsFound = self.applyThreshold(weightsAIS)
        row, col = self.getNonzeroIndices(binaryInputPiece)
        targetIndxs = [(r, c) for r, c in zip(row, col)]

        # Moving average filter; note this filter while effective falls prey to high noise
        if oscFlag%10 == 0:
            rawFilter = ndimage.uniform_filter(self.pShape.input_array,
                                               size=oscFlag+3, mode='mirror')
            weightsFilter = ndimage.uniform_filter(weightsAIS, size=oscFlag+3,
                                                   mode='mirror')
            weightsFilter = weightsAIS/weightsFilter
            normWeighted = self.pShape.input_array-(rawFilter*weightsFilter)
            num = np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
            indxs = np.c_[np.unravel_index(np.argpartition(normWeighted.ravel(),-num)[-num:], normWeighted.shape)]
            filterIndxs = [tuple(x) for x in indxs.tolist()]

        likelyTargIndxs = list(set(filterIndxs) & set(targetIndxs))
        # possTargIndxs = list(set(filterIndxs) ^ set(targetIndxs))

        # Recursive base case return value
        if self.sparseNum['low'] <= len(likelyTargIndxs) <= self.sparseNum['high']:
            confidenceFlag = True
            return likelyTargIndxs, confidenceFlag
        elif ( (self.sparseNum['low'] <= targetsFound <= self.sparseNum['high'])
               or oscFlag==100
              ):
            confidenceFlag = False
            return targetIndxs, confidenceFlag

        # Base case criteria not met; set parameters and initiate recursive search
        oscFlagUpdated = self.noiseAdjustments(targetsFound, prevTargetsFound,
                                               binaryInputPiece, oscFlag)
        prevTargetsFound = targetsFound

        return self.applyReceptiveField(prevTargetsFound, oscFlagUpdated, filterIndxs)


    def calcWeights(self):
        '''Helper function to retrieve ChC weights and adjust according to AIS
        placement.'''

        RF_AvgToMax = np.mean(self.pShape.input_array)/self.pShape.MAX_INPUT

        numRows = self.pShape.input_array.shape[0]
        numCols = self.pShape.input_array.shape[1]

        weights = np.zeros([numRows, numCols])
        for i in range(numRows):
            for j in range(numCols):
                weights[i, j] = self.attachedChC.total_Active_Weight(PyC_array_element=(i,j),
                                                                     avgPercentFR_RF=RF_AvgToMax)
                weights[i, j] -= self.AIS.ais[i, j]
                if weights[i, j] < 0:
                    weights[i, j] = 0

        return weights


    def applyThreshold(self, weightsAIS):
        '''Threshold image according to ChCs.

        Inputs:
        weightsAIS       - np array of ChC weights adjusted according to AIS
                           placement

        Returns:
        binaryInputPiece - np binary array of same size as input with value 1
                           wherever input>threshold and 0 elsewhere
        targetsFound     - int count of targets found

        '''

        chcStep = self.pShape.MAX_INPUT/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT

        # Check/Set threshold
        if self.threshold.any() < 0:
            self.threshold[:] = np.mean(self.pShape.input_array)/chcStep

        result = np.zeros([self.pShape.input_array.shape[0], self.pShape.input_array.shape[1]])

        result = self.pShape.input_array - (chcStep*weightsAIS)
        binaryInputPiece = np.where(result > self.threshold, 1, 0)

        targetsFound = np.count_nonzero(binaryInputPiece > 0)

        return binaryInputPiece, targetsFound


    def noiseAdjustments(self, targetsFound, prevTargetsFound, binaryInputPiece,
                         oscFlag):
        '''Helper function to adjust AIS position and/or threshold for targeted
        input cells depending on the number of targets found.

        Returns:

        oscFlag     - integer value used for updating moving average filter and
                      terminating base case for recursive function
                      'applyReceptiveField'

        '''

        # Note AIS moves opposite to self.threshold; decrease AIS means closer to cell body
        # i.e. increase ChC which can affect output
        if targetsFound > self.sparseNum['high']:
            self.moveAIS(binaryInputPiece, 'decrease')
            if prevTargetsFound > self.sparseNum['high']: # assume gradient; punish those at the higher end
                self.adjustThreshold(binaryInputPiece, 'up')
            else:
                oscFlag += 1
        if targetsFound < self.sparseNum['low']:
            binaryInvert = 1-binaryInputPiece
            self.moveAIS(binaryInvert, 'increase')
            if prevTargetsFound < self.sparseNum['low']: # assume gradient; boost those at the lowest end
                self.adjustThreshold(binaryInputPiece, 'down')
            else:
                oscFlag += 1

        return oscFlag


    def moveAIS(self, binaryInputPiece, direction, max_wt=40):
        '''Helper function to get indices of nonzero elements and adjust AIS
        value for each of those elements.'''

        nonzero = binaryInputPiece.nonzero()
        row = nonzero[0]
        col = nonzero[1]
        # self.getNonzeroIndices(binaryInputPiece)
        for idx1, idx2 in zip(row, col):
            if direction == 'decrease':
                self.AIS.ais[idx1, idx2] = max(0, self.AIS.ais[idx1, idx2]-1)
            if direction == 'increase':
                self.AIS.ais[idx1, idx2] = min(max_wt, self.AIS.ais[idx1, idx2]+1)


    def adjustThreshold(self, binaryInputPiece, direction):
        '''Helper function to adjust the threshold values on hits up to help
        remove stray targets or threshold values on misses down to help locate
        additional targets.'''

        row_hit, col_hit = self.getNonzeroIndices(binaryInputPiece)
        hits = [(r, c) for r, c in zip(row_hit, col_hit)]

        binaryInvert = 1-binaryInputPiece
        row_miss, col_miss = self.getNonzeroIndices(binaryInvert)
        misses = [(r, c) for r, c in zip(row_miss, col_miss)]

        dist_char = binaryInputPiece.size*0.35

        if direction == 'up':
            for i, hit in enumerate(hits):
                if len(misses) == 0: # guard against edge case where all values are hits
                    dist = 0
                else:
                    dist = self.computeMinDist(hit, misses)
                self.threshold[hit[0], hit[1]] += np.exp(dist/dist_char)

        if direction == 'down':
            for i, miss in enumerate(misses):
                if len(hits) == 0:
                    dist = 0
                else:
                    dist = self.computeMinDist(miss, hits)
                self.threshold[miss[0], miss[1]] -= np.exp(dist/dist_char)
                if self.threshold[miss[0], miss[1]] < 0:
                    self.threshold[miss[0], miss[1]] = 0


    def getNonzeroIndices(self, binaryInputPiece):
        '''Helper function to get indices of nonzero elements.'''
        nonzero = binaryInputPiece.nonzero()
        row = nonzero[0]
        col = nonzero[1]

        return row, col


    def computeMinDist(self, subject, listOfTargs):
        '''Helper function to compute distance from thresholded zero elements to
        farthest distance from ALL postive targets.'''
        dist = 10e10
        for targ in listOfTargs:
            d = np.sqrt((targ[0]-subject[0])**2 + (targ[1]-subject[1])**2)
            if d < dist:
                dist = d

        return dist


    def internalMove(self, targetIndxs, clearTargIndxCounter=True,
                     numSaccades=5, noiseFilter=3):
        '''Internal movement to sift out noise.  Note this is a recursive
        function built on top of another recursive function
        (applyReceptiveField).

        Inputs:
        targetIndxs      - list of indices of input cells above the (direct
                           AIS/ChC modulated) and (indirect dynamic -- meant to
                           simulate AIS length changing) self.threshold

        Returns:
        targetIndxs      - list of row, col indices for found targets
        '''

        if clearTargIndxCounter:
            self.targsINTERNAL.clear()

        self.countINTERNAL_MOVE += 1
        originalInput = self.pShape.input_array.copy()

        print('internal move', self.countINTERNAL_MOVE)
        print('before update targsInternal', self.targsINTERNAL)
        self.targsINTERNAL.update(targetIndxs)
        print('after update', self.targsINTERNAL)

        for i in range(numSaccades):
            self.pShape.add_Noise()
            newIndxs, confidenceFlag = self.applyReceptiveField()
###################3 could return flag from applyRF and if targs from filter and other then update counter more!???
            self.targsINTERNAL.update(newIndxs)
            self.pShape.input_array = originalInput.copy() # restore input

        print('after internalmovement', self.targsINTERNAL)

        suspectedFalseTargsDueToNoise = []
        targetIndxs = []
        for targ in self.targsINTERNAL:
            if self.targsINTERNAL[targ] < noiseFilter:
                suspectedFalseTargsDueToNoise.append(targ)
            else:
                targetIndxs.append(targ)

        if self.countINTERNAL_MOVE > 1:
            print('length targs internal', len(self.targsINTERNAL))
            print('current target indexes', targetIndxs)

        targetsFound = len(targetIndxs)

        if (self.sparseNum['low'] <= targetsFound <= self.sparseNum['high']):
            self.internalNoiseFlag = False
            return targetIndxs
        else:
            self.suspectFalseTargsInternal.update(suspectedFalseTargsDueToNoise)
            self.thresholdOutSuspFalseTargs(suspectedFalseTargsDueToNoise)
            self.pShape.input_array = originalInput.copy() # restore input to re-process with noisy input cells thresholded out
            targetIndxs, confidenceFlag = self.applyReceptiveField()
            unchecked = self.pShape.input_array.size-len(self.targsINTERNAL)
            if unchecked <= self.sparseNum['high']:
                self.internalNoiseFlag = True
                bestGuess = self.targsINTERNAL.most_common(self.sparseNum['low'])
                targetIndxs = []
                for indxAndCount in bestGuess:
                    targetIndxs.append(indxAndCount[0])
                return targetIndxs
            else:
                print('hey look i run')
                return self.internalMove(targetIndxs, clearTargIndxCounter=False)


    def thresholdOutSuspFalseTargs(self, suspectedFalseTargsDueToNoise):
        '''Helper function to remove suspected false targets from consideration.'''

        for falseTarg in suspectedFalseTargsDueToNoise:
            self.AIS.ais[falseTarg] = 0 # place ais at cell body
            self.threshold[falseTarg] += 0.1*self.pShape.MAX_INPUT # inhibit these cells


    def externalMove(self, targetIndxs):
        '''External movement to simulate changing gradients across input space.
        Note this is a recursive function built on top of 2 nested recursive
        functions (internalMove and applyReceptiveField).

        Inputs:
        targetIndxs       - list of indices of input cells above the (direct
                            AIS/ChC modulated) and (indirect dynamic -- meant to
                            simulate AIS length changing) self.threshold

        Returns:
        targetIndxs       - list of row, col indices for found targets
        sdrFoundWholeFlag - Boolean indicating that if true then indexes
                            corresponding to SDR match the lower bound of ground
                            truth indices.  If false, targetIndxs will be
                            incomplete i.e. less than sparse lower bound
                            indicating ambiguity.
        '''

        originalInput = self.pShape.input_array.copy()
        print('targetindxs external move', targetIndxs)

        while True:
            suspectedTargs = set(targ for targ in targetIndxs)
            correctTargs = suspectedTargs & self.trueTargs
            incorrect = suspectedTargs - self.trueTargs

            self.correctTargsFound.update(correctTargs)
            self.falseTargsFound.update(incorrect)

            if len(correctTargs) >= self.sparseNum['low']:
                return True, list(correctTargs)
            if len(self.correctTargsFound) >= self.sparseNum['low']:
                return False, list(correctTargs)

            self.simulateExternalMove()
            newIndxs, confidenceFlag = self.applyReceptiveField()
            newIndxs = self.internalMove(newIndxs)

            self.countEXTERNAL_MOVE += 1

            self.pShape.input_array = originalInput.copy()

            return self.externalMove(newIndxs)


    def simulateExternalMove(self):
        '''Helper function to adjust contrast across input space.'''

        randFirst = np.random.randint(0, self.pShape.MAX_INPUT)
        randSecond = np.random.randint(0, self.pShape.MAX_INPUT-randFirst)
        rowStart = np.random.randint(0, self.pShape.input_array.shape[1])
        colStart = np.random.randint(0, self.pShape.input_array.shape[0])

        if np.random.randint(0, 2) == 0:
            self.pShape.create_Gradient(is_horizontal=True, gradStop=randFirst,
                                        rowStart=rowStart)
            self.pShape.create_Gradient(is_horizontal=False, gradStop=randSecond,
                                        colStart=colStart)
        else:
            self.pShape.create_Gradient(is_horizontal=False, gradStop=randFirst,
                                        colStart=colStart)
            self.pShape.create_Gradient(is_horizontal=False, gradStop=randSecond,
                                        rowStart=rowStart)

        return



    def calcInterference(self, result, threshold):
        '''Examine values within receptive field and calculate interference to
        determine how confident receptive field chcStep is.

        Inputs:
        result - numpy square array with length equal to receptive field and
                 values corresponding to input values after chandelier cell filtration.
        self.threshold - scalar self.threshold value

        Returns:
        firingRateOutput - Scalar value representing strength of output (input
                           to next hierarchical layer)
        '''
        aboveThresh = np.where(result > threshold, result, 0)
        aboveThresh = np.reshape(aboveThresh, -1)
        aboveThresh = aboveThresh[aboveThresh!=0]

        belowThresh = np.where(result <= threshold, result, 0)
        belowThresh = np.reshape(belowThresh, -1)
        belowThresh = belowThresh[belowThresh!=0]

        uValue, pValue = stats.wilcoxon(aboveThresh, belowThresh)

        # key concept in wilcoxon rank sum test uValue ranges from 0 (= complete sepatation) to n1*n2 (= no separation) where n1 and n2 are number of samples from each distribution (here size of receptive field).  Note, bc of the threshold splitting above and below the uValue will be zero.  The pValue though tells the probability that this is true.
        return min(1/pValue, self.maxFiringRateOutput)
