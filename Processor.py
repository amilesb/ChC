''' This script moves information from one type of encoding to the next to
form a regenerative hierarchically designed cycle.'''

import numpy as np
import random
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

    def __init__(self, sparseType, sparseLow=0.02, sparseHigh=0.04,
                 gaussBlurSigma=0, noiseLevel=0, display=False, **kwargs):
        '''Initialize Processor object with pShape, chandelier cells, sparsity
        parameters, and noise parameters.

        Inputs:
        sparseType       - String: either 'Percent' or 'Exact' for target number
        sparseLow/High   - target sparsity level for extracted SDR
        gaussBlurSigma   - float standard deviation for blurring input
        noiseLevel       - float value for random noise added to input
        **kwargs         - object to be defined as pShape (input_array),
                           attachedChC, self.seq, self.topo or parameters to
                           build polygon and attachedChC'''

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
        self.sparseNum = {}
        self.internalMovesCounter=[]

        if kwargs:
            try:
                pShape = kwargs.get('pShape')
                attachedChC = kwargs.get('attachedChC')
            except NameError:
                pShape, attachedChC = self.buildPolygonAndAttachChC(**kwargs)

        # Set instance variables
        self.pShape = pShape
        self.attachedChC = attachedChC

        self.threshold = np.zeros((pShape.input_array.shape[0],
                                   pShape.input_array.shape[1]))
        self.threshold[:] = -1

        self.trueTargs = set(pShape.activeElements)
        self.totTargVal = 0
        for indx in pShape.activeElements:
            self.totTargVal += self.pShape.input_array[indx[0], indx[1]]
        self.uncorruptedInput = self.pShape.input_array.copy()
        if display:
            plotTitle = 'Raw Target Input'
            self.pShape.display_Polygon(pShape.input_array, plotTitle)

        self.gaussBlurSigma = gaussBlurSigma
        self.pShape.blur_Array(sigma=gaussBlurSigma)
        if display:
            plotTitle = 'Input After Gaussian Blurring'
            self.pShape.display_Polygon(pShape.input_array, plotTitle)

        self.noiseLevel = noiseLevel
        self.pShape.add_Noise(scale=noiseLevel)
        if display:
            plotTitle='Input With Noise Added'
            self.pShape.display_Polygon(pShape.input_array, plotTitle)

        if sparseType=='Percent':
            sparseLow = int(np.round(pShape.size*sparseLow))
            sparseHigh = int(np.round(pShape.size*sparseHigh))
        elif sparseType == 'Exact':
            sparseLow = int(sparseHigh)
        if sparseLow > len(self.trueTargs):
            sparseLow = int(len(self.trueTargs))
        self.sparseNum = {'low': sparseLow, 'high': sparseHigh}

        # build ais
        self.AIS = AIS(pShape, attachedChC)

        # Collect initial weights arrays
        self.initalAISWeights = self.AIS.ais.copy()
        ChCWeightsOnly = self.calcWeights()
        self.initialChCWeights = ChCWeightsOnly.copy()


    @staticmethod
    def buildPolygonAndAttachChC(array_size=10, form='rectangle', x=4,
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


    def extractSDR(self):
        ''' Top level function to control process flow and extract an SDR from
        an input.

        Returns:
        SDR              - extracted SDR
        sdrFoundWholeFlag- Boolean indicating that if true then indexes
                           corresponding to SDR match the lower bound of ground
                           truth indices.  If false, targetIndxs will be
                           incomplete i.e. less than sparse lower bound
                           indicating ambiguity.
        '''

        targetIndxs, confidenceFlag = self.applyReceptiveField()

        print('finished applyRF', sorted(targetIndxs))
        targetIndxs = self.internalMove(targetIndxs)
        print('finished internalMove', targetIndxs)
        if self.internalNoiseFlag:
            pass
            ###### Need to implement REWIRING!!

        sdrFoundWholeFlag, targetIndxs = self.externalMove(targetIndxs)

        if sdrFoundWholeFlag:
            # Reward chandelier cell weights for those target indices and save these as abstract SDR X.
            pass

        targsNotFoundYet = list(set(self.pShape.activeElements) - set(self.correctTargsFound))
        correctHits = list(set(self.pShape.activeElements) & set(self.correctTargsFound))
        misses = list(set(self.falseTargsFound) - set(self.pShape.activeElements))
        plotTitle=f'External Movement Target Indexes'
        self.pShape.display_Polygon(self.pShape.input_array, plotTitle,
                                        targsNotFoundYet=targsNotFoundYet,
                                        correctHits=correctHits, misses=misses
                                       )

        # firingRateOutput = self.calcInterference(result, self.threshold)

        # if not self.seq:
        #     self.seq = Sequencer(self.sp)

        # if objectToTrain == 'seq':
        #     self.seq.evalActiveColsVersusPreds(winningColumnsInd)
        #
        # if objectToTrain == 'topo':
        #     self.trainTopo()


        return sdrFoundWholeFlag, targetIndxs


    def refineSDR(self, targs):
        '''Take given list of targs and adjust thresholding to narrow down to
        sparser representation of true targets.

        Inputs:
        targs       - list of target indices to inspect

        Returns:
        targetIndxs   - list of target indexes restricted to length
                        sparseNum['high']
        '''
            numRows = self.pShape.input_array.shape[0]
            numCols = self.pShape.input_array.shape[1]

            bin = np.zeros([self.pShape.input_array.shape[0], self.pShape.input_array.shape[1]])
            for idx in targs:
                bin[idx[0], idx[1]] = 1

            targetIndxs, _ = self.applyReceptiveField(mode='Refine', mask=bin)
            targetIndxs = self.internalMove(targetIndxs)




            sdrFoundWholeFlag, targetIndxs = self.externalMove(targetIndxs)

    def applyReceptiveField(self, mode='Seek', mask=None):
        ''' Recursive BIG function returns an array of the same size as the
        receptive field filtered by the self.threshold i.e. generates an SDR!

        Inputs:
        mode           - String equal to 'Seek', 'Refine', or 'Infer'
        mask           - binary numpy array with which to threshold out
                         unwanted indices in order to refine a search

        Returns:
        targetIndxs      - list of row, col indices for found targets
        confidenceFlag   - boolean based on if targets were found through 2
                           seperate filtrations or not
        '''

        self.countAPPLY_RF += 1

        weightsAIS = self.calcWeights()

        size = self.pShape.input_array.shape[0]


        # Process input with moving average filter; note the use of the filter
        # in 'Seek' and 'Refine' abstracts out concept of AIS and ChC weights
        if mode=='Seek':
            rawFilter = ndimage.uniform_filter(self.pShape.input_array,
                                               size=size, mode='mirror')
            normWeighted = self.pShape.input_array-rawFilter
        elif mode=='Refine':
            thresholded = self.pShape.input_array.copy()*mask
            thresholdedFilter = ndimage.uniform_filter(thresholded, size=size,
                                                       mode='mirror')
            normWeighted = thresholded-thresholdedFilter
        elif mode=='Infer':
            step = self.pShape.input_array.MAX_INPUT/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
            weightsAdjusted = self.pShape.input_array-(weightsAIS*step)
            weightsAdjusted[weightsAdjusted<0] = 0
            adjustedFilter = ndimage.uniform_filter(weightsAdjusted, size=size,
                                                    mode='mirror')
            normWeighted = weightsAdjusted-adjustedFilter
        num = self.selectNum()
        if num > self.pShape.input_array.size:
            num = self.pShape.input_array.size
        # This next line is abstraction; moving up and down to threshold input relative to ChC weights
        # Bc in software I can arbitrarily select out top 'num' in input!
        indxs = np.c_[np.unravel_index(np.argpartition(normWeighted.ravel(),-num)[-num:], normWeighted.shape)]
        targetIndxs = [tuple(x) for x in indxs.tolist()]

        return targetIndxs, False


    def calcWeights(self, RF_AvgToMax=1):
        '''Helper function to retrieve ChC weights and adjust according to AIS
        placement.  Note, RF_AvgToMax set to 1 collects total active weight for
        ChC.'''

        numRows = self.pShape.input_array.shape[0]
        numCols = self.pShape.input_array.shape[1]

        weights = np.zeros([numRows, numCols])
        for i in range(numRows):
            for j in range(numCols):
                weights[i, j] = self.attachedChC.total_Active_Weight(PyC_array_element=(i,j),
                                                                     avgPercentFR_RF=RF_AvgToMax)
                    weightReduction = self.AIS.ais[i, j]/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
                    weights[i, j] *= (1-weightReduction)

        return weights


    def selectNum(self):
        '''Helper function to select integer number of targets.
        Returns a random number between sparseLow and sparseHigh multiplied by
        self.countEXTERNAL_MOVE and updates self.num to store the number selected.
        '''

        num1 = np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
        num2 = num1*(self.countEXTERNAL_MOVE+1)
        if hasattr(self, 'num'):
            if num2 < self.num:
                num2 = self.num + num1

        self.num = num2

        return num2


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

        self.targsINTERNAL.update(targetIndxs)

        for i in range(numSaccades):
            self.pShape.input_array = self.uncorruptedInput.copy()
            self.pShape.blur_Array(sigma=self.gaussBlurSigma)
            self.pShape.add_Noise(scale=self.noiseLevel)
            newIndxs, confidenceFlag = self.applyReceptiveField()
            self.targsINTERNAL.update(newIndxs)

        self.pShape.input_array = originalInput.copy() # restore input

        noiseFilter = numSaccades-self.countEXTERNAL_MOVE+1

        suspectedFalseTargsDueToNoise = []
        targetIndxs = []
        for targ in self.targsINTERNAL:
            if self.targsINTERNAL[targ] < noiseFilter:
                suspectedFalseTargsDueToNoise.append(targ)
            else:
                targetIndxs.append(targ)

        if self.sparseNum['low'] <= len(targetIndxs):# <= self.sparseNum['high']:
            self.internalNoiseFlag = False
            return targetIndxs
        else:
            self.suspectFalseTargsInternal.update(suspectedFalseTargsDueToNoise)
            self.thresholdOutSuspFalseTargs(suspectedFalseTargsDueToNoise)
            self.pShape.input_array = originalInput.copy() # restore input to re-process with noisy input cells thresholded out
            targetIndxs, confidenceFlag = self.applyReceptiveField()
            penaltySearch = len(self.targsINTERNAL) * np.exp(self.countINTERNAL_MOVE)#* (np.exp(self.sparseNum['low']/self.sparseNum['high']))
            unchecked = self.pShape.input_array.size-abs(penaltySearch)
            if unchecked <= self.sparseNum['high']:
                print('hiya!!!')
                self.internalNoiseFlag = True
                noiseEst = self.noiseEstimate(targetIndxs)
                listLength = np.int_(np.round(self.sparseNum['low']/noiseEst))
                targetIndxs = self.selectFromMostCommon(listLength)
                return targetIndxs
            else:
                return self.internalMove(targetIndxs, clearTargIndxCounter=False)


    def thresholdOutSuspFalseTargs(self, suspectedFalseTargsDueToNoise):
        '''Helper function to remove suspected false targets from consideration.'''

        for falseTarg in suspectedFalseTargsDueToNoise:
            self.AIS.ais[falseTarg] = np.floor(self.AIS.ais[falseTarg]/2) # move ais away from cell body
            self.threshold[falseTarg] += 0.1*self.pShape.MAX_INPUT # inhibit these cells


    def noiseEstimate(self, targetIndxs):
        '''Estimate noise from total value of targets found versus total
        value found in input array.

        Inputs:
        targetIndxs     - list of indices of suspected targets

        Returns:
        noiseEst        - value equal to total value of targs / sum of input
        '''

        totalValOfTargsFound = 0
        for i in targetIndxs:
            totalValOfTargsFound += self.pShape.input_array[i[0], i[1]]
        noiseEst = totalValOfTargsFound/np.sum(self.pShape.input_array)

        return noiseEst


    def selectFromMostCommon(self, listLength):
        ''' select most common indexes up to a certain list length.
        Inputs:
        listLength      - integer number of indices to select
        Returns:
        targetIndxs     - list of target indexes selected from most common
        '''

        bestGuess = self.targsINTERNAL.most_common(listLength)
        items = []
        count = []
        for c in bestGuess:
            items.append(c[0])
            count.append(c[1])
        tot = sum(count)
        prob = [c/tot for c in count]
        if self.sparseNum['low'] > self.sparseNum['high']:
            self.sparseNum['low'] = self.sparseNum['high']
        num = np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
        indxIntoItems = np.random.choice(len(items), size=num, replace=False, p=prob)
        targetIndxs = [items[indx] for indx in indxIntoItems]

        return targetIndxs


    def externalMove(self, targetIndxs, allPrevTargIndxs=None):
        '''External movement to simulate changing gradients across input space.
        Note this is a recursive function built on top of 2 nested recursive
        functions (internalMove and applyReceptiveField).

        Inputs:
        targetIndxs       - list of indices of input cells above the (direct
                            AIS/ChC modulated) and (indirect dynamic -- meant to
                            simulate AIS length changing) self.threshold

        Returns:
        sdrFoundWholeFlag - Boolean indicating that if true then indexes
                            corresponding to SDR match the lower bound of ground
                            truth indices.  If false, targetIndxs will be
                            incomplete i.e. less than sparse lower bound
                            indicating ambiguity.
        suspectedTargs    - list of row, col indices for found targets
        '''

        originalInput = self.pShape.input_array.copy()

        while True:
            if allPrevTargIndxs:
                suspectedTargs = set(targ for targ in targetIndxs)|allPrevTargIndxs
            else:
                suspectedTargs = set(targ for targ in targetIndxs)
            correctTargs = suspectedTargs & self.trueTargs
            incorrect = suspectedTargs - self.trueTargs

            self.correctTargsFound.update(correctTargs)
            self.falseTargsFound.update(incorrect)

            # For debugging and visulization
            targsNotFoundYet = list(set(self.pShape.activeElements) - suspectedTargs)
            correctHits = list(set(self.pShape.activeElements) & suspectedTargs)
            misses = list(incorrect)
            plotTitle=f'From External Movement Target Indexes'
            self.pShape.display_Polygon(self.pShape.input_array, plotTitle,
                                            targsNotFoundYet=targsNotFoundYet,
                                            correctHits=correctHits, misses=misses
                                           )

            if len(correctTargs) >= self.sparseNum['low']:
                if len(suspectedTargs) > self.sparseNum['high']:
                    return False, list(suspectedTargs)
                else:
                    return True, list(suspectedTargs)

            self.simulateExternalMove(self.noiseLevel)
            self.countEXTERNAL_MOVE += 1

            # Move AIS on suspected targs to bias network to look for new suspects
            for indx in suspectedTargs:
                self.AIS.ais[indx] -= 1 # Move AIS farther from cell body

            newIndxs, confidenceFlag = self.applyReceptiveField()
            self.internalMovesCounter.append(self.countINTERNAL_MOVE)
            self.countINTERNAL_MOVE = 0
            newIndxs = self.internalMove(newIndxs, numSaccades=self.countEXTERNAL_MOVE+5)

            self.pShape.input_array = originalInput.copy()

            return self.externalMove(targetIndxs=newIndxs,
                                     allPrevTargIndxs=suspectedTargs)


    def simulateExternalMove(self, noiseLevel=1, blur=None, arrayNoise=None):
        '''Helper function to randomly readjust target contrast.'''

        # Reset input array
        self.pShape.input_array = self.uncorruptedInput.copy()

        # Redistribute true target values stochastically
        noise = np.random.normal(0, noiseLevel)
        redistribute = self.totTargVal + noise
        trueTargsList = list(self.trueTargs)
        np.random.shuffle(trueTargsList)
        for i, idx in enumerate(trueTargsList):
            targsLeft = len(trueTargsList)-i
            if targsLeft==1:
                self.pShape.input_array[idx[0], idx[1]] = np.maximum(redistribute, 0)
            else:
                val = np.round(np.random.normal(redistribute//targsLeft, noiseLevel))
                self.pShape.input_array[idx[0], idx[1]] = np.maximum(val, 0)
                redistribute -= val

        # Reconstruct blur and noise - default is same as initial input.
        # (adjustment from default represents crude knobs to simulate information
        # conveyed via lateral connectivity between columns)
        if blur:
            self.pShape.blur_Array(sigma=blur)
        else:
            self.pShape.blur_Array(sigma=self.gaussBlurSigma)
        if arrayNoise:
            self.pShape.add_Noise(scale=arrayNoise)
        else:
            self.pShape.add_Noise(scale=self.noiseLevel)


    def collectNearbyIndices(self, targetIndxs):
        '''Helper function to grab nearby target indices.
        Inputs:
        targetIndxs - list of indices
        Returns:
        neighborhood - list of indices in the neighborhood (including) target indices
        '''

        neighborhood = []
        for i in targetIndxs:
            if i[0] == 0:
                row_s = 1
            elif i[0] == self.pShape.input_array.shape[0]:
                row_s = self.pShape.input_array.shape[0] - 1
            else:
                row_s = i[0]
            if i[1] == 0:
                col_s = 1
            elif i[1] == self.pShape.input_array.shape[1]:
                col_s = self.pShape.input_array.shape[1] - 1
            else:
                col_s = i[1]
            for i in range(row_s-1, row_s+1):
                for j in range(col_s-1, col_s+1):
                    if (i, j) not in neighborhood:
                        neighborhood.append((i, j))

        return neighborhood


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

        #############3333 use dynamic feedback to AIS change in external move!!!!!

        # key concept in wilcoxon rank sum test uValue ranges from 0 (= complete sepatation) to n1*n2 (= no separation) where n1 and n2 are number of samples from each distribution (here size of receptive field).  Note, bc of the threshold splitting above and below the uValue will be zero.  The pValue though tells the probability that this is true.
        return min(1/pValue, self.maxFiringRateOutput)




    # def moveAIS(self, binaryInputPiece, direction, max_wt=40):
    #     '''Helper function to get indices of nonzero elements and adjust AIS
    #     value for each of those elements.'''
    #
    #     nonzero = binaryInputPiece.nonzero()
    #     row = nonzero[0]
    #     col = nonzero[1]
    #     # self.getNonzeroIndices(binaryInputPiece)
    #     for idx1, idx2 in zip(row, col):
    #         if direction == 'decrease':
    #             self.AIS.ais[idx1, idx2] = max(0, self.AIS.ais[idx1, idx2]-1)
    #         if direction == 'increase':
    #             self.AIS.ais[idx1, idx2] = min(max_wt, self.AIS.ais[idx1, idx2]+1)

    # def applyThreshold(self, weightsAIS):
    #     '''Threshold image according to ChCs.
    #
    #     Inputs:
    #     weightsAIS       - np array of ChC weights adjusted according to AIS
    #                        placement
    #
    #     Returns:
    #     binaryInputPiece - np binary array of same size as input with value 1
    #                        wherever input>threshold and 0 elsewhere
    #     targetsFound     - int count of targets found
    #
    #     '''
    #
    #     chcStep = self.pShape.MAX_INPUT/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
    #
    #     # Check/Set threshold to sparseNum['high']th biggest value
    #     if np.any(self.threshold < 0):
    #         self.threshold[:] = np.partition(self.pShape.input_array.flatten(),
    #                                          -self.sparseNum['high'])[-self.sparseNum['high']]
    #
    #     result = np.zeros([self.pShape.input_array.shape[0],
    #                        self.pShape.input_array.shape[1]])
    #     result = self.pShape.input_array - (chcStep*weightsAIS)
    #
    #     binaryInputPiece = np.where(result > self.threshold, 1, 0)
    #
    #     targetsFound = np.count_nonzero(binaryInputPiece > 0)
    #
    #     return binaryInputPiece, targetsFound
    #
    #
    # def noiseAdjustments(self, targetsFound, prevTargetsFound, binaryInputPiece,
    #                      oscFlag):
    #     '''Helper function to adjust AIS position and/or threshold for targeted
    #     input cells depending on the number of targets found.
    #
    #     Returns:
    #
    #     oscFlag     - integer value used for updating moving average filter and
    #                   terminating base case for recursive function
    #                   'applyReceptiveField'
    #
    #     '''
    #
    #     # Note AIS moves opposite to self.threshold; decrease AIS means closer to cell body
    #     # i.e. increase ChC which can affect output
    #     if targetsFound > self.sparseNum['high']:
    #         self.moveAIS(binaryInputPiece, 'decrease')
    #         if prevTargetsFound > self.sparseNum['high']: # assume gradient; punish those at the higher end
    #             self.adjustThreshold(binaryInputPiece, 'up')
    #         else:
    #             oscFlag += 1
    #     if targetsFound < self.sparseNum['low']:
    #         binaryInvert = 1-binaryInputPiece
    #         self.moveAIS(binaryInvert, 'increase')
    #         if prevTargetsFound < self.sparseNum['low']: # assume gradient; boost those at the lowest end
    #             self.adjustThreshold(binaryInputPiece, 'down')
    #         else:
    #             oscFlag += 1
    #
    #     return oscFlag




    # def adjustThreshold(self, binaryInputPiece, direction):
    #     '''Helper function to adjust the threshold values on hits up to help
    #     remove stray targets or threshold values on misses down to help locate
    #     additional targets.'''
    #
    #     row_hit, col_hit = self.getNonzeroIndices(binaryInputPiece)
    #     hits = [(r, c) for r, c in zip(row_hit, col_hit)]
    #
    #     binaryInvert = 1-binaryInputPiece
    #     row_miss, col_miss = self.getNonzeroIndices(binaryInvert)
    #     misses = [(r, c) for r, c in zip(row_miss, col_miss)]
    #
    #     dist_char = binaryInputPiece.size*0.35
    #
    #     if direction == 'up':
    #         for i, hit in enumerate(hits):
    #             if len(misses) == 0: # guard against edge case where all values are hits
    #                 dist = 0
    #             else:
    #                 dist = self.computeMinDist(hit, misses)
    #             self.threshold[hit[0], hit[1]] += np.exp(dist/dist_char)
    #
    #     if direction == 'down':
    #         for i, miss in enumerate(misses):
    #             if len(hits) == 0:
    #                 dist = 0
    #             else:
    #                 dist = self.computeMinDist(miss, hits)
    #             self.threshold[miss[0], miss[1]] -= np.exp(dist/dist_char)
    #             if self.threshold[miss[0], miss[1]] < 0:
    #                 self.threshold[miss[0], miss[1]] = 0
    #
    #
    # def getNonzeroIndices(self, binaryInputPiece):
    #     '''Helper function to get indices of nonzero elements.'''
    #     nonzero = binaryInputPiece.nonzero()
    #     row = nonzero[0]
    #     col = nonzero[1]
    #
    #     return row, col
    #
    #
    # def computeMinDist(self, subject, listOfTargs):
    #     '''Helper function to compute distance from thresholded zero elements to
    #     farthest distance from ALL postive targets.'''
    #     dist = 10e10
    #     for targ in listOfTargs:
    #         d = np.sqrt((targ[0]-subject[0])**2 + (targ[1]-subject[1])**2)
    #         if d < dist:
    #             dist = d
    #
    #     return dist






        # totalValOfTargsFound = 0
        # for i in targetIndxs:
        #     totalValOfTargsFound += self.pShape.input_array[i[0], i[1]]
        # searchFactor = totalValOfTargsFound/np.sum(self.pShape.input_array)
        # if searchFactor == 0:
        #     searchFactor = 1 # to prevent division by zero
        # P = np.exp(-self.countEXTERNAL_MOVE/searchFactor)
        # if P > np.random.default_rng().random():
        #     neighborhood = self.collectNearbyIndices(targetIndxs)
        #     for idx in neighborhood:
        #         c = (idx, self.attachedChC.PyC[idx])
        #         self.attachedChC.change_Synapse_Weight(connection=c,
        #                                                change=change)
        # else:
        #     for i in range(20):
        #         x = np.random.randint(self.pShape.input_array.shape[0])
        #         y = np.random.randint(self.pShape.input_array.shape[1])
        #         idx = (x, y)
        #         c = (idx, self.attachedChC.PyC[idx])
        #         self.attachedChC.change_Synapse_Weight(connection=c,
        #                                                change=change)
            # search far away targets found i.e. decrease chc weights farther away
            # Note this is basic function to achieve crude search to refine after calc interference made
