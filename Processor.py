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
# 1) find targs  = COMPLETED!
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
        self.countREFINE_SDR = 0
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

        if not os.path.isdir('ChC_handles'):
            os.makedirs(ChC_handles)

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


    def extractSDR(self, plot=True, mode='Seek'):
        '''High level function to control process flow and extract an SDR from
        an input.

        Inputs:
        sdrName          - string for sdr Object name (default is None to create
                           new object name)
        plot             - boolean to display sdr object or not

        Returns:
        SDR              - extracted SDR
        sdrFoundWholeFlag- Boolean indicating that if true then indexes
                           corresponding to SDR match the lower bound of ground
                           truth indices.  If false, targetIndxs will be
                           incomplete i.e. less than sparse lower bound
                           indicating ambiguity.
        '''

        targetIndxs = self.applyReceptiveField(mode=mode)

        # print('finished applyRF', sorted(targetIndxs))
        targetIndxs = self.internalMove(targetIndxs, mode=mode)
        # print('finished internalMove', targetIndxs)

        if mode == 'Seek':
            sdrFoundWholeFlag, targetIndxs = self.externalMove(targetIndxs)
        # print('finished externalMove', targetIndxs)

        if plot:
            self.displayInputSearch(plotTitle='From externalMove Target Indices')

        if not sdrFoundWholeFlag:
            targetIndxs = self.refineSDR(targetIndxs)

        # firingRateOutput = self.calcInterference(result, self.threshold)

        # if not self.seq:
        #     self.seq = Sequencer(self.sp)

        # if objectToTrain == 'seq':
        #     self.seq.evalActiveColsVersusPreds(winningColumnsInd)
        #
        # if objectToTrain == 'topo':
        #     self.trainTopo()
# during learning extract --> refine --> sdr acquired store in chc weights
### during inference extract SDR with applyRF mode = inference, if match success use if not enter learning mode

        return sdrFoundWholeFlag, targetIndxs


    def applyReceptiveField(self, mode='Seek', num=None):
        ''' Computes a 2D moving average to filter over input and extract
        selectNum number of target indices.

        Inputs:
        mode           - String equal to 'Seek', or 'Infer'
        mask           - binary numpy array with which to threshold out
                         unwanted indices in order to refine a search

        Returns:
        targetIndxs      - list of row, col indices for found targets
        confidenceFlag   - boolean based on if targets were found through 2
                           seperate filtrations or not
        '''

        self.countAPPLY_RF += 1
        size = self.pShape.input_array.shape[0]

        # Process input with moving average filter; note, this is abstraction of
        # AIS and ChC weights bc arbitrarily selects out top 'num'!
        if mode=='Seek':
            size=np.random.randint(size)
            rawFilter = ndimage.uniform_filter(self.pShape.input_array, size=3,
                                               mode='mirror')
            normWeighted = self.pShape.input_array-rawFilter
            num = self.selectNum()
            if num > self.pShape.input_array.size:
                num = self.pShape.input_array.size
            indxs = np.c_[np.unravel_index(np.argpartition(normWeighted.ravel(),-num)[-num:], normWeighted.shape)]
            targetIndxs = [tuple(x) for x in indxs.tolist()]
        else: # mode is 'inference'
            weightsAIS = self.calcWeights()
            step = self.pShape.input_array.max()/self.attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
            weightsAdjusted = self.pShape.input_array-(weightsAIS*step)
            weightsAdjusted[weightsAdjusted<0] = 0
            adjustedFilter = ndimage.uniform_filter(weightsAdjusted, size=size,
                                                    mode='mirror')
            normWeighted = weightsAdjusted-adjustedFilter
            if not num:
                num = np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
            indxs = np.c_[np.unravel_index(np.argpartition(normWeighted.ravel(),-num)[-num:], normWeighted.shape)]
            targetIndxs = [tuple(x) for x in indxs.tolist()]

        return targetIndxs


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
                     num=None, mode='Seek'):
        '''Internal movement to sift out noise.  Note this is a split function
        with 2 modes: 'Seek' which aims to collect more suspected targets and
        'inference' which aims to extract SDR through potential recursive call.

        Inputs:
        targetIndxs          - list of indices of input cells filtered as
                               significantly above neighbors
        clearTargIndxCounter - boolean to clear counter on internal move target
                               indices
        mode                 - String to control logic for Seek versus inference

        Returns:
        targetIndxs      - list of row, col indices for found targets
        '''

        if clearTargIndxCounter:
            self.targsINTERNAL.clear()

        # Setup for main while loop
        self.targsINTERNAL.update(targetIndxs)
        originalInput = self.pShape.input_array.copy()
        noiseEst = []
        KeepGoing=True
        saccadeNum=0
        minSaccades=3

        while KeepGoing:
            # Calculate noise and set realted variables (minSaccades noiseFilter)
            self.countINTERNAL_MOVE += 1
            saccadeNum += 1
            self.pShape.input_array = originalInput.copy() # restore input
            noiseEst.append(self.noiseEstimate(targetIndxs))
            noise = sum(noiseEst)/len(noiseEst)
            minSaccades = max(minSaccades*noise, 7)
            noiseFilter = saccadeNum*np.exp(-noise)
            if mode=='Seek':
                sigma = self.gaussBlurSigma
                scale = self.noiseLevel
            else:
                sigma = scale = sum(noiseEst)/len(noiseEst)

            # Execute internal movement and extract new targets to be added to internal counter
            self.pShape.blur_Array(sigma=sigma)
            self.pShape.add_Noise(scale=scale)
            newIndxs = self.applyReceptiveField(mode=mode, num=num)
            self.targsINTERNAL.update(newIndxs)

            # Filter collected targets in counter based on noise criteria set above
            # Note structure of noise in if statements - increase noise results in more saccades but relaxed filtration-
            if saccadeNum > minSaccades:
                suspectedFalseTargsDueToNoise = []
                targetIndxs = []
                for targ in self.targsINTERNAL:
                    if self.targsINTERNAL[targ] < noiseFilter:
                        suspectedFalseTargsDueToNoise.append(targ)
                    else:
                        targetIndxs.append(targ)

            # Collect target indices
            if mode == 'Seek':
                stopCheck = np.random.randint(saccadeNum)
                if stopCheck >= minSaccades:
                    KeepGoing=False # Terminate while loop
                    targetIndxs = self.selectFromMostCommon()
            else: # mode is 'inference'
                if self.sparseNum['high'] < len(targetIndxs):
                    targetIndxs = [i[0] for i in self.targsINTERNAL.most_common(self.sparseNum['high'])]
                if self.sparseNum['low'] <= len(targetIndxs):
                    try:
                        if flag:
                            self.internalNoiseFlag = True
                    except:
                        self.internalNoiseFlag = False
                    KeepGoing=False
                else:
                    if num:
                        num += noiseFilter**2
                        num = min(self.pShape.input_array.size, num)
                    else:
                        num = self.sparseNum['high']
                    flag = True
                    newIndxs = self.applyReceptiveField(mode=mode, num=num)

        return targetIndxs


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
        surround = np.sum(self.pShape.input_array)-totalValOfTargsFound
        surroundAvg = surround/max(self.pShape.input_array.size-len(targetIndxs), 1)
        normalizedSA = max(surroundAvg*len(targetIndxs), 1) # to avoid negative or zero division
        noiseEst = normalizedSA/max(totalValOfTargsFound, 1)

        return noiseEst


    def thresholdOutSuspFalseTargs(self, suspectedFalseTargsDueToNoise):
        '''Helper function to remove suspected false targets from consideration.'''
        for falseTarg in suspectedFalseTargsDueToNoise:
            self.AIS.ais[falseTarg] = np.floor(self.AIS.ais[falseTarg]/2) # move ais away from cell body


    def selectFromMostCommon(self):
        ''' select most common indexes up to a certain list length.

        Returns:
        targetIndxs     - list of target indexes selected from most common
        '''

        num=np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
        num = np.int_(num*(self.countEXTERNAL_MOVE+1))
        if num > len(self.targsINTERNAL):
            num = len(self.targsINTERNAL)
        bestGuess = self.targsINTERNAL.most_common(num)
        items = []
        count = []
        for c in bestGuess:
            items.append(c[0])
            count.append(c[1])
        tot = sum(count)
        prob = [c/tot for c in count]
        indxIntoItems = np.random.choice(len(items), size=num, replace=False, p=prob)
        targetIndxs = [items[indx] for indx in indxIntoItems]

        return targetIndxs


    def externalMove(self, targetIndxs, mode='Seek'):
        '''External movement to simulate changing gradients across input space..

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
        allPrevTargIndxs=None

        while True:
            # Compare selected indices to true targets
            suspectedTargs = set(targ for targ in targetIndxs)
            if (mode == 'Seek') and allPrevTargIndxs:
                suspectedTargs.update(allPrevTargIndxs)
            correctTargs = suspectedTargs & self.trueTargs
            incorrect = suspectedTargs - self.trueTargs

            # Used for plotting results
            if mode == 'Seek':
                self.correctTargsFound.update(correctTargs)
                self.falseTargsFound.update(incorrect)

            # Termination of while loop criteria
            if len(correctTargs) >= self.sparseNum['low']:
                if len(suspectedTargs) <= self.sparseNum['high']:
                    return True, list(suspectedTargs)
                elif mode=='Seek':
                    return False, list(suspectedTargs)

            # Terimnation critieria not met; execute external movement
            noiseEst = self.noiseEstimate(suspectedTargs)
            noiseEst = int(np.rint(noiseEst*np.max(self.pShape.input_array)))
            targOutputStrengths = self.calcInterference(targetIndxs, noiseEst)
            self.simulateExternalMove(noiseEst)
            self.countEXTERNAL_MOVE += 1

            # Move AIS on suspected targs to bias network to look for new suspects
            for indx in self.attachedChC.PyC_points:
                if indx in suspectedTargs:
                    self.AIS.ais[indx] = max(0, self.AIS.ais[indx]-1) # Move AIS farther from cell body
                else:
                    self.AIS.ais[indx] = max(self.AIS.MAX, self.AIS.ais[indx]+1) # Move AIS closer to cell body

            # Collect new target indices
            allPrevTargIndxs = targetIndxs
            targetIndxs = self.applyReceptiveField(mode=mode)
            self.internalMovesCounter.append(self.countINTERNAL_MOVE)
            self.countINTERNAL_MOVE = 0
            targetIndxs = self.internalMove(targetIndxs, mode=mode)

            if mode == 'Infer':
                overlap = self.findNamesForMatchingSDRs(targetIndxs)
                P.setChCWeightsFromMatchedSDRs(overlap)

            self.pShape.input_array = originalInput.copy()



    def simulateExternalMove(self, noiseLevel=1, blur=None, arrayNoise=None):
        '''Helper function to randomly readjust target contrast.

        ***************

        NOTE:

        noiseLevel is a dynamic variable to adjust contrast relative to average
        (potentially blur and arrayNoise could be controlled by other
        interneurons)

        ***************
        '''

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


    def refineSDR(self, targs, numReduce=None):
        '''Take given list of targs and adjust thresholding to narrow down to
        sparser representation of true targets.

        Inputs:
        targs         - list of target indices to inspect

        Returns:
        targetIndxs   - list of target indexes restricted to length
                        sparseNum['high']
        '''

        prevTargs = targs.copy()
        if not numReduce:
            numReduce = np.random.randint(len(targs)-self.sparseNum['high']+1)

        targsCut, refinedTargIndxs = self.splitTargs(targs, numReduce)

        suspectedTargs = set(targ for targ in refinedTargIndxs)
        correctTargs = suspectedTargs & self.trueTargs

        if self.sparseNum['low'] <= len(correctTargs):
            if len(suspectedTargs) <= self.sparseNum['high']: # Base case
                return refinedTargIndxs
            else: # Continue Refining
                return self.refineSDR(refinedTargIndxs)
        else: # At least 1 correct target was refined out
            numReduce = max(1, int(np.floor(numReduce/2)))
            self.countREFINE_SDR += 1
            return self.refineSDR(prevTargs, numReduce)


    def splitTargs(self, targs, numReduce):
        '''Function to split targets into 2 piles: ones to cut and ones to keep.

        Inputs:
        targs               - list of target indices
        numReduce           - integer value of how many targs to cut

        Returns:
        targsCut            - list of target indices cut
        refinedTargIndxs    - list of target indices kept
        '''

        random.shuffle(targs)
        targsCut = targs[:numReduce]
        refinedTargIndxs = targs[numReduce:]

        # Holding for potential future reference if desire to use more sophisticated logic than random search
        # # Move AIS to bias network to only look at supsected targets
        # for indx in self.attachedChC.PyC_points:
        #     if indx in targs:
        #         self.AIS.ais[indx] = self.AIS.MAX
        #         self.attachedChC
        #     else:
        #         self.AIS.ais[indx] = 0

        return targsCut, refinedTargIndxs


    def updateChCWeightsMatchedToSDR(self, refinedTargIndxs, sdrName=None):
        '''Set weights for attachedChC and store as pickled object.

        Inputs:
        refinedTargIndxs    - list of target indices kept
        sdrName             - string to name object

        Returns:
        None                - function creates or updates existing object
        '''

        DIR1 = 'ChC_handles/Objects'
        DIR2 = 'ChC_handles/Targs'

        if not os.path.isdir(DIR1):
            os.makedirs(DIR1)
        if not os.path.isdir(DIR2):
            os.makedirs(DIR2)

        if not sdrName:
            sdrName = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])
            # Save target index list as set
            with open(f'ChC_handles/Targs/targs_{sdrName}', 'wb') as targs_handle:
                pickle.dump(self.trueTargs, targs_handle)
        else:
            with open(f'ChC_handles/Objects/ChC_{sdrName}', 'rb') as ChC_handle:
                attachedChC = pickle.load(ChC_handle)

        for indx in self.attachedChC.PyC_points:
            connection = indx, self.attachedChC.PyC[indx]
            dist = self.computeMinDist(indx, refinedTargIndxs)
            if dist==0:
                self.attachedChC.change_Synapse_Weight(connection, change=-3)
            elif dist < 1.5:
                self.attachedChC.change_Synapse_Weight(connection, change=-2)
            elif dist < 2.5:
                self.attachedChC.change_Synapse_Weight(connection, change=-1)
            elif 4.5 > dist >= 3.5:
                self.attachedChC.change_Synapse_Weight(connection, change=1)
            elif 6.5 > dist >= 4.5:
                self.attachedChC.change_Synapse_Weight(connection, change=2)
            else:
                self.attachedChC.change_Synapse_Weight(connection, change=3)

        with open(f'ChC_handles/Objects/ChC_{sdrName}', 'wb') as ChC_handle:
            pickle.dump(self.attachedChC, ChC_handle)


    def computeMinDist(self, subject, listOfTargs):
        '''Helper function to compute distance from thresholded zero elements to
        farthest distance from ALL postive targets.'''
        dist = 10e10
        for targ in listOfTargs:
            d = np.sqrt((targ[0]-subject[0])**2 + (targ[1]-subject[1])**2)
            if d < dist:
                dist = d

        return dist


    def calcInterference(self, targetIndxs, shuffleStrength=0):
        '''Examine values within receptive field and calculate interference to
        determine how confident values idenitied in SDR are.

        Inputs:
        targetIndxs         - list of target indexes
        shuffleStrength     - integer value of number of times to shuffle
                              indices in receptive field sub-arrays

        Returns:
        targOutputStrengths  - dictionary with keys equal to target indices and
                               values equal to interference strength (output
                               firing rate).
        '''

        # Set up variables
        arr = self.pShape.input_array
        intTargIndxs = {}
        for targ in targetIndxs:
            intTargIndxs[targ] = int(f'{targ[0]}{targ[1]}')

        highVal = max(len(targetIndxs), self.sparseNum['high']+1)
        numRF=np.random.randint(self.sparseNum['low'], highVal)
        splitSize = np.rint(np.sqrt(numRF))

        # Split array into smaller receptive fields
        arrIndx = np.arange(arr.size)
        arrIndx = np.reshape(arrIndx, (arr.shape[0], arr.shape[1]))
        slicedIDX = [m for subA in np.array_split(arrIndx, splitSize, axis=0)
                       for m in np.array_split(subA, splitSize, axis=1)]
        slicedINPUT = [m for subA in np.array_split(arr, splitSize, axis=0)
                         for m in np.array_split(subA, splitSize, axis=1)]
        pieceCount = len(slicedINPUT)

        # Shuffle input values in receptive fields to desired shuffle strength
        for i in range(shuffleStrength):
            if np.random.randint(2):
                s_idx, slice, row, col = self.getRandSliceAndIdxs(pieceCount, slicedINPUT)
                s_idx2, slice2, row2, col2 = self.getRandSliceAndIdxs(pieceCount, slicedINPUT)
                temp = slice[row, col]
                slice[row, col] = slice2[row2, col2]
                slice2[row2, col2] = temp

        # Compute interference
        targOutputStrengths = {}
        for indxPiece, arrPiece in zip(slicedIDX, slicedINPUT):
            for targ, targString in intTargIndxs.items():
                if targString in indxPiece:
                    ix = np.where(targString==indxPiece)
                    surroundAvg = max( (np.sum(arrPiece)-arrPiece[ix])/max(arrPiece.size-1, 1), 1) # to avoid negative or zero division in next step
                    boostFactor = arrPiece[ix]/self.pShape.MAX_INPUT + 1 # to reward higher absolute inputs
                    targOutputStrengths[targ] = min(np.round(boostFactor*arrPiece[ix]/surroundAvg), self.pShape.MAX_INPUT)

        targOutputStrengths = {k: v for k, v in sorted(targOutputStrengths.items(), key=lambda x:x[1])}

        return targOutputStrengths


    def getRandSliceAndIdxs(self, pieceCount, slicedINPUT):
        rand_slice_idx = np.random.randint(0, pieceCount)
        slice = slicedINPUT[rand_slice_idx]
        rand_idx_row = np.random.randint(0, slice.shape[0])
        rand_idx_col = np.random.randint(0, slice.shape[1])

        return rand_slice_idx, slice, rand_idx_row, rand_idx_col


    def findNamesForMatchingSDRs(self, indxs, knownSDRs=None, threshold=8):
        '''Accepts a list of indices and identifies any SDRs that match above a
        threshold and collects the ChC weights attached to each of those.  Uses
        these matches to compute overlap.

        Inputs:
        indxs       - list of indices that were selected from the input
        knownSDRs   - list of individual lists of previously identified SDRs
                      indices

        Returns:
        overlap     - dictionary with keys as sdr names and values as strengths
        '''

        if not knownSDRs:
            try:
                knownSDRs = self.knownSDRs
            except:
                print('No Known SDRs stored!')

        indxs = set(indxs)
        overlap = []
        matchLength = []
        for count, SDR in enumerate(knownSDRs):
            match = indxs & set(SDR)
            if len(match) >= 8:
                overlap.append(count)
                matchLength.append(len(match))

        OL = self.calcSDRMatchStrength(overlap, matchLength)

        return OL


    def calcSDRMatchStrength(self, overlap, matchLength):
        '''Accepts a list of sdr names with corresponding list of overlap match
        length and computes a strength weight for each sdr.

        Inputs:
        sdr_names   - list of sdr names
        matchLength - list of overlapping matches lengths

        Returns:
        OL          - dictionary with keys as sdr names and values as strengths
        '''

        olStrength = [m/max(matchLength) for m in matchLength]
        OL = {}
        for name, strength in zip(overlap, olStrength):
            OL[name] = strength

        return OL


    def setChCWeightsFromMatchedSDRs(self, overlap):
        '''Accepts a list of indices and identifies any SDRs that match above a
        threshold and collects the ChC weights attached to each of those.  Uses
        these matches to compute overlap.

        Inputs:
        overlap       - dictionary with keys as sdr names and values as strengths


        Returns:
        compositeChC  - weights for a composite set of attached ChCs based on all
                        sdrs in list provided
        '''

        if len(overlap) == 0:
            array_size = P.pShape.input_array.shape[0]
            with open(f'ChC_handles/ChC_size_{array_size}', 'rb') as ChC_handle:
                self.attachedChC = pickle.load(ChC_handle)
            for indx in self.attachedChC.PyC_points:
                connection = indx, self.attachedChC.PyC[indx]
                self.attachedChC.change_Synapse_Weight(connection=connection,
                                                       change='RANDOM')
        else:
            connected = []
            for sdrName in overlap.keys():
                with open(f'ChC_handles/Objects/ChC_{sdrName}', 'rb') as ChC_handle:
                    attachedChC = pickle.load(ChC_handle)
                connected.append(attachedChC)

            for indx in self.attachedChC.PyC_points:
                w = []
                for i, attached in enumerate(connected):
                    w.append(attached.total_Active_Weight(PyC_array_element=indx))
                    w[i] *= list(overlap.values())[i]
                    avg_intW = np.round(sum(w)/len(w))
                connection = indx, self.attachedChC.PyC[indx]
                self.attachedChC.change_Synapse_Weight(connection=connection,
                                                       change='SET',
                                                       target_tot_wght=avg_intW)


    def displayInputSearch(self, plotTitle='Target Indices'):

        targsNotFoundYet = list(set(self.pShape.activeElements) - set(self.correctTargsFound))
        correctHits = list(set(self.pShape.activeElements) & set(self.correctTargsFound))
        misses = list(set(self.falseTargsFound) - set(self.pShape.activeElements))
        self.pShape.display_Polygon(self.pShape.input_array, plotTitle,
                                        targsNotFoundYet=targsNotFoundYet,
                                        correctHits=correctHits, misses=misses
                                       )

################################################################################


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



    # def collectNearbyIndices(self, targetIndxs):
    #     '''Helper function to grab nearby target indices.
    #     Inputs:
    #     targetIndxs - list of indices
    #     Returns:
    #     neighborhood - list of indices in the neighborhood (including) target indices
    #     '''
    #
    #     neighborhood = []
    #     for i in targetIndxs:
    #         if i[0] == 0:
    #             row_s = 1
    #         elif i[0] == self.pShape.input_array.shape[0]:
    #             row_s = self.pShape.input_array.shape[0] - 1
    #         else:
    #             row_s = i[0]
    #         if i[1] == 0:
    #             col_s = 1
    #         elif i[1] == self.pShape.input_array.shape[1]:
    #             col_s = self.pShape.input_array.shape[1] - 1
    #         else:
    #             col_s = i[1]
    #         for i in range(row_s-1, row_s+1):
    #             for j in range(col_s-1, col_s+1):
    #                 if (i, j) not in neighborhood:
    #                     neighborhood.append((i, j))
    #
    #     return neighborhood



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





# From internal move
        # if self.sparseNum['low'] <= len(targetIndxs):# <= (self.countINTERNAL_MOVE+1)*self.sparseNum['high']:
        #     self.internalNoiseFlag = False
        #     return targetIndxs
        # else:
        #     if mode == 'inference':
        #         self.suspectFalseTargsInternal.update(suspectedFalseTargsDueToNoise)
        #         self.thresholdOutSuspFalseTargs(suspectedFalseTargsDueToNoise)
        #         targetIndxs, confidenceFlag = self.applyReceptiveField()
        #     penaltySearch = len(self.targsINTERNAL) * np.exp(self.countEXTERNAL_MOVE)* (np.exp(self.sparseNum['low']/self.sparseNum['high']))
        #     unchecked = self.pShape.input_array.size-abs(penaltySearch)
        #     print('uncheck', unchecked)
        #     if unchecked <= self.sparseNum['high']:
        #         self.internalNoiseFlag = True
        #         listLength = np.int_(self.sparseNum['high']*(self.countEXTERNAL_MOVE+1))
        #         print('liistleng', listLength)
        #         targetIndxs = self.selectFromMostCommon(listLength)
        #         return targetIndxs
        #     else:
        #         return self.internalMove(targetIndxs, clearTargIndxCounter=False)











    # def internalMove(self, targetIndxs, clearTargIndxCounter=True,
    #                  numSaccades=5, num=None, mode='Seek'):
    #     '''Internal movement to sift out noise.  Note this is a split function
    #     with 2 modes: 'Seek' which aims to collect more suspected targets and
    #     'inference' which aims to extract SDR through potential recursive call.
    #
    #     Inputs:
    #     targetIndxs      - list of indices of input cells filtered as
    #                        significantly above neighbors
    #
    #     Returns:
    #     targetIndxs      - list of row, col indices for found targets
    #     '''
    #
    #     if clearTargIndxCounter:
    #         self.targsINTERNAL.clear()
    #
    #     self.countINTERNAL_MOVE += 1
    #     originalInput = self.pShape.input_array.copy()
    #
    #     self.targsINTERNAL.update(targetIndxs)
    #
    #     noiseEst = self.noiseEstimate(targetIndxs)
    #     for i in range(numSaccades):
    #         self.pShape.input_array = originalInput.copy()
    #         if mode=='Seek':
    #             sigma = self.gaussBlurSigma
    #             scale = self.noiseLevel
    #         else:
    #             sigma = noiseEst
    #             scale = noiseEst
    #         self.pShape.blur_Array(sigma=sigma)
    #         self.pShape.add_Noise(scale=scale)
    #         newIndxs, confidenceFlag = self.applyReceptiveField(mode=mode)
    #         self.targsINTERNAL.update(newIndxs)
    #         noiseEst = (noiseEst + self.noiseEstimate(newIndxs))/(i+2)
    #
    #     self.pShape.input_array = originalInput.copy() # restore input
    #
    #     if mode=='Seek':
    #         noiseFilter = max(numSaccades-self.countINTERNAL_MOVE, 1)
    #     else:
    #         noiseFilter = max(numSaccades-(numSaccades*(1-noiseEst)), 2)
    #
    #     suspectedFalseTargsDueToNoise = []
    #     targetIndxs = []
    #     for targ in self.targsINTERNAL:
    #         if self.targsINTERNAL[targ] < noiseFilter:
    #             suspectedFalseTargsDueToNoise.append(targ)
    #         else:
    #             targetIndxs.append(targ)
    #
    #     if mode == 'Seek':
    #         if self.sparseNum['low'] <= len(targetIndxs):
    #             self.internalNoiseFlag = False
    #             return targetIndxs
    #         else:
    #             self.internalNoiseFlag = True
    #             listLength = np.int_(self.sparseNum['high']*(self.countEXTERNAL_MOVE+1))
    #             targetIndxs = self.selectFromMostCommon(listLength)
    #             return targetIndxs
    #     else: # mode is 'inference'
    #         if self.sparseNum['high'] < len(targetIndxs):
    #             targetIndxs = [i[0] for i in self.targsINTERNAL.most_common(self.sparseNum['high'])]
    #         if self.sparseNum['low'] <= len(targetIndxs) <= self.sparseNum['low']:
    #             self.internalNoiseFlag = False
    #             return targetIndxs
    #         else:
    #             if num:
    #                 num += noiseFilter**2
    #                 num = min(self.pShape.input_array.size, num)
    #             else:
    #                 num = self.sparseNum['high']
    #             newIndxs, confidenceFlag = self.applyReceptiveField(mode=mode, num=num)
    #             self.internalMove(newIndxs, clearTargIndxCounter=False, mode='Infer')
    #             # recursive call avoids infinite loop by keeping the target index counter running
    # def selectFromMostCommon(self, listLength):
    #     ''' select most common indexes up to a certain list length.
    #     Inputs:
    #     listLength      - integer number of indices to select
    #     Returns:
    #     targetIndxs     - list of target indexes selected from most common
    #     '''
    #
    #     bestGuess = self.targsINTERNAL.most_common(listLength)
    #     items = []
    #     count = []
    #     for c in bestGuess:
    #         items.append(c[0])
    #         count.append(c[1])
    #     tot = sum(count)
    #     prob = [c/tot for c in count]
    #     if self.sparseNum['low'] > self.sparseNum['high']:
    #         self.sparseNum['low'] = self.sparseNum['high']
    #     num = np.random.randint(self.sparseNum['low'], self.sparseNum['high']+1)
    #     if num > len(bestGuess):
    #         num = len(bestGuess)
    #     indxIntoItems = np.random.choice(len(items), size=num, replace=False, p=prob)
    #     targetIndxs = [items[indx] for indx in indxIntoItems]
    #
    #     return targetIndxs















    #
    # def externalMoveSEEK(self, targetIndxs, allPrevTargIndxs=None, mode='Seek'):
    #     '''External movement to simulate changing gradients across input space..
    #
    #     Inputs:
    #     targetIndxs       - list of indices of input cells above the (direct
    #                         AIS/ChC modulated) and (indirect dynamic -- meant to
    #                         simulate AIS length changing) self.threshold
    #
    #     Returns:
    #     sdrFoundWholeFlag - Boolean indicating that if true then indexes
    #                         corresponding to SDR match the lower bound of ground
    #                         truth indices.  If false, targetIndxs will be
    #                         incomplete i.e. less than sparse lower bound
    #                         indicating ambiguity.
    #     suspectedTargs    - list of row, col indices for found targets
    #     '''
    #
    #     originalInput = self.pShape.input_array.copy()
    #     keepGoing = True
    #
    #     while keepGoing:
    #         if allPrevTargIndxs:
    #             suspectedTargs = set(targ for targ in targetIndxs)|allPrevTargIndxs
    #         else:
    #             suspectedTargs = set(targ for targ in targetIndxs)
    #         correctTargs = suspectedTargs & self.trueTargs
    #         incorrect = suspectedTargs - self.trueTargs
    #
    #         self.correctTargsFound.update(correctTargs)
    #         self.falseTargsFound.update(incorrect)
    #
    #         # For debugging and visulization
    #         # self.displayInputSearch(self, plotTitle='from internalMove Target Indices')                             )
    #
    #         if len(correctTargs) >= self.sparseNum['low']:
    #             if len(suspectedTargs) > self.sparseNum['high']:
    #                 return False, list(suspectedTargs)
    #             else:
    #                 return True, list(suspectedTargs)
    #
    #         self.simulateExternalMove(self.noiseLevel)
    #         self.countEXTERNAL_MOVE += 1
    #
    #         # Move AIS on suspected targs to bias network to look for new suspects
    #         for indx in self.attachedChC.PyC_points:
    #             if indx in suspectedTargs:
    #                 self.AIS.ais[indx] = max(0, self.AIS.ais[indx]-1) # Move AIS farther from cell body
    #             else:
    #                 self.AIS.ais[indx] = max(self.AIS.MAX, self.AIS.ais[indx]+1) # Move AIS closer to cell body
    #
    #         newIndxs, confidenceFlag = self.applyReceptiveField()
    #         newIndxs = self.internalMove(newIndxs)
    #         # self.internalMovesCounter.append(self.countINTERNAL_MOVE)
    #         # self.countINTERNAL_MOVE = 0
    #
    #         self.pShape.input_array = originalInput.copy()
    #
    #         return self.externalMoveSEEK(targetIndxs=newIndxs,
    #                                      allPrevTargIndxs=suspectedTargs)
    #
    #
    # def externalMoveINFER(self, targetIndxs, mode='Inference'):
    #     '''External movement to simulate changing gradients across input space..
    #
    #     Inputs:
    #     targetIndxs       - list of indices of input cells above the (direct
    #                         AIS/ChC modulated) and (indirect dynamic -- meant to
    #                         simulate AIS length changing) self.threshold
    #
    #     Returns:
    #     sdrFoundWholeFlag - Boolean indicating that if true then indexes
    #                         corresponding to SDR match the lower bound of ground
    #                         truth indices.  If false, targetIndxs will be
    #                         incomplete i.e. less than sparse lower bound
    #                         indicating ambiguity.
    #     suspectedTargs    - list of row, col indices for found targets
    #     '''
    #
    #     originalInput = self.pShape.input_array.copy()
    #
    #     while True:
    #         suspectedTargs = set(targ for targ in targetIndxs)
    #         correctTargs = suspectedTargs & self.trueTargs
    #
    #         if len(correctTargs) >= self.sparseNum['low']:
    #             return True, list(suspectedTargs)
    #
    #         # For debugging and visulization
    #         # self.displayInputSearch(self, plotTitle='from internalMove Target Indices'))
    #
    #         noiseEst = self.noiseEstimate(suspectedTargs)
    #         self.simulateExternalMove(noiseEst)
    #         self.countEXTERNAL_MOVE += 1
    #
    #         # Move AIS on suspected targs to bias network to look for new suspects
    #         for indx in self.attachedChC.PyC_points:
    #             if indx in suspectedTargs:
    #                 self.AIS.ais[indx] = max(0, self.AIS.ais[indx]-3) # Move AIS farther from cell body
    #             else:
    #                 self.AIS.ais[indx] = max(self.AIS.MAX, self.AIS.ais[indx]+3) # Move AIS closer to cell body
    #
    #
    #         newIndxs, confidenceFlag = self.applyReceptiveField(mode='Infer')
    #         self.internalMovesCounter.append(self.countINTERNAL_MOVE)
    #         self.countINTERNAL_MOVE = 0
    #         newIndxs = self.internalMove(newIndxs, mode='Infer')
    #         overlap = self.findNamesForMatchingSDRs(newIndxs)
    #         P.setChCWeightsFromMatchedSDRs(overlap)
    #
    #
    #         self.pShape.input_array = originalInput.copy()
    #
    #         return self.externalMove(targetIndxs=newIndxs,
    #                                  allPrevTargIndxs=suspectedTargs)
    #
    #     #############3333 use dynamic feedback from interference based firing rate to AIS change in external move!!!!!
    #
##### RECURSIVE

        # def externalMove(self, targetIndxs, allPrevTargIndxs=None, mode='Seek'):
        #     '''External movement to simulate changing gradients across input space..
        #
        #     Inputs:
        #     targetIndxs       - list of indices of input cells above the (direct
        #                         AIS/ChC modulated) and (indirect dynamic -- meant to
        #                         simulate AIS length changing) self.threshold
        #     allPrevTargIndxs  - set of previously selected target indices
        #
        #     Returns:
        #     sdrFoundWholeFlag - Boolean indicating that if true then indexes
        #                         corresponding to SDR match the lower bound of ground
        #                         truth indices.  If false, targetIndxs will be
        #                         incomplete i.e. less than sparse lower bound
        #                         indicating ambiguity.
        #     suspectedTargs    - list of row, col indices for found targets
        #     '''
        #
        #     originalInput = self.pShape.input_array.copy()
        #
        #     while True:
        #         suspectedTargs = set(targ for targ in targetIndxs)
        #         if (mode == 'Seek') and allPrevTargIndxs:
        #             suspectedTargs.update(allPrevTargIndxs)
        #
        #         correctTargs = suspectedTargs & self.trueTargs
        #         incorrect = suspectedTargs - self.trueTargs
        #
        #         if mode == 'Seek':
        #             self.correctTargsFound.update(correctTargs)
        #             self.falseTargsFound.update(incorrect)
        #
        #         if len(correctTargs) >= self.sparseNum['low']:
        #             if len(suspectedTargs) <= self.sparseNum['high']:
        #                 return True, list(suspectedTargs)
        #             elif mode=='Seek':
        #                 return False, list(suspectedTargs)
        #
        #         noiseEst = self.noiseEstimate(suspectedTargs)
        #         self.simulateExternalMove(noiseEst)
        #         self.countEXTERNAL_MOVE += 1
        #
        #         # Move AIS on suspected targs to bias network to look for new suspects
        #         for indx in self.attachedChC.PyC_points:
        #             if indx in suspectedTargs:
        #                 self.AIS.ais[indx] = max(0, self.AIS.ais[indx]-1) # Move AIS farther from cell body
        #             else:
        #                 self.AIS.ais[indx] = max(self.AIS.MAX, self.AIS.ais[indx]+1) # Move AIS closer to cell body
        #
        #         newIndxs, confidenceFlag = self.applyReceptiveField(mode=mode)
        #         if mode == 'Infer':
        #             self.internalMovesCounter.append(self.countINTERNAL_MOVE)
        #             self.countINTERNAL_MOVE = 0
        #
        #         newIndxs = self.internalMove(newIndxs, mode=mode)
        #
        #         if mode == 'Infer':
        #             overlap = self.findNamesForMatchingSDRs(newIndxs)
        #             P.setChCWeightsFromMatchedSDRs(overlap)
        #
        #         self.pShape.input_array = originalInput.copy()
        #
        #         return self.externalMove(targetIndxs=newIndxs,
        #                                  allPrevTargIndxs=suspectedTargs,
        #                                  mode=mode)
