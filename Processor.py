''' This script moves information from one type of encoding to the next to
form a regenerative hierarchically designed cycle.'''

import numpy as np
import os.path
import pickle
import scipy.stats as stats

from ChC import ChC
from Polygon import Polygon



class Processor:

    def __repr__(self):
        return (f'''This class processes information input in analog form to
        extract a binary sparse distributed representation carrying
        spatial information which is complementary to the raw rate encoded
        analog information.  It then pools the relevant information to
        regenerate an analog output which can then cycle through the process
        hierarchically.''')

    def __init__(self):#, encoder, input_array):
        ''''''

        # self.input_array = input_array

        ## use ChC total synapse weight to apply threshold chcStep to input
        self.REC_FLD_LENGTH = 4
        self.maxFiringRateOutput = 255
        self.countAPPLY_RF = 0
        self.countINTERNAL_MOVE = 0
        self.countEXTERNAL_MOVE = 0
        # stride = np.ceil(pShape.pShape[0]/REC_FLD_LENGTH) # 256/4 = 32
        # num_receptive_fields = stride**2 # 1024
        pass

    def extractSDR(self, sparseType, sparseLow=0.02, sparseHigh=0.04, **kwargs):
        ''' Top level function to take an input and run through the network.

        Inputs:
        sparseType       - String: either 'Percent' or 'Exact' for target number
        sparseLow/High   - target sparsity level for extracted SDR
        **kwargs         - dictionary with boolean flag for 'polygon' or 'target'
                           object to be defined as pShape (input_array),
                           attachedChC, self.seq, self.topo

        Returns:
        SDR              - extracted SDR
        '''

        # from kwargs
        if not pShape or not attachedChC:
            pShape, attachedChC = self.buildPolygonAndAttachChC()

        trueTargs = set(pShape.activeElements)

        # build ais
        self.AIS = AIS(pShape, attachedChC)

        fldHEIGHT = pShape.input_array.shape[0]
        fldWIDTH = pShape.input_array.shape[1]
        intValArray = np.zeros(fldHEIGHT, fldWIDTH)
        threshold = np.zeros((fldHEIGHT, fldWIDTH))
        threshold[:] = -1

        if sparseType=='Percent':
            sparseLow = np.round(pShape.size*sparseLow)
            sparseHigh = np.round(pShape.size*sparseHigh)
        elif sparseType == 'exact':
            sparseLow = sparseHigh
        if sparseLow > len(trueTargs):
            sparseLow = trueTargs
        sparseNum = {'low': sparseLow, 'high': sparseHigh}

        targetIndxs = self.applyReceptiveField(pShape, attachedChC, threshold,
                                               sparseNum)

        targetIndxs = self.internalMove(pShape, attachedChC, threshold,
                                        sparseNum, targetIndxs)


############
        self.countEXTERNAL_MOVE += 1

        ##### change contrast REPEAt
        correctTargsFound = set()

        suspectedTargs = set(targ for targ in targetIndxs)
        correctTargs = suspectedTargs & trueTargs
        incorrect = suspectedTargs - trueTargs

        if len(correctTargs) >= sparseLow:
            return trueTargs
        else:
            correctTargsFound.add(correctTargs)

        randH_Start = np.random.randint(pShape.MAX_INPUT)
        randH_Stop = np.random.randint(pShape.MAX_INPUT)       
        randVert = np.random.randint(pShape.MAX_INPUT)
        pShape.create_Gradient(is_horizontal=True, stop=randHoriz)
        pShape.create_Gradient(is_horizontal=False, stop=randVert)



        for i in range(5):
            input_array = pShape.add_Noise()
            pShape.input_array = input_array
            newIndxs = self.applyReceptiveField(pShape, attachedChC, threshold,
                                                sparseNum)
            falseTargsDueToNoise.append(set(newIndxs) ^ set(targetIndxs))
            targetIndxs = [val for val in targetIndxs if value in newIndxs]

        targetsFound = len(targetIndxs)

        if (sparseNum['low'] <= targetsFound <= sparseNum['high']):
            return targetIndxs
        else:
            for falseTarg in falseTargsDueToNoise:
                self.AIS.ais[falseTarg[0], falseTarg[1]] = 0 # place ais at cell body
                threshold[falseTarg[0], falseTarg[1]] += 0.1*pShape.MAX_INPUT # inhibit these cells
                targetIndxs = self.applyReceptiveField(pShape, attachedChC,
                                                       threshold, sparseNum)
                return self.internalMove(pShape, attachedChC, threshold, sparseNum)
        # self.externalMove()

        # firingRateOutput = self.calcInterference(result, threshold)

        '''
        output dynamically shapes input receptive field so narrow triangle and wide triangle
        even though different in lower level but output next layer input is constant!!!
        note example using triangle in 2d but in reality is sdr in 3d and potentially
        covering n dims feature space!
        '''


        # if not self.seq:
        #     self.seq = Sequencer(self.sp)

        if objectToTrain == 'seq':elf.applyReceptiveField

            self.seq.evalActiveColsVersusPreds(winningColumnsInd)

        if objectToTrain == 'topo':
            self.trainTopo()


    def buildPolygonAndAttachChC(self, array_size=10, form='rectangle', x=4,
                                 y=4, wd=4, ht=3, angle=0,
                                 useTargetSubclass=False, numTargets=False,
                                 numClusters=0):
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
            pShape = Target(array_size, numTargets, numClusters)
            pShape.insert_Targets()

        else:
            pShape = Polygon(array_size=array_size, form=form, x=x, y=y, width=wd,
                             height=ht, angle=angle)

            pShape.insert_Polygon()

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


    def applyReceptiveField(self, pShape, attachedChC, threshold, sparseNum,
                            prevTargetsFound=0, oscFlag=0):
        ''' Recursive BIG function returns an array of the same size as the
        receptive field filtered by the threshold i.e. generates an SDR!

        Inputs:
        pShape           - Polygon object
        attachedChC      - ChC object representing map of array coordinates
                           to connected chandelier cells
        threshold        - np array of float values to compare against input
        sparseNum        - dictionary with desired number (low/high) of targets
        prevTargetsFound - float
        oscFlag          - Flag to prevent infinite recursion

        Returns:
        targetIndxs      - list of row, col indices for found targets
        '''

        self.countAPPLY_RF += 1

        input_array = pShape.input_array
        array_MAX = pShape.MAX_INPUT
        avgInputValInRF = np.mean(input_array)
        avgPercFR_of_RF_arrayMAX = avgInputValInRF/array_MAX
        chcStep = array_MAX/attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
        if threshold.any() < 0:
            threshold[:] = avgInputValInRF/chcStep


        result = np.zeros([input_array.shape[0], input_array.shape[1]])
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                weight = attachedChC.total_Active_Weight(PyC_array_element=(i,j),
                                                         avgPercentFR_RF=avgPercFR_of_RF_arrayMAX)
                weight -= self.AIS.ais[i, j]
                if weight < 0:
                    weight = 0
                result[i, j] = max(input_array[(i,j)] - chcStep*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        targetsFound = np.count_nonzero(binaryInputPiece > 0)

        if (sparseNum['low'] <= targetsFound <= sparseNum['high']) or oscFlag == 100:
            row, col = self.getNonzeroIndices(binaryInputPiece)
            targetIndxs = [(r, c) for r, c in zip(row, col)]
            return targetIndxs

        '''
        # print('targs', targetsFound)
        # print('threshold', threshold)
        # print('chcStep', chcStep)
        # print('binary', binaryInputPiece.nonzero())
        '''

        # Note AIS moves opposite to threshold; decrease AIS means closer to cell body
        # i.e. increase ChC which can affect output
        if targetsFound > sparseNum['high']:
            self.moveAIS(binaryInputPiece, 'decrease')
            if prevTargetsFound > sparseNum['high']: # assume gradient; punish those at the higher end
                self.adjustThreshold(binaryInputPiece, threshold, 'up')
            else:
                oscFlag += 1
        if targetsFound < sparseNum['low']:
            binaryInvert = 1-binaryInputPiece
            self.moveAIS(binaryInvert, 'increase')
            if prevTargetsFound < sparseNum['low']: # assume gradient; boost those at the lowest end
                self.adjustThreshold(binaryInputPiece, threshold, 'down')
            else:
                oscFlag += 1


        prevTargetsFound = targetsFound

        return self.applyReceptiveField(pShape, attachedChC, threshold,
                                        sparseNum, prevTargetsFound, oscFlag)


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


    def adjustThreshold(self, binaryInputPiece, threshold, direction):
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
                threshold[hit[0], hit[1]] += np.exp(dist/dist_char)

        if direction == 'down':
            for i, miss in enumerate(misses):
                if len(hits) == 0:
                    dist = 0
                else:
                    dist = self.computeMinDist(miss, hits)
                threshold[miss[0], miss[1]] -= np.exp(dist/dist_char)
                if threshold[miss[0], miss[1]] < 0:
                    threshold[miss[0], miss[1]] = 0


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


    def internalMove(pShape, attachedChC, threshold, sparseNum, targetIndxs):
        '''Internal movement to sift out noise.  Note this is a recursive
        function built on top of another recursive function
        (applyReceptiveField).

        Inputs:
        pShape           - Polygon object
        attachedChC      - ChC object representing map of array coordinates
                           to connected chandelier cells
        threshold        - np array of float values to compare against input
        sparseNum        - dictionary with desired number (low/high) of targets
        targetIndxs      - list of indices of input cells above the (direct
                           AIS/ChC modulated) and (indirect dynamic -- meant to
                           simulate AIS length changing) threshold

        Returns:
        targetIndxs      - list of row, col indices for found targets
        '''

        self.countINTERNAL_MOVE += 1

        falseTargsDueToNoise = []
        for i in range(5):
            input_array = pShape.add_Noise()
            pShape.input_array = input_array
            newIndxs = self.applyReceptiveField(pShape, attachedChC, threshold,
                                                sparseNum)
            falseTargsDueToNoise.append(set(newIndxs) ^ set(targetIndxs))
            targetIndxs = [val for val in targetIndxs if value in newIndxs]

        targetsFound = len(targetIndxs)

        if (sparseNum['low'] <= targetsFound <= sparseNum['high']):
            return targetIndxs
        else:
            for falseTarg in falseTargsDueToNoise:
                self.AIS.ais[falseTarg[0], falseTarg[1]] = 0 # place ais at cell body
                threshold[falseTarg[0], falseTarg[1]] += 0.1*pShape.MAX_INPUT # inhibit these cells
                targetIndxs = self.applyReceptiveField(pShape, attachedChC,
                                                       threshold, sparseNum)
                return self.internalMove(pShape, attachedChC, threshold, sparseNum)


    def calcInterference(self, result, threshold):
        '''Examine values within receptive field and calculate interference to
        determine how confident receptive field chcStep is.

        Inputs:
        result - numpy square array with length equal to receptive field and
                 values corresponding to input values after chandelier cell filtration.
        threshold - scalar threshold value

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
