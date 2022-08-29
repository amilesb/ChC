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
        # stride = np.ceil(pShape.pShape[0]/REC_FLD_LENGTH) # 256/4 = 32
        # num_receptive_fields = stride**2 # 1024
        pass

    def extractSDR(self, sparseType='Percent', sparsity=0.02, **kwargs):
        ''' Top level function to take an input and run through the network.

        Inputs:
        sparseType  - String: either 'Percent' or 'Exact' for target number
        sparsity    - target sparsity level for extracted SDR
        **kwargs    - dictionary with pShape (input_array), attachedChC,
                      self.seq, self.topo

        Returns:
        SDR         - extracted SDR
        '''

        # input sys arg inputs and process
        if not pShape or not attachedChC:
            pShape, attachedChC = self.buildPolygonAndAttachChC()

        # build ais
        self.AIS = AIS(pShape, attachedChC)

        fldHEIGHT = pShape.input_array.shape[0]
        fldWIDTH = pShape.input_array.shape[1]
        intValArray = np.zeros(fldHEIGHT, fldWIDTH)

        if sparseType=='Percent':
            sparseNum = np.round(pShape.size*sparsity)
        else:
            sparseNum = sparsity

        '''selectZone
        execute movement
        trueCountFound = counfound above and again with new random zones
        sparseNum left to find = sparseNum - trueCountFound
        subdivide array
        '''

        threshold = False
        input_array = pShape.input_array
        array_MAX = pShape.MAX_INPUT

        targetIndxs = self.applyReceptiveField(input_array, attachedChC,
                                               array_MAX, threshold, sparseNum)




        # firingRateOutput = self.calcInterference(result, threshold)

        '''
        use the average input signal relative to max to determine how many chcs to turn on
        so if input is near ceiling all chcs on
        if input very low very few chcs turned on

        create ais and move up or down to reduce impact of chcs to find targets

        movement consists of random set of new chcs (same initial number as above)

        output dynamically shapes input receptive field so narrow triangle and wide triangle
        even though different in lower level but output next layer input is constant!!!
        note example using triangle in 2d but in reality is sdr in 3d and potentially
        covering n dims feature space!
        '''

        # if not self.seq:
        #     self.seq = Sequencer(self.sp)

        if objectToTrain == 'seq':
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


    def applyReceptiveField(self, input_array, attachedChC, array_MAX, threshold,
                            sparseNum, factor=1, flag=None,
                            prevTargetsFound=0):
        ''' Recursive function returns an array of the same size as the
        receptive field filtered by the threshold.

        Inputs:
        input_array      - Numpy array representing a grayscale image
        attachedChC      - ChC object representing map of array coordinates
                           to connected chandelier cells
        array_MAX        - float that represents max allowed value in input
        threshold        - float to compare against filtered input
        sparseNum        - desired number of targets
        factor           - momentum factor to accelerate or (decelerate) changes
                           to threshold
        flag             - string to represent which way to change threshold
        prevTargetsFound - float

        Returns:
        binarySDR        - binary 2D numpy array extracted from input
        threshold        - float value
        '''

        avgInputValInRF = np.mean(input_array)
        avgPercFR_of_RF_arrayMAX = avgInputValInRF/array_MAX
        chcStep = array_MAX/attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT

        if not threshold:
            threshold = avgInputValInRF/chcStep
        if threshold < 0:
            threshold = 0

        result = np.zeros([input_array.shape[0], input_array.shape[1]])
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                weight = attachedChC.total_Active_Weight(PyC_array_element=(i,j),
                                                         avgPercentFR_RF=avgPercFR_of_RF_arrayMAX)
                weight -= self.AIS[i, j]
                if weight < 0:
                    wieght = 0
                result[i, j] = max(input_array[(i,j)] - chcStep*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        targetsFound = np.count_nonzero(binaryInputPiece > 0)

        if targetsFound == sparseNum:
            return np.nonzero(binaryInputPiece)

        print('targs', targetsFound)
        print('factor', factor)
        print('threshold', threshold)
        print('chcStep', chcStep)

        if targetsFound > sparseNum:
            if prevTargetsFound > sparseNum:
                for i, j in binaryInputPiece:
                    AIS[i, j] -= 1
        if targetsFound < sparseNum:
            if prevTargetsFound < sparseNum:
                for i, j in binaryInputPiece:
                    AIS[i, j] += 1

        prevTargetsFound = targetsFound

        self.applyReceptiveField(input_array, attachedChC, chcStep, threshold,
                                 sparseNum, factor, flag, prevTargetsFound)


        # if flag == 'UP':
        #     factor *= 2
        # if flag == 'DOWN':
        #     factor /= 2
        #
        # if targetsFound == sparseNum: # or chcStep*factor < 1:
        #     return np.nonzero(binaryInputPiece)
        #
        # print('targs', targetsFound)
        # print('factor', factor)
        # print('threshold', threshold)
        # print('chcStep', chcStep)
        #
        # if targetsFound > sparseNum:
        #     threshold += chcStep*factor
        #     if prevTargetsFound == None:
        #         flag = None
        #     elif prevTargetsFound > sparseNum:
        #         flag = 'UP'
        #     else:
        #         flag = 'DOWN'
        # if targetsFound < sparseNum:
        #     threshold -= chcStep*factor
        #     if prevTargetsFound == None:
        #         flag = None
        #     elif prevTargetsFound < sparseNum:
        #         flag = 'UP'
        #     else:
        #         flag = 'DOWN'


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
