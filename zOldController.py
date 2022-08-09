'''OLD CONTROLLER '''

''' This is the top level script which integrates all the components to run
the actual experiments.'''

import numpy as np
import os.path
import pickle
import scipy.stats as stats

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon
from Spatial_Pooler import Spatial_Pooler



class Controller:

    def __repr__(self):
        return (f'''This class implements a handler to process an input and
        coordinate handoff(s) between other objects.''')

    def __init__(self):#, encoder, input_array):
        ''''''

        # self.input_array = input_array

        ## use ChC total synapse weight to apply threshold filter to input
        self.REC_FLD_LENGTH = 4
        self.maxFiringRateOutput = 255
        # stride = np.ceil(pShape.pShape[0]/REC_FLD_LENGTH) # 256/4 = 32
        # num_receptive_fields = stride**2 # 1024
        pass

    def processInput(self, objectToTrain, **kwargs):
        ''' Top level function to take an input and run through the network.

        Inputs:
        objectToTrain - string identifier to determine what logic to run
        **kwargs      - dictionary with pShape, attachedChC, self.sp, self.seq
                        self.topo, and prevCenterRF passed in if already defined
                        else created or handled in main function

        Returns:
        None
        '''

        # input sys arg inputs and process

        if not pShape or not attachedChC:
            pShape, attachedChC = self.buildPolygonAndAttachChC()
        binaryPieces, salience = self.extractPieces(pShape, attachedChC)

        if not self.encoder:
            S1 = (pShape[0]-self.REC_FLD_LENGTH)**2
            S2 = self.REC_FLD_LENGTH**2
            self.encoder = Encoder(fullInputArraySize=S1, receptiveFieldSize=S2)

        if not self.sp:
            firstInputPiece = self.encoder.build_Encoding(list(binaryPieces.values())[0],
                                                          list(binaryPieces.keys())[0]
                                                         )
            self.sp = Spatial_Pooler(len(firstInputPiece))

        if not self.seq:
            self.seq = Sequencer(self.sp)

        # choose a feature (centerRF) with attention filter built in (salience)
        centerRF = np.random.choice(list(salience.keys()), replace=True,
                                    p=list(salience.value()))
        try:
            movement = prevCenterRF - centerRF
        except:
            movement = 0

        # Process through encoder and into the spatial pooler
        spInput = self.encoder.build_Encoding(binaryPieces[centerRF], centerRF,
                                              movement)
        overlapScore = self.sp.computeOverlap(currentInput=spInput)
        winningColumnsInd = self.sp.computeWinningColumns(overlapScore)

        if objectToTrain == 'sp':
            self.sp.updateSynapseParameters(winningColumnsInd, overlapScore,
                                            spInput)
        if objectToTrain == 'seq':
            self.seq.evalActiveColsVersusPreds(winningColumnsInd)

        ############## need to construct object SDR
        ##########################
        print('need logic to create metric to determine if sp has settled and/or seq')
        '''could use permanence values from spatial pooler i.e. sp.synapses[c]['permanence']
        for every column and extract the distribution every ~100 iterations
        then compute statistics for how quickly they are changing and stop once the change is
        small.
        '''

        if objectToTrain == 'topo':
            self.trainTopo()

        ########### work on how movement affects SDR  -- object is set of features at location
        #### need to incorporate movement into Enncoder?!  maybe not  perhaps movement is inherent within receptive field of column!!




    def buildPolygonAndAttachChC(self, array_size=10, form='rectangle', x=4,
                                 y=4, wd=4, ht=3, angle=0):
        '''Draw a polygon from the polygon class (numpy array) and then attach
        Chandelier Cells to that input.

        Inputs:
        array_size: integer
        form: string
        x, y: integers representing center of polygon
        wd, ht: integers corresponding to dimensions of 2D polygon
        angle: float representing rotation angle of polygon

        returns:
        pShape: numpy array with specified shape perimeter defined within
        attachedChC: dictionary which maps chandelier cells to the input_array
        '''
        pShape = Polygon(array_size=array_size, form=form, x=x, y=y, width=wd,
                         height=ht, angle=angle)

        # pShape = Polygon(array_size=array_size, form='rectangle', x=125, y=125, wd=40,
        #                     ht=30, angle = 0)
        pShape.insert_Polygon()

        # pShape.display_Polygon(pShape.input_array, angle=pShape.angle)

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


    def extractPieces(self, input_array, ChCs):
        '''Process a Polygon object to pass a filter with receptive field
        defined below (length) and extract filtered pieces after striding across
        entire image.

        Inputs:
        input_array - Polygon object which essentially is a numpy array
                      representing a grayscale image with polygon inserted
        ChCs        - ChC object representing map of polygon object coordinates
                      to connected chandelier cells

        returns:
        binaryPieces - dictionary with center of the receptive field as key and
                       2D binary array generated from applied filter as the
                       corresponding values
        salience     - dicitonary with center of the receptive field as key and
                       a salience score as the values.
        '''
        binaryPieces = {}
        length = self.REC_FLD_LENGTH
        s = (input_array.shape[0]-length, input_array.shape[1]-length)

        intValArray = np.zeros(s)
        filter = np.amax(input_array)/ChCs.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT


        for i in range(s[0]):
            for j in range(s[1]):
                cornerStart = i, j
                centerRF = self.calcCenterRF(cornerStart)
                binaryInputPiece = self.applyReceptiveField(input_array,
                                                            cornerStart, filter,
                                                            ChCs)
                bin1D = binaryInputPiece.copy().flatten()
                intValFromBin = bin1D.dot(2**np.arange(bin1D.size)[::-1])
                intValArray[i, j] = intValFromBin

                binaryPieces[centerRF] = binaryInputPiece

        connectedGraph = self.findConnectedComponents(intValArray)

        salience = self.calcSalience(connectedGraph, intValArray, binaryPieces)

        return binaryPieces, salience


    def findConnectedComponents(self, intValArray):
        '''Search through the values obtained from applying a receptive field
        and identify idnetical values are neighbors to one another in the input
        space.

        Inputs:
        intValArray - array with the integer representation of a 2d binary array

        Returns:
        g           - graph object which stores a dictionary called 'islandDict'
                      with the integer representation(s) as the key(s) and size
                      of connected components as the values.

        '''
        intValuesFromBin = set(np.unique(intValArray))

        g = Graph(intValArray.shape[0], intValArray.shape[1], intValArray)
        for value in intValuesFromBin:
            g.countIslands(value)

        return g


    def applyReceptiveField(self, input_array, cornerStart, filter, attachedChC,
                            length=4, dynamicThreshold=False, threshold=None):
        ''' Function returns an array of the same size as the receptive field
        filtered by the threshold set from the connected chandelier cell weights.

        Inputs:
        input_array      - Polygon object which essentially is a numpy array
                           representing a grayscale image with polygon inserted
        cornerStart      - x, y integer coordinates for corresponding array indices
        filter           - float value which sets dynamic range of input values
        attachedChC      - ChC object representing map of polygon object coordinates
                           to connected chandelier cells
        length           - square receptive field size
        dynamicThreshold - boolean
        threshold        - float value to compare against filtered input

        Returns:
        binaryInputPiece - binary 2D numpy array of dimensions length x length
        '''
        start_x, start_y = cornerStart

        avgInputValInRF = np.mean(input_array[start_x:start_x+length,
                                              start_y:start_y+length])
        if dynamicThreshold:
            threshold = avgInputValInRF/filter
        if not threshold:
            threshold = 0

        result = np.zeros([length, length])
        for i in range(start_x, start_x+length):
            for j in range(start_y, start_y+length):
                weight = attachedChC.total_Synapse_Weight(PyC_array_element=(i,j))
                result[i-start_x, j-start_y] = max(input_array[(i,j)] - filter*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        firingRateOutput = self.calcInterference(result, threshold)

        return binaryInputPiece, firingRateOutput


    def calcInterference(self, result, threshold):
        '''Examine values within receptive field and calculate interference to
        determine how confident receptive field filter is.

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


    def calcCenterRF(self, cornerStart):
        '''Helper function to find start indices for upper left corner of a
        receptive field.

        Inputs:
        cornerStart - list of 2 float values corresponding to the x, y starting
                      positions of the receptive field

        Returns:
        centerRF - x,y integer values corresponding to center of receptive field
        '''

        if self.REC_FLD_LENGTH%2 == 0:
            center_x = cornerStart[0]+self.REC_FLD_LENGTH/2 - 1
            center_y = cornerStart[1]+self.REC_FLD_LENGTH/2 - 1
        else:
            center_x = cornerStart[0]+self.REC_FLD_LENGTH//2
            center_y = cornerStart[1]+self.REC_FLD_LENGTH//2

        return (center_x, center_y)


    def calcSalience(self, g, intValArray, binaryPieces, strength=0.1):
        '''Calculate a salience score for each thresoholded binary input piece.
        NOTE: This is a soft surrogate that bluntly serves as a spatial low
        pass filter i.e. increases salience of non repeat elements in input!

        Inputs:
        g             -- Graph object that analyzes how many connected
                         components are present in the input space.
        intValArray   -- Array corresponding to compression of binary input pieces
                        into single decimal value.
        binaryPieces  -- dictionary with centerRFs as keys and values binary
                         arrays with size equal to the receptive field length
        strength      -- value that controls how much to boost isolated components
                         versus different connected components based upon a
                         gaussian redistribution that favors less connected
                         components (i.e. unique inputs).
                         Note: strength of 0 treats all inputs the same; default=0.1

        Returns:
        salience -- dictionary with keys equal to centerRFs and values the
                    salience score for that binary piece
        '''

        salience = {}
        maxConnected = 1
        flatIntValArray = np.reshape(intValArray, -1)

        for intVal in set(np.unique(intValArray)):
            maxConnectedComp = max(g.islandDict[intVal]['islandSize'])
            if maxConnectedComp > maxConnected:
                maxConnected = maxConnectedComp

        for index, centerRF in enumerate(list(binaryPieces.keys())):
            intVal = flatIntValArray[index]
            connectedList = g.islandDict[intVal]['islandSize']
            avgConnectedVal = sum(connectedList)/len(connectedList)

            rawSal = 1/(avgConnectedVal * flatIntValArray.size)
            P = np.exp(-strength*avgConnectedVal/maxConnected)

            salience[centerRF] = P*rawSal

        total = sum(salience.values(), 0)
        salience = {k: v/total for k, v in salience.items()}

        return salience






class Graph:

    def __repr__(self):
        return (f'''This class implements an object to index the number of
        connected components in an undirected graph (islands in 2D array).''')

    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.graph = g
        self.islandDict = {}

    def isSafe(self, i, j, visited, value):
        '''Function to check if a given cell (row, col) can be included in the
        depth first search (DFS) recursive function.

        Inputs:
        i, j - Integers representing row, col position in 2D array
        visited - boolean array of same dimensions as graph for whether cell has been checked

        Returns:
        Boolean value
        '''

        return (i>=0 and i<self.ROW and j>=0 and j<self.COL
                and not visited[i][j] and self.graph[i][j] == value)


    def DFS(self, i, j, visited, islandCenters, islandSize, value, index):
        '''Depth first search for 4 adjacent neighbors

        Inputs:
        i, j - Integers representing row, col position in 2D array
        visited - boolean array of same dimensions as graph for whether cell has been checked
        islandCenters - list of centerRF for each connected component
        islandSize - list of integers corresponding to the size of each island
        value - integer value that is target value for connected components.
        index - integer value that represents which island is being explored

        '''

        centerRF = islandCenters[index]
        islSize = islandSize[index]

        rowNbr = [-1, 0, 0, 1]
        colNbr = [0, -1, 1, 0]

        visited[i][j] = True

        for k in range(4):
            # new x, y cell to try
            x, y = i + rowNbr[k], j + colNbr[k]
            if self.isSafe(x, y, visited, value):
                islSize += 1
                centerRF = self.calcMovingAvgCenterRF(islSize, centerRF, x, y)
                islandCenters[index] = centerRF
                islandSize[index] = islSize
                self.islandDict[value].update({'islandCenters': islandCenters,
                                               'islandSize': islandSize})
                self.DFS(x, y, visited, islandCenters, islandSize, value, index)

        return


    def calcMovingAvgCenterRF(self, islandSize, centerRF, x, y):
        '''Function to update the coordinates (centerRF) of a particular island.

        Inputs:
        islandSize - Integer value used for normalization
        centerRF - integers representing current x, y position
        x, y - integers representing added component to island.

        Returns:
        centerRF - integers updated to center point.
        '''

        centerRF[0] = (centerRF[0]*(islandSize-1)+x)/islandSize
        centerRF[1] = (centerRF[1]*(islandSize-1)+y)/islandSize

        return centerRF


    def countIslands(self, value):
        '''Main function which creates a boolean array of visited cells and
        counts number of connected components.

        Inputs:
        value - integer value that is present in the graph
        centerRF - x, y integers corresponding to center of first island.

        Returns:
        self.islandDict - nested dictionary with top level key corresponding to
                          each binary_to_integer value which refers to an inner
                          dictionary that contains a list of centerRFs for each
                          connected component along with the number of elements
                          in that connected component.
        '''


        islandCenters = []
        islandSize = []
        self.islandDict[value] = {'islandCenters': islandCenters,
                                  'islandSize': islandSize}
        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

        index = 0
        for i in range(self.ROW):
            for j in range(self.COL):
                if visited[i][j] == False and self.graph[i][j] == value:
                    centerRF = [i, j]
                    islandCenters.append(centerRF)
                    islandSize.append(1)
                    self.DFS(i, j, visited, islandCenters, islandSize, value, index)
                    index += 1

        return
