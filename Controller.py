''' This is the top level script which integrates all the components to run
the actual experiments.'''

import numpy as np
import os.path
import pickle

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon



class Controller:

    def __repr__(self):
        return (f'''This class implements a handler to process an input and
        coordinate handoff(s) between other objects.''')

    def __init__(self, spatialPooler, sequencer, encoder, input_array):
        ''''''
        self.sp = spatialPooler
        self.seq = sequencer
        self.encoder = encoder
        self.input_array = input_array

        ## use ChC total synapse weight to apply threshold filter to input
        self.REC_FLD_LENGTH = 4
        stride = np.ceil(pShape.pShape[0]/REC_FLD_LENGTH) # 256/4 = 32
        num_receptive_fields = stride**2 # 1024
        self.enc = Encoder()

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
        pShape = Polygon(array_size=array_size, form=form, x=x, y=y, wd=wd,
                        ht=ht, angle=angle)

        # pShape = Polygon(array_size=array_size, form='rectangle', x=125, y=125, wd=40,
        #                     ht=30, angle = 0)
        pShape.insert_Polygon()

        pShape.display_Polygon(pShape.input_array, angle=pShape.angle)

        if os.path.exists(f'ChC_size_{array_size}'):
            with open(f'ChC_size_{array_size}', 'rb') as dicts_handle:
                (ChC_dict, pyc_dict) = pickle.load(dicts_handle)
        else:
            attachedChC = ChC(pShape)
            ChC_dict, pyc_dict = attachedChC.attach_ChC_Coords(debug=False)
            attachedChC.sort_ChC()
            attachedChC.sort_PyC()
            with open(f'ChC_size_{array_size}', 'wb') as dicts_handle:
                pickle.dump((ChC_dict, pyc_dict), dicts_handle)

        return pShape, attachedChC

    def extractEncodedFeatures(self, input_array, ChCs):
        encodedFeatures = {}
        binaryPieces = self.extractBinaryPieces(input_array=pShape, ChCs=attachedChC)
        intValuesFromBin = set(binaryPieces.values()[0])


        encodedFeatures[centerRF] = [length, encoding]
        # encoding = self.enc.build_Encoding(binaryInputPiece)
        pass

    def extractBinaryPieces(self, input_array, ChCs):
        binaryPieces = {}
        filter = pShape.MAX_INPUT/attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT #default 255/40
        length = self.REC_FLD_LENGTH


        for i in range(pShape.shape[0]-length):
            for j in range(pShape.shape[1]-length):
                cornerStart = (i, j)
                centerRF = self.calcCenterRF(cornerStart, pShape.shape[0],
                                             pShape.shape[1])
                binaryInputPiece = self.applyReceptiveField(input_array,
                                                            cornerStart, filter,
                                                            attachedChC)
                bin1D = binaryInputPiece.copy().flatten()
                intValFromBin = bin1D.dot(2**np.arange(bin1D.size)[::-1])

                binaryPieces[centerRF] = [intValFromBin, binaryInputPiece]

        return binaryPieces



    def applyReceptiveField(input_array, cornerStart, filter, attachedChC,
                            length=4, dynamicThreshold=False, threshold=None):
        '''centerRF is point in the input space designating upper left corner of the
        receptive field.  Receptive field length represents the size of the
        receptive field.  Function returns an array of the same size as the
        receptive field filtered by the threshold set from the connected chandelier
        cell weights.'''

        start_x, start_y = cornerStart

        avgInputValInRF = np.average(input_array[start_x:length, start_y:length])
        if dynamicThreshold:
            threshold = avgInputValInRF/filter
        if not threshold:
            threshold = 0

        result = np.zeros([length, length])
        for i in range(start_x, start_x+length):
            for j in range(start_y, start_y+length):
                weight = attachedChC.total_Synapse_Weight(self, PyC_array_element=(i,j))
                result[i, j] = max(input_array[(i,j)] - filter*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        return binaryInputPiece
        # encoding = self.enc.build_Encoding(binaryInputPiece)


    def calcCenterRF(cornerStart, ht, wd):
        '''Helper function to find start indices for upper left corner of a
        receptive field.

        Inputs:
        cornerStart - list of 2 float values corresponding to the x, y starting
                      positions of the receptive field

        Returns:
        centerRF - x,y integer values corresponding to center of receptive field
        '''

        if self.REC_FLD_LENGTH%2 == 0:
            center_x = conrnerStart[0]+self.REC_FLD_LENGTH/2 - 1
            center_y = conrnerStart[1]+self.REC_FLD_LENGTH/2 - 1
        else:
            center_x = conrnerStart[0]+self.REC_FLD_LENGTH//2
            center_y = conrnerStart[1]+self.REC_FLD_LENGTH//2

        return [center_x, center_y]

        # if self.REC_FLD_LENGTH%2 == 0:
        #     start_x = pShape.shape[0]-centerRF-(1+self.REC_FLD_LENGTH/2)
        #     start_y = pShape.shape[1]-centerRF-(1+self.REC_FLD_LENGTH/2)
        # else:
        #     start_x = pShape.shape[0]-centerRF-(1+self.REC_FLD_LENGTH//2)
        #     start_y = pShape.shape[1]-centerRF-(1+self.REC_FLD_LENGTH//2)

        # # check if receptive field falls off input space and adjust
        # if start_x < 0:
        #     start_x = 0
        # if start_y < 0:
        #     start_y = 0
        # if start_x > pShape.shape[0]-self.REC_FLD_LENGTH:
        #     start_x = pShape.shape[0]-self.REC_FLD_LENGTH
        # if start_y > pShape.shape[1]-self.REC_FLD_LENGTH:
        #     start_y = pShape.shape[0]-self.REC_FLD_LENGTH
        #
        # return start_x, start_y








# build SDR from encoding!







## create a 4x4 reeceptive field and pass into encoder to create SDR for each receptive field.
## also need to calculate interference within receptive field to compute output weights
## pass output to spatial pooler

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
                print('value', value)
                print(centerRF)
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

    def countIslands(self, value, centerRF):
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
                    islandCenters.append(centerRF)
                    islandSize.append(1)
                    self.DFS(i, j, visited, islandCenters, islandSize, value, index)
                    index += 1

        return
