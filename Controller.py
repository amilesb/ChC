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

    # def extractEncodedFeatures(self, input_array=pShape, ChCs=attachedChC):
    #     encodedFeatures = {}
    #     centerRF = [pShape.shape[0]/2, pShape.shape[1]/2]
    #     filter = pShape.MAX_INPUT/attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT #default 255/40
    #     informationPresent = True
    #     dynamicStride = self.REC_FLD_LENGTH
    #     lgth = self.REC_FLD_LENGTH
    #     directionToExpand = ['LT_UP', 'LT_DOWN', 'RT_UP', 'RT_DOWN']
    #     directionToAdvance = ['LT', 'RT', 'UP', 'DOWN']
    #
    #     ######### need to either expand receptive field or move it around and reset
    #
    #     while informationPresent:
    #         encoding = self.applyReceptiveField(input_array, centerRF, filter,
    #                                             attachedChC)
    #         encodedFeatures[centerRF] = [lgth, encoding]
    #         stepDirection = directionToExpand.pop(0)
    #         directionToExpand.append(stepDirection) # recycle choice at end of list
    #         centerRF = self.moveCenterRF(centerRF, stepDirection)
    #         nextEncoding = self.applyReceptiveField(input_array, centerRF,
    #                                                 filter, attachedChC)
    #         if nextEncoding == encoding:
    #             wd =pass##############
    #
    #     return


    def applyReceptiveField(input_array, centerRF, filter, attachedChC,
                            lgth=4, dynamicThreshold=False):
        '''centerRF is point in the input space designating upper left corner of the
        receptive field.  Receptive field length represents the size of the
        receptive field.  Function returns an array of the same size as the
        receptive field filtered by the threshold set from the connected chandelier
        cell weights.'''

        start_x, start_y = self.findCornerStart(centerRF, input_array)

        avgInputValInRF = np.average(input_array[start_x:lgth, start_y:lgth])
        if dynamicThreshold:
            threshold = avgInputValInRF/filter
        else:
            threshold = 0
        result = np.zeros([lgth, lgth])
        for i in range(start_x, start_x+lgth):
            for j in range(start_y, start_y+lgth):
                weight = attachedChC.total_Synapse_Weight(self, PyC_array_element=(i,j))
                result[i, j] = max(input_array[(i,j)] - filter*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        encoding = self.enc.build_Encoding(binaryInputPiece)


    def findCornerStart(centerRF, pShape):
        '''Helper function to find start indices for upper left corner of a
        receptive field.

        Inputs:
        centerRF - list of 2 float values corresponding to the x, y center of
                   the RF

        Returns:
        start_x, start_y - integer values corresponding to upper left corner
        '''

        if self.REC_FLD_LENGTH%2 == 0:
            start_x = pShape.shape[0]-centerRF-(1+self.REC_FLD_LENGTH/2)
            start_y = pShape.shape[1]-centerRF-(1+self.REC_FLD_LENGTH/2)
        else:
            start_x = pShape.shape[0]-centerRF-(1+self.REC_FLD_LENGTH//2)
            start_y = pShape.shape[1]-centerRF-(1+self.REC_FLD_LENGTH//2)

        # check if receptive field falls off input space and backup if so
        if start_x > pShape.shape[0]-self.REC_FLD_LENGTH:
            start_x = pShape.shape[0]-self.REC_FLD_LENGTH
        if start_y > pShape.shape[1]-self.REC_FLD_LENGTH:
            start_y = pShape.shape[0]-self.REC_FLD_LENGTH

        return start_x, start_y


    def moveCenterRF(centerRF, stepDirection):
        ''' Helper function to the receptive field center point to new location.

        Inputs:
        centerRF - list of 2 float values corresponding to the x, y center of
                   the RF
        stepDirection - string indicating one of 8 cardinal directions

        Returns:
        newCenterRF - list of 2 float values corresponding to new x, y center
        '''

        if stepDirection == 'LT_UP':
            if centerRF[0]%2 == 0:
                lgth += 1
            centerRF[0] = centerRF[0]-self.REC_FLD_LENGTH
            centerRF[1] = centerRF[1]-self.REC_FLD_LENGTH
        if stepDirection == 'LT_DOWN':
            centerRF[0] = centerRF[0]-self.REC_FLD_LENGTH
            centerRF[1] = centerRF[1]+self.REC_FLD_LENGTH
        if stepDirection == 'RT_UP':
            centerRF[0] = centerRF[0]+self.REC_FLD_LENGTH
            centerRF[1] = centerRF[1]-self.REC_FLD_LENGTH
        if stepDirection == 'RT_DOWN':
            centerRF[0] = centerRF[0]+self.REC_FLD_LENGTH
            centerRF[1] = centerRF[1]+self.REC_FLD_LENGTH

        return centerRF





# build SDR from encoding!







## create a 4x4 reeceptive field and pass into encoder to create SDR for each receptive field.
## also need to calculate interference within receptive field to compute output weights
## pass output to spatial pooler

class Graph:

    def __repr__(self):
        return (f'''This class implements an object to count the number of
        connected components in an undirected graph (islands in 2D array).''')

    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.graph = g

    def isSafe(self, i, j, visited):
        '''Function to check if a given cell (row, col) can be included in the
        depth first search (DFS) recursive function.

        Inputs:
        i, j - Integers representing row, col position in 2D array
        visited - boolean array of same dimensions as graph for whether cell has been checked

        Returns:
        Boolean value
        '''

        return (i>=0 and i<self.ROW and j>=0 and j<self.COL
                and not visited[i][j] and self.graph[i][j])

    def DFS(self, i, j, visited):
        '''Depth first search for 4 adjacent neighbors

        Inputs:
        i, j - Integers representing row, col position in 2D array
        visited - boolean array of same dimensions as graph for whether cell has been checked
        '''

        rowNbr = [-1, 0, 0, 1]
        colNbr = [0, -1, 1, 0]

        visited[i][j] = True

        for k in range(4):
            if self.isSafe(i + rowNbr[k], j + colNbr[k], visited):
                self.DFS(i + rowNbr[k], j + colNbr[k], visited)

    def countIslands(self):
        '''Main function which creates a boolean array of visited cells and
        counts number of connected components.

        Returns:
        count - Integer number of connected components
        '''

        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

        count = 0
        for i in range(self.ROW):
            for j in range(self.COL):
                if visited[i][j] == False and self.graph[i][j] == 1:
                    self.DFS(i, j, visited)
                    count += 1

        return count
