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
                                 y=4, width=4, height=3, angle=0):
        ''' Inputs:
        array_size: integer
        form: string
        x, y: integers representing center of polygon
        width, height: integers corresponding to dimensions of 2D polygon
        angle: float representing rotation angle of polygon

        Function uses polygon class to draw a polygon in numpy array and then
        attaches Chandelier Cells to that input.

        returns:
        pShape: numpy array with specified shape perimeter defined within
        attachedChC: dictionary which maps chandelier cells to the input_array
        '''
        pShape = Polygon(array_size=array_size, form=form, x=x, y=y, width=width,
                        height=height, angle=angle)

        # pShape = Polygon(array_size=array_size, form='rectangle', x=125, y=125, width=40,
        #                     height=30, angle = 0)
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

    def extractEncodedFeatures(self, input_array=pshape, ChCs=attachedChC):
        encodedFeatures = {}
        centerRF = (pShape.shape[0]//2, pShape.shape[1]//2)
        filter = pShape.MAX_INPUT/attachedChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT #default 255/40

        encoding = self.applyReceptiveField(centerRF, filter, attachedChC)




    def applyReceptiveField(length=self.REC_FLD_LENGTH, threshold=0,
                            centerRF, filter, attachedChC):
        '''centerRF is point in the input space designating upper left corner of the
        receptive field.  Receptive field length represents the size of the
        receptive field.  Function returns an array of the same size as the
        receptive field filtered by the threshold set from the connected chandelier
        cell weights.'''

        # check if receptive field falls off input space and backup if so
        if centerRF[0] > pShape.shape[0]-length:
            start_x = pShape.shape[0]-length
        else:
            start_x = centerRF[0]
        if centerRF[1] > pShape.shape[1]-length:
            start_y = pShape.shape[0]-length
        else:
            start_y = centerRF[1]


        result = np.zeros([length, length])
        for i in range(start_x, start_x+REC_FLD_LENGTH):
            for j in range(start_y, start_y+REC_FLD_LENGTH):
                weight = attachedChC.total_Synapse_Weight(self, PyC_array_element=(i,j))
                result[i, j] = max(pShape[(i,j)] - filter*weight, 0)
        binaryInputPiece = np.where(result > threshold, 1, 0)

        encoding = self.enc.build_Encoding(binaryInputPiece)





# build SDR from encoding!







## create a 4x4 reeceptive field and pass into encoder to create SDR for each receptive field.
## also need to calculate interference within receptive field to compute output weights
## pass output to spatial pooler
