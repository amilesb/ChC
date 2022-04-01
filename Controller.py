''' This is the top level script which integrates all the components to run
the actual experiments.'''

import numpy as np
import os.path
import pickle

from ChC import ChC
from Encoder import Encoder
from Polygon import Polygon





## build a polygon, attach a chandelier cell network
array_size = 10
test_shape = Polygon(array_size=array_size, form='rectangle', x=4, y=4, width=4,
                    height=3, angle = 0)
# test_shape = Polygon(array_size=array_size, form='rectangle', x=125, y=125, width=40,
#                     height=30, angle = 0)
test_shape.insert_Polygon()

test_shape.display_Polygon(test_shape.input_array, angle=test_shape.angle)


if os.path.exists(f'ChC_size_{array_size}'):
    with open(f'ChC_size_{array_size}', 'rb') as dicts_handle:
        (ChC_dict, pyc_dict) = pickle.load(dicts_handle)
else:
    test_ChC = ChC(test_shape)
    ChC_dict, pyc_dict = test_ChC.attach_ChC_Coords(debug=False)
    test_ChC.sort_ChC()
    test_ChC.sort_PyC()
    with open(f'ChC_size_{array_size}', 'wb') as dicts_handle:
        pickle.dump((ChC_dict, pyc_dict), dicts_handle)



## use ChC total synapse weight to apply threshold filter to input
REC_FLD_LENGTH = 4
stride = np.ceil(test_shape.shape[0]/REC_FLD_LENGTH) # 256/4 = 32
num_receptive_fields = stride**2 # 1024

start = (0,0)



receptive_fields = []
for i in range(receptive_fields_per_length):
    for j in range(receptive_fields_per_length):
        field = [x, y for ]
        threshold = test_ChC.total_Synapse_Weight(self, PyC_array_element=(i,j))
        receptive_field =

def apply_Receptive_Field(length=REC_FLD_LENGTH, start):
    '''start is point in the input space designating upper left corner of the
    receptive field.  Receptive field length represents the size of the
    receptive field.  Function returns an array of the same size as the
    receptive field filtered by the threshold set from the connected chandelier
    cell weights.'''

    # check if receptive field falls off input space and backup if so
    if start[0] > test_shape.shape[0]-length:
        start_x = test_shape.shape[0]-length
    else:
        start_x = start[0]
    if start[1] > test_shape.shape[1]-length:
        start_y = test_shape.shape[0]-length
    else:
        start_y = start[1]

    for i in range(start_x, start_x+REC_FLD_LENGTH):
        for j in range(start_y, start_y+REC_FLD_LENGTH):
            threshold = test_ChC.total_Synapse_Weight(self, PyC_array_element=(i,j))

            # pull out threshold and apply to input





## create a 4x4 reeceptive field and pass into encoder to create SDR for each receptive field.
## also need to calculate interference within receptive field to compute output weights
## pass output to spatial pooler
