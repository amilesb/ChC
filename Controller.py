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

max_gray = test_shape.MAX_INPUT # 255
max_tot_chc_wght = test_ChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT # 40

def calculate_Threshold_Filter(max_gray, max_tot_chc_wght):
    '''Inputs are scalar values representing the maximum grascale pixel
    intensity and the maximum total chandelier cell combined weight at an
    active point in the input space.  Function returns a scalar filter value
    that is used to apply a threshold to the an arbitrary value in the input
    space.'''
    return max_gray/max_tot_chc_wght

corner_start = (test_shape.shape[0]//2, test_shape.shape[1]//2)
filter = calculate_Threshold_Filter(max_gray, max_tot_chc_wght)


def apply_Receptive_Field(length=REC_FLD_LENGTH, corner_start=corner_start,
                          threshold=threshold):
    '''corner_start is point in the input space designating upper left corner of the
    receptive field.  Receptive field length represents the size of the
    receptive field.  Function returns an array of the same size as the
    receptive field filtered by the threshold set from the connected chandelier
    cell weights.'''

    # check if receptive field falls off input space and backup if so
    if corner_start[0] > test_shape.shape[0]-length:
        start_x = test_shape.shape[0]-length
    else:
        start_x = corner_start[0]
    if corner_start[1] > test_shape.shape[1]-length:
        start_y = test_shape.shape[0]-length
    else:
        start_y = corner_start[1]

    rec_fld_filter = np.zeros([length, length])
    for i in range(start_x, start_x+REC_FLD_LENGTH):
        for j in range(start_y, start_y+REC_FLD_LENGTH):
            weight = test_ChC.total_Synapse_Weight(self, PyC_array_element=(i,j))
            result = max(test_shape[(i,j)] - filter*weight, 0)
            if result > threshold:
                rec_fld_filter[(i,j)] = result
    encoding = Encoder(rec_fld_filter).build_Encoding()

# build SDR from encoding!







## create a 4x4 reeceptive field and pass into encoder to create SDR for each receptive field.
## also need to calculate interference within receptive field to compute output weights
## pass output to spatial pooler
