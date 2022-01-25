import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from pprint import pprint

from Polygon import Polygon

from PIL import Image
from skimage.draw import polygon_perimeter
from scipy.ndimage.filters import gaussian_filter

test_shape = Polygon(array_size=10, form='rectangle', x=5, y=5, width=4,
                    height=3, angle = 0)
test_shape.insert_Polygon()


# test_shape = Polygon(array_size=256, form='rectangle', x=125, y=125, width=40,
#                     height=30, angle = 0)
# test_shape.insert_Polygon()
# test_shape.display_Polygon(test_shape.input_array, angle=test_shape.angle)

Class ChC:

    def __repr__(self):
        return (f'''This class returns a dictionary which maps chandelier cells
                    to elements in an array.''')

    def __init__(self, Polygon_Array):
        self.Polygon_Array = Polygon_Array
        self.HT = Polygon_Array.input_array.shape[0]
        self.WD = Polygon_Array.input_array.shape[1]
        self.array_size = Polygon_Array.input_array.size
        self.ChCs = {}

def create_ChC_Coords(Polygon_Array, debug=False):
    '''Accepts a Polygon object (numpy array with shape inserted) and returns a
    ChC dictionary object which defines coordinates for ChCs within the array.
    Note, for convenience the ChC are placed at equal intervals within the array
    with any miscellaneous extra selected randomly.'''
    HT = test_shape.input_array.shape[0]
    WD = test_shape.input_array.shape[1]
    array_size = test_shape.input_array.size
    Coordinates = []
    ChC = {}
    PyC = {}
    PyC_points = [(i, j) for i in range(HT) for j in range(WD)]

    if array_size > 3025:
        # match ChC number to biological values in column
        numberChC = array_size//300
    else:
        numberChC = 10 #serve as simple base value for smaller test cases
    SPLIT = int(np.sqrt(numberChC))
    REMAINING_ChC = numberChC-SPLIT**2

    for i in range(1, SPLIT+1):
        for j in range(1, SPLIT+1):
            Coordinates.append((i*HT//SPLIT-np.ceil(SPLIT/2),
                                j*WD//SPLIT-np.ceil(SPLIT/2)))

    for k in range(REMAINING_ChC):
        Coordinates.append((np.random.randint(0, HT), np.random.randint(0, WD)))

    rng = default_rng()
    percent_connected = 40
    for center in Coordinates:
        board = np.zeros((HT,WD), dtype=bool)
        N_points = min(round(board.size * percent_connected/100), 1200)
                   # 1200 is appx biological connection value for ChC
        dist_char = 35 # characteristic distance where probability decays to 1/e
        flat_board = board.ravel()
        endpoints = []
        while N_points:
            idx = rng.integers(flat_board.size)
            while flat_board[idx]:
                idx += 1
                if idx >= flat_board.size:
                    idx = 0
            if not flat_board[idx]:
                pt_coords = (idx // board.shape[0], idx % board.shape[0])
                point = np.array(pt_coords) # to enable distance calculation
                dist = np.sqrt(np.sum((center-point)**2))
                P = np.exp(-dist/dist_char)
                if rng.random() < P:
                    flat_board[idx] = True
                    # if check_PyC_Connection():
                    ''' need add reverse PyC connection and check if too many
                    if counter =8 can't attach.'''

                    endpoints.append(pt_coords)
                    N_points -= 1
        ChC[center] = endpoints

''' next need to reorganize so look at each pyc and check if at least 4 separate connections
if <4 find cell with at least 5 and randomly disconnect one provided it is not 1 already connected
 and attach to other  continue for x time i.e. if same cells getting connected / disconeected then
 add ChC'''

    if debug:
        print ('Coordinates', Coordinates)
        print('array size:',array_size)
        print('ChC number:',numberChC)
        print('height:',HT)
        print('width:', WD)
        print('Split:',SPLIT)
        print('REMAINING_ChC:',REMAINING_ChC)

    return ChC, PyC
# def check_PyC_Connection()


''''each ChC connects to ~40% (effectlively 1200 PyC targets per ChC in column)
of input space near it with 1-7 synapses per ChC.  Each PyC receives input from
at least 4 ChC.  So if 55x55 = 3025 PyC then appx. 3025x4 = 12100 ChC
connections.  12100/1200 = ~10 ChC for small column.'''

#
# def connect_ChC(Polygon_Array, ChC):
#
#
#
#
# plt.figure()
# for ep in endpoints:
#     plt.plot(*zip(center, ep), c = "red")
# plt.show()

''' create super list that stores all cells connected in input space to check
and make sure at least 4 but no more than 8...'''


ChC, _ = create_ChC_Coords(test_shape, debug=True)
pprint(ChC)
