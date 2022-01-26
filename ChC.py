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

class ChC:
    '''create a chandelier cell dictionary which records which chandelier cells
    are attached to which input cells (PyC columns) and the strength of each of
    those connections'''

    def __repr__(self):
        return (f'''This class returns a dictionary which maps chandelier cells
                    to elements in an array.''')

    def __init__(self, Polygon_Array):
        self.Polygon_Array = Polygon_Array
        self.HT = Polygon_Array.input_array.shape[0]
        self.WD = Polygon_Array.input_array.shape[1]
        self.array_size = Polygon_Array.input_array.size
        self.ChCs = {}
        self.PyC = {}

    def create_ChC_Coords(self, debug=False):
        '''Accepts a Polygon object (numpy array with shape inserted) and returns a
        ChC dictionary object which defines coordinates for ChCs within the array.
        Note, for convenience the ChC are placed at equal intervals within the array
        with any miscellaneous extra selected randomly.'''

        # ChC_Coordinates, PyC Points = create

        ChC_Coordinates = []
        PyC_points = [(i, j) for i in range(self.HT) for j in range(self.WD)]

        if self.array_size > 3025:
            # match ChC number to biological values in column
            numberChC = self.array_size//300
        else:
            numberChC = 10 #serve as simple base value for smaller test cases
        SPLIT = int(np.sqrt(numberChC))
        REMAINING_ChC = numberChC-SPLIT**2

        for i in range(1, SPLIT+1):
            for j in range(1, SPLIT+1):
                ChC_Coordinates.append((i*self.HT//SPLIT-np.ceil(SPLIT/2),
                                    j*self.WD//SPLIT-np.ceil(SPLIT/2)))

        for k in range(REMAINING_ChC):
            ChC_Coordinates.append( (np.random.randint(0, self.HT),
                                 np.random.randint(0, self.WD)) )

        # build chandelier cell map to input
        rng = default_rng()
        percent_connected = 40
        for center in ChC_Coordinates:
            board = np.zeros((self.HT, self.WD), dtype=bool)
            N_points = min(round(board.size * percent_connected/100), 1200)
                       # 1200 is appx biological chc_connection value for ChC
            dist_char = 35 # distance where probability decays to 1/e
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
                        ''' need add reverse PyC chc_connection and check if too many
                        if counter =8 can't attach.'''
                        synaptic_strength = np.random.randint(1,9)
                        endpoints.append({pt_coords: synaptic_strength})
                        N_points -= 1
            self.ChCs[center] = endpoints

        # build reciprocal map of input space to attached chandelier cells
        for point in PyC_points:
            attached_chcs = []
            for chc_point,connected_points in self.ChCs.items():
                for attached_point in connected_points:
                    if point in attached_point.keys():
                        synaptic_strength = list(attached_point.values())[0]
                        attached_chcs.append({chc_point: synaptic_strength})
            self.PyC[point] = attached_chcs



        # if debug:
            # print ('ChC_Coordinates', ChC_Coordinates)
            # print('array size:',self.array_size)
            # print('ChC number:',numberChC)
            # print('height:',self.HT)
            # print('width:', self.WD)
            # print('Split:',SPLIT)
            # print('REMAINING_ChC:',REMAINING_ChC)
            # print('PyC points:', PyC_points)


        return self.ChCs, self.PyC

    ''' next need to reorganize so look at each pyc and check if at least 4 separate connections
    if <4 find cell with at least 5 and randomly disconnect one provided it is not 1 already connected
     and attach to other  continue for x time i.e. if same cells getting connected / disconeected then
     add ChC'''


    # def check_PyC_Connection()

''' next need to reorganize so look at each pyc and check if at least 4 separate connections
if <4 find cell with at least 5 and randomly disconnect one provided it is not 1 already connected
 and attach to other  continue for x time i.e. if same cells getting connected / disconeected then
 add ChC'''

''''each ChC connects to ~40% (effectlively 1200 PyC targets per ChC in column)
of input space near it with 1-7 synapses per ChC.  Each PyC receives input from
at least 4 ChC.  So if 55x55 = 3025 PyC then appx. 3025x4 = 12100 ChC
connections.  12100/1200 = ~10 ChC for small column.'''


test_ChC = ChC(test_shape)
ChC_dict, pyc_dict = test_ChC.create_ChC_Coords(debug=True)

# plt.figure()
# for ep in endpoints:
#     plt.plot(*zip(center, ep), c = "red")
# plt.show()

''' create super list that stores all cells connected in input space to check
and make sure at least 4 but no more than 8...'''



pprint(ChC_dict)

pprint(pyc_dict)
