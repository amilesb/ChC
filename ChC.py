import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from pprint import pprint
import random

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

    def attach_ChC_Coords(self, debug=False):
        '''Accepts a Polygon object (numpy array with shape inserted) and returns a
        ChC dictionary object which defines coordinates for ChCs within the array.
        Note, for convenience the ChC are placed at equal intervals within the array
        with any miscellaneous extra selected randomly.'''

        self.ChC_Coordinates, self.PyC_points = self.create_Coords()
        rng = default_rng()

        # build base chandelier cell map to input space
        for center in self.ChC_Coordinates:
            percent_connected = np.random.randint(30,50)
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
                        synapse_wght = np.random.randint(1,9)
                        endpoints.append({pt_coords: synapse_wght})
                        N_points -= 1
            self.ChCs[center] = endpoints

        # build reciprocal map of input space to attached chandelier cells
        self.build_Reciprocal_Map_Of_Input_To_ChC()

        return self.ChCs, self.PyC

    def build_Reciprocal_Map_Of_Input_To_ChC(self):
        for point in self.PyC_points:
            attached_chcs = []
            # Generate initial reciprocal mapping of input to ChCs
            for chc_point,connected_points in self.ChCs.items():
                for attached_point in connected_points:
                    chc_points_in_use = set()
                    if point in attached_point.keys():
                        synapse_wght = list(attached_point.values())[0]
                        attached_chcs.append({chc_point: synapse_wght})
                        chc_points_in_use.add(chc_point)



            # ensure each input column connected to at least 4 chcs
            while len(attached_chcs) < 4:
                free_chcs = self.ChC_Coordinates - chc_points_in_use
                new_connection = random.choice(list(free_chcs))
                chc_points_in_use.add(new_connection)
                synapse_wght = np.random.randint(1,9)
                attached_chcs.append({new_connection: synapse_wght})
                #add the new connection to ChC dictionary
                self.ChCs[new_connection].append({point: synapse_wght})

            self.PyC[point] = attached_chcs
            total_synapse_wght = self.total_Synapse_Weight(point)
            print('point:', point, 'total_synapse_wght:', total_synapse_wght)
            if total_synapse_wght > 40:
                self.change_Synapse_Weight(connection=(point, attached_chcs))


            ######## make this separate function


            # ensure each input column not over connected with ChCs
            stopper = 0
            while len(attached_chcs) > 7 and stopper <100:
                diconnected = []
                # dict_keys is unhashable type: solution is to convert to list
                # and simply use first element.
                pt = [point]
                max_connected = list(attached_chcs[0].keys())
                print('point:', point)
                print('length attached chcs:', len(attached_chcs))
                print('attached_chcs:', attached_chcs)
                for chc in attached_chcs:
                    chc_point = list(chc.keys())
                    print('chc_point:',chc_point)
                    if len(self.ChCs[chc_point[0]]) > len(self.ChCs[max_connected[0]]):
                        max_connected = chc_point
                    print('max_connected:', max_connected)
                attached_chcs = [i for i in attached_chcs if not
                                 list(i.keys()) == max_connected]
                print('new length of attached_chcs:', len(attached_chcs))
                print('attached_chcs:', attached_chcs)
                pyc_list = self.ChCs[max_connected[0]]
                new_pyc = [i for i in pyc_list if not list(i.keys()) == pt]
                print('length chc dict before:', len(self.ChCs[max_connected[0]]))
                self.ChCs.update({max_connected[0]: new_pyc})
                print('length chc dict after:', len(self.ChCs[max_connected[0]]))

                stopper += 1

            self.PyC[point] = attached_chcs

    def total_Synapse_Weight(self, PyC_array_element):
        '''Takes a point in the PyC input space and returns the total synapse
        weight attached to it.'''
        return sum([list(chc.values())[0] for chc in self.PyC[PyC_array_element]])


    def change_Synapse_Weight(self, connection, change=None):
        '''Connection is a tuple representing the connection between a list of
        chandelier cell points with a connected point in the input space.
        Change is the weight change of the synapse.  The function returns a new
        total synapse weight for the connected chandelier cells along with
        updated individual weights for each chandelier cell.'''
        PyC_key, ChC_keys = connection
        current_Total_Weight = self.total_Synapse_Weight(PyC_key)
        if change:
            new_Weight = current_Total_Weight + change
            new_Weight = self.check_Weight_Change(new_Weight)
        else:
            new_Weight = current_Total_Weight + np.random.randint(1,41)
        while new_Weight > current_Total_Weight:
            #reduce individual chc weights update both dicts
            pass
        while new_Weight < current_Total_Weight:
            #increase individual chc weights update both dicts
            pass

    def check_Weight_Change(self, new_Weight):
        ''' Check that total weight is between 0-40'''
        if new_Weight < 0:
            return 0
        elif new_Weight > 39:
            return 40
        else:
            return new_Weight

# note check_ChCs and check_PyC_Connection could create infinite loop so need logic to break out of
# like counter that gets passed back and forth a few times more than 3 than do X

    def check_ChCs(self):
        pass
        # while chc_connect_more60%inputspace: disconnect ChC update PyC
        # while chc connect < 20% : connect ChC update PyC

    def check_PyC_Connection():
        pass
        #while len(attached_chcs)< 4
        # while len (attached_chcs)>8  or total synapse wght > 40
        # remove bulkiest and update ChC
        # NOTE once this function defined can put into PyC map above


# # If decide to go observer route
# class Synapse:
#     def __init__(self):
#         self.observers = []
#         self._connection = None
#         self._weight = 0

    def update_Synapse_Wght(self):
        pass
    #  need to create observer like method here for each dictionary


        # if debug:
            # print ('self.ChC_Coordinates', self.ChC_Coordinates)
            # print('array size:',self.array_size)
            # print('ChC number:',numberChC)
            # print('height:',self.HT)
            # print('width:', self.WD)
            # print('Split:',SPLIT)
            # print('REMAINING_ChC:',REMAINING_ChC)
            # print('PyC points:', self.PyC_points)

            # plt.figure()
            # for ep in endpoints:
            #     plt.plot(*zip(center, ep), c = "red")
            # plt.show()



    def create_Coords(self):
        self.ChC_Coordinates = set()
        self.PyC_points = [(i, j) for i in range(self.HT) for j in range(self.WD)]

        if self.array_size > 3025:
            # match ChC number to biological values in column
            numberChC = self.array_size//300
        else:
            numberChC = 10 #serve as simple base value for smaller test cases
        SPLIT = int(np.sqrt(numberChC))
        REMAINING_ChC = numberChC-SPLIT**2

        for i in range(1, SPLIT+1):
            for j in range(1, SPLIT+1):
                self.ChC_Coordinates.add( (i*self.HT//SPLIT-np.ceil(SPLIT/2),
                                      j*self.WD//SPLIT-np.ceil(SPLIT/2)) )

        for k in range(REMAINING_ChC):
            self.ChC_Coordinates.add( (np.random.randint(0, self.HT),
                                 np.random.randint(0, self.WD)) )

        return self.ChC_Coordinates, self.PyC_points


''''each ChC connects to ~40% (effectlively 1200 PyC targets per ChC in column)
of input space near it with 1-7 synapses per ChC.  Each PyC receives input from
at least 4 ChC.  So if 55x55 = 3025 PyC then appx. 3025x4 = 12100 ChC
connections.  12100/1200 = ~10 ChC for small column.'''


test_ChC = ChC(test_shape)
ChC_dict, pyc_dict = test_ChC.attach_ChC_Coords(debug=True)


pprint(ChC_dict)

pprint(pyc_dict)

synapse_sum = test_ChC.total_Synapse_Weight((9,9))
pprint(synapse_sum)
