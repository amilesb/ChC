import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from pprint import pprint
import random

from Polygon import Polygon

from PIL import Image
from skimage.draw import polygon_perimeter


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
        self.MIN_CHC_ATTACHED = 4
        self.MAX_CHC_ATTACHED = 8
        self.MAX_CHC_WEIGHT = 9
        self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT = 40

    def attach_ChC_Coords(self, debug=False):
        '''Accepts a Polygon object (numpy array with shape inserted) and returns a
        ChC dictionary object which defines coordinates for ChCs within the array.
        Note, for convenience the ChC are placed at equal intervals within the array
        with any miscellaneous extra selected randomly.'''

        self.ChC_Coordinates, self.PyC_points = self.create_Coords()

        # build base chandelier cell map to input space
        for center in self.ChC_Coordinates:
            percent_connected = np.random.randint(30,50)
            board = np.zeros((self.HT, self.WD), dtype=bool)
            N_points = min(round(board.size * percent_connected/100), 1200)
                       # 1200 is appx biological chc_connection value for ChC
            dist_char = board.size*0.35 # distance where probability decays to 1/e
            flat_board = board.ravel()
            endpoints = []
            while N_points:
                idx = default_rng().integers(flat_board.size)
                while flat_board[idx]:
                    idx += 1
                    if idx >= flat_board.size:
                        idx = 0
                if not flat_board[idx]:
                    pt_coords = (idx // board.shape[0], idx % board.shape[0])
                    point = np.array(pt_coords) # to enable distance calculation
                    dist = np.sqrt(np.sum((center-point)**2))
                    P = np.exp(-dist/dist_char)
                    if P > default_rng().random():
                        flat_board[idx] = True
                        synapse_wght = np.random.randint(0,self.MAX_CHC_WEIGHT)
                        endpoints.append({pt_coords: synapse_wght})
                        N_points -= 1
            self.ChCs[center] = endpoints
        print('made it')
        self.build_Reciprocal_Map_Of_Input_To_ChC()

        return self.ChCs, self.PyC

    def build_Reciprocal_Map_Of_Input_To_ChC(self):
        '''Constructs a reciprocal connectivity map with synaptic weights from
        input space to attached chandelier cells.'''
        for point in self.PyC_points:
            attached_chcs = []
            chc_points_in_use = set()
            # Generate initial reciprocal mapping of input to ChCs
            for chc_point,connected_points in self.ChCs.items():
                for attached_point in connected_points:
                    if point in attached_point.keys():
                        synapse_wght = list(attached_point.values())[0]
                        attached_chcs.append({chc_point: synapse_wght})
                        chc_points_in_use.add(chc_point)

            self.check_PyC_Connection(attached_chcs, point, chc_points_in_use)

            total_synapse_wght = self.total_Synapse_Weight(point)
            if total_synapse_wght > self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT:
                self.change_Synapse_Weight(connection=(point, attached_chcs))


    def check_PyC_Connection(self, attached_chcs, point, chc_points_in_use):
        '''Checks each input cell to ensure at least 4 chandelier cells
        connected and no more than 8. Updates both ChCs and PyC dictionaries.'''

        while len(attached_chcs) < self.MIN_CHC_ATTACHED:
            free_chcs = self.ChC_Coordinates - chc_points_in_use
            new_connection = random.choice(list(free_chcs))
            chc_points_in_use.add(new_connection)
            synapse_wght = np.random.randint(0,self.MAX_CHC_WEIGHT)
            attached_chcs.append({new_connection: synapse_wght})
            #add the new connection to ChC dictionary
            self.ChCs[new_connection].append({point: synapse_wght})
        self.PyC[point] = attached_chcs

        stopper = 0
        while len(attached_chcs) > self.MAX_CHC_ATTACHED and stopper <100:
            # dict_keys is unhashable type: solution is to convert to list
            # and simply use first element.
            pt = [point]
            max_connected = self.find_Max_Connected_ChC(attached_chcs)
            attached_chcs = [i for i in attached_chcs if not
                             list(i.keys()) == max_connected]
            pyc_list = self.ChCs[max_connected[0]]
            new_pyc = [i for i in pyc_list if not list(i.keys()) == pt]
            self.ChCs.update({max_connected[0]: new_pyc})
            stopper += 1
        self.PyC[point] = attached_chcs


    def total_Synapse_Weight(self, PyC_array_element):
        '''Takes a point in the PyC input space and returns the total synapse
        weight attached to it.'''
        return sum([list(chc.values())[0] for chc in self.PyC[PyC_array_element]])


    def find_Max_Connected_ChC(self, attached_chcs):
        '''Takes a list of attached chandelier cells in dictionary form with
        their corresponding weights and returns the one with the most
        connections to the input space.'''
        max_connected = list(attached_chcs[0].keys())
        for chc in attached_chcs:
            chc_point = list(chc.keys())
            if len(self.ChCs[chc_point[0]]) > len(self.ChCs[max_connected[0]]):
                max_connected = chc_point
        return max_connected


    def find_Least_Connected_ChC(self, attached_chcs):
        '''Takes a list of attached chandelier cells as tuples (without their
        corresponding weights) and returns the one with the least connections to
        the input space.'''
        # print('least connected function attached chcs', attached_chcs)
        least_connected = attached_chcs[0]
        for chc_pt in attached_chcs:
            if len(self.ChCs[chc_pt]) < len(self.ChCs[least_connected]):
                least_connected = chc_pt
        return least_connected


    def change_Synapse_Weight(self, connection,
                              change='RANDOM', target_tot_wght=None):
        '''Connection is a tuple representing the connection between a list of
        chandelier cell points with a connected point in the input space.
        Change is the weight change of the synapse.  The function returns a new
        total synapse weight for the connected chandelier cells along with
        updated individual weights for each chandelier cell.'''
        PyC_point, attached_chcs = connection
        current_tot_wght = self.total_Synapse_Weight(PyC_point)
        FLAG_NEW_CONNECTION = False

        if change == 'RANDOM':
            target_tot_wght = np.random.randint(0,self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT)

        if target_tot_wght or target_tot_wght==0:
            target_tot_wght = target_tot_wght
        else:
            target_tot_wght = current_tot_wght+change
            target_tot_wght = self.check_Weight_Change(target_tot_wght)
        inc, chc_index = self.select_Chand(target_tot_wght, current_tot_wght,
                                           attached_chcs)
        chc = attached_chcs[chc_index[0]]
        chc_pt = list(chc.keys())[0]

        if 0 <= chc[chc_pt]+inc < self.MAX_CHC_WEIGHT:
            chc[chc_pt] += inc
        elif inc==1 and len(attached_chcs) < self.MAX_CHC_ATTACHED:
            free_chcs = self.ChC_Coordinates-set([list(i.keys())[0] for
                                                  i in attached_chcs])
            new_connection = self.find_Least_Connected_ChC(list(free_chcs))
            attached_chcs.append({new_connection: 1})
            FLAG_NEW_CONNECTION = True
        elif inc==-1 and current_tot_wght == 0:
            return
        else:
            unchecked_chcs = [x for i,x in enumerate(attached_chcs)
                              if i!=chc_index and
                              0 <= list(x.values())[0]+inc < self.MAX_CHC_WEIGHT]
            chc = random.choice(unchecked_chcs)
            chc_pt = list(chc.keys())[0]
            chc[chc_pt] += inc

        self.update_PyC_and_ChCs_dicts(PyC_point, attached_chcs, chc_pt, chc,
                                       FLAG_NEW_CONNECTION)

        current_tot_wght = current_tot_wght+inc
        new_change = current_tot_wght-target_tot_wght

        if new_change != 0:
            self.change_Synapse_Weight(connection, change=new_change,
                                       target_tot_wght=target_tot_wght)
        else:
            return

    def check_Weight_Change(self, target_tot_wght):
        ''' Check that total weight is between 0 and the total allowed max ChC
        attached weight.  Return the target total weight satisfying the above
        bound.'''
        if target_tot_wght < 0:
            return 0
        elif target_tot_wght > self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT:
            return self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT
        else:
            return target_tot_wght

    def select_Chand(self, target_tot_wght, current_tot_wght, attached_chcs):
        '''Creates a weighted list of chandelier cell indexes from the
        attached_chcs provided as a list of dictionaries with (chand_cell: weight)
        to randomly choose from to adjust weight on.  Function returns chandelier
        cell index as a list with 1 element and the increment to adjust
        (either +1 or -1).'''
        chc_lengths = []
        for attached in attached_chcs:
            chc = list(attached.keys())
            chc_lengths.append(len(chc[0]))
        index = list(range(len(chc_lengths)))

        if target_tot_wght<current_tot_wght:
            inc = -1
            flipped = [-x for x in chc_lengths]
            chc_lengths_copy = [x for x in chc_lengths]
            weights_in_reverse = [None] * len(chc_lengths)
            for i in range(len(flipped)):
                index_F = flipped.index(max(flipped))
                flipped[index_F] = -10^80
                index_C = chc_lengths_copy.index(max(chc_lengths_copy))
                max_wt = chc_lengths_copy.pop(index_C)
                weights_in_reverse[index_F] = max_wt
            chc_index = random.choices(index, weights=weights_in_reverse)
        else:
            inc = 1
            chc_index = random.choices(index, weights=chc_lengths)

        return inc, chc_index

    def update_PyC_and_ChCs_dicts(self, PyC_point, attached_chcs, chc_pt, chc,
                                  FLAG_NEW_CONNECTION):
        '''Inputs: PyC_point is a tuple representing the pyramidal cell column
        location in the input space,
        attached_chcs is a list of dictionaries containing the connected
        chandelier cells as keys and weights as the values,
        chc_pt is a tuple representing the chandelier cell coordinates.
        chc is the corresponding dictionary containing chandelier cell as key
        and weight as value.
        FLAG_NEW_CONNECTION is a boolean.
        This function updates both mappings from pyramidal cell input space to
        chandelier cells and chandelier cells to the input space along with the
        shared weight value.'''
        self.PyC[PyC_point] = attached_chcs
        attached_pycs = self.ChCs[chc_pt]
        for attached_pyc in attached_pycs:
            if list(attached_pyc.keys())[0] == PyC_point:
                attached_pyc.update({PyC_point: chc[chc_pt]})
        if FLAG_NEW_CONNECTION:
            attached_pycs.append({PyC_point: 1})
        self.ChCs.update({chc_pt: attached_pycs})


    def create_Coords(self):
        '''Generate a set of coordinates for each chandelier cell along with a
        list of points in the input space.'''
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


    def sort_ChC(self):
        '''For each chandelier cell, this function sorts the list of connected
        points for easier viewing.'''
        for chc_pt in self.ChC_Coordinates:
            connected_points = self.ChCs[chc_pt]
            connected_points.sort(key=lambda x: list(x.keys())[0])

    def sort_PyC(self):
        '''For each pyramidal cell column in the input space, this function
        sorts the list of connected chandelier cells for easier viewing.'''
        for pyc_pt in self.PyC_points:
            connected_chcs = self.PyC[pyc_pt]
            connected_chcs.sort(key=lambda x: list(x.keys())[0])


''''each ChC connects to ~40% (effectlively 1200 PyC targets per ChC in column)
of input space near it with 1-7 synapses per ChC.  Each PyC receives input from
at least 4 ChC.  So if 55x55 = 3025 PyC then appx. 3025x4 = 12100 ChC
connections.  12100/1200 = ~10 ChC for small column.'''


if __name__ == '__main__':

    test_shape = Polygon(array_size=10, form='rectangle', x=5, y=5, width=4,
                        height=3, angle = 0)
    test_shape.insert_Polygon()
    #
    #
    # # test_shape = Polygon(array_size=256, form='rectangle', x=125, y=125, width=40,
    # #                     height=30, angle = 0)
    # # test_shape.insert_Polygon()
    # # test_shape.display_Polygon(test_shape.input_array, angle=test_shape.angle)


    test_ChC = ChC(test_shape)
    ChC_dict, pyc_dict = test_ChC.attach_ChC_Coords(debug=False)


    test_ChC.sort_ChC()
    test_ChC.sort_PyC()

    # pprint(ChC_dict)
    #
    # pprint(pyc_dict)

    synapse_sum = test_ChC.total_Synapse_Weight((9,9))
    # pprint(synapse_sum)
