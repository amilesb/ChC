''' Unit tests for ChC module in ChC package. '''

import unittest
from unittest import mock
import random
import numpy as np
from numpy.random import default_rng
from pprint import pprint
import matplotlib.pyplot as plt
import copy

from Polygon import Polygon
from ChC import ChC


Polygons = ['triangle', 'rect', 'pent', 'hex', 'complex', 'rotate30',
            'gradient', 'blur', 'noise']

class TestChC(unittest.TestCase):

    def setUp(self):
        self.small_input = Polygon(array_size=10, form='rectangle', x=5, y=5,
                                width=4, height=3, angle = 0)
        self.small_input.insert_Polygon()
        self.small_ChC = ChC(self.small_input)

        self.large_input = Polygon(array_size=256, form='rectangle', x=125, y=125,
                              width=40, height=30, angle = 0)
        self.large_input.insert_Polygon()
        self.large_ChC = ChC(self.large_input)

        self.fake_ChC_dict = {
                              (1,1): [{(0,0): 0}, {(1,1): 1}, {(2,2): 2}],
                              (2,2): [{(0,0): 0}, {(1,1): 1}, {(2,2): 2}],
                              (3,3): [{(0,0): 0}, {(1,1): 1}, {(2,2): 2}],
                              (4,4): [{(0,0): 0}, {(1,1): 1}, {(2,2): 2}],
                              (5,5): [{(0,0): 3}, {(1,1): 4}, {(2,2): 5}]
                             }
        self.fake_PyC_dict = {
                              (0,0): [{(1,1): 0}, {(2,2): 0}, {(3,3): 0}, {(4,4): 0}, {(5,5): 3}],
                              (1,1): [{(1,1): 1}, {(2,2): 1}, {(3,3): 1}, {(4,4): 1}, {(5,5): 4}],
                              (2,2): [{(1,1): 2}, {(2,2): 2}, {(3,3): 2}, {(4,4): 2}, {(5,5): 5}],
                             }
        self.fake_PyC_points = [(0,0), (1,1), (2,2)]

    def build_ChC_Dict_From_PyC(self, PyC_dict):
        chand_dict = {}
        for pt, attached_chcs in PyC_dict.items():
            for ch_pt_with_weight_dict in attached_chcs:
                ch_pt = list(ch_pt_with_weight_dict.keys())[0]
                weight = list(ch_pt_with_weight_dict.values())[0]
                if ch_pt in chand_dict:
                    chand_dict[ch_pt].append({pt: weight})
                else:
                    chand_dict[ch_pt] = [{pt: weight}]
        return chand_dict


    def test_Create_Coords(self):
        mocked_random_int = lambda x,y : 7
        with mock.patch('numpy.random.randint', mocked_random_int):
            ChC_coordinates, PyC_points = self.small_ChC.create_Coords()

        points_list = [(i, j) for i in range(10) for j in range(10)]

        assert ChC_coordinates == {(1.0, 1.0), (1.0, 4.0), (1.0, 8.0),
                                   (4.0, 1.0), (4.0, 4.0), (4.0, 8.0), (7, 7),
                                   (8.0, 1.0), (8.0, 4.0), (8.0, 8.0)}

        assert PyC_points == points_list


    @mock.patch('ChC.ChC.build_Reciprocal_Map_Of_Input_To_ChC')
    @mock.patch('numpy.random.randint', lambda x,y : 40)
    @mock.patch('ChC.default_rng')
    def test_Attach_ChC_Coords(self, mocked_default_rng,
                               mocked_build_Reciprocal_Map_Of_Input_To_ChC,
                               debug=False):
        synapse_weight = 40

        mocked_default_rng.return_value.integers.side_effect=list(range(40))*10
        mocked_default_rng.return_value.random.side_effect=[0]*100000
        ChCs_dict, _ = self.small_ChC.attach_ChC_Coords()

        assert list(ChCs_dict[(8.0,8.0)][0].values())[0] == synapse_weight

        assert len(ChCs_dict[(8.0, 8.0)]) == 40

        a = [{(i, j):40} for i in range(4) for j in range(10)]
        b = {(1, 1): a}

        assert b.items() <= ChCs_dict.items()

    @mock.patch('ChC.ChC.total_Synapse_Weight')
    def test_Build_Reciprocal_Map_Of_Input_To_ChC_AND_check_PyC_Connection(self,
                                                                           mocked_Total):
        ''' attach_ChC_Coords calls build_Reciprocal_Map_Of_Input_To_ChC which
        in turn calls check_PyC_Connection.  This test function tests both.'''
        min_attached = 3
        max_attached = 9
        mocked_Total.return_value = 0
        ChCs_dict, PyC_dict = self.small_ChC.attach_ChC_Coords()

        #Chc_dict values are a list of other dicts sorting critical for equality check below
        self.small_ChC.sort_ChC()

        # test check_PyC_Connection method
        for k, v in PyC_dict.items():
            assert min_attached < len(v) < max_attached

        # use helper function defined above
        chand_dict = self.build_ChC_Dict_From_PyC(PyC_dict)

        # test build_Reciprocal_Map_Of_Input_To_ChC
        assert chand_dict == ChCs_dict

    def test_Total_Synapse_Weight(self):
        self.PyC = self.fake_PyC_dict
        sum1 = ChC.total_Synapse_Weight(self, (1,1))
        sum2 = ChC.total_Synapse_Weight(self, (2,2))
        assert sum1 == 8
        assert sum2 == 13

    @mock.patch('random.uniform')
    def test_total_Active_Weight(self, uniform_mock):
        uniform_mock.return_value = 0.5
        self.PyC = self.fake_PyC_dict
        sum1 = ChC.total_Active_Weight(self, (1,1), 0.75)
        sum2 = ChC.total_Active_Weight(self, (2,2), 0.25)
        assert sum1 == 8
        assert sum2 == 0

    def test_Find_Max_Connected_ChC(self):
        j = 0
        for k, v in self.fake_ChC_dict.items():
            j += 1
            if j==2:
                self.fake_ChC_dict[k] = v
            else:
                self.fake_ChC_dict[k] = [{(0,0): 1}]
        self.ChCs = self.fake_ChC_dict # to substitute in ChC module
        attached_chcs = self.fake_PyC_dict[(0,0)]

        max_connected = ChC.find_Max_Connected_ChC(self, attached_chcs)
        assert max_connected == [(2,2)]

    def test_Find_Least_Connected_ChC(self):
        j = 0
        for k, v in self.fake_ChC_dict.items():
            j += 1
            if j==2:
                self.fake_ChC_dict[k] = [{(0,0): 1}]
            else:
                self.fake_ChC_dict[k] = v
        self.ChCs = self.fake_ChC_dict # to substitute in ChC module
        attached_chcs = [(i,i) for i in range(1,6)]

        least_connected = ChC.find_Least_Connected_ChC(self, attached_chcs)
        assert least_connected == (2,2)


    def test_Change_Synapse_Weight(self):
        ChCs_dict, PyC_dict = self.small_ChC.attach_ChC_Coords()
        self.small_ChC.sort_ChC()
        self.small_ChC.sort_PyC()
        ChC_before = copy.deepcopy(ChCs_dict)
        PyC_before = copy.deepcopy(PyC_dict)
        pt = (0,0)


        ##### Test when change = RANDOM  #############
        attached_chcs = PyC_dict[(0,0)]
        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs))
        total_Synapse_Weight = self.small_ChC.total_Synapse_Weight(pt)
        assert 0 <= total_Synapse_Weight < self.small_ChC.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT

        ###### Test when change = 1 (condition 1; INC = 1)  #############
        total_Synapse_Weight_Before = self.small_ChC.total_Synapse_Weight(pt)
        attached_chcs = PyC_dict[(0,0)]
        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                             change = 1)

        total_Synapse_Weight_After = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_After-1 == total_Synapse_Weight_Before

        for k, v in PyC_dict.items():
            attached_chcs_before = PyC_before[k]
            attached_chcs_after = v
            if k != (0,0):
                assert attached_chcs_before == attached_chcs_after

        ###### Test when change = 1 AND all attached chc weights maxed out #####
        total_Synapse_Weight_Before = self.small_ChC.total_Synapse_Weight(pt)
        attached_chcs = PyC_dict[(0,0)]
        PyC_dict[(0,0)] = attached_chcs[0:4]# ensures chc list isn't maxed out so can append new connection
        attached_chcs = attached_chcs[0:4]
        for d in PyC_dict[(0,0)]:
            d.update((k, 8) for k, v in d.items()) # forcing all weight values to be greater than increment allowed
        length_before = len(attached_chcs)
        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                             change = 1)
        attached_chcs_after = PyC_dict[(0,0)]
        length_after = len(attached_chcs_after)

        assert length_after-1 == length_before

        ##### Test change = -1 AND total weight already equal to zero #######
        attached_chcs = PyC_dict[(0,0)]
        PyC_dict[(0,0)] = attached_chcs[0:4]
        for d in PyC_dict[(0,0)]:
            d.update((k, 0) for k, v in d.items()) # force all weights to be zero
        total_Synapse_Weight_Before = self.small_ChC.total_Synapse_Weight(pt)

        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                             change = -1)
        total_Synapse_Weight_After = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_Before == 0
        assert total_Synapse_Weight_After == 0

        ##### Test final condition when change=-1 and first selected chc_pt weight already zero #####
        attached_chcs = PyC_dict[(0,0)]
        PyC_dict[(0,0)] = attached_chcs[0:4]
        j = 0
        for d in PyC_dict[(0,0)]:
            if j == 0:
                d.update((k, 5) for k, v in d.items())
                j += 1
            else:
                d.update((k, 0) for k, v in d.items())
        total_Synapse_Weight_Before = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_Before == 5

        with mock.patch('ChC.ChC.select_Chand') as mocked:
            mocked.return_value = (-1, [0])
            self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                                 change = -5)

        total_Synapse_Weight_Before = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_After == 0

        ##### Integrated Test of function with recursion +/- ######
        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                             change = 50)

        total_Synapse_Weight_Positive = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_Positive == 40

        self.small_ChC.change_Synapse_Weight(connection=(pt, attached_chcs),
                                             change = -50)

        total_Synapse_Weight_Negative = self.small_ChC.total_Synapse_Weight(pt)

        assert total_Synapse_Weight_Negative == 0

    def test_Check_Weight_Change(self):
        self.TOTAL_MAX_ALL_CHC_ATTACHED_WEIGHT = 40
        condition1 = ChC.check_Weight_Change(self, -5)
        condition2 = ChC.check_Weight_Change(self, 15)
        condition3 = ChC.check_Weight_Change(self, 41)
        assert condition1 == 0
        assert condition2 == 15
        assert condition3 == 40

    def test_Select_Chand(self):
        target_less = 10
        target_greater = 35
        current_tot_wght = 30
        attached_chcs = self.fake_PyC_dict[(0,0)]
        set_indices = {i for i in range(len(attached_chcs))}

        inc, chc_index = ChC.select_Chand(self, target_less,
                                          current_tot_wght, attached_chcs)
        assert inc == -1
        assert chc_index[0] in set_indices

        inc2, chc_index2 = ChC.select_Chand(self, target_greater,
                                          current_tot_wght, attached_chcs)
        assert inc2 == 1
        assert chc_index2[0] in set_indices

    def test_Sort_ChC(self):
        self.ChC_Coordinates = {(i,i) for i in range(1,6)}
        self.ChCs = self.fake_ChC_dict
        ChC.sort_ChC(self)
        assert self.ChCs == self.fake_ChC_dict

    def test_Sort_PyC(self):
        self.PyC_points = self.fake_PyC_points
        self.PyC = self.fake_PyC_dict
        ChC.sort_PyC(self)
        assert self.PyC == self.fake_PyC_dict

    def test_Update_PyC_and_ChCs_dicts(self):
        self.PyC = self.fake_PyC_dict
        self.ChCs = self.fake_ChC_dict
        ChC.update_PyC_and_ChCs_dicts(self,
                                      PyC_point=(100,100),
                                      attached_chcs=[{(5,5):1}],
                                      chc_pt=(5,5),
                                      chc={(5,5):1},
                                      FLAG_NEW_CONNECTION=True
                                     )
        chand_dict = self.build_ChC_Dict_From_PyC(self.PyC)

        assert self.ChCs == chand_dict

        ChC.update_PyC_and_ChCs_dicts(self,
                                      PyC_point=(100,100),
                                      attached_chcs=[{(5,5):10}],
                                      chc_pt=(5,5),
                                      chc={(5,5):10},
                                      FLAG_NEW_CONNECTION=False
                                     )
        chand_dict2 = self.build_ChC_Dict_From_PyC(self.PyC)

        assert self.ChCs == chand_dict2


    def test_display_Weights(self):
        pass

    # def check_ChCs(self):
    # PLACEHOLDER  -- NOT IMPLEMENTED IN ChC module
    #     # while chc_connect_more60%inputspace: disconnect ChC update PyC
    #     # while chc connect < 20% : connect ChC update PyC

    ############### INTEGRATION TEST ########################

    # def test_Large_ChC(self):
    #     print('Takes about 25 mins to run for 256x256 input space')
    #     ChCs_dict, PyC_dict = self.large_ChC.attach_ChC_Coords()
    #     self.large_ChC.sort_ChC()
    #     self.large_ChC.sort_PyC()
    #
    #     chand_dict = self.build_ChC_Dict_From_PyC(self.PyC)
    #
    #     assert self.ChCs == chand_dict


if __name__ == '__main__':
    unittest.main()


'''NOTES'''

'''each ChC connects to ~40% (effectlively 1200 PyC targets per ChC in column)
of input space near it with 1-7 synapses per ChC.  Each PyC receives input from
at least 4 ChC.  So if 55x55 = 3025 PyC then appx. 3025x4 = 12100 ChC
connections.  12100/1200 = ~10 ChC for small column.'''

########### Common calls for quick reference ##########
# test_shape = Polygon(array_size=10, form='rectangle', x=5, y=5, width=4,
#                     height=3, angle = 0)
# test_shape.insert_Polygon()
# #
# #
# # # test_shape = Polygon(array_size=256, form='rectangle', x=125, y=125, width=40,
# # #                     height=30, angle = 0)
# # # test_shape.insert_Polygon()
# # # test_shape.display_Polygon(test_shape.input_array, angle=test_shape.angle)
#
#
# test_ChC = ChC(test_shape)
# ChC_dict, pyc_dict = test_ChC.attach_ChC_Coords(debug=False)
#
#
# test_ChC.sort_ChC()
# test_ChC.sort_PyC()
#
# # pprint(ChC_dict)
# #
# # pprint(pyc_dict)
#
# synapse_sum = test_ChC.total_Synapse_Weight((9,9))
# pprint(synapse_sum)
