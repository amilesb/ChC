''' Unit tests for Encoder module in ChC package. '''

import unittest
from unittest import mock
import numpy as np
from Encoder import Encoder

class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.a = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]])
        self.encoded = Encoder(self.a)

    def test_Prox_Score(self):
        num_active = self.encoded.num_active
        proximity = self.encoded.proximity

        assert num_active == 6
        assert proximity == 35.48650249088339

    def test_Compute_Index(self):
        type_input = ['input_piece', 'num_active', 'prox_Score']
        value = [21889, 6, 35.48650249088339]
        index_result = [127, 142, 52]
        for i in range(len(type_input)):
            index = self.encoded.compute_Index(value[i], type_input[i])
            assert index == index_result[i]

        with self.assertRaises(ValueError):
            self.encoded.compute_Index(0,'wrong type')

    def test_Build_Encoding(self):
        result = self.encoded.build_Encoding()

        assert len(result) == 1200

        counter = 0
        for i in result:
            if i == 1:
                counter += 1

        assert counter == 60


if __name__ == '__main__':
    unittest.main()


# ''' for a 4x4 grid with 1 and 0 there are 2^16 permutations = 65536'''
#
# ''' Pseudocode to build an encoder
# 1 choose range of values to represent
# 2 compute the range = maxval - minval
# 3 choose a number of bukcets to split values into
# 4 choose the number of active bits to have in each representation, w
# 5 compute total number of bits n = bukcets + w -1
# 6 for given value, v determine the bucket, i that it falls into
#   i = floor(buckets*(v-minval)/range) create representation starting with n unset bits
#   then 1 at i and 1's for w-1 next bits followed by unset bits to end
# 7 * OPTIONAL * implement a hashing function to take step 6 to SDR style
#   000000000000001111111111111110000000000000
# to
#    001010101111000110000010000000001100101000
#
# '''
