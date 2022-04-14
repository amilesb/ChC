import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class Encoder:
    '''Create a geometric encoder to map active points in 2D space to a fixed
    encoding which can then be passed on to a spatial pooling algorithm.'''

    def __repr__(self):
        return ('''This class accepts a piece of the input space after a binary
        threshold has been overlaid and then returns an encoding of that input
        which is then subsequently ready for spatial pooling analysis.  Note,
        the size of the input piece encoded represents the recptive field of the
        layer above.''')

    def __init__(self, input_piece):
        ''' input_piece is a binary array which arises from threshold filter(s)
        being overlaid on top of the input space and then (combined) to equal
        the desired receptive field for the output.'''
        self.flat_input = input_piece.ravel()
        # self.MAXVAL = 2^self.flat_input.size
        # self.MINVAL = 0
        # self.RANGE = self.MAXVAL-self.MINVAL
        self.w = 20 # number of active bits in the encoding - 1
        self.n = 400 # number of bits
        self.buckets = self.n+1-self.w
        self.valInput = int("".join(str(x) for x in self.flat_input), 2)

        coordinates = []
        for x in range(input_piece.shape[0]):
            for y in range(input_piece.shape[0]):
                if input_piece[x, y] == 1:
                    coordinates.append([x, y])
        self.num_active, self.proximity = self.prox_Score(coordinates)
        self.variable_types = {'input_piece': self.valInput,
                               'num_active': self.num_active,
                               'prox_Score': self.proximity}

    def prox_Score(self, coordinates):
        '''Accepts a list of coordinates and computes a proximity score based on
        distance from one another.  Returns this score along with the number of
        active elements.'''
        pts = np.asarray(coordinates)
        dist = np.sqrt(np.sum((pts[np.newaxis, :, :]-pts[:, np.newaxis, :])**2, axis=2))
        num_active = dist.shape[0]
        proximity = np.sum(dist[np.triu_indices(pts.shape[0], 1)].tolist())

        return num_active, proximity

    def build_Encoding(self):
        '''Create the multi-encoding of the input space based on 3 implicit
        variables.  Return the concatenated sparse distributed array.'''

        multiEncoding = []

        for type, value in self.variable_types.items():
            index = self.compute_Index(value,type)
            encoding = np.zeros(self.n)
            encoding[index:index+self.w] = 1
            multiEncoding.append(encoding)

        multiEncoding = np.concatenate(multiEncoding)

        return multiEncoding

    def compute_Index(self, VALUE, type):
        '''Find the index for a specific value for a scalar encoding.  Returns a
        scalar index in the range of MIN and MAX.'''
        if type == 'input_piece':
            min = 0.
            max = 2**self.flat_input.size
        elif type == 'num_active':
            min = 0.
            max = 16.
        elif type == 'prox_Score':
            min = 0.
            max = 257.02591608065126
        else:
            raise ValueError('wrong or no type provided.')
        range = max-min
        index = int(np.floor(self.buckets*(VALUE-min)/range))

        return index
