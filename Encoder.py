import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


#### Note on encoding; in mammals ~20+ retinal ganglion cells (RGC) allow higher dimensional spatial information to be encoded so can use exta dimensions to represent more than single on bit in input (i.e. high firing rate implies 4 bits on in square configuration within receptive field versus 4 random bits on etc...)

class Encoder:
    '''Create a geometric encoder to map active points in 2D space to a fixed
    encoding which can then be passed on to a spatial pooling algorithm.'''

    def __repr__(self):
        return ('''This class accepts a piece of the input space after a binary
        threshold has been overlaid and then returns an encoding of that input
        which is then subsequently ready for spatial pooling analysis.  Note,
        the size of the input piece encoded represents the receptive field of
        the layer above.''')

    def __init__(self, fullInputArraySize=64, receptiveFieldSize=16):
        ''' input_piece is a binary array which arises from threshold filter(s)
        being overlaid on top of the input space and then (combined) to equal
        the desired receptive field for the output.'''

        # self.MAXVAL = 2^self.flat_input.size
        # self.MINVAL = 0
        # self.RANGE = self.MAXVAL-self.MINVAL
        self.w = 20 # number of active bits in the encoding - 1
        self.n = 300 # number of bits
        self.buckets = self.n+1-self.w
        self.variable_types = ['input_piece', 'num_active', 'prox_Score', 'location']
        self.size = receptiveFieldSize
        self.fullInputArraySize = fullInputArraySize

    def build_Encoding(self, input_piece, location):
        '''Create the multi-encoding of the input space based on 3 implicit
        variables.

        Inputs:
        input_piece: a binary numpy array (of square length equal to receptive
                     field size) which arises from a threshold
        location:    x, y coordinates of center of receptive field

        Returns:
        multiEncoding: a list representing the concatenated sparse distributed array.
        '''

        valInput, coordinates = self.retrieveBinaryValueInputAndCoordinates(input_piece)
        valNumActive, valProximity = self.prox_Score(coordinates)
        values = [valInput, valNumActive, valProximity, location]

        multiEncoding = []

        for type, value in zip(self.variable_types, values):
            index = self.compute_Index(type, value)
            encoding = np.zeros(self.n)
            encoding[index:index+self.w] = 1
            multiEncoding.append(encoding)

        multiEncoding = np.concatenate(multiEncoding)

        return multiEncoding

    def retrieveBinaryValueInputAndCoordinates(self, input_piece):
        ''' Transform 2D binary array into binary digit and store coordinate
        positions for active input bits.

        Inputs:
        input_piece: a binary numpy array which arises from a threshold
        filter being overlaid on top of the raw analog input.

        Returns:
        valInput: flattened binary integer representation of the input
        coordinates: a list of x, y positions of the on bits in the input_piece
        '''
        flat_input = input_piece.ravel()
        valInput = int("".join(str(x) for x in flat_input), 2)

        coordinates = []
        for x in range(input_piece.shape[0]):
            for y in range(input_piece.shape[0]):
                if input_piece[x, y] == 1:
                    coordinates.append([x, y])

        return valInput, coordinates


    def prox_Score(self, coordinates):
        '''Computes a proximity score based on distance from one another.

        Inputs:
        coordinates: a list of x, y positions of the on bits in the input_piece

        Returns:
        valNumActive: Integer number of on bits.
        valProximity: float64 value representing proximity of on bits to each other
        '''
        pts = np.asarray(coordinates)
        dist = np.sqrt(np.sum((pts[np.newaxis, :, :]-pts[:, np.newaxis, :])**2, axis=2))
        valNumActive = dist.shape[0]
        valProximity = np.sum(dist[np.triu_indices(pts.shape[0], 1)].tolist())

        return valNumActive, valProximity


    def compute_Index(self, type, VALUE):
        '''Find the index for a specific value for a scalar encoding.  Returns a
        scalar index in the range of MIN and MAX.'''
        if type == 'input_piece':
            min = 0.
            max = 2**self.size
        elif type == 'num_active':
            min = 0.
            max = self.size
        elif type == 'prox_Score':
            min = 0.
            max = self.helperMaxValueProximityScore(array_length=np.floor(np.sqrt(self.size))) # 257.02591608065126 = default for array length = 4
        elif type == 'location':
            min = 0.
            max = self.fullInputArraySize
            VALUE = np.sqrt(max)*VALUE[0]+VALUE[1]
        else:
            raise ValueError('wrong or no type provided.')
        range = max-min
        index = int(np.floor(self.buckets*(VALUE-min)/range))

        return index

    def helperMaxValueProximityScore(self, array_length):
        '''A simple helper function that calculates the proximity score for an
        array with all bits on i.e. the array corresponding to the maximum
        proximity score'''
        array_length = array_length.astype(np.int64)
        x = np.ones([array_length, array_length], dtype=int)
        e = Encoder()
        valInput, coordinates = self.retrieveBinaryValueInputAndCoordinates(x)
        valNumActive, maxValProximity = self.prox_Score(coordinates)

        return maxValProximity
