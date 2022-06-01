import numpy as np
from numpy.random import default_rng
from collections import defaultdict
import random

from Spatial_Pooler import Spatial_Pooler
from Sequncer import SequenceMemory
from Encoder import Encoder

class Topographer:

    def __repr__(self):
        return (f'''This class implements the rewiring of the input space.''')


    def __init__(self, spatialPooler, sequencer):
        '''Num mini-columns = 2048, 40 is roughly 2% of 2048, the percent of
        inputs in radius initialized to be in column's potential synapses note
        want ~ 50% or 15-20 on at beginning == (40*0.5*0.75 = 15).  The
        connectedPerm value is arbitrary but the inc and dec are relative to
        it.'''
