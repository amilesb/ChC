import numpy as np
from numpy.random import default_rng

from Spatial_Pooler import Spatial_Pooler
from Encoder import Encoder


sp = Spatial_Pooler()

def processSP:
    overlapScore = sp.computeOverlap(currentInput)
    winningColumnsInd = sp.computeWinningCOlumns(overlapScore)
    sp.updateSynapseParameters(winningColumnsInd, overlapScore, currentInput) # learning in spatial pooler
