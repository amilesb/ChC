'''This script represents a high level call to run specific experiments and
generate plots for research paper.'''

import numpy as np
from matplotlib import pyplot as plt

from Processor import Processor

############ EXPERIMENT 1  ############

def createFigure1():

    internal = []
    external = []
    for i in range(2):
        P = Processor()
        sdrFoundWholeFlag, targetIndxs = P.extractSDR('Exact', sparseHigh=20,
                                                      array_size=16,
                                                      numTargets=20,
                                                      useTargetSubclass=True
                                                      )
        P.pShape.display_Polygon(P.pShape.input_array)
        internal.append(P.countINTERNAL_MOVE)
        external.append(P.countEXTERNAL_MOVE)
##### figure out why second display is happening!!
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.hist(internal)
    # ax.hist(external)








if __name__ == '__main__':

    createFigure1()



# array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0,
# useTargetSubclass=True, numTargets=20, numClusters=0
