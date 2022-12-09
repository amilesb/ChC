'''This script represents a high level call to run specific experiments and
generate plots for research paper.'''

import numpy as np
from matplotlib import pyplot as plt
import time

from Processor import Processor

############ EXPERIMENT 1  ############

def createFigure1():

    applyRF = []
    internal = []
    external = []
    applyRF_N = []
    internalN = []
    externalN = []

    standardizedInputs = dict(array_size=32, numTargets=20,
                              useTargetSubclass=True, maxInput=255,
                              useVariableTargValue=False)

##################  NEed to find when I miss by just a few!!!!!!!!!! how to find thme???

    for i in range(7):
        start = time.time()
        # Simple setup - no noise, blurring, or gradient
        # pShape, attachedChC = Processor.buildPolygonAndAttachChC(**standardizedInputs)
        # P = Processor('Exact', sparseHigh=20, pShape=pShape, attachedChC=attachedChC)
        # sdrFoundWholeFlag, targetIndxs = P.extractSDR()
        # internal.append(P.countINTERNAL_MOVE)
        # external.append(P.countEXTERNAL_MOVE)

        # Setup - with noise and blurring but no gradient
        pShape, attachedChC = Processor.buildPolygonAndAttachChC(**standardizedInputs)
        P_Noise = Processor('Exact', sparseHigh=20, gaussBlurSigma=i/2,
                            noiseLevel=1.5, display=True, pShape=pShape,
                            attachedChC=attachedChC
                            )
        print('True Targets', sorted(pShape.activeElements))
        sdrFoundWholeFlag, targetIndxs = P_Noise.extractSDR()
        applyRF_N.append(P_Noise.countAPPLY_RF)
        internalN.append(P_Noise.countINTERNAL_MOVE)
        externalN.append(P_Noise.countEXTERNAL_MOVE)

        end = time.time()
        print(f'time for noise with standard deviation equal to {i/2} added: {end-start:.1f}s')


##### figure out why second display is happening!!
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.hist(internal)
    # ax.hist(external)

    print('applyRF', applyRF_N)
    print('internal', internalN)
    print('external', externalN)






if __name__ == '__main__':

    createFigure1()



# array_size=10, form='rectangle', x=4, y=4, wd=4, ht=3, angle=0,
# useTargetSubclass=True, numTargets=20, numClusters=0
