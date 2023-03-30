'''This script represents a high level call to run specific experiments and
generate plots for research paper.'''

import numpy as np
from matplotlib import pyplot as plt
import time

from Processor import Processor

############ EXPERIMENT 1  ############

def createFigure1():

    applyRF_N = []
    internalN = []
    externalN = []
    noiseAdded = []

    standardizedInputs = dict(array_size=32, numTargets=20,
                              useTargetSubclass=True, maxInput=255,
                              useVariableTargValue=True)

    for i in range(10):
        start = time.time()
        # Simple setup - no noise, blurring, or gradient
        # pShape, attachedChC = Processor.buildPolygonAndAttachChC(**standardizedInputs)
        # P = Processor('Exact', sparseHigh=20, pShape=pShape, attachedChC=attachedChC)
        # sdrFoundWholeFlag, targetIndxs = P.extractSDR()
        # internal.append(P.countINTERNAL_MOVE)
        # external.append(P.countEXTERNAL_MOVE)

        # Setup - with noise and blurring but no gradient
        pShape, attachedChC = Processor.buildPolygonAndAttachChC(**standardizedInputs)
        P_Noise = Processor('Exact', sparseHigh=20, gaussBlurSigma=0.01*i,
                            noiseLevel=0.01*i, display=False, pShape=pShape,
                            attachedChC=attachedChC
                            )
        print('True Targets', sorted(pShape.activeElements))
        sdrFoundWholeFlag, targetIndxs = P_Noise.extractSDR(plot=False)
        P_Noise.updateChCWeightsMatchedToSDR(targetIndxs)
        applyRF_N.append(P_Noise.countAPPLY_RF)
        internalN.append(P_Noise.countINTERNAL_MOVE)
        # internalN.append(P_Noise.internalMovesCounter)
        externalN.append(P_Noise.countEXTERNAL_MOVE)
        noiseAdded.append(i)

        end = time.time()
        print(f'time for noise with standard deviation equal to {0.05*i} added: {end-start:.1f}s')


#### NOTE applyRF, Internal, and external are effectively all the same plot!
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot(noiseAdded, applyRF_N, label='ApplyRF')
    ax2.plot(noiseAdded, internalN, label='Internal Movement')
    ax3.plot(noiseAdded, externalN, label='External Movement')
    ax1.set_title('ApplyRF')
    ax1.set_xlabel('Noise')
    ax1.set_ylabel('Number')
    ax2.set_title('Internal')
    ax2.set_xlabel('Noise')
    ax2.set_ylabel('Number')
    ax3.set_title('External')
    ax3.set_xlabel('Noise')
    ax3.set_ylabel('Number')

    plt.show()


    print('applyRF', applyRF_N)
    print('internal', internalN)
    print('external', externalN)



############## EXPERIMENT 2 #####################
'''Once objects learned; found faster with weights set'''



############# EXPERIMENT 3 ######################



'''learning a new object consists of activating more and more units until object
threshold (certain number of target indexes are predicted) reached; then paring
down until desired sparsity reached (note this may mean some false positives are
included!!!)

Weights get stored; a new object is learned

new objects keep getting added until threshold of collision with other objects
is reached.

At this point,network can tighten sparsity to increase stored items OR change
number of active targets in each object to move towards average value i.e. targ
with 8 gains new input connections to generate more and targ with 40 loses some.
'''

''' fig 1 = learning
fig 2 = inference - seq mem and movement
fig 3 = abstraction - sdr of sdrs; each mc learns weight slider
fig 4 = invariance - transfer learning
fig 5 = cycling - fr and feedback to control between learning and inference
fig 6 = topology - rearranging input in inference
fig 7 = pruning - learning is always trying to predict more; < 40 RF expands to
include more potential targs and search; > 40 targs and tighten rf if loses
sdr then need for topo else can prune

discuss:
IN and neuromodulators control state transitions
ChC in L2/3 and L5
L6
'''


''' figure 1 = learning show that network can find target SDR and refine to
desired with varying levels of noise sparsity (plot # applyRF, externalMove, and
refineSDR for various noise parameters)

Figure 2 = create objects with varying # of targets 1-1024 show that network
can store maximal representations with sparsity in range of 20-40
note, if pattens drawn from random distribution improves robustness i.e. less
chance collision but if 20 activate pattern a and 19same 1 new activates z
instead of b then durning inference rearrange

figure 3 = Once objects learned; found faster with weights set

figure 4 = information is abstracted; weight combos enable dynamic selection
i.e. hierarchical layer recieving inputs as lower level ChC weight SDRs then
it can learn/predict dynamic combinations -- 2 parts a 1.3 parts b
(this extends capacity of lower layer!  by itself can only learn small # but
layer above can blend disinct SDRs to dynamically extend capacity -- also
removes need for SPARSE SDRs)

figure 5 = interference produces output for FR
use FR to back project to control learn vs inference
add noise in input

Create data with no noise or gradient?
figure 6 = corrupt input so 20 targs but 1 is in wrong location
demonstrate rewiring
Increase corruption
compare to CNN

Figure 7
Add noise repeat fig 6?

fig ?
Try MNIST with uncorrupted / corrupted cells see if it can identify how many objects to create?

Couple figs main theory

Supp figs Theory and HTM review


Use feedback from interference in output to determine whether factor should
bias network to look for targs or instead initiate topographical rearrangement
(initical plan to implement with count external move?)


Things to do:
collect frames to make movie
add fxn call in external move to predict (set) chc wieghts and ais
examine internal move to improve filter ?



Homeostasis-
* during inference if too many activated = rearrange
* during learning if too many activated = refinement


'''



if __name__ == '__main__':

    createFigure1()
