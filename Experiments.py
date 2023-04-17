'''This script represents a high level call to run specific experiments and
generate plots for research paper.'''

import numpy as np
from matplotlib import pyplot as plt
import time
import os

from Processor import Processor

############ FIGURE 1  ############

def createFigure1():

    applyRF_N = []
    internalN = []
    externalN = []
    noiseAdded = []

    standardizedInputs = dict(array_size=32, numTargets=20,
                              useTargetSubclass=True, maxInput=255,
                              useVariableTargValue=True)

    for i in range(2): #100
        start = time.time()

        pShape, attachedChC = Processor.buildPolygonAndAttachChC(**standardizedInputs)
        P_Noise = Processor('Exact', sparseHigh=20, gaussBlurSigma=0.1*i,
                            noiseLevel=0.1*i, display=False, pShape=pShape,
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



############## FIGURE 2 #####################
'''Once objects learned; found faster with weights set'''

def createFigure2():
    # Simple cleaning up
    DIR1 = 'ChC_handles/Objects'
    DIR2 = 'ChC_handles/Targs'

    if os.path.isdir(DIR1):
        for f in os.listdir(DIR1):
            os.remove(os.path.join(DIR1, f))
    if os.path.isdir(DIR2):
        for f in os.listdir(DIR2):
            os.remove(os.path.join(DIR2, f))

    # Simple setup - no noise, blurring, or gradient

    inputs = dict(array_size=32, numTargets=20, useTargetSubclass=True,
                  maxInput=255, useVariableTargValue=True)
    P_Objs = []
    for i in range(5): # 25
        pShape, attachedChC = Processor.buildPolygonAndAttachChC(**inputs)
        P = Processor('Exact', sparseHigh=20, gaussBlurSigma=5,
                      noiseLevel=5, display=False, pShape=pShape,
                      attachedChC=attachedChC
                     )
        P_Objs.append(P)

    results_AvgApplyRF = []
    for i in range(2): # 10
        applyRF_Global = []
        for P in P_Objs:
            applyRF_Local = []
            for j in range(10): # 20
                if i == 1:
                    P.noiseLevel += 1
                    P.gaussBlurSigma += 1
                sdrFoundWholeFlag, targetIndxs = P.extractSDR(plot=False)
                P.updateChCWeightsMatchedToSDR(targetIndxs)
                applyRF_Local.append(P.countAPPLY_RF)
                P.countAPPLY_RF = 0
            applyRF_Global.append(applyRF_Local)

        arrays = [np.array(x) for x in applyRF_Global]
        avgApplyRF = [np.mean(k) for k in zip(*arrays)]
        results_AvgApplyRF.append(avgApplyRF)

    plt.plot(results_AvgApplyRF[0])
    plt.plot(results_AvgApplyRF[1])
    plt.title('Average Number of Iterative Steps to Find SDR')
    plt.xlabel('Training Number')
    plt.ylabel('Number of Iterations')

    plt.show()

############# FIGURE 3 ######################

def createFigure3():

    # Setup
    P_Objs = []
    knownSDRs = []
    for i in range(1000):
        num = np.random.randint(20, 41)
        inputs = dict(array_size=32, numTargets=num, useTargetSubclass=True,
                      maxInput=255, useVariableTargValue=False)
        pShape, attachedChC = Processor.buildPolygonAndAttachChC(**inputs)
        P = Processor('Exact', sparseHigh=num, gaussBlurSigma=0,
                      noiseLevel=0, display=False, pShape=pShape,
                      attachedChC=attachedChC
                     )
        knownSDRs.append(P.pShape.activeElements)
        for j in range(10):
            sdrFoundWholeFlag, targetIndxs = P.extractSDR(plot=False)
            P.updateChCWeightsMatchedToSDR(targetIndxs)
        P_Objs.append(P)


    min=0
    max=255
    targBoost=100
    for i in range(100):
        P = np.random.choice(P_Objs)
        P.knownSDRs = knownSDRs
        indexSDR = P.pShape.activeElements
        len = P.pShape.input_array.shape[0]
        randInput = np.random.randint(min, max, size=(len, len))
        for idx in indexSDR:
            randInput[idx] = min(randInput[idx]+targBoost, max)
        P.pShape.input_array = randInput
        targetIndxs, _ = P.applyReceptiveField(mode='Seek')
        overlap = P.findNamesForMatchingSDRs(targetIndxs, knownSDRs)
        P.AIS.resetAIS()
        P.setChCWeightsFromMatchedSDRs(overlap)
        sdrFoundWholeFlag, targetIndxs = P.extractSDR(plot=False, mode='Infer')



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

figure 2 = Once objects learned; found faster with weights set

Figure 3 = Feed into spatial pooler 1000 sdrs; show that random activation
represents x% sdrA, y% sdrB, etc...

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

Figure XXX = create objects with varying # of targets 1-1024 show that network
can store maximal representations with sparsity in range of 20-40
note, if pattens drawn from random distribution improves robustness i.e. less
chance collision but if 20 activate pattern a and 19same 1 new activates z
instead of b then durning inference rearrange


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
    createFigure2()
