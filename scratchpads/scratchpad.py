import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
from scipy.ndimage.filters import gaussian_filter
from scipy import stats, ndimage


class Scratchpad():

    def __init__(self):
        self.counterTest = Counter()
        self.track = 1


    def argpart(self):
        a = np.arange(100).reshape(10, -1)
        num=5
        # https://stackoverflow.com/questions/57103908/finding-indices-of-k-top-values-in-a-2d-array-matrix
        top = np.c_[np.unravel_index(np.argpartition(a.ravel(),-5)[-5:],a.shape)]

        print(top)
        print('hellow',type(top))
        indxs = [tuple(x) for x in top.tolist()]
        print(indxs)
        print(type(indxs))


    def gauss_blur(self):
        sigma = 0
        array = np.arange(100, step=1).reshape((10,10))
        print(array)
        array = gaussian_filter(array, sigma)
        print(array)

        sigma=2.5
        mu=3
        s=np.random.normal(mu, sigma, size=(2, 4))
        count, bins, ignored = plt.hist(s, 30, density=True)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        plt.show()


    def internalMove(self, targetIndxs):

        print('counter at start of fxn', self.counterTest)

        if self.track == 1:
            print('I should only run once!')
            self.counterTest.clear()

        print('before update indx counter', self.counterTest)
        self.counterTest.update(targetIndxs)
        print('after update indx counter', self.counterTest)

        newIndx = self.track + 1

        if self.track >= 10:
            return targetIndxs
        else:
            self.track += 1
            self.internalMove([(newIndx, newIndx)])


    def commonPick(self):
        best = Counter(apple=1, banana=2, melon=3, orange=4)
        bestGuess = best.most_common(10)
        items = []
        count = []
        for c in bestGuess:
            items.append(c[0])
            count.append(c[1])
        tot = sum(count)
        prob = [c/tot for c in count]
        print(count, 'c')
        print('prob', prob)

        print('bg',bestGuess)
        print('itmes', items)
        print('count', count)

        targetIndxs = np.random.choice(items, size=20,
                                       replace=True, p=prob)
        print(targetIndxs)


    def recrusiveCounter(self, i):
        print('at start of recursion', self.counterTest)

        if i == 10:
            return


        if i%2 == 0:
            self.counterTest.update(['apples', 'oranges'])
        else:
            self.counterTest.update(['apples'])
        i += 1
        self.recrusiveCounter(i)


    def filtering(self):
        test = np.arange(100.00).reshape(10,10)
        print(test)
        mask = np.zeros((10, 10))
        mask[1,3] = 1
        mask[4,5] = 1
        mask[4,6] = 1
        mask[4,7] = 1
        mask[4,8] = 1
        mask[5,6] = 1
        mask[5,7] = 1
        mask[5,8] = 1
        mask[5,9] = 1
        mask[6,7] = 1
        test[1,3] = .01
        print('m',mask)
        result = test*mask
        print('result', result)
        thresholdedFilter = ndimage.uniform_filter(result, size=100)
        print('Threshold')
        print(thresholdedFilter)
        normWeighted = result-thresholdedFilter
        print('Normed')
        print(normWeighted)
        indxs = np.c_[np.unravel_index(np.argpartition(normWeighted.ravel(),-10)[-10:], normWeighted.shape)]
        print('indxs', indxs)

applyRFFFFFF
    elif mode=='Refine':
        if refineTargs==None:
            print('Failed to set target indices to refine in applyRF mode=Refine')
        # bin = np.zeros([self.pShape.input_array.shape[0], self.pShape.input_array.shape[1]])
        # for idx in targs:
        #     bin[idx[0], idx[1]] = 1
        # numTargsToRefine = np.sum(bin)
        # num = np.round(numTargsToRefine*(1-PERCENT_REFINE))
        #
        # thresholded = self.pShape.input_array.copy()*bin
        # thresholdedFilter = ndimage.uniform_filter(thresholded, size=size,
        #                                            mode='mirror')
        # normWeighted = thresholded-thresholdedFilter
        indxSort = np.c_[np.unravel_index(np.sort(normWeighted.ravel()), normWeighted.shape)]
        targetIndxs=[]
        i=0
        while len(targetIndxs) < numRefine or i<normWeighted.size:
            if indxSort[i] in refineTargs:
                targetIndxs.append(indxSort[i])
            i += 1


if __name__ == '__main__':

    scratch = Scratchpad()
    # scratch.internalMove([('aa', 'aa')])
    # scratch.argpart()
    # scratch.gauss_blur()
    # scratch.commonPick()
    # scratch.recrusiveCounter(0)
    # print('after recursive function exit', scratch.counterTest)
    scratch.filtering()
