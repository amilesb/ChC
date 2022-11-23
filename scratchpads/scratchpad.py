import numpy as np
import os.path
import pickle
import scipy.stats as stats
from collections import Counter
from scipy.ndimage.filters import gaussian_filter


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


if __name__ == '__main__':

    scratch = Scratchpad()
    # scratch.internalMove([('aa', 'aa')])
    # scratch.argpart()
    scratch.gauss_blur()
