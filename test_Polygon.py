''' Unit tests for Polygon module in ChC package. '''
# see calcresource.com/geom-pentagon.html for general polygon formula

import unittest
import random

import numpy as np
import Polygon

data = np.load("/home/amilesb/Desktop/ChC/Polygon_arrays.npz")

Polygons = ['triangle', 'rect', 'pent', 'hex', 'complex', 'rotate30',
            'gradient', 'blur', 'noise']

class TestPoly(unittest.TestCase):

    def setUp(self):
        ''' Polygon is the module name and name of class so to instantiate a
        Polygon object must call the module.function (Polygon.Polygon()).
        Note, complex shape and add_Noise function both use a random number.
        A random seed of 5 is used before both are called to reset test case to
        match the base data array imported above.'''
        self.triangle = Polygon.Polygon(array_size=16, form='triangle', x = 7,
                                        y = 5, width=8, height=12, angle = 0)
        self.rect = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                    y = 5, width=8, height=6, angle = 0)
        self.pent = Polygon.Polygon(array_size=12, form='pentagon', x = 5,
                                    y = 5, width=9, height=8, angle = 0)
        self.hex = Polygon.Polygon(array_size=12, form='hexagon', x = 5, y = 5,
                                   width=8, height=6, angle = 0)
        np.random.seed(5)
        self.complex = Polygon.Polygon(array_size=12, form='complex', x = 5,
                                       y = 5, width=8, height=6, angle = 0)
        self.rotate30 = Polygon.Polygon(array_size=16, form='triangle', x = 7,
                                          y = 5, width=8, height=12, angle = 30)
        self.gradient = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                        y = 5, width=8, height=6, angle = 0)
        self.blur = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                        y = 5, width=8, height=6, angle = 0)
        self.noise = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                        y = 5, width=8, height=6, angle = 0)

###################### VISUALIZATION ###########################
    def test_insert_Polygon(self):
        for shape in Polygons:
            x = getattr(self, shape)
            if shape == 'complex':
                x.insert_Polygon(complex=True, vert_number=5, Rmax=7)
            else:
                x.insert_Polygon()
            if shape == 'gradient':
                x.create_Gradient(is_horizontal=True)
                x.create_Gradient(is_horizontal=False)
            if shape == 'blur':
                x.blur_Array()
            if shape == 'noise':
                np.random.seed(5)
                x.add_Noise(2) # base array used for testing employed a scale of 2
            print(f'Now testing: test_{shape}')
            print(x)

            np.testing.assert_array_almost_equal_nulp(x.input_array,
                                                      data[f'test_{shape}'])

    def test_create_Gradient(self):
        self.grad = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                            y = 5, width=8, height=6, angle = 0)
        self.grad.insert_Polygon()
        self.grad.create_Gradient(is_horizontal=True)
        self.grad.create_Gradient(is_horizontal=False)
        print(f'''Testing an embedded rectangle with a gradient applied along
                both axes''')
        self.grad.display_Polygon(self.grad.input_array)

        # self.grad.display_Polygon(self.grad.input_array, angle =self.grad.angle,
                                    # form=self.grad.form)

    def test_blur_Array(self):
        self.blurred = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                        y = 5, width=8, height=6, angle = 0)
        self.blurred.insert_Polygon()
        self.blurred.blur_Array(0.8)
        self.blurred.display_Polygon(self.blurred.input_array, angle=self.blurred.angle)



    def test_add_Noise(self):
        self.noisy = Polygon.Polygon(array_size=12, form='rectangle', x = 5,
                                        y = 5, width=8, height=6, angle = 0)
        self.noisy.insert_Polygon()
        self.noisy.add_Noise(40)
        self.noisy.display_Polygon(self.noisy.input_array, angle=self.noisy.angle)

class TestTarget(unittest.TestCase):

    def setUp(self):
        self.targ = Polygon.Target(array_size=32, numClusters=5)

    def test_insert_Targets(self):
        self.targ.insert_Targets()
        self.targ.display_Polygon(self.targ.input_array)

        assert self.targ.numActiveElements == self.targ.numTargets


if __name__ == '__main__':
    unittest.main()
