import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.draw import polygon_perimeter
from scipy.ndimage.filters import gaussian_filter

class Polygon:
    '''Generates a square array and inserts a polygon for the shape specified
    inside the array at an offset position of x and y and rotated with respect
    to angle.'''

    POLYGON_DICT = {'triangle': 3, 'rectangle': 4, 'pentagon': 5, 'hexagon': 6}

    def __repr__(self):
        return (f'A {self.form} shape of width {self.cx*2} and height'
               f' {self.cy*2} embedded in an array of size'
               f' {self.input_array.shape} at angle {self.angle}')

    def __init__(self, array_size=128, form='rectangle', x=0, y=0,
                 width=5, height=5, angle=0, maxInput=255):
        self.input_array = np.zeros([array_size,array_size])
        self.cx = np.floor(width/2)
        self.cy = np.floor(height/2)
        self.x = x
        self.y = y
        self.angle = angle
        self.form = form
        self.MAX_INPUT = maxInput
        self.activeElements = []

    def create_Simple_Polygon(self, form):
        ''' Returns the coordinates of the vertex points for a polygon of
        the specified shape.  Used for simple testing.'''

        vertex_number = self.POLYGON_DICT[form]
        if vertex_number == 3:
            polygon_vertices = np.array([(-self.cy, 0),
                                         (self.cy, -self.cx),
                                         (self.cy, self.cx)
                                        ])
        if vertex_number == 4:
            polygon_vertices = np.array([(-self.cx, -self.cy),
                                         (self.cx, -self.cy),
                                         (self.cx, self.cy),
                                         (-self.cx, self.cy)
                                        ])
        if vertex_number == 5:
            a = (-self.cy, 0)
            b = (np.floor(-self.cy*np.cos(2*np.pi/5)),
                 np.floor(self.cx*np.sin(2*np.pi/5)))
            c = (np.floor(self.cy*np.cos(np.pi/5)),
                 np.floor(self.cx*np.sin(np.pi/5)))
            d = (np.floor(self.cy*np.cos(np.pi/5)),
                 np.floor(-self.cx*np.sin(np.pi/5)))
            e = (np.floor(-self.cy*np.cos(2*np.pi/5)),
                 np.floor(-self.cx*np.sin(2*np.pi/5)))
            polygon_vertices = np.array([a, b, c, d, e])
        if vertex_number == 6:
            a = (0, -self.cx)
            b = (-self.cy, np.sqrt(3)/2*-self.cx)
            c = (-self.cy, np.sqrt(3)/2*self.cx)
            d = (0, self.cx)
            e = (self.cy, np.sqrt(3)/2*self.cx)
            f = (self.cy, np.sqrt(3)/2*-self.cx)
            polygon_vertices = np.array([a, b, c, d, e, f])

        return polygon_vertices

    def create_Complex_Shape(self, vert_number, Rmax):
        ''' Returns the coordinates of the vertex points for a random
        non-self-intersecting polygon with the specified number of vertices
        which are by default 2 dimensional.'''
        polygon_vertices = np.zeros([vert_number, 2])
        # np.random.seed(5) #for testing only
        for n in range(vert_number):
            angle = np.pi * np.random.uniform(0,2)
            x = Rmax * np.random.uniform(0,1) * np.cos(angle)
            y = Rmax * np.random.uniform(0,1) * np.sin(angle)
            polygon_vertices[n] = (x, y)

        return polygon_vertices

    def insert_Polygon(self, complex=False, vert_number=False, Rmax=False,
                       display=False, useVariableTargValue=False):
        ''' Inserts a polygon perimeter into a numpy array.'''
        if complex:
            poly = self.create_Complex_Shape(vert_number, Rmax)
        else:
            poly = self.create_Simple_Polygon(self.form)
        theta = (np.pi/180.0)*self.angle
        rotator = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        offset = np.array([self.x, self.y])
        transformed_poly = (np.dot(poly, rotator) + offset).round()
        rr = transformed_poly[:,0].tolist()
        cc = transformed_poly[:,1].tolist()
        # polygon_perimeter is an imported library function
        rr, cc = polygon_perimeter(rr, cc, shape=self.input_array.shape,
                                   clip=False)
        if useVariableTargValue:
            self.input_array[rr, cc] = np.random.randint(self.MAX_INPUT)
        else:
            self.input_array[rr, cc] = self.MAX_INPUT
        [self.activeElements.append((r, c)) for r, c in zip(rr, cc) if (r, c) not in self.activeElements]
        if display:
            self.display_Polygon(self.input_array, angle=self.angle,
                                 form=self.form, polygon=transformed_poly)

        self.numActiveElements = self.countActiveElements()

        return self.input_array

    def display_Polygon(self, array, **kwargs): #angle, shape, polygon):
        '''For troubleshooting and guiding visual intuition.'''
        if array.size > 1025:
            print(f'Array too big.  Array size is {array.shape}')
        else:
            if kwargs:
                print(kwargs)
                print('Array:')
                print(array)
            plt.imshow(np.uint8(array), cmap='gray')
            plt.show()
            # for debugging:
            # img.save(f'test_{array.shape}_{kwargs["angle"]}_{kwargs["form"]}.png')

    def blur_Array(self, sigma=0.5):
        '''Returns a Guassian blurring function applied across the 2d array.'''
        array = np.asarray(self.input_array, dtype=np.float64)
        # gaussian_filter is imported library function
        self.input_array = gaussian_filter(array, sigma)

        return

    def create_Gradient(self, is_horizontal, gradStart=None, gradStop=None,
                        rowStart=0, colStart=0):
        '''Return gradient across the image in either the horizontal or vertical
        dimension.  Note if run twice- once for horizontal and once for vertical
        will produce a diagonal gradient.  Furthermore, by manipulating gradStart
        and gradStop to different values can create any amount of diagonal gradient.
        Default option is to apply 50% gradient i.e. 0-127 on 255 grayscale.'''
        array = np.asarray(self.input_array, dtype=np.float64)
        height = array.shape[0]
        width = array.shape[1]
        if not gradStart:
            gradStart = 0
        if not gradStop:
            gradStop = self.MAX_INPUT//2
        if is_horizontal:
            gradient = np.tile(np.linspace(gradStart, gradStop, width), (height, 1))
            if rowStart > 0:
                widthToEnd = width-rowStart
                tempEnd = gradient[:, 0:widthToEnd].copy()
                tempStart = gradient[:, widthToEnd:width].copy()
                gradient[:, rowStart:] = tempEnd
                gradient[:, 0:rowStart] = tempStart
        else:
            gradient = np.tile(np.linspace(gradStart, gradStop, height), (width, 1)).T
            if colStart > 0:
                heightToEnd = height-colStart
                tempEnd = gradient[0:heightToEnd, :].copy()
                tempStart = gradient[heightToEnd:height, :].copy()
                gradient[colStart:, :] = tempEnd
                gradient[0:colStart, :] = tempStart
        self.input_array += gradient
        self.input_array = np.clip(self.input_array, 0, self.MAX_INPUT)

        return


    def add_Noise(self, scale=1):
        '''Add Gaussian noise to each array element to simulate effect of
        movement of a sensor.'''
        # np.random.seed(5) # for testing
        noise = np.random.normal(0, scale, (self.input_array.shape))
        self.input_array += noise
        self.input_array = np.clip(self.input_array, 0, self.MAX_INPUT)
        # self.display_Polygon(self.input_array, angle=self.angle, form=self.form)

        return

    def countActiveElements(self):
        return np.count_nonzero(self.input_array)


class Target(Polygon):
    '''Subclass polygon to create a simple target array with some bits on
    (targets) and rest off.'''

    def __repr__(self):
        return (f'An array of size {self.input_array.shape} with '
                f'{self.numTargets} active elements inserted.')

    def __init__(self, array_size=32, numTargets=20, numClusters=0, maxInput=255):
        self.input_array = np.zeros([array_size,array_size])
        self.numTargets = numTargets
        if numTargets<numClusters:
            self.numClusters = numTargets
        else:
            self.numClusters = numClusters
        self.MAX_INPUT = maxInput
        self.activeElements = []


    def insert_Targets(self, useVariableTargValue=False):
        ''' Inserts specified number of targets into a numpy array.'''

        dimX = self.input_array.shape[0]
        dimY = self.input_array.shape[1]
        numTargets = self.numTargets

        if self.numClusters == 0:
            targsPerCluster = 1
            remaining = 0
        else:
            targsPerCluster, remaining = divmod(self.numTargets, self.numClusters)

        while numTargets > 0:
            row = np.random.randint(dimX)
            col = np.random.randint(dimY)
            numTargsToInsert = targsPerCluster
            if remaining > 0:
                numTargsToInsert += 1
                remaining -= 1

            runLength = int(np.ceil(np.sqrt(numTargsToInsert)))
            row, col = self.findRowAndColStart(row, col, runLength)
            rowStart, colStart = row, col
            rowStop, colStop = row+runLength, col+runLength

            if self.input_array[rowStart:rowStop, colStart:colStop].any():
                continue

            while numTargsToInsert:
                if useVariableTargValue:
                    self.input_array[row, col] = np.random.randint(self.MAX_INPUT)
                else:
                    self.input_array[row, col] = self.MAX_INPUT
                self.activeElements.append((row, col))
                numTargets -= 1
                numTargsToInsert -= 1
                row += 1
                if row > rowStop:
                    row = rowStart
                    col += 1

        self.numActiveElements = self.countActiveElements()

        return


    def findRowAndColStart(self, row, col, runLength):
        '''Helper function to compute safe place to start adding on bits'''

        if row-runLength < 0:
            row = 0
        else:
            row = row-runLength
        if col-runLength < 0:
            col = 0
        else:
            col = col-runLength

        return int(row), int(col)
