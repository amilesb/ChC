import numpy as np
from skimage.draw import polygon_perimeter
from PIL import Image

#Create input space
input_array = np.zeros([16,16])

class Polygon:

    '''Generates a square array and inserts a polygon for the shape specified
    inside the array at an offset position of x and y and rotated with respect
    to angle.'''
    POLYGON_DICT = {'triangle': 3, 'rectangle': 4, 'pentagon': 5, 'hexagon': 6}

    def __init__(self, array_size=128, shape='rectangle', x=0, y=0,
                 width=5, height=5, angle=0):
        self.input_array = np.zeros([array_size,array_size])
        self.cx = np.floor(width/2)
        self.cy = np.floor(height/2)
        self.x = x
        self.y = y
        self.angle = angle
        self.shape = shape

    def create_Polygon(self, shape):
        vertex_number = self.POLYGON_DICT[shape]
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
            pass
        if vertex_number == 6:
            pass

        return polygon_vertices

    def insert_Polygon(self):
        poly = self.create_Polygon(self.shape)
        theta = (np.pi/180.0)*self.angle
        rotator = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        offset = np.array([self.x,self.y])
        transformed_poly = (np.dot(poly, rotator) + offset).round()
        rr = transformed_poly[:,0].tolist()
        cc = transformed_poly[:,1].tolist()
        print(self.cx)
        print(rr)
        print(cc)
        # polygon_perimeter imported function
        rr, cc = polygon_perimeter(rr, cc, shape=self.input_array.shape,
                                   clip=False)
        print(transformed_poly)
        self.input_array[rr, cc] = 255
        np.reshape(self.input_array, (16,16))
        print(self.input_array)
        img = Image.fromarray(self.input_array,'L')
        img.save('test.png')

test_Rect = Polygon(array_size=16, shape='triangle', x = 7, y = 7, width=12, height=6, angle = 30)
test_Rect.insert_Polygon()


random_input = np.random.randint(0,1000,(128,128))

# Class ChC(dict):
#
#     def __init__(self, inputDimensionX, inputDimensionY):
#         numberChC = inputDimensionX * inputDimensionY
#         for key in numberChC:
#             self[]
