import numpy as np
from skimage.draw import polygon_perimeter

#Create input space
input_array = np.zeros([16,16])

def insert_rect(input_array, x, y, width, height, angle):
    '''Takes an input array and generates a rectangle perimeter inside the
    array at the desired angle.  x and y represent the offset coordinates for
    the rectangle's position.'''
    cx = np.floor(width/2)
    cy = np.floor(height/2)
    print(cx)
    rect = np.array([(-cx, -cy), (cx, -cy), (cx, cy), (-cx, cy)])
    theta = (np.pi/180.0)*angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x,y])
    transformed_rect = (np.dot(rect, R) + offset).round()
    rr = transformed_rect[:,0].tolist()
    cc = transformed_rect[:,1].tolist()
    print(rr)
    print(cc)
    rr, cc = polygon_perimeter(rr, cc, shape=input_array.shape, clip=False)
    print(transformed_rect)
    input_array[rr, cc] = 1
    print(input_array)

test = insert_rect(input_array, 8, 8, 5, 5, 20)

random_input = np.random.randint(0,1000,(128,128))

# Class ChC(dict):
#
#     def __init__(self, inputDimensionX, inputDimensionY):
#         numberChC = inputDimensionX * inputDimensionY
#         for key in numberChC:
#             self[]
