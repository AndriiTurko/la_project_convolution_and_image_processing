import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import copy
import math
img1_path ='image/flows.jpg'
image1 = cv2.imread(img1_path)
resized1 = cv2.resize(image1, (300, 300))
cv2.imwrite('/resized1.jpg', resized1)


def multiply(matr1, matr2):
    if type(matr1) is 'numpy.ndarray':
        matr1 = matr1.tolist()
    if type(matr2) is 'numpy.ndarray':
        matr2 = matr2.tolist()

    if not check_rows_size(matr1) or not check_rows_size(matr2):
        return
    elif len(matr1[0]) != len(matr2):
        print("Wrong sizes of matrices!")
        return
    else:
        result = [[0 for i in range(len(matr2[0]))] for j in range(len(matr1))]
        for i in range(len(result)):
            for m in range(len(matr1[i])):            
                for j in range(len(result[0])):
        
                    result[i][j] += round(matr1[i][m] * matr2[m][j])
        return result


def check_rows_size(matrix):
    for i in range(1, len(matrix)):
        if type(matrix[i]) is 'numpy.ndarray':
            matrix[i] = matrix[i].tolist() # change type to check length properly
        if type(matrix[i-1]) is 'numpy.ndarray':
          matrix[i-1] = matrix[i-1].tolist() # change type to check length properly
        if len(matrix[i]) != len(matrix[i-1]):
            print("Wrong size of rows in matrix")
            return False
    return True


def black_and_white(image):
    black_and_white = [
            [0.07, 0.07, 0.07],

            [0.72, 0.72, 0.72],

            [0.21, 0.21, 0.21]
        ]


    # change of colour to black and white shadows
    b_w_image = []
    for i in image:
        temp_matr = []
        for j in i:
            temp_matr.append(multiply([j.tolist()], black_and_white)[0][::-1])
        b_w_image.append(temp_matr)
    return b_w_image

def filter_colour(col, image):  
    if col == "cyan":
        filter =   [
        [0, 0, 0],

        [0, 1, 0],

        [0, 0, 1]
        ]
    elif col == "magenta":
        filter = [
        [1, 0, 0],

        [0, 0, 0],

        [0, 0, 1]
        ]  
    
    elif col == "yellow":
        filter = [
        [1, 0, 0],

        [0, 1, 0],

        [0, 0, 0]
        ]
    
    else:
        print("Please, enter the colour properly. Image was not changed")
        filter = [
        [1, 0, 0],

        [0, 1, 0],

        [0, 0, 1]
        ]

    filtered_image = []
    for i in resized1:
        temp_matr = []
        for j in i:
            temp_matr.append(multiply([j.tolist()], filter)[0][::-1])
        filtered_image.append(temp_matr)
    return filtered_image


def convolve(m1, m2):
    if len(m1) != len(m1[0]) or len(m2) != len(m2[0]):
        print("Matrices must be nxn!")
        return

    elif len(m2) > len(m1):
        print("Matrix 1 should be bigger than matrix 2!")
        return

    size1 = len(m1)
    size2 = len(m2)

    result = [[0 for x in range(size1-size2+1)] for y in range(size1-size2+1)]

    for y in range(size1 - size2 + 1):
        for x in range(size1 - size2 + 1):
            res = 0 
            for y_i in range(size2):
                for x_i in range(size2):
                    res += m1[y+y_i][x+x_i] * m2[y_i][x_i]
            result[y][x] = res
    
    return result


def sobel_filters(img):
    horizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    vertical = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    new_img = [[img[y][x][0] for x in range(len(img))] for y in range(len(img))]
    
    Ix = convolve(new_img, horizontal)
    Iy = convolve(new_img, vertical)
    
    G = np.hypot(np.asarray(Ix), np.asarray(Iy))
    result = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return result


def transform(img, M, type_of_transformation): # M - transformation matrix
    size = len(img)
    check = 0
    if type_of_transformation == "rotate":
        new_size = 2 * size
        check = size - 1
    elif type_of_transformation == "mirror":
        new_size = size
    elif type_of_transformation == "scale":
        new_size = math.ceil(max(M[0][0], M[1][1]) * size)
    else:
        print("No such type of transformation!")
        return

    transformation = [[[0, 0, 0] for _x in range(new_size)]for _y in range(new_size)]
    for y in range(size):
      for x in range(size):
        [[x_new], [y_new]] = multiply(M, [[x], [y]])
        transformation[y_new][x_new+check] = img[y][x]
    return np.asarray(transformation)


# helper functions to find matrices of transformations

def get_horizontal_mirror_marix():
  horizontal_mirror_marix = [[0 for i in range(2)] for j in range(2)]
  horizontal_mirror_marix [0][0] = -1
  horizontal_mirror_marix [1][1] = 1

  return horizontal_mirror_marix 


def get_scaling_matrix(scale_factor_x, scale_factor_y):
  scaling_matrix = [[0 for i in range(2)] for j in range(2)]
  scaling_matrix[0][0] = scale_factor_x
  scaling_matrix[1][1] = scale_factor_y

  return scaling_matrix


def get_rotation_matrix(phi):
    phi = math.radians(phi)
    transformation_matrix = [[0 for i in range(2)] for j in range(2)]
    transformation_matrix[0][0] = math.cos(phi)
    transformation_matrix[0][1] = -math.sin(phi)
    transformation_matrix[1][0] = math.sin(phi)
    transformation_matrix[1][1] = math.cos(phi)
    
    return transformation_matrix


def blur(b, image):
    result = image.tolist()
    root = math.floor(b ** (1/2))
    size = len(result[0])

    for y in range(0, size-root, root):
        for x in range(0, size-root, root):
            temp_rgb = [0, 0, 0]
            for i in range(root):
                for j in range(root):
                    for l in range(3):
                        temp_rgb[l] += result[y+i][x+j][l]
        block_colour = [i/b for i in temp_rgb]
        for i in range(root):
            for j in range(root):
                result[y+i][x+j] = block_colour

    return result

import time

# test multiplication via its usage in image transformation
# comparison to function warpAffine() of cv2 library

def get_horizontal_mirror_marix_for_cv2():
    horizontal_mirror_marix = np.zeros((2,3))
    horizontal_mirror_marix [0][0] = -1
    horizontal_mirror_marix [1][1] = 1
    return horizontal_mirror_marix 


test_resized = cv2.resize(image1, (200, 200))
# cv2.imwrite('/resized1.jpg', resized1)

test_cv2 = get_horizontal_mirror_marix_for_cv2()
start = time.time()
cv2.warpAffine(test_resized, test_cv2, (300, 300),  flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
finish = time.time()
# print(start, "\n", finish)
time_cv2 = finish - start


test_our = get_horizontal_mirror_marix()
start = time.time()
transform(test_resized, test_our, "mirror")
finish = time.time()
time_our = finish - start

print("cv2 implementation time:", time_cv2, "\nOur implementation time:", time_our)


# test convolution
# comparison to function ndimage.filters.convolve() of scipy library

from scipy import ndimage

test_array = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
test_resized2 = [[test_resized[y][x][0] for x in range(len(test_resized))] for y in range(len(test_resized))]

test_np_array = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
start = time.time()
ndimage.filters.convolve(test_resized2, test_np_array)
finish = time.time()
time_scipy = finish - start


test_our = get_horizontal_mirror_marix()
start = time.time()
convolve(test_resized2, test_array)
finish = time.time()
time_our = finish - start

print("\nscipy implementation time:", time_scipy, "\nOur implementation time:", time_our)


# All together

cv2_imshow(resized1)

print("\nBlack and white filter")

b_w_image = black_and_white(resized1)
cv2_imshow(np.asarray(b_w_image))

print("\nColour filters")

cv2_imshow(np.asarray(filter_colour("yellow", resized1)))
cv2_imshow(np.asarray(filter_colour("magenta", resized1)))
cv2_imshow(np.asarray(filter_colour("cyan", resized1)))

print("\nMirrored image")

M = get_horizontal_mirror_marix()
mirror = transform(resized1, M, "mirror")
cv2_imshow(mirror)

print("\nEdges detected")

edges = sobel_filters(b_w_image)
cv2_imshow(edges)

print("\nStrong blur")

b = 200
blurred = blur(b, resized1)
cv2_imshow(np.asarray(blurred))

print("Weak blur")

b = 25
blurred = blur(b, resized1)
cv2_imshow(np.asarray(blurred))