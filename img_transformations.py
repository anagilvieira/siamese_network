import cv2
import os
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np

folder = f'./DadosSinteticos/'
  

def add_noise(img):
  
    # Getting the dimensions of the image
    row , col = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img


def elastic(image, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)


    #print(random_state)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dz = np.zeros_like(dx)


    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))


    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='nearest')  #wrap,reflect, nearest


    return distored_image.reshape(image.shape)


for filename in sorted(os.listdir(folder), key=len): 
    if filename.endswith(".png"):
        im_gray = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        #im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_CIVIDIS)  
        cv2.imwrite(f'./DadosSinteticos_3/{filename}', elastic(im_gray, alpha=5000, sigma=60, random_state=None))


"""
o_img = cv2.imread(f'./DadosSinteticos/100.png')
elMat = elastic(o_img, alpha=5000, sigma=12, random_state=None)


cv2.namedWindow('origin',0)
cv2.imshow('origin', o_img)
cv2.namedWindow('elastic',0)
cv2.imshow('elastic', elMat)


cv2.waitKey(0)
""" 
