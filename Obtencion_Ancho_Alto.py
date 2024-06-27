# -*- coding: utf-8 -*-

#Calibracion ancho y altura

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, filters, img_as_float32
from skimage.color import rgb2gray
import cv2 as cv
from skimage import data, io, filters, color

gx = [[-1,0,1],[-1,0,1],[-1,0,1]]
gy = [[-1,-1,-1],[0,0,0],[1,1,1]]
g45 = [[-1,-1,0],[-1,0,1],[0,1,1]]
g135 = [[0,1,1],[-1,0,1],[0,-1,-1]]

ima = io.imread('imm3.jpeg')
ima=rgb2gray(ima)
plt.figure(0)
plt.imshow(ima, cmap = 'gray')

imaa = img_as_float32(ima)
gxx = filters.edges.convolve(imaa,gx)
borde = np.where(np.abs(gxx)>0.5,1,0)

gyy = filters.edges.convolve(imaa,gy)
borde = np.where(np.abs(gxx)>0.5,1,0)

g455 = filters.edges.convolve(imaa,g45)
borde = np.where(np.abs(g455)>0.5,1,0)

g335 = filters.edges.convolve(imaa,g135)
borde = np.where(np.abs(g335)>0.5,1,0)

salida = np.zeros(ima.shape)
for i in range(ima.shape[0]):
    for j in range(ima.shape[1]):
        vector = np.max([np.abs(gxx[i,j]),np.abs(gyy[i,j]),np.abs(g455[i,j]),np.abs(g335[i,j])])
        salida[i,j] = vector
borde = np.where(salida>0.4,1,0)
# kernel = np.ones((5,5),np.uint8)
# opening = cv.morphologyEx(borde, cv.MORPH_OPEN, kernel)
#ima = data.camera()
plt.figure(1)
plt.imshow(borde, cmap = 'gray')

# plt.figure(2)
# plt.imshow(salida, cmap = 'gray')