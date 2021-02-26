#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto2 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import nibabel
from scipy.io import loadmat
import os
import glob
import numpy as np
import skimage.io as io
import requests
from scipy.signal import correlate2d
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#se crean kernels propuestos en la guía
kernel_a=np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel_b=(1/9)*kernel_a
kernel_c=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_d=np.array([[1,2,1],[0,0,0],[-1,2,-1]])
#print(kernel_c) #print(kernel_d)
#con la función indicada en la guía se crea filtro gaussiano
def gaussian_kernel(size, sigma):
    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/(2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
    return g
filtro_Gauss=gaussian_kernel(3,5)
#print(filtro_Gauss)

def MyCCorrelation_201719942_201822262(image, kernel, boundary_condition="fill"):
    CCorrelation=0
    if boundary_condition=="fill":
        fill_image=np.insert(image, 0, 0, axis=1)
        fill_image=np.insert(fill_image, 0, 0, axis=0)
        fill_image=np.insert(fill_image, fill_image.shape[0], 0, axis=0)
        fill_image=np.insert(fill_image, fill_image.shape[1], 0, axis=1)
        CCorrelation = np.zeros((len(image)+1, len(image[0])+1))
        #print(CCorrelation.shape)
        for filas in range(1,len(fill_image)-1):
            for columnas in range(1,len(fill_image[0])-1):
                i_fila=filas-1
                for multi_i in range(len(kernel)):
                    j_column=columnas-1
                    for multi_j in range(len(kernel[0])):
                        #CCorrelation[i][j]+=fill_image[filas-1][columnas-1]*kernel[multi_i][multi_j]
                        CCorrelation[filas][columnas]+=fill_image[i_fila][j_column]*kernel[multi_i][multi_j]
                        #print("entra")
                        j_column+=1
                    i_fila+=1
    elif boundary_condition=="symm":
        True
    elif boundary_condition=="valid":
        True
    return CCorrelation

rosas=io.imread("roses.jpg")
rosas_noise=io.imread("noisy_roses.jpg")
rosas=rgb2gray(rosas) #se le quita 3D a la imágen para convertirla en una imagen blanco-negro
print(rosas.shape) #print(kernel_a.shape) print(len(rosas))
#Comparaciones de resultados función creada con función propia de scipy.signal: correlate2d
prueba_ka=MyCCorrelation_201719942_201822262(rosas,kernel_a)
prueba_scipy=correlate2d(rosas,kernel_a,boundary="fill")
print("fin")
print(prueba_scipy.shape)
io.imshow(prueba_scipy)
plt.figure()
io.imshow(prueba_ka)
##
a = np.array([[1, 1,1], [2, 2,2], [3, 3,3]])
b=np.insert(a, 0, 0, axis=1)
b=np.insert(b, 0, 0, axis=0)
b=np.insert(b, b.shape[0], 0, axis=0)
b=np.insert(b, b.shape[1], 0, axis=1)
print(b)
print(b.shape)
print(a.shape)