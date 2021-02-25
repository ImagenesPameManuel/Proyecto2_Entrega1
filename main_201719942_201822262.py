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
from  scipy.signal import correlate2d
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

def MyCCorrelation_201719942_201822262(image, kernel, boundary_condition):
    CCorrelation=0
    return CCorrelation

#Comparaciones de resultados función creada con función propia de scipy.signal: correlate2d
#prueba_scipy=correlate2d()