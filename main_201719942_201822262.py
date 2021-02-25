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

def MyCCorrelation_201719942_201822262(image, kernel, boundary_condition):
    CCorrelation=0
    return CCorrelation

prueba_scipy=correlate2d()