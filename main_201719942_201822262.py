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
    if boundary_condition=="fill": #FALTA ARREGLAR
        fill_image=np.insert(image, 0, 0, axis=1)
        fill_image=np.insert(fill_image, 0, 0, axis=0)
        fill_image=np.insert(fill_image, fill_image.shape[0], 0, axis=0)
        fill_image=np.insert(fill_image, fill_image.shape[1], 0, axis=1)
        CCorrelation = np.zeros((len(image)+2, len(image[0])+2))
        #print(CCorrelation.shape)
        for filas in range(1,len(fill_image)-1):
            for columnas in range(1,len(fill_image[0])-1):
                i_fila=filas-1
                for multi_i in range(len(kernel)):
                    j_column=columnas-1
                    for multi_j in range(len(kernel[0])):
                        #CCorrelation[i][j]+=fill_image[filas-1][columnas-1]*kernel[multi_i][multi_j]
                        CCorrelation[filas][columnas]+=fill_image[i_fila][j_column]*kernel[multi_i][multi_j]
                        j_column+=1
                    i_fila+=1
    elif boundary_condition=="symm":
        True
    elif boundary_condition=="valid":
        CCorrelation=np.zeros((len(image)-2,len(image[0])-2))
        for filas in range(1,len(image)-1):
            for columnas in range(1,len(image[0])-1):
                i_fila=filas-1
                for multi_i in range(len(kernel)):
                    j_column=columnas-1
                    for multi_j in range(len(kernel[0])):
                        CCorrelation[filas-1][columnas-1]+=image[i_fila][j_column]*kernel[multi_i][multi_j]
                        j_column+=1
                    i_fila+=1
    return CCorrelation

rosas=io.imread("roses.jpg")
rosas_noise=io.imread("noisy_roses.jpg")
rosas=rgb2gray(rosas) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
rosas_noise=rgb2gray(rosas_noise) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
#print(rosas.shape) #print(kernel_a.shape) print(len(rosas))
#Comparaciones de resultados función creada con función propia de scipy.signal: correlate2d
prueba_ka=MyCCorrelation_201719942_201822262(rosas,kernel_a)
prueba_scipy=correlate2d(rosas,kernel_a,boundary="fill")
prueba_ka_v=MyCCorrelation_201719942_201822262(rosas,kernel_a,boundary_condition="valid")
prueba_scipy_v=correlate2d(rosas,kernel_a,mode="valid")
#print(prueba_scipy.shape)
print(prueba_scipy_v.shape)
print(prueba_ka_v.shape)
##
io.imshow(prueba_scipy)
plt.figure()
io.imshow(prueba_ka)
##
io.imshow(prueba_scipy_v)
plt.figure()
io.imshow(prueba_ka_v)
##5.1.1. Función MyCCorrelation 2.1
plt.figure("original_funcionpython")
plt.subplot(1,2,1)
plt.title("Imagen original escala grises")
plt.imshow(rosas,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Imagen correlate2d con kernel a")
plt.imshow(prueba_scipy,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##5.1.2. Aplicaciones de Cross-Correlación 1.
noise_kernel_a=MyCCorrelation_201719942_201822262(rosas_noise,kernel_a)
noise_kernel_b=MyCCorrelation_201719942_201822262(rosas_noise,kernel_b)
plt.figure("original_kernel_ab")
plt.subplot(1,3,1)
plt.title("Imagen original escala grises")
plt.imshow(rosas_noise,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con kernel a")
plt.imshow(noise_kernel_a,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con kernel b")
plt.imshow(noise_kernel_b,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##5.1.2. Aplicaciones de Cross-Correlación 3. FALTA ARREGLAR fill porque no funciona con filtros más grandes de 3x3 por dimensiones estables BOTA ERROR
filtro_Gauss_punto3=gaussian_kernel(5,1)
cross_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro_Gauss_punto3)
plt.figure("Original_kernel_b_Gauss")
plt.subplot(1,3,1)
plt.title("Imagen original escala grises")
plt.imshow(rosas_noise,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con kernel b")
plt.imshow(noise_kernel_b,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n5x5 y σ = 1")
plt.imshow(cross_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##5.1.2. Aplicaciones de Cross-Correlación 5.  YO LOS VEO IGUAL
filtro1_Gauss_punto5=gaussian_kernel(3,1)
filtro2_Gauss_punto5=gaussian_kernel(3,50)
filtro3_Gauss_punto5=gaussian_kernel(3,100)
cross1P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro1_Gauss_punto5)
cross2P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro2_Gauss_punto5)
cross3P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro3_Gauss_punto5)
plt.figure("Gausstamanofijosigmavariable")
plt.subplot(1,3,1)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 1")
plt.imshow(cross1P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 50")
plt.imshow(cross2P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 100")
plt.imshow(cross3P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##5.1.2. Aplicaciones de Cross-Correlación 7.  VA A BOTAR ERROR función no sirve para filtros con tamaño diferente 3x3
filtro1_Gauss_punto7=gaussian_kernel(3,5)
filtro2_Gauss_punto7=gaussian_kernel(5,5)
filtro3_Gauss_punto7=gaussian_kernel(7,5)
cross1P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro1_Gauss_punto7)
cross2P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro2_Gauss_punto7)
cross3P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro3_Gauss_punto7)
plt.figure("Gausstamanovariablesigmafijo")
plt.subplot(1,3,1)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 5")
plt.imshow(cross1P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con filtro Gauss:\n5x5 y σ = 5")
plt.imshow(cross2P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n7x7 y σ = 5")
plt.imshow(cross3P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##
a = np.array([[1, 1,1], [2, 2,2], [3, 3,3]])
b=np.insert(a, 0, 0, axis=1)
b=np.insert(b, 0, 0, axis=0)
b=np.insert(b, b.shape[0], 0, axis=0)
b=np.insert(b, b.shape[1], 0, axis=1)
print(b)
print(b.shape)
print(a.shape)