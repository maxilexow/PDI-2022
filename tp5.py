# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:22:44 2022

@author: Maximiliano Lexow
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import convolve2d

# matplotlib inline
MAT_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596,-0.275,-0.321],
                        [0.211,-0.523, 0.311]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

def rgb2yiq(img):
    return apply_matrix(img, MAT_RGB2YIQ)

def yiq2rgb(img):
    return apply_matrix(img, np.linalg.inv(MAT_RGB2YIQ))

def rmse(img1, img2):
    return np.sqrt(np.mean((img1-img2)**2))

def plot_kernel(data, ax=None):
    rows, cols = data.shape
    y, x = np.meshgrid(np.arange(rows),np.arange(cols),indexing='ij')
    if ax == None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    _min, _max = (np.min(data), np.max(data))
    ax.plot_surface(x, y, data.T, cmap=cm.jet, vmin=_min, vmax=_max)

def plot_images_and_kernel(img, img_filt, kernel, kernel_name):
    fig = plt.figure(figsize=(17,5))
    plt.title(kernel_name)      # Agrega un título al plot
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, 'gray')
    ax1.title.set_text('Input image')
    ax2 = fig.add_subplot(132)
    ax2.imshow(img_filt, 'gray')
    ax2.title.set_text('Filtered image')
    ax3 = fig.add_subplot(133, projection='3d')
    plot_kernel(kernel, ax=ax3)
    ax3.title.set_text('Kernel')
    plt.show()
    
img = imageio.imread('imageio:camera.png')/255
plt.imshow(img, 'gray')




#%% Implementación de FILTOS PASA BAJOS

# 1. Box (cuadrado)
N = 3
cuadrado = np.ones((N,N)) / (N**2)
img_filt = convolve2d(img, cuadrado, 'same')

plot_images_and_kernel(img, img_filt, cuadrado, "Filtro cuadrado de orden {}".format(N))

#%%
# 2. Circular


#%%
# 3. Bartlett

# Escribo el KERNEL de Bartlett

#N = int(input("Ingrese el orden del filtro (N). Puede ser cualquier número impar mayor a 1: "))
N = 5
# Defino el filtro Bartlett de orden N ingresado por el usuario.
# Para ello primero defino un vector, y hago la multiplicación para crear la matriz

def bartlett(N) :
    matriz = []
    fila = np.zeros(N)
    for i in range(N):
        if i <= N/2:
            fila[i] = i + 1
        else:
            for k in range(int(np.ceil(N/2))):
                fila[i] = np.abs (i + 1 - (k+1)*2)
        
    matriz = np.outer(fila,np.transpose(fila))
    matriz = matriz / np.sum(matriz)
    return matriz

kernel_bartlett = bartlett(N)


img_filt = convolve2d(img, kernel_bartlett, 'same')

plot_images_and_kernel(img, img_filt, kernel_bartlett, "Filtro Bartlett de orden {}".format(N))

#%%
# 4. Gaussiano

# Kernel del filtro gaussiano
N = 5
std = 1.4

def gaussiano(N,std) :
    matriz = []
    fila = np.linspace(-(N - 1) / 2., (N - 1) / 2., N)
    gauss = np.exp(-0.5 * np.square(fila) / np.square(std))
        
    matriz = np.outer(gauss,np.transpose(gauss))
    matriz = matriz / np.sum(matriz)
    return matriz

kernel_gaussiano = gaussiano(N,std)


img_filt = convolve2d(img, kernel_gaussiano, 'same')

plot_images_and_kernel(img, img_filt, kernel_gaussiano, "Filtro Gaussiano de orden {} y desvío estándar {}.".format(N,std))    

#%% 
# 1. Laplaciano
# Defino kernel del filtro Laplaciano

neighbors = 8 # Puede valer 4 u 8

def laplaciano(neighbors):
    if neighbors == 4:
        kernel = np.array([[0, -1, 0], 
                           [-1, 4, -1],
                           [0, -1, 0]])
    elif neighbors == 8:
        kernel = np.array([[1, 1, 1], 
                           [1, -8, 1],
                           [1, 1, 1]])
    else:
        print("Ingresó un parámetro no especificado.")
    return kernel

kernel_laplaciano = laplaciano(neighbors)

img_filt = convolve2d(img, kernel_laplaciano, 'same')

plot_images_and_kernel(img, img_filt, kernel_laplaciano, "Filtro Laplaciano de {} vecinos.".format(neighbors))    

#%%
# 2. Pasaaltos a partir de un pasabajo elegido (Cuadrado, bartlett o gaussiano)

print("Elija el filtro pasabajos a partir del cual se armará un pasaaltos: ")
print("(1) cuadrado, (2) bartlett, (3) gaussiano")
# Como no funciona el input le fijo un filtro:
print("Ingrese el orden del filtro: ")

filter_name = "gaussiano"
N = 5
std = 1.4

filtro = gaussiano(N,std)    

pasaaltos = np.array(np.ones((N,N))) / np.square(N) - filtro

img_filt = convolve2d(img, pasaaltos, 'same')

plot_images_and_kernel(img, img_filt, pasaaltos, "Filtro pasaaltos a partir de un filtro {}:".format(filter_name))

#%%
# PASABANDA
# 1. Diferencia de Gaussianos

# Llamo dos filtros gaussianos con diferente desvío estándar y los resto:
# N debe ser el mismo lógicamente

N = 5
std1 = 0.5
std2 = 1
# std1 debe ser menor a std 2

gauss1 = gaussiano(N,std1)
gauss2 = gaussiano(N,std2)

# Defino a mi filtro por dif de gaussianas:
DoG = gauss1 - gauss2

img_filt = convolve2d(img, DoG, 'same')

plot_images_and_kernel(img, img_filt, DoG, "Filtro por diferencia de Gaussianas entre dos filtros de tamaño: N = {}, y desvío estándar: std1 = {}, std2 = {}".format(N,std1,std2))

#%% 
# OTROS
# 1. Mejora de contraste

# Utilizo un filtro laplaciano 

K = 0.8     # K es la constante de proporción del pasaaltos
filtro = laplaciano(4)
N = 3

# Filtro primero la img con el laplaciano
img_filt_laplace = convolve2d(img, filtro, 'same')
img_filt = img + K * img_filt_laplace 
img_filt = np.clip(img_filt,0,1)    # Clampeo la imagen de 0 a 1


# img_filt = convolve2d(img, MdC, 'same')

plot_images_and_kernel(img, img_filt, filtro, "Filtro de mejora de contraste con constante de proporción = {}".format(K))

#%% 1.2 IMPLEMENTAR LOS SIGUIENTES FILTROS DIRECCIONALES (ASIMÉTRICOS)

# Defino los kernel para los gradientes en 8 direcciones (Sobel 3x3)

Gx = np.array([[-1, 0, 1], 
               [-2, 0, 2],
               [-1, 0, 1]])
Gy = np.array([[1, 2, 1], 
               [0, 0, 0],
               [-1, -2, -1]])

img_filt_Gx = convolve2d(img, Gx, 'same')

plot_images_and_kernel(img, img_filt_Gx, Gx, "Gradiente sobel 3x3 oeste")

img_filt_Gy = convolve2d(img, Gy, 'same')

plot_images_and_kernel(img, img_filt_Gy, Gy, "Gradiente sobel 3x3 sur")

# Interpreto la imagen como número complejo
img_filt = img_filt_Gx + 1j * img_filt_Gy

# El módulo sirve para detectar bordes
# La fase indica hacia donde apunta el gradiente

modulo = np.abs(img_filt)
modulo = modulo / abs(np.max(modulo))   # Normalizo el módulo
fase = np.angle(img_filt)
fase = fase / abs(np.max(fase))  # Normalizo fase

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(modulo, 'gray')
ax1.title.set_text('Módulo')
ax2 = fig.add_subplot(122)
ax2.imshow(fase, 'coolwarm')
ax2.title.set_text('Fase')

# No puedo plotear la barra de colores

#%%
# Es necesario correr la celda anterior que implementa el filtro Sobel, para obtener la
# variable módulo.

# Aplico un umbral u para binarizar 
u = 0.2     # Umbral

img_bordes = np.zeros(modulo.shape)

img_bordes[modulo > u] = 1      # Esto ve donde modulo tiene pixeles mayor a u

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(img_bordes, 'gray')
ax1.title.set_text('Bordes binarizados')

#%%
# 1.4 Explicación del algoritmo detector de bordes Canny
# 
# Primero necesitamos una imagen en escala de grises. Luego le aplicamos un filtro gaussiano
# para hacerla más suave y quitarle ruido. Una vez hecho esto aplicamos el filtro sobel en X
# y en Y, para hallar el módulo del gradiente y la orientación (fase).
# Teniendo esta imagen ya procesada, debemos achicar los bordes ya detectados para que estos
# sean de 1 pixel. 
# Luego lo que debemos hacer es eliminar los bordes no dominantes (ruido). Eso se hace 
# eligiendo un rango de umbral adecuado. Lo que esté por encima del umbral max cuenta como
# borde, lo que está por debajo del umbral min se descarta, y lo que está en el medio se 
# analiza si los píxeles están en contacto con algún borde fuerte.
