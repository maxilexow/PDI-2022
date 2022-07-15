# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:31:37 2022

@author: Study
"""
import imageio
import numpy as np
import matplotlib.pyplot as plt 
import tp7_functions as tp7
from scipy.signal import convolve2d

MAT_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596,-0.275,-0.321],
                        [0.211,-0.523, 0.311]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

def rgb2yiq(img):
    return apply_matrix(img, MAT_RGB2YIQ)

def yiq2rgb(img):
    return apply_matrix(img, np.linalg.inv(MAT_RGB2YIQ))

def plots(title, img_rgb, img_gray, img_processed, title2):
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    plt.suptitle(title)
    axes[0].imshow(img_rgb)
    axes[0].title.set_text('Input image')
    axes[1].imshow(img_gray, 'gray')
    axes[1].title.set_text('Gray image')
    axes[2].imshow(img_processed, 'gray')
    axes[2].title.set_text(title2)   
    
# Cierro gráficos
plt.close('all')

# Cargo la imagen
filename = 'shark.jpg'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]

# =============================================================================
# Downsampling sampleando cada 2 pixeles
# =============================================================================
img_downsampled1 = tp7.downsampling1(img_gray, 2)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
plt.suptitle('Downsampling')
axes[0].imshow(img_rgb)
axes[0].title.set_text('Input image')
axes[1].imshow(img_gray, 'gray')
axes[1].title.set_text('Gray image')
axes[2].imshow(img_downsampled1, 'gray')
axes[2].title.set_text('Sampling cada 2 pixeles')

# =============================================================================
# Downsampling con cuadricula 2x2
# =============================================================================
img_downsampled2 = tp7.downsampling2(img_gray, 2)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
plt.suptitle('Downsampling')
axes[0].imshow(img_rgb)
axes[0].title.set_text('Input image')
axes[1].imshow(img_gray, 'gray')
axes[1].title.set_text('Gray image')
axes[2].imshow(img_downsampled2, 'gray')
axes[2].title.set_text('Dowsnampling promediando con cuadricula de 2x2')

# =============================================================================
# Ahora aplico un gaussiano y luego downsampleo
# =============================================================================
# Kernel gaussiano
N = 5
std = 1
kernel_gaussiano = tp7.gaussiano(N,std)
img_gray_gauss = convolve2d(img_gray, kernel_gaussiano, 'same')
img_downsampled3 = tp7.downsampling1(img_gray_gauss, 2)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
plt.suptitle('Downsampling')
axes[0].imshow(img_rgb)
axes[0].title.set_text('Input image')
axes[1].imshow(img_gray, 'gray')
axes[1].title.set_text('Gray image')
axes[2].imshow(img_downsampled3, 'gray')
axes[2].title.set_text('Downsampling con gaussiano y luego sampleando cada 2 pixeles')

#%%
# =============================================================================
# Upsampling repitiendo pixeles en una grilla 2x2
# =============================================================================
plt.close('all')
# Cargo la imagen
filename = 'shark_pixelado.jpg'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]


img_upsampled = tp7.upsampling2x2(img_gray)

plots('Upsampling', img_rgb, img_gray, img_upsampled, 'Upsampling con matriz 2x2')

# =============================================================================
# Upsampling utilizando una interpolación bilineal
# =============================================================================

img_upsampled = tp7.up_bilineal(img_gray)

plots('Upsampling', img_rgb, img_gray, img_upsampled, 'Upsampling con interp bilineal')

# =============================================================================
# Upsampling utilizando una interpolación bicubica
# =============================================================================

img_upsampled = tp7.up_bicubic(img_gray)
       
plots('Upsampling', img_rgb, img_gray, img_upsampled, 'Upsampling con interp bicubica')

# =============================================================================
# Ahora aplico un filtro gaussiano al upsampling de 2x2
# =============================================================================

img_upsampled = tp7.upsampling2x2(img_gray)
# APlico gaussiano: 
N = 5
std = 1
kernel_gaussiano = tp7.gaussiano(N,std)
img_upsampled = convolve2d(img_upsampled, kernel_gaussiano, 'same')

plots('Upsampling', img_rgb, img_gray, img_upsampled, 'Upsampling con gaussiano y matriz 2x2')

#%% 
# =============================================================================
#               Downsampling / Upsampling utilizando la fft
# =============================================================================
plt.close('all')
# Cargo la imagen
filename = 'shark_pixelado.jpg'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]

# Defino las filas y columnas de mi imagen
rows, cols = img_gray.shape
# Y luego las filas y columnas de la imagen a modificar
new_shape = (rows//2, cols*2)

# llamo a la funcion que hace el resampling utilizando la fft
img2 = tp7.fft_resampling(img_gray, new_shape)


plots('a', img_rgb, img_gray, img2, 'Resampling con fft')

#%%
# =============================================================================
#   Cuantización con método de cuantización uniforme
# =============================================================================
plt.close('all')
# Cargo la imagen
filename = 'imageio:coffee.png'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]

# Con cuantos niveles se quiere cuantizar la imagen
levels = 4

# Primero hago un downsampling para notar más la cuantización
img_down = tp7.downsampling2(img_gray, 2)

img_cuant = tp7.cuantiz_uniforme(img_down, levels)

plots('Cuantización', img_rgb, img_gray, img_cuant, 'Cuantización uniforme')

#%%
# =============================================================================
#   Cuantización con método de cuantización dithering scanline
# =============================================================================
plt.close('all')
# Cargo la imagen
filename = 'imageio:coffee.png'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]

# Con cuantos niveles se quiere cuantizar la imagen
levels = 4

# Primero hago un downsampling para notar más la cuantización
img_down = tp7.downsampling2(img_gray, 2)

#img_cuant = tp7.cuantiz_dithering_scanline(img_down, levels)
# No me anda la funcion cuantiz_dithering_scanline, así que lo hago acá mismo:
    
img = img_down
rows, cols = img.shape
img2 = np.zeros(img.shape)
error = 0
for i in range(rows):
    error = 0
    for j in range(cols):
        img2[i,j] = np.round((img[i,j] + error) * (levels-1))  / (levels) 
        error = (img[i,j] - img2[i,j])
        
plots('Cuantización', img_rgb, img_gray, img2, 'Cuantización dithering scanline')

#%%
# =============================================================================
#   Cuantización con método de cuantización floyd steinberg
# =============================================================================
plt.close('all')
# Cargo la imagen
filename = 'imageio:coffee.png'
img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]

# Con cuantos niveles se quiere cuantizar la imagen
levels = 4

# Primero hago un downsampling para notar más la cuantización
img_down = tp7.downsampling2(img_gray, 2)
    
img = img_down
rows, cols = img.shape
img2 = np.zeros(img.shape)
error = 0
for i in range(rows-1):
    error = 0
    for j in range(cols-1):
        img2[i+1,j  ] = np.round((img[i,j] + error*7/16) * (levels-1))  / (levels) 
        img2[i-1,j+1] = np.round((img[i,j] + error*3/16) * (levels-1))  / (levels) 
        img2[i  ,j+1] = np.round((img[i,j] + error*5/16) * (levels-1))  / (levels) 
        img2[i+1,j+1] = np.round((img[i,j] + error*1/16) * (levels-1))  / (levels) 
        error = (img[i,j] - img2[i,j])
        
plots('Cuantización', img_rgb, img_gray, img2, 'Cuantización floyd steinberg')










