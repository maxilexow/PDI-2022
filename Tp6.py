# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:24:07 2022

@author: Maximiliano Lexow
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import canny

MAT_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596,-0.275,-0.321],
                        [0.211,-0.523, 0.311]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

def rgb2yiq(img):
    return apply_matrix(img, MAT_RGB2YIQ)

def yiq2rgb(img):
    return apply_matrix(img, np.linalg.inv(MAT_RGB2YIQ))

img_gray = imageio.imread('imageio:camera.png')/255
img_rgb = imageio.imread('imageio:chelsea.png')/255
#img_bin = canny(rgb2yiq(img_rgb)[:,:,0], sigma=2)
img_bin = canny(img_gray, sigma=2)

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_bin, 'gray')
axes[2].imshow(img_rgb)


#%% PARTE 1 Implementar funciones que devuelvan los siguientes structuring elements. 
# Son similares a un kernel convolucional pero el tipo de dato es booleano (True o False). 
# Incluso pueden construirse aplicando un threshold a un kernel. Ej: se_box = kernel_box > 0

# Matriz de NxN de booleanos TRUE:
N = 3
# ----------------------------------------------------------------------------#
def se_box(N):
    se_box = np.ones((N,N), dtype=bool)
    return se_box
# ----------------------------------------------------------------------------#
     # llamo la funcion para guardar el resultado en una variable

# Defino una función que me devuelve un structuring element circular de radio R
# El radio se cuenta a partir del centro (un punto equivale a radio 0), un structuring 
# element de radio 3 por ejemplo, sería una matriz de 7x7

# ----------------------------------------------------------------------------#
def se_circle(R):
    kernel = np.zeros((2*R+1, 2*R+1))
    y,x = np.ogrid[-R:R+1, -R:R+1]
    mask = x**2 + y**2 <= R**2
    kernel[mask] = 1
    kernel = kernel.astype(bool)
    return kernel
# ----------------------------------------------------------------------------# 

    
def se_rectang_horiz(N):
    kernel = np.ones((int(np.ceil(N/3)),N)).astype(bool)
    return kernel

def se_rectang_vert(N):
    kernel = np.transpose(se_rectang_horiz(N))
    return kernel
                            #%% FUNCIONES VARIAS

# Aplica una operación morfológica a una imagen.
# Parámetros: img = imagen, se = elemento estructurante, fcn = funcion
# Si la fcn es el mín será una erosión, si es el máx, una dilatación, etc

def _morph_op(img, se, fcn):
    '''Implements a general morphologic operation.'''
    se_flip = np.flip(se, axis=[0,1])
    rk, ck = se_flip.shape
    img_pad = np.pad(img, ((rk//2, rk//2), (ck//2, ck//2)), 'edge')
    img_out = np.zeros(img.shape)
    for r,c in np.ndindex(img.shape):
        img_out[r,c] = fcn(img_pad[r:r+rk,c:c+ck][se_flip])
    return img_out

def _morph_multiband(img, se, argfcn):
    '''Implements a general morphologic operation on a mutichannel image based on the first channel.'''
    se_flip = np.flip(se, axis=[0,1])
    rk, ck = se_flip.shape
    img_pad = np.pad(img, ((rk//2, rk//2), (ck//2, ck//2), (0,0)), 'edge')
    img_out = np.zeros(img.shape)
    rse, cse = np.where(se_flip)
    for r,c in np.ndindex(img.shape[:2]):
        loc = argfcn(img_pad[r:r+rk,c:c+ck,0][se_flip])
        img_out[r,c] = img_pad[r+rse[loc],c+cse[loc]]
    return img_out

def _morph_color(img, se, argfcn):
    '''Applies a morphological operation to a color image based on the 
    Y-channel.
    '''
    img2 = (rgb2yiq(img)[:, :, 0])[:, :, np.newaxis]
    img2 = np.concatenate((img2, img),axis=2)
    result = _morph_multiband(img2, se, argfcn)[:, :, 1:]
    return result

def plot_images(img, img_bin, img_morph, title):
    fig = plt.figure(figsize=(17,5))
    plt.title(title)      # Agrega un título al plot
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, 'gray')
    ax1.title.set_text('Input image')
    ax2 = fig.add_subplot(132)
    ax2.imshow(img_bin, 'gray')
    ax2.title.set_text('Gray image')
    ax3 = fig.add_subplot(133)
    ax3.imshow(img_morph, 'gray')
    ax3.title.set_text('Output image')
    plt.show()
    
def plot_images2(img, img_gray, img3, title_img3, img_morph, se, title):
    fig = plt.figure(figsize=(17,10))
    plt.suptitle(title)      # Agrega un título al plot
    ax1 = fig.add_subplot(231)
    ax1.imshow(img, 'gray')
    ax1.title.set_text('Input image')
    
    ax2 = fig.add_subplot(232)
    ax2.imshow(img_gray, 'gray')
    ax2.title.set_text('Gray image')
    
    ax3 = fig.add_subplot(233)
    ax3.imshow(img3, 'gray')
    ax3.title.set_text(title_img3)
    
    ax4 = fig.add_subplot(234)
    ax4.imshow(img_morph, 'gray')
    ax4.title.set_text('Output image')
    
    if se.dtype == bool:
        se = np.array(se, dtype=int)        
    ax5 = fig.add_subplot(235)
    ax5.imshow(se)
    ax5.title.set_text('Structuring element')
    plt.show()
    #plt.tight_layout()
# ----------------------------------------------------------------------------#
#%%
# =============================================================================
#                      Definición de funciones varias
# =============================================================================
# Funcion erosión:
def im_ero(img, se):
    if img.ndim == 2 :
        return _morph_op(img, se, np.min)
    else:
        return _morph_color(img, se, np.argmin)
    
# Funcion dilatación:
def im_dilat(img, se):
    if img.ndim == 2 :
        return _morph_op(img, se, np.max)
    else:
        return _morph_color(img, se, np.argmax)

# Funcion auxiliar que calcula la mediana y posición  
def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]

# Funcion mediana:
def im_median(img, se):
    if img.ndim == 2 :
        return _morph_op(img, se, np.median)
    else:
        return _morph_color(img, se, argmedian)
    
# Función borde externo
def borde_ext(img, se):
    return im_dilat(img, se) - img

# Función borde interno
def borde_int(img, se):
    return img - im_ero(img, se)
    
# Funcion gradiente
def grad_img(img, se):
    return im_dilat(img, se) - im_ero(img, se)

# Función apertura
def apertura(img, se):
    return im_dilat(im_ero(img, se),se)

# Función cierre
def cierre(img, se):
    return im_ero(im_dilat(img,se),se)

# Función top hat
def top_hat(img, se):
    return img - apertura(img, se)

# Función bottom hat
def bot_hat(img, se):
    return cierre(img, se) - img

# CO
def CO(img, se):
    return apertura(cierre(img_gray,se), se)

# OC
def OC(img, se):
    return cierre(apertura(img_gray,se), se)

# Suavizado 
def suavizado(img, se, nivel):
    if nivel == 1 :
        img_suaviz = (im_ero(img,se)+im_dilat(img,se))/2
    elif nivel == 2:
        img_suaviz = (apertura(img,se) + cierre(img,se))/2
    else:
        img_suaviz = (CO(img,se) + OC(img,se))/2
    return img_suaviz
#%% 
# =============================================================================
# Operaciones básicas (Nivel 1)
# =============================================================================
plt.close('all')
# Nombre de la imagen que quiero utilizar
filename = 'text_supression_example.png'

img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]
img_bin = canny(img_gray, sigma=0.2)  

# Elijo el structuring element (box, circle o line)
se = se_circle(3)

# Erosiono la imagen:
img_ero = im_ero(img_gray, se)

plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_ero, se, "Erosion")
    
# Dilato la imagen
img_dilat = im_dilat(img_gray, se)
    
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_dilat, se, "Dilatación")
    
# Aplico el filtro de mediana 
img_median = im_median(img_gray, se)
    
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_median, se, "Mediana")
    
#%%
# =============================================================================
# Operaciones de suma/resta 
# =============================================================================
plt.close('all')

# Nombre de la imagen que quiero utilizar
filename = 'retina.jpg'

img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]
img_bin = canny(img_gray, sigma=2)  

se = se_circle(3)

# 1. Borde externo (dilatación - original)
img_borde_ext = borde_ext(img_gray, se)
    
plot_images(img_rgb, img_gray, img_borde_ext, "Borde exterior")

# 2. Borde interno (original - erosion)
img_borde_int = borde_int(img_gray, se)
    
plot_images(img_rgb, img_gray, img_borde_int, "Borde interior")

# 3. Gradiente (dilatación - erosion)
img_grad = grad_img(img_gray, se)

plot_images(img_rgb, img_gray, img_grad, "Gradiente")

#%%
# =============================================================================
# Operaciones concatenando erosion y dilatacion (Nivel 2)
# =============================================================================
plt.close('all')

# Nombre de la imagen que quiero utilizar
filename = 'text_supression_example.png'

img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]
img_bin = canny(img_gray, sigma=0.2)  

# Elijo el structuring element:
se = se_circle(3)

# Apertura: Erosion seguido de dilatacion. (borra los detalles finitos)
img_apertura = apertura(img_gray, se)
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_apertura, se, 'Apertura')

# Cierre: Dilatacion seguido de erosion. Tapa agujeros
img_cierre = cierre(img_gray, se)
#plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_cierre, se, 'Cierre')

# Top hat (imagen - apertura) Retiene elementos más pequeños que se
img_top_hat = top_hat(img_gray, se)
plot_images2(img_rgb, img_gray, img_apertura, 'Apertura', img_top_hat, se, 'Top Hat (img-apertura)')

# Bottom hat (cierre - imagen)
img_bot_hat = bot_hat(img_gray, se)
plot_images2(img_rgb, img_gray, img_cierre, 'Cierre', img_bot_hat, se, 'Bottom Hat (cierre-img)')
#%%
# =============================================================================
# Operaciones concatenando cierre y apertura (Nivel 3)
# =============================================================================
plt.close('all')
# Nombre de la imagen que quiero utilizar
filename = 'retina.jpg'

img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]
img_bin = canny(img_gray, sigma=0.2)  

# Elijo el structuring element:
se = se_circle(2)

# OC Cierre y luego apertura

img_OC = OC(img_gray ,se)
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_OC, se, 'OC')

# CO Apertura y luego cierre
img_CO = CO(img_gray ,se)

plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_CO, se, 'CO')

#%%
# =============================================================================
# Implementación de suavizado y realce de contrastes a partir de op morf 
# =============================================================================
plt.close('all')
# Nombre de la imagen que quiero utilizar
filename = 'prueba.png'

img_rgb = imageio.imread(filename)/255
img_gray =  rgb2yiq(img_rgb)[:,:,0]
img_bin = canny(img_gray, sigma=0.2) 

# Suavizado
# El usuario debería ingresar el nivel
nivel = 1

img_suavizada = suavizado(img_gray, se, nivel)
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_suavizada, se, 'Suavizado')

# Realce de contraste
k = 0.3

img_realce_cont = np.clip ( img_gray - k * img_suavizada , 0, 1)
plot_images2(img_rgb, img_gray, img_bin, 'Binary image', img_realce_cont, se, 'Realce contraste')






