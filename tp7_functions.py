# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:34:54 2022

@author: Study
"""
import numpy as np
from scipy.signal import convolve2d

def downsampling1(img, step):
    '''inputs: img: imagen que desea downsamplear, step: cada cuanto debe tomarse 1 pixel'''
    rows = len(img[:,0])
    cols = len(img[0,:])
    imgdown = np.zeros((rows//step, cols//step))
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            imgdown[i//step,j//step] = img[i,j]
    return imgdown

def downsampling2(img, step):
    '''inputs: img: imagen que desea downsamplear, step: orden del downsampleo'''
    rows = len(img[:,0])
    cols = len(img[0,:])
    imgdown = np.zeros((rows//step, cols//step))
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            imgdown[i//step,j//step] = (img[i,j] + img[i+1,j] + img[i,j+1] + img[i+1,j+1]) / 4
    return imgdown

def gaussiano(N,std) :
    matriz = []
    fila = np.linspace(-(N - 1) / 2., (N - 1) / 2., N)
    gauss = np.exp(-0.5 * np.square(fila) / np.square(std))
        
    matriz = np.outer(gauss,np.transpose(gauss))
    matriz = matriz / np.sum(matriz)
    return matriz

def upsampling2x2(img):
    '''realiza un upsampling repitiendo pixeles en una grilla de 2x2'''
    rows = len(img[:,0])
    cols = len(img[0,:])
    imgup = np.zeros((rows*2, cols*2))
    for i in range(0, rows*2):
        for j in range(0, cols*2):
            imgup[i:i+1,j:j+1] = img[i//2,j//2]
    return imgup

def up_bilineal(img):
    ''' realizo un upsampling con interpolación bilineal'''
    rows = len(img[:,0])
    cols = len(img[0,:])
    imgaux = np.zeros((rows*2, cols*2))
    for i in range(0, rows*2, 2):
        for j in range(0, cols*2, 2):
            imgaux[i,j] = img[i//2,j//2]
    
    vector = [1/2, 1, 1/2]
    kernel = np.outer(vector,np.transpose(vector))  
    
    imgup = convolve2d(imgaux, kernel, 'same')
    return imgup

def up_bicubic(img):
    ''' realizo un upsampling con interpolación bicúbica'''
    rows = len(img[:,0])
    cols = len(img[0,:])
    imgaux = np.zeros((rows*2, cols*2))
    for i in range(0, rows*2, 2):
        for j in range(0, cols*2, 2):
            imgaux[i,j] = img[i//2,j//2]
    
    vector = [-1/8, 0, 5/8, 1, 5/8, 0, -1/8]
    kernel = np.outer(vector,np.transpose(vector))  
    
    imgup = convolve2d(imgaux, kernel, 'same')
    return imgup

def fft_resampling(img, new_shape):
    
    X = np.fft.fftshift(np.fft.fft2(img))
    
    rows, cols = img.shape
        
    new_rows = new_shape[0]
    new_cols = new_shape[1]

    # Upsampling
    # Para hacer upsampling le tengo que añadir ceros alrededor de la fft, así, al hacer
    # la ifft recupero una imagen más grande, pero sin nueva información
    # Downsampling
    # Para hacer downsampling tengo que recortar la fft

    if new_rows > rows:
        rows_pad = (new_rows - rows)//2
        X = np.pad(X, [(rows_pad,rows_pad),(0,0)], mode='constant', constant_values=0)
    elif new_rows < rows:
        X = X[(rows-new_rows)//2 : rows - (rows-new_rows)//2, :]
    else:
        print('new_rows = rows')
        
    if new_cols > cols:
        cols_pad = int((new_cols - cols)//2)
        X = np.pad(X, [(0,0), (cols_pad,cols_pad)], mode='constant', constant_values=0)
    elif new_cols < cols:
        X = X[:, (cols-new_cols)//2 : cols - (cols-new_cols)//2]
    else:
        print('new_cols = cols')
    
    x = np.abs((np.fft.ifft2(X)))
    
    return x

def cuantiz_uniforme(img, levels):
    img2 = np.round(img*(levels-1))/(levels-1)
    return img2

def cuantiz_dithering_scanline(img, levels):
    rows, cols = img.shape
    img2 = np.zeros(img.shape)
    error = 0
    for i in range(rows):
        error = 0
        for j in range(cols):
            img2[i,j] = np.round((img[i,j] + error) * (levels-1))  / (levels) 
            error = (img[i,j] - img2[i,j])
    return img2