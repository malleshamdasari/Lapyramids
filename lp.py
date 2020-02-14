import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_pass_filter(img_in):
	
   dft = np.fft.fft2(img_in)
   dft_shift = np.fft.fftshift(dft)
   rows, cols = img_in.shape
   crow,ccol = rows/2 , cols/2
   mask = np.zeros((rows, cols), np.uint8)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
   fshift = dft_shift*mask
   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)
   img_out = np.abs(img_back)

   return True, img_out

def high_pass_filter(img_in):

   dft = np.fft.fft2(img_in)
   dft_shift = np.fft.fftshift(dft) 
   rows, cols = img_in.shape
   crow,ccol = rows/2 , cols/2
   dft_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
   f_ishift = np.fft.ifftshift(dft_shift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)
   img_out = np.abs(img_back)

   return True, img_out
   
def deconvolution(img_in):
   
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
  
   def ft(im, newsize=None):
      dft = np.fft.fft2(np.float32(im),newsize)
      return np.fft.fftshift(dft)

   def ift(shift):
      f_ishift = np.fft.ifftshift(shift)
      img_back = np.fft.ifft2(f_ishift)
      return np.abs(img_back)

   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
   gkf = ft(gk, (img_in.shape[0],img_in.shape[1])) # so we can multiple easily 
   imconvf = np.divide(imf, gkf)
   deconv_img = ift(imconvf)
   img_out = deconv_img*255

   return True, img_out

def laplacian_pyramids(img_in1):
   levels = 3
   A = img_in1[:,:img_in1.shape[0]]
 
   img_out = img_in1
  
   # generate Gaussian pyramid for A
   G = A.copy()
   gpA = [G]
   for i in xrange(levels):
       G = cv2.pyrDown(G)
       gpA.append(G)
       cv2.imwrite('g_'+str(i)+'.jpg', G)
   # generate Laplacian Pyramid for A
   lpA = [gpA[levels]]
   for i in xrange(levels-1,0,-1):
       GE = cv2.pyrUp(gpA[i])
       L = cv2.subtract(gpA[i-1],GE)
       lpA.append(L)
       cv2.imwrite('l_'+str(i)+'.jpg', L)

input_image1 = cv2.imread('input1.jpg', cv2.IMREAD_COLOR);
laplacian_pyramids(input_image1)
