import numpy
import os
import math
import mahotas
import scipy as sc
from scipy import signal
from scipy import ndimage
import scipy.ndimage
import sys
import skimage.morphology as sm
import pylab as pl
from scipy import signal



    #//////////////////////Pre-processing\\\\\\\\\\\\\\\\\\\\\


def normalize_mask(mask):
    mk = mask.copy()
    mk = mk.astype(numpy.uint8)
    for i in range(mask.shape[0]):
        for j in range (mask.shape[1]):
        	if (mask[i,j] > 0):mk[i,j] = 1
        	else: mk[i,j] = 0
    return mk

def padding(image, pading):
    pad = round(pading / 2)
    M = image.shape[0]
    N = image.shape[1] 
    im_pad = numpy.zeros(((M + pading), (N + pading)))
    im_pad[pad: -pad, pad: -pad] = image
    
    return im_pad

def preprocessing(image, mask, pad):
    # Tranforms the mask into 0 and 1, so it will not change the image values
    mask = normalize_mask(mask) 
    c_verde = image[:,:,1] # Select the green channel 
    inv = 255 - c_verde # Inverts the image
    # Multiplies for the mask to eliminate background noise
    img_inv = inv * mask 
    le_img = padding(img_inv,pad)
    
    return le_img

def bordas(image, mask, l, pad):
    mask = padding(mask,pad)
    image_p = image.copy()
    mdp = int(l / 2)
    
    for i in range(image.shape[0] -1):
        for j in range(image.shape[1] -1):
            counter = 0
            n_p = 0
            if ((mask[i, j-1] == 0 and mask[i,j + 1] > 0)  
            	or (mask[i-1, j] == 0 and mask[i,j] > 0) 
            	or (mask[i,j + 1] == 0 and mask[i,j] > 0) 
            	or (mask[i + 1, j] == 0 and mask[i,j] > 0)):

                for x in range(-l, l + 1):
                    for y in range (-l, l + 1):
                        
                        if mask[i + x, j + y + 1] > 0:
                            n_p = n_p + 1
                            counter = counter + image_p[i + x, j + y]
                media = counter / n_p
                
                for x in range(-l, l + 1):
                    for y in range (-l, l + 1):
                        if (mask[i + x, j + y] == 0 or (mask[i+x-1,j+y]==0 and mask[ i + x , j + y] >0 )):
                            image_p[i + x, j + y] = media
    return image_p

#///////////////////////Kernels Creation\\\\\\\\\\\\\\\\\\\\\

def rotation(kernel):  
    mask = kernel.copy()
    l = kernel.shape[0]
    c = kernel.shape[1]
    rotated_kernel = numpy.zeros((l,c))
   
    for i in range(l):
        for j in range(c):
            # Mirrors the kernel in the y axis
            rotated_kernel[i,j] = mask[i, c-j-1]
    return rotated_kernel


def draw_line(p0, p1, kernel, l):
    
    image = kernel.copy()
    line = get_line(p0,p1,l)
    print('Creating a kernel...')
   # Substitute the points of the kernel
    for n in range(line.shape[0]): 
        i = line[n,0]
        j = line[n,1]    
        image[i,j] = 1 
    
    return image 


def get_line(p0, p1, l):
    
    x0 = int(p0[1]-1) # Coluns correspond to the y axis
    y0 = int(p0[0]-1) # Lines correspond to the x axis
    x1 = int(p1[1]-1)
    y1 = int(p1[0]-1)
    dif_x = x1-x0
    dif_y = y1-y0
    line = numpy.zeros((l,2),dtype = int)
    index = 0
    if abs(dif_y) > abs(dif_x): 
        for y in range (y0 + 1): # Run the lines from top to bottom
        # Calculates the value of the corresponding colun
            x = round((dif_x / dif_y) * (y-y1) + x1) 
            # Adds the point to the matrix which belongs to the line 
            line[index,:] = [y,x]  
            index += 1 
    else: 
        for x in range (x1 + 1): # Run the coluns from right to left
            y = round((dif_y / dif_x) * (x - x0) + y0) 
            # Calculate the value of the corresponding colun
            line[index,:] = [y,x] 
            # adds the point which correspond to th lines
            index += 1 
            
    return line


def kernel (theta, l):
    kernel_c = numpy.zeros((l,l))
    midpoint = (l-1) / 2
    
    if theta == 0:   # Line
        kernel_c[int(midpoint),:] = 1
        
    elif theta == (math.pi / 2):  # Colun
        kernel_c[:, int(midpoint)] = 1
    else:
        if theta <= (math.pi / 4): # Oposite leg calculation
            x0 = - midpoint 
            y0 = round(x0 * (math.sin(theta) / math.cos(theta))) 
            x1 = midpoint
            y1 = round(x1 * (math.sin(theta) / math.cos(theta))) 
        else: # Adjacent leg calculation
            y0 =-int(midpoint)
            x0 = round(y0 * (math.cos(theta) / math.sin(theta))) 
            y1 = int(midpoint)
            x1 = round(y1 * (math.cos(theta) / math.sin(theta))) 
        p0 = [midpoint-y0+1, midpoint+x0+1] # Superior rigth value of kernel
        p1 = [midpoint-y1+1, midpoint+x1+1] # Inferior left value of kernel
        kernel_c = draw_line(p0,p1,kernel_c,l)

    return kernel_c


def get_line_detector(angulo, l):
    theta = angulo * (math.pi / 180)
    if theta > (math.pi / 2):
        s_line = kernel((math.pi-theta),l)
        line = rotation(s_line)
    else:
        line = kernel(theta,l)   	
    return numpy.matrix(line)


#//////////////////Get S and S0\\\\\\\\\\\\\\\\\\\\\\\\\

def average_level(image, mask, l, pad):  
    mdp = int(l / 2)
    mask_x = (1 / (l * l)) * numpy.ones((l,l))
    avg = signal.correlate2d(image, mask_x, mode = 'same') 
    # Gets the mean value from the neigbhourd lxl
    Feature_n = numpy.zeros((image.shape[0],image.shape[1])) 
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0: 
                Feature_n[i,j] = avg[i,j]
    return avg


def get_line_detector_ortogonal(angulo, l):
    midpoint = int((l-1) / 2)
    line_o = numpy.zeros((l, l))
    
    if (angulo <= 15 or angulo > 150):
        line_o[midpoint - 1:midpoint + 2,midpoint] = 1 # 90º ortogonal angle
    elif (angulo > 60 and angulo <= 90):
        line_o[midpoint, midpoint - 1:midpoint + 2] = 1   # 0º ortogonal angle
    elif (angulo > 15 and angulo <= 60):
        for i in range(-1,2):
            line_o[midpoint + i,midpoint + i] = 1 # 135º ortogonal angle
    elif (angulo > 90 and angulo <= 150):
        for i in range(-1,2):
            line_o[midpoint + i,midpoint - i] = 1 # 45º ortogonal angle
    line = line_o / numpy.sum(line_o)
    return numpy.array(line)


def get_value_l(image, l):
    n = image.shape[0]
    m = image.shape[1]
    line_l = numpy.zeros((n,m))
    img = numpy.zeros((n,m))

    for angulo in range(0, 180, 15):
        line_k = get_line_detector(angulo, 15)
        line = line_k / numpy.sum(line_k)
        line_ls = signal.correlate2d(image, line, mode = 'same')
        # Compares the new value line with the before one, with new rotation 
        # The biggest value is kept
        line_l = numpy.maximum(line_l,line_ls)
        # Return of matrix with biggest values
    return line_l


def get_value_s(image, mask, l, pad):
    img = bordas(preprocessing(image, mask, pad), mask, l, pad)
    m = normalize_mask(padding(mask, pad))
    feature_l = get_value_l(img, l)
    feature_n = average_level(img, m, l, pad)
    feature_s = (feature_l - feature_n) * m
    return feature_s


def get_value_s0(image, mask, l, pad):
    img = bordas(preprocessing(image, mask, pad), mask, l, pad)
    n = img.shape[0]
    m = img.shape[1]
    line_l = numpy.zeros((n,m))
    line_s = numpy.zeros((n,m))
    mdp = int(l / 2)
    m_n = normalize_mask(padding(mask,pad))
    avg = average_level(img,m_n,l,pad)
    x = 0
    print('starting s0...')
    for angulo in range(0,180,15):
        line_k = get_line_detector(angulo,15)
        line = line_k / numpy.sum(line_k)
        line_ls = signal.correlate2d(img, line, mode = 'same')
        # Compares the new value line with the before one, with new rotation 
        # The biggest value is kept
        line_l = numpy.maximum(line_l,line_ls) 
        # Return of matrix with biggest values
        
        for i in range(n):
            for j in range(m):
                if m_n[i,j] > 0:
                    if line_l[i,j] == line_ls[i,j]:
                        x = x + 1
                        kernel_l0 = get_line_detector_ortogonal(angulo,l)
                        a = numpy.sum(kernel_l0 * (img[i - mdp:i + mdp + 1,j - mdp:
                        	j + mdp + 1]))

                        line_s[i,j] = a
    image_s0 = (line_s - avg) * m_n
    return image_s0

#//////////////////////////Pos-Processing\\\\\\\\\\\\\\\\\\\\\\\\\\

def threshold(image):
    img_th = image.copy()
    for i in range(image.shape[0]):
        for j in range (image.shape[1]):
            if (img_th[i,j] > 5):
                img_th[i,j] = 255
            else: 
                img_th[i,j] = 0
    return img_th

#/////////////////////////////Metrics\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def metrics(image, maskv, mask):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if mask[i,j] > 0:
                if maskv[i,j] == 255: # Pixel its a vein
                    if image[i,j] == 255: # Pixel was segmented as vein
                        TP += 1
                    elif image[i,j] == 0: # Pixel was segmented as background
                        FN += 1
                if maskv[i,j] == 0: # Pixel its background
                    if image[i,j] == 255: # Pixel was segmented as vein
                        FP += 1 
                    if image[i,j] == 0: # Pixel was segmented as background
                        TN += 1
    accuracy = round((((TP + TN) / (TP + TN + FP + FN)) * 100),2)
    sensitive = round(((TP / (TP + FN)) * 100),2)
    precision = round(((TP / (TN + FP)) * 100),2)
    specificity = round(((TN / (TN + FP)) * 100),2)
    metric_txt = open("metrics.txt", "w+")
    metric_txt.write("accuracy %f \r\n" % (accuracy))
    metric_txt.write("sensitive %f \r\n" % (sensitive)) 
    metric_txt.write("precision %f \r\n" % (precision)) 
    metric_txt.write("specificity %f \r\n" % (specificity))     
    metric_txt.close()
    return (accuracy,sensitive, precision, specificity)

