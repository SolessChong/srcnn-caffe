import numpy as np
import matplotlib.pyplot as plt
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/solesschong/workspace/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import caffe


def colorize(y, ycrcb):
    y[y>255] = 255
    
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    
    return img

# PSNR measure, from ANR's code
def PSNR(pred, gt):
    f = pred.astype(float)
    g = gt.astype(float)
    e = (f - g).flatten()
    n = len(e)
    rst = 10*np.log10(n/e.dot(e))
    
    return rst
