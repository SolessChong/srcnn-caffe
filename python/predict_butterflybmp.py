import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/solesschong/workspace/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'workspace/SR-g/python')

import caffe
from evaluation import *



# Parameters
zooming = 2

net = caffe.Net(caffe_root + 'workspace/SR-2x/SRCNN_deploy_butterfly.prototxt',
                       caffe_root + 'workspace/SR-2x/SRCNN_iter_120024')

input_dir = caffe_root + 'data/SR_NE_ANR/Cropped-g/data/'
gt_dir    = caffe_root + 'data/SR_NE_ANR/Cropped-g/label/'
project_dir = caffe_root + 'workspace/SR-2x/'

# Inputs
im_raw = cv2.imread('/home/solesschong/workspace/caffe-master/data/SR_NE_ANR/Test/butterfly_GT.bmp')
ycrcb = cv2.cvtColor(im_raw, cv2.COLOR_RGB2YCR_CB)
im_raw = ycrcb[:,:,0]
# im_raw = im_raw.reshape((im_raw.shape[0], im_raw.shape[1], 1))

im_small = cv2.resize(im_raw, (int(im_raw.shape[1]/zooming), int(im_raw.shape[0]/zooming)))

im_blur = cv2.resize(im_small, (im_raw.shape[1], im_raw.shape[0]))
im_blur = im_blur.reshape(im_blur.shape[0], im_blur.shape[1], 1)
im_raw = im_raw.reshape((im_raw.shape[0], im_raw.shape[1], 1))

# Switch input
im_input = im_raw

# plt.show()

net.set_phase_test()
net.set_mode_cpu()
#image_mean = np.load(input_dir + 'image_mean.npy')
# net.set_mean('data', image_mean)

net.set_raw_scale('data', 255.0)
out = net.forward_all(data=np.asarray([net.preprocess('data', im_input.astype(float)/255)]))

# Predict results
mat = out['recon'][0]

# Show
ycrcb = ycrcb[6:-6,6:-6,:]
im_pred = colorize(mat[0,:,:], ycrcb)
im_input = colorize(im_input[6:-6,6:-6,0], ycrcb)
im_raw = colorize(im_raw[6:-6,6:-6,0], ycrcb)

# PSNR
print PSNR(im_pred.astype(float)/255, im_raw.astype(float)/255)

f, arr = plt.subplots(1, 4)
arr[0].imshow(im_raw)
arr[0].set_title("raw")
arr[1].imshow(im_input)
arr[1].set_title("input")
arr[2].imshow(im_pred)
arr[2].set_title('predict')
diff = im_raw - im_pred
arr[3].imshow(diff)
arr[3].set_title('diff')
plt.savefig(project_dir + 'results/butterfly_blur_%dx.png' % zooming)
plt.show()