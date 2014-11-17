import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import cv2
import caffe

input_dir = caffe_root + 'data/SR_NE_ANR/Cropped-g/data/'
gt_dir    = caffe_root + 'data/SR_NE_ANR/Cropped-g/label/'
project_dir = caffe_root + 'workspace/SR-2x/'

net = caffe.Net(project_dir + 'SRCNN_deploy.prototxt',
                project_dir + 'SRCNN_iter_440088')

for fn in os.listdir(input_dir):
    # Inputs
    im = caffe.io.load_image(input_dir + fn, color=False)
    # plt.show()
    
    net.set_phase_test()
    net.set_mode_cpu()
    image_mean = np.load(input_dir + 'image_mean.npy')
    # net.set_mean('data', image_mean)
    
    net.set_raw_scale('data', 255.0)
    out = net.forward_all(data=np.asarray([net.preprocess('data', im)]))

    # Predict results
    mat = out['recon'][0]
    
    # Ground truth
    gt = caffe.io.load_image(gt_dir + fn, 1)
    
    # Show
    """
    mat = cv2.equalizeHist(mat[0,:,:].astype('uint8'))
    im = cv2.equalizeHist(im[6:-6,6:-6,0].astype('uint8'))
    gt = cv2.equalizeHist((gt[:,:,0]*255).astype('uint8'))
    """
    mat = (mat[0,:,:]).astype('uint8')
    im = im[6:-6,6:-6,0].astype('uint8')
    gt = (gt[:,:,0]*255).astype('uint8')
    diff = gt - mat
    # Print Euclidean_loss_layer output
    print (np.dot(diff.flatten(), diff.flatten())).astype(float) / (mat.shape[0]*mat.shape[1]) / 2

    f, axarr = plt.subplots(2, 4, figsize=(15,8))
    axarr[0, 0].imshow(im, cmap='gray')
    axarr[0, 0].set_title('input')
    axarr[0, 1].imshow(mat, cmap='gray')
    axarr[0, 1].set_title('predict')
    axarr[0, 2].imshow(gt, cmap='gray')
    axarr[0, 2].set_title('groundtruth')
    axarr[1, 0].hist(im.flatten(), 256, range=(0.0,255.0), fc='k', ec='k')
    axarr[1, 1].hist(mat.flatten(), 256, range=(0, 255))
    axarr[1, 2].hist(gt.flatten(), 256, range=(0, 255))
    
    axarr[0, 3].imshow(diff)
    axarr[0, 3].set_title('diff')
    axarr[1, 3].hist((diff).flatten(), 100)
    
    plt.savefig(project_dir + 'results/' + fn[0:-4] + '.png')
    plt.show()