import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    """
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    
    padding = ((0,0), (0,0), (0,1), (0,1))
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    #data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    #data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    data = data.transpose(0,2,3,1)
    data = data.reshape((n,n, data.shape[1],data.shape[2],data.shape[3]))
    data = data.transpose(0,2,1,3,4)
    data = data.reshape((n*data.shape[1], n*data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    """
    n = int(np.ceil(np.sqrt(data.shape[0])))
    f, arr = plt.subplots(n, n)
    for i in range(data.shape[0]):
        arr[i/n, i%n].imshow(data[i,0,:,:], cmap='gray')
    plt.show()

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier(caffe_root + 'workspace/SR-g/SRCNN_deploy.prototxt',
                       caffe_root + 'workspace/SR-g/SRCNN_iter_470000')

net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

# Visualize layers
filters = net.params['conv1'][0].data

vis_square(filters)
