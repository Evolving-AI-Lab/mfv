# Adapt the following paths for your system
model_path = "/path/to/your/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
model_definition   = './deploy_alexnet_updated.prototxt'
caffe_root = "/path/to/your/caffe/python" 
gpu = True

# Layers of AlexNet
fc_layers = ["fc6", "fc7", "fc8", "prob"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

import numpy as np
mean = np.float32([104.0, 117.0, 123.0])
