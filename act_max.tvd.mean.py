#!/usr/bin/env python
'''
This code is to reproduce the result of "mean initialization method" from the paper:

Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks
A Nguyen, J Yosinski, J Clune - arXiv preprint arXiv:1602.03616, 2016

Code is a fork from https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb
The jittering technique is originally from Google Inceptionism.

Feel free to email Anh Nguyen <anh.ng8@gmail.com> if you have questions.
'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from skimage.restoration import denoise_tv_bregman

pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

# Layers of AlexNet
fc_layers = ["fc6", "fc7", "fc8"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

mean = np.float32([104.0, 117.0, 123.0])

if settings.gpu:
  caffe.set_mode_gpu()

net = caffe.Classifier(settings.model_definition, settings.model_path,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, xy, step_size=1.5, end='fc8', clip=True, unit=None, denoise_weight=0.1, margin=0, w=224, h=224):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    
    dst = net.blobs[end]
    acts = net.forward(end=end)

    if end in fc_layers:
        fc = acts[end][0]
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        obj_act = fc[unit]
        # print "unit: %s [%.2f], obj: %s [%.2f]" % (best_unit, fc[best_unit], unit, obj_act)
    
    one_hot = np.zeros_like(dst.data)
    
    if end in fc_layers:
      one_hot.flat[unit] = 1.
    elif end in conv_layers:
      one_hot[:, unit, xy, xy] = 1.
    else:
      raise Exception("Invalid layer type!")

    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]

    # Mask out gradient to limit the drawing region
    if margin != 0:
      mask = np.zeros_like(g)

      for dx in range(0 + margin, w - margin):
        for dy in range(0 + margin, h - margin):
          mask[:, dx, dy] = 1
      g *= mask

    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 

    # Run a separate TV denoising process on the resultant image
    asimg = deprocess( net, src.data[0] ).astype(np.float64)
    denoised = denoise_tv_bregman(asimg, weight=denoise_weight, max_iter=100, eps=1e-3)

    src.data[0] = preprocess( net, denoised )
    
    # reset objective for next step
    dst.diff.fill(0.)

    return best_unit, best_act, obj_act

def save_image(output_folder, filename, unit, img):
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(unit).zfill(4))
    PIL.Image.fromarray(np.uint8(img)).save(path)

    return path


def max_activation(net, layer, xy, base_img, octaves, random_crop=True, debug=True, unit=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "start optimizing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    
    iter = 0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
        
        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x * o['window'], 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y * o['window'], 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            denoise_weight = o['start_denoise_weight'] - (o['start_denoise_weight'] - (o['end_denoise_weight']) * i) / o['iter_n']

            best_unit, best_act, obj_act = make_step(net, xy, end=layer, clip=clip, unit=unit, 
                      step_size=step_size, denoise_weight=denoise_weight, margin=o['margin'], w=w, h=h)

            print "iter: %s\t unit: %s [%.2f]\t obj: %s [%.2f]" % (iter, best_unit, best_act, unit, obj_act)

            if debug:
                img = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    img = img*(255.0/np.percentile(img, 99.98))
                if i % 1 == 0:
                    save_image(".", "iter_%s" % str(iter).zfill(4), unit, img)
           
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]

            iter += 1   # Increase iter

        print "octave %d image:" % e
            
    # returning the resulting image
    return deprocess(net, image)



def main():
    # Hyperparams for AlexNet
    octaves = [
        {
            'margin': 0,
            'window': 0.3, 
            'iter_n':190,
            'start_denoise_weight':0.4,
            'end_denoise_weight': 0.4,
            'start_step_size':1,
            'end_step_size':2
        },
        {
            'margin': 0,
            'window': 0.3,
            'scale':1.2,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':1.,
            'end_step_size':1.
        },
        {
            'margin': 50,
            'window': 0.2,
            'scale':1.0,
            'iter_n':20,
            'start_denoise_weight':0.1,
            'end_denoise_weight': 2,
            'start_step_size':2.,
            'end_step_size':2.
        },
        {
            'margin': 0,
            'window': 0.3, 
            'iter_n':10,
            'start_denoise_weight':0.4,
            'end_denoise_weight': 2,
            'start_step_size':2.0,
            'end_step_size':2.0
        }
    ]

    # get original input size of network
    original_w = net.blobs['data'].width
    original_h = net.blobs['data'].height

    # which imagenet class to visualize
    unit = int(sys.argv[1]) # unit
    filename = str(sys.argv[2])
    layer = str(sys.argv[3])    # layer
    xy = int(sys.argv[4])       # spatial position
    seed = int(sys.argv[5])     # random seed

    print "----------"
    print "unit: %s \tfilename: %s\tlayer: %s\txy: %s\tseed: %s" % (unit, filename, layer, xy, seed)

    # Set random seed
    np.random.seed(seed)

    # the background color of the initial image
    background_color = np.float32([175.0, 175.0, 175.0])

    # generate initial random image
    start_image = np.random.normal(background_color, 8, (original_w, original_h, 3))
    output_folder = '.' # Current folder

    if len(sys.argv) >= 7:
        image_path = str(sys.argv[6])
        start_image = np.float32(PIL.Image.open(image_path))

        # output folder
        if len(sys.argv) == 8:
          output_folder = str(sys.argv[7])

        print "Loaded start image: %s %s" % (image_path, start_image.shape)
        print "Output: %s" % output_folder
        print "-----------"

    # generate class visualization via octavewise gradient ascent
    output_image = max_activation(net, layer, xy, start_image, octaves, unit=unit, 
                     random_crop=True, debug=False)

    # save image
    path = save_image(output_folder, filename, unit, output_image)
    print "Saved to %s" % path

if __name__ == '__main__':
    main()
