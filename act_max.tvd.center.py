#!/usr/bin/env python
'''
This code is to reproduce the result of "center-bias regularization method" from the paper:

Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks
A Nguyen, J Yosinski, J Clune - arXiv preprint arXiv:1602.03616, 2016

Code is a fork from https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb
The jittering technique is originally from Google Inceptionism.

Feel free to email Anh Nguyen <anh.ng8@gmail.com> if you have questions.
'''

import act_max_tvd

def main():
    parser = act_max_tvd.get_parser('center-bias regularization method', mean=False)
    args = parser.parse_args()
    # Hyperparams for AlexNet
    octaves = [
        {
            'margin': 0,
            'window': 0.3,
            'layer':'prob',
            'iter_n':190,
            'start_denoise_weight':0.001,
            'end_denoise_weight': 0.05,
            'start_step_size':11.,
            'end_step_size':11.
        },
        {
            'margin': 0,
            'window': 0.3,
            'layer':'prob',
            'scale':1.2,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 0.08,
            'start_step_size':6.,
            'end_step_size':6.
        },
        {
            'margin': 0,
            'window': 0.3,
            'layer':'fc8',
            'scale':1.2,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':1.,
            'end_step_size':1.
        },
        {
            'margin': 50,
            'window': 0.1,
            'layer':'fc8',
            'scale':1.0,
            'iter_n':30,
            'start_denoise_weight':0.1,
            'end_denoise_weight': 2,
            'start_step_size':3.,
            'end_step_size':3.
        },
        {
            'margin': 0,
            'window': 0.3,
            'layer':'fc8',
            'iter_n':10,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':6.,
            'end_step_size':3.
        }
    ]
    act_max_tvd.run(args.unit, args.filename, args.xy, args.seed, octaves)

if __name__ == '__main__':
    main()
