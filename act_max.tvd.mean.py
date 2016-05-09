#!/usr/bin/env python
'''
This code is to reproduce the result of "mean initialization method" from the paper:

Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks
A Nguyen, J Yosinski, J Clune - arXiv preprint arXiv:1602.03616, 2016

Code is a fork from https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb
The jittering technique is originally from Google Inceptionism.

Feel free to email Anh Nguyen <anh.ng8@gmail.com> if you have questions.
'''

import act_max_tvd

def main():
    parser = act_max_tvd.get_parser('mean initialization method')
    args = parser.parse_args()
    layer = args.layer
    # Hyperparams for AlexNet
    octaves = [
        {
            'margin': 0,
            'window': 0.3,
            'layer': layer,
            'iter_n':190,
            'start_denoise_weight':0.4,
            'end_denoise_weight': 0.4,
            'start_step_size':1,
            'end_step_size':2
        },
        {
            'margin': 0,
            'window': 0.3,
            'layer': layer,
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
            'layer': layer,
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
            'layer': layer,
            'iter_n':10,
            'start_denoise_weight':0.4,
            'end_denoise_weight': 2,
            'start_step_size':2.0,
            'end_step_size':2.0
        }
    ]
    act_max_tvd.run(args.unit, args.filename, args.xy, args.seed, octaves,
                    output_folder=args.output_folder,
                    image_path=args.start_image)

if __name__ == '__main__':
    main()
