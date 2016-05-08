# Multifaceted Feature Visualization

This is the code base used to reproduce the experiments in the paper:

[Nguyen A](http://anhnguyen.me), [Yosinski J](http://yosinski.com/), [Clune J](http://jeffclune.com). ["Multifaceted Feature Visualization: Uncovering the different types of features learned by each neuron in deep neural networks"](http://www.evolvingai.org/files/nguyen_mfv_2016.pdf). arXiv:1602.03616, 2016

**If you use this software in an academic article, please cite:**

    @article{nguyen2016multifaceted,
      title={Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks},
      author={Nguyen, Anh and Yosinski, Jason and Clune, Jeff},
      journal={arXiv preprint arXiv:1602.03616},
      year={2016}
    }

For more information regarding the paper, please visit www.evolvingai.org/mfv

## Installation
* Install Caffe and its Python interface
* Some Python libraries are required and can be installed quickly via e.g. Anaconda

## Usage
* Starting optimization from mean images. Here I provided 10 mean images for two example classes: "bell pepper" and "movie theater".
```bash
    ./opt_from_mean.sh 945  # bell pepper
    ./opt_from_mean.sh 498  # movie theater
```

* Optimizing images with "center-bias reqgularization" (CBR)
```bash
   ./opt_center_bias.sh 945  
```

## Notes
* Examples are not provided, but with the code, you could also try other experiments like:
 * Running CBR from a mean image (Fig. S8 in the paper)
 * Starting from a real image (Fig. S1 in the paper)

* In the paper, we show different ways to compute the mean image (e.g. from training or val set). The code for this is not include here, but is quite straightforward to implement (see paper for more).

Feel free to create github issues. We will help you as we can.

## License

MIT License.
