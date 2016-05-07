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

## Notes

This code base is a fork from the [initial project] (https://github.com/Evolving-AI-Lab/fooling).

* It has support for the latest Caffe.
* Experiments are in [here](https://github.com/Evolving-AI-Lab/innovation-engine/tree/master/sferes/exp/images/x)
  * [Novelty Search](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_novelty_images_imagenet.cpp)
  * [MAP-Elites](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_map_elites_images.cpp)
    * [CPPN without Sine waves](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_map_elites_images_no_sine.cpp) experiment shows to create a lot of *recognizable* images.
  * [Single-class EA](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_rank_simple_images.cpp)
  


## Requirements and Installation

Feel free to create github issues. We will help you as we can.

## License

MIT License.
