# Innovation Engine

This is the code base used to reproduce the experiments in the paper:

[Nguyen A](http://anhnguyen.me), [Yosinski J](http://yosinski.com/), [Clune J](http://jeffclune.com). ["Innovation Engines: Automated creativity and improved stochastic optimization via Deep Learning"](http://www.evolvingai.org/files/InnovationEngine_gecco15.pdf). Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '15), 2015.

**If you use this software in an academic article, please cite:**

    @inproceedings{nguyen2015innovation,
      title={Innovation Engines: Automated Creativity and Improved Stochastic Optimization via Deep Learning},
      author={Nguyen, Anh and Yosinski, Jason and Clune, Jeff},
      booktitle={Genetic and Evolutionary Computation Conference (GECCO), 2015 IEEE Conference on},
      year={2015}
    }

For more information regarding the paper, please visit www.evolvingai.org/InnovationEngine

## Notes

This code base is a fork from the [initial project] (https://github.com/Evolving-AI-Lab/fooling).

* It has support for the latest Caffe.
* Experiments are in [here](https://github.com/Evolving-AI-Lab/innovation-engine/tree/master/sferes/exp/images/x)
  * [Novelty Search](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_novelty_images_imagenet.cpp)
  * [MAP-Elites](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_map_elites_images.cpp)
    * [CPPN without Sine waves](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_map_elites_images_no_sine.cpp) experiment shows to create a lot of *recognizable* images.
  * [Single-class EA](https://github.com/Evolving-AI-Lab/innovation-engine/blob/master/sferes/exp/images/x/gecco15/dl_rank_simple_images.cpp)
  


## Requirements and Installation

To start off, please refer to the documentation in the initial [project] (https://github.com/Evolving-AI-Lab/fooling)

Feel free to create github issues. We will help you as we can.

## License

MIT License.
