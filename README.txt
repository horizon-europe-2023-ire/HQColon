# HQColon

HQColon is a tool for high-resolution segmentation of the colon from CT Colonography scans. This model aims to segment both air filled and fluid pockets to cover the whole volume of the colon.

![HQColon Segmentation examples!](/assets/segmentation-examples.png "HQColon Segmentation examples")

## Citation

Please cite the [following paper](https://arxiv.org/abs/2502.21183) when using HQColon:

<html>
    <head>
        
    @misc{finocchiaro2025hqcolonhybridinteractivemachine,
      title={HQColon: A Hybrid Interactive Machine Learning Pipeline for High Quality Colon Labeling and Segmentation}, 
      author={Martina Finocchiaro and Ronja Stern and Abraham George Smith and Jens Petersen and Kenny Erleben and Melanie Ganz},
      year={2025},
      eprint={2502.21183},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.21183}, 
    }
</html>

## Installation

### Operating System

HQColon has been developed on Linux (Ubuntu).

Please check hardware requirements of nnUNet for training: [Hardware Requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

### Installation of nnUNet

First set up a CUDA:

1. Load the cuda modul you want to use e.g. module load cuda/11.8
2. Install Pytorch according to what cuda module you loaded: [pytorch]https://pytorch.org/get-started/locally/

Set up a virtual environment similar to the one below:

``` Console
conda create -n HQColon python=3.9 -y
conda activate HQColon      
conda install numpy     
conda install scipy     
conda install matplotlib
conda install paraview
```

For use of the out-of-the-box segmentation algorithm install nnunetv2:

```
pip install nnunetv2
```

make sure dependecies like pip install blosc2 are installed

This also directly installs SimpleITK. Installing SimpleITK before installing nnunetv2 might lead to problems. Use the following command to install SimpleITK separately:

```
conda install -c conda-forge simpleitk
```

For more information check the documentation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).


Install further dependencies:



### Installation of visualization tools

To use nnUNet the data needs to be saved in a very specific hierarchy and naming convention. Check instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

Last install ITK-Snap or 3D Slicer as visualization tools.
* http://www.itksnap.org/pmwiki/pmwiki.php
* https://www.slicer.org/

## Example Usage



## Acknowledgments

Created by members of the [Image Section of the University of Copenhagen](https://di.ku.dk/english/research/image/) as part of the [Horizon Europe 2023 Intelligent Robotic Endoscopes (IRE)](https://ire4health.eu/) project.

For any questions and feedback please contact: ???
