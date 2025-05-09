# HQColon

HQColon is a tool for high-resolution segmentation of the colon from CT Colonography scans. This model aims to segment both air filled and fluid pockets to cover the whole volume of the colon.

In this repository you will find code to either train your own nnunetv2 model using your own data or to use one of our pre-trained models to predict segmentations for your CT Colonography scans. In both cases you will need to install nnunetv2 and other necessary libraries listed below.

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

Please check hardware requirements of nnUNet before installing it: [Hardware Requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

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


### Install MetricsReload

In order to use the evaluation script MetricsRedload has to be installed.

1. Check out the website: https://github.com/Project-MONAI/MetricsReloaded

2. Clone the GitHub repository

3. On top of the evaluation (evaluation.py) script add the path to your local MetricsReload repository:
```
import sys
sys.path.append('/path/to/MetricsReloaded')
```

### Installation of visualization tools

Install ITK-Snap or 3D Slicer as visualization tools.
* [http://www.itksnap.org/pmwiki/pmwiki.php](http://www.itksnap.org/pmwiki/pmwiki.php)
* [https://www.slicer.org/](https://www.slicer.org/)

## Example Usage

You can either train your own hqcolon nnunet or use existing checkpoints to predict the segmentation for your colons. For both tasks it is crucial to have your data in the expected format.

To use nnUNet the data needs to be saved in a very specific hierarchy and naming convention. Check instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).


### Train your own network

Go to the [IRE OSF project](https://osf.io/8tkpm/) and download all zip files. Unzip all folders locally and create a new folder data in this repository where all unzipped folders should be move.

Further download the corresponding dicom images from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ct-colonography/#citations). Use the meta-data.json file on OSF to know which series to download from the TCIA. The field "InstanceUID" indicates the unique TCIA scan.
Save all .mha files in the folder data/CTC Scans.

Now you should have CTC Scans, Segmentation Air, Segmentation Air and Fluid and Masks TotalSegmentator and the meta-data.json in your data folder.

You can now specify what dataset configuration you want to use to train your nnunet. To do that open the run_nnunet.sh file in the hqcolon directory.
- Alter the name of your dataset and model in line 7. Remember that the name needs to have format DatasetXXX_name where XXX are 3 unique digits (you have only one dataset / model at the same time with the same unique digits).
- Once you updated the dataset name you need to put your unique 3 digits in line 8 as dataset_numnber. Those two numbers must be the same for the pipeline to work smoothly.
- In line 33 define whether you want to use flags --masked an / or --fluid. If masked is set then the input images will be masked using the dilated masks in data/Masks TotalSegmentatr. If flag fluid is set, the model will segment both air and fluid.

This script will first split your dataset into train and test splits (adapt file dataset_split_creator.py to change splitting logic). Then it will create the dataset by copying and renaming the needed files from the data folder. Last it will prepare the dataset (nnunetv2 logic) and train and validate after. In a final step the model predicts segmentations for our test set and evaluates those predictions using the ground truth labels.


### Predict segmentations using a pre-trained model

Download the model checkpoints from ....
Find a datasetname in nnunet formal like: DatasetXXX_name where XXX are exactly 3 digits and name is any name of your choice. This dataset name and the belonging digit is crucial to identify the model that should be loaded. This name has nothing to do with the data you want to predict segmentations for. It is part of the nnunet internal logic. See more here: [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).
Create new folders: nnunet_results/DatasetXXX_name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/.
Unzip your model checkpoints and move them into the folder above.

Adapt file predict.sh:
- Alter the name of your dataset and model in line 7. Remember that the name needs to have format DatasetXXX_name where XXX are 3 unique digits (you have only one dataset / model at the same time with the same unique digits).
- Once you updated the dataset name you need to put your unique 3 digits in line 8 as dataset_numnber. Those two numbers must be the same for the pipeline to work smoothly.
- Alter in line 10 input_path_to_test_dir: the path where your data is stored (the CTC scans for with you would like to get some predictions). By default it is set to the paths used in HQColon.
- Alter in line 11 output_path_to_predictions: the path where you want to save predictions. By default it is set to the paths used in HQColon.
- Line 23 adapt checkpoint-best.pth to your model filename which is saved in your nnunet_results/DatasetXXX_name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ folder.

## Acknowledgments

Created by members of the [Image Section of the University of Copenhagen](https://di.ku.dk/english/research/image/) as part of the [Horizon Europe 2023 Intelligent Robotic Endoscopes (IRE)](https://ire4health.eu/) project.

For any questions and feedback please contact: martina.finocchiaro@di.ku.dk
