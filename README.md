# HQColon

**HQColon** is a high-resolution colon segmentation tool for CT Colonography scans. It can segment both air-filled or air- and fluid-filled parts of the colon, enabling accurate volume extraction for research and clinical applications.

This repository provides:
- Original HQColon results
- Code to train your own segmentation model using `nnUNetV2`
- Tools to apply pre-trained models to your data
- Instructions for setting up your environment and visualizing results

![HQColon Segmentation examples!](/assets/segmentation-examples.png "HQColon Segmentation examples")
On top is a visual representation of our results using an nnunetv2 model trained on the 105_regiongrowing_qc_fluid dataset. A comparison with Totalsegmentator is included in this figure.

---

## Quick Links
- [Citation](#citation)
- [Installation](#installation)
- [Example Usage](#example-usage)
  - [Train Your Own Network](#train-your-own-network)
  - [Predict with Pre-Trained Model](#predict-segmentations-using-a-pre-trained-model)
- [Original HQColon results](#original-hqcolon-results)
- [Acknowledgments](#acknowledgments)

---

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

---

## Installation

### Operating System

HQColon has been developed on Linux (Ubuntu).

Please check hardware requirements of nnUNet before installing it: [Hardware Requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

### Installation of nnUNet

First set up a CUDA and PYTorch:

1. Load a compatible CUDA version: Example: module load cuda/11.8
2. Install PyTorch based on your CUDA version: [pytorch]https://pytorch.org/get-started/locally/

Create a Virtual Environment

``` Console
conda create -n HQColon python=3.9 -y
conda activate HQColon      
conda install numpy scipy matplotlib paraview
```

Install nnUNetV2

```
pip install nnunetv2
```

Ensure required dependencies (e.g., blosc2) are installed:

```
pip install blosc2
```

Install SimpleITK

Install only after nnunetv2:

```
conda install -c conda-forge simpleitk
```

For more information check the documentation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).


### Install MetricsReload

Used for model evaluation.

1. Check out the website: https://github.com/Project-MONAI/MetricsReloaded

2. Clone it locally.

3. Add its path in your evaluation.py:
```
import sys
sys.path.append('/path/to/MetricsReloaded')
```

### Installation of visualization tools

Install ITK-Snap or 3D Slicer as visualization tools. This step is not mandatory but a help to verify segmentation results.
* [http://www.itksnap.org/pmwiki/pmwiki.php](http://www.itksnap.org/pmwiki/pmwiki.php)
* [https://www.slicer.org/](https://www.slicer.org/)

---

## Example Usage

You can either train your own hqcolon nnunet or use existing checkpoints to predict the segmentation for your colons. For both tasks it is crucial to have your data in the expected format. To use nnUNet the data must be stored in a very specific hierarchy and naming convention. Check instructions:[here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).


### Train your own network

In this example, we will use **435 colon segmentations** from the HQColon paper as training and test data.

#### Step 1: Download Training Data

- Visit the [IRE OSF Project](https://osf.io/8tkpm/) and **download all zip files**.
- Unzip all folders locally.
- Create a folder named `data/` inside this repository and move all unzipped folders into it.

Your `data/` folder should eventually include:

- `CTC Scans/`
- `Segmentation Air/`
- `Segmentation Air and Fluid/`
- `Masks TotalSegmentator/`
- `meta-data.json`

#### Step 2: Download CT Scans

- Go to the [CT Colonography dataset on TCIA](https://www.cancerimagingarchive.net/collection/ct-colonography/#citations).
- Use the `meta-data.json` from the OSF project to find the correct series. Look for the `InstanceUID` field to identify each scan.
- Convert and save all relevant DICOMs as `.mha` files in:  
  `data/CTC Scans/`

#### Step 3: Configure the Training Script

Open the file `run_hqcolon.sh` in the hqcolon folder of the repository and make the following changes:

- **Line 7**: Set your dataset name using the format `DatasetXXX_name` (replace `XXX` with a unique 3-digit ID).
- **Line 8**: Set `dataset_number=XXX` (must match the digits used in Line 7).
- **Line 33**: Optional flags:
  - `--masked`: use dilated masks from `Masks TotalSegmentator/` to preprocess input.
  - `--fluid`: segment both air and fluid regions.

### What the Script Does

The `run_hqcolon.sh` script performs the following steps:

1. **Splits** your dataset into training and  test sets stratified by sex and position, keeping patients grouped by split.
   *(To modify this logic, edit `dataset_split_creator.py`)*

2. **Creates the dataset** by copying and renaming required files from your `data/` directory.

3. **Prepares** the dataset using nnUNetV2's internal logic.

4. **Trains and validates** the model using the provided data.

5. **Predicts** segmentations on the test set.

6. **Evaluates** the model's predictions against ground truth labels.

---

### Predict Segmentations Using a Pre-Trained Model

We provide **four different pre-trained HQColon models**.  
Each model was trained using the same train/test split and input images.  
The models differ in two ways:

1. **Input Type**: Whether the original input image was **masked** using **dilated TotalSegmentator masks**.
2. **Label**: Whether the model was trained to segment only the **air-filled** part or both **air and fluid-filled** parts of the colon.

#### Model Comparison Table

| Checkpoint Name      | Labels           | Input Type       | Dataset Name            |
|----------------------|------------------|------------------|-------------------------|
| `air.pth`            | Air              | Original image   | Dataset001_air          |
| `air-masked.pth`     | Air              | Masked image     | Dataset002_air-masked   |
| `fluid.pth`          | Air & Fluid      | Original image   | Dataset003_fluid        |
| `fluid-masked.pth`   | Air & Fluid      | Masked image     | Dataset004_fluid-masked |

We recommend working with model `fluid.pth`.

#### Step 1: Download Model Checkpoints

- Download the pre-trained model checkpoints from the official source (link to be provided).
- Use a dataset name with the format `DatasetXXX_name`, where:
  - `XXX` is a 3-digit unique ID (e.g., `123`)
  - `name` is a descriptive label (e.g., `hqcolon_airfluid`)

> This dataset name is used **only** for model identification within nnUNet. It does not need to match your input data name.

More info: [Setting up paths in nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)

#### Step 2: Organize Model Files

Create the following folder structure:
nnunet_results/
└── DatasetXXX_name/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
└── fold_0/
└── checkpoint_best.pth

- Replace DatasetXXX_name with your chosen dataset name (e.g., Dataset123_hqcolon).
- Move the downloaded checkpoint_best.pth file into the fold_0/ directory.

#### Step 3: Edit the Prediction Script (predict.sh)

Open the predict.sh file and modify the following lines:

- **Line 7** – Set the dataset name: DATASET_NAME="Dataset123_hqcolon"

- **Line 8** – Set the matching dataset number: DATASET_NUMBER=123

- **Line 10** – Set the path to your input test data (e.g., the CTC scans): input_path_to_test_dir="/path/to/data/CTC Scans"

- **Line 11** – Set the output directory for saving predictions: output_path_to_predictions="/path/to/output/predictions"

- **Line 23** – Confirm the correct checkpoint file name: --checkpoint_name checkpoint_best.pth

#### Step 4: Run the Script

Make the script executable and run it:

```
chmod +x predict.sh
. predict.sh
```

#### Step 3: Rename Data files

Make sure your data files have the nnunetv2 filename format. For example files can be named: colon_XXX_0000.mha, where XXX is a unique digit. It is crucial you have the same ending (here 0000.mha) as in the trained model. 

### What the Script Does

1. **Predicts** segmentations on your data set.

### Optional Evaluation

In case you have ground truth data for your images, include the in the folder nnunet_raw/<dataset_name>/labelsTs with the correct nnunet filenames. Here a name like colon_XXX.mha is expected.

Note: You will only get mean and median results if your dataset includes more than 1 sample.

---

## Original HQColon Results

We provide the original evaluation results for four HQColon-trained nnUNetv2 models. All models were trained for binary segmentation: distinguishing only between background and segmentation, without differentiating between multiple segmentation classes.

The four dataset configurations differ along two key dimensions:
1. Input Type: Whether the input image was masked using dilated TotalSegmentator masks or left unmasked.
2. Label Type: Whether the model segments only air-filled regions or both air and fluid-filled regions of the colon.

### Dataset Configurations Overview

| Labels           | Input Type       | Dataset Name                             |
|------------------|------------------|------------------------------------------|
| Air              | Original image   | Dataset101_regiongrowing_qc              |
| Air              | Masked image     | Dataset102_regiongrowing_qc_masked       |
| Air & Fluid      | Original image   | Dataset105_regiongrowing_qc_fluid        |
| Air & Fluid      | Masked image     | Dataset106_regiongrowing_qc_fluid_masked |

**We recommend working wiht the model trained on Dataset105_regiongrowing_qc_fluid.**

### Results Folder: `original hqcolon results`

This folder includes:

- All four dataset configurations listed above  
- Comparative evaluations using **TotalSegmentator** colon segmentations  
  - Both for **air** and **air + fluid** segmentation  

We applied **two evaluation strategies**:
- **Postprocessed Evaluation**: Removes small segmentation artifacts and retains only large connected regions ("islands"). This approach showed improved reliability.
- **Raw Evaluation**: Uses the raw output predictions without postprocessing.

### Result Files

Each evaluation strategy produces the following files:

| **Filename**                   | **Description**                                         |
|-------------------------------|----------------------------------------------------------|
| `island_results.jsonl`        | Postprocessed results across the full dataset            |
| `island_results_details.jsonl`| Postprocessed results per individual sample              |
| `results.jsonl`               | Raw segmentation results across the full dataset         |
| `results_details.jsonl`       | Raw results per individual sample                        |

### Evaluation Metrics


In our evaluation, we focus on a comprehensive set of metrics to assess model performance.

#### Per-Sample Metrics

These metrics are computed for each individual test sample and are stored in the `*_details.jsonl` files:

- **Confusion Matrix**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
- **Accuracy**
- **Matthews Correlation Coefficient (MCC)**
- **Average Symmetric Surface Distance (ASSD)**
- **Mean Absolute Surface Distance (MASS)**
- **Normalized Surface Dice (NSD)**
- **Intersection over Union (IoU)**
- **Hausdorff 95% Percentile (HD_perc)**
- **Hausdorff Distance (HD)**
- **Dice Coefficient**


For evaluating performance across the entire test set, we focus on the following spatial metrics:

- **ASSD**
- **MASS**
- **Hausdorff 95% Percentile**
- **Hausdorff Distance**

For each of these, we report the following statistical summaries:

- **Median**
- **Confidence Interval of the Median** (CI Low, CI High)
- **Mean**
- **Standard Deviation**

All aggregated results are reported in the respective `*.jsonl` summary files.

---

## Acknowledgments

Created by members of the [Image Section of the University of Copenhagen](https://di.ku.dk/english/research/image/) as part of the [Horizon Europe 2023 Intelligent Robotic Endoscopes (IRE)](https://ire4health.eu/) project.

For any questions and feedback please contact: Martina Finocchiaro (martina.finocchiaro.mf@gmail.com).
