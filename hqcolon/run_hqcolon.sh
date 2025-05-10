#!/bin/bash

# Set the required environment variables for nnUNet
export nnUNet_raw="../nnunet_raw"  # Base directory for raw dataset
export nnUNet_preprocessed="../nnUNet_preprocessed"     # Directory for preprocessed data
export nnUNet_results="../nnunet_results"          # Directory to store trained model results
export dataset_name="Dataset001_fluid_masked"       # Name of the dataset
export dataset_number="001"                             # Dataset number
export fold=0                            # Fold number
export input_path_to_test_dir="$nnUNet_raw/$dataset_name/imagesTs"
export output_path_to_predictions="$nnUNet_results/$dataset_name/predictions"

# Print the environment variables for debugging
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "dataset: $dataset_name"
echo "dataset number: $dataset_number"
echo "input_path_to_test_dir: $input_path_to_test_dir"
echo "output_path_to_predictions: $output_path_to_predictions"

## split the dataset into test and train
python dataset_split_creator.py

# here we create the local dataset the following flags can be set:
# --masked: mask images, else we use the original images using the dilated totalsegmentator segmentation
# --fluid: add fluid labels to existing air labels
python prepare_dataset.py $dataset_name --masked --fluid

# this creates an additional file in the dataset folder, needed by the nnunet to configure the dataset
python create_dataset_json.py $dataset_name

# Run preprocessing and verify dataset integrity
nnUNetv2_plan_and_preprocess -d $dataset_number --verify_dataset_integrity -c 3d_fullres
# -t 001: Specifies the task ID for your dataset (001 for Dataset001_colon_automatic)
# --verify_dataset_integrity: Checks for any inconsistencies in the dataset format

# Start training the nnUNet model
nnUNetv2_train $dataset_number 3d_fullres $fold -tr nnUNetTrainer
# -num_gpus all: Utilizes all available GPUs for training.
# --c: Enables checkpointing during training, allowing you to continue from the latest checkpoint if it exists.

# Predictions for test set
nnUNetv2_predict -i $input_path_to_test_dir -o $output_path_to_predictions -d $dataset_number -c 3d_fullres -f $fold -chk checkpoint_best.pth --verbose

# evaluate predictions and compute metrics
python ../evaluation/evaluation.py $dataset_name
