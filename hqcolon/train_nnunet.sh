#!/bin/bash

# Set the required environment variables for nnUNet
export nnUNet_raw="./nnunet_raw"  # Base directory for raw dataset
export nnUNet_preprocessed="./nnUNet_preprocessed"     # Directory for preprocessed data
export nnUNet_results="./nnunet_results"          # Directory to store trained model results
export dataset_name="Dataset001_test"       # Name of the dataset
export dataset_number="001"                             # Dataset number
export fold=0                            # Fold number

# Print the environment variables for debugging
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "dataset: $dataset_name"
echo "dataset number: $dataset_number"

# Activate your virtual environment if needed
# source /home/amin/anaconda3/bin/activate IRE-nnunetv2  # Activate Python virtual environment

# Run preprocessing and verify dataset integrity

nnUNetv2_plan_and_preprocess -d $dataset_number --verify_dataset_integrity
# -t 001: Specifies the task ID for your dataset (001 for Dataset001_colon_automatic)
# --verify_dataset_integrity: Checks for any inconsistencies in the dataset format

# Start training the nnUNet model
nnUNetv2_train $dataset_number 3d_fullres $fold -tr nnUNetTrainer
# -num_gpus all: Utilizes all available GPUs for training.
# --c: Enables checkpointing during training, allowing you to continue from the latest checkpoint if it exists.

# Run evaluation on the trained model
export input_path_to_test_dir = $nnUNet_raw/$dataset_name/imagesTs
export output_path_to_predictions = $nnUNet_results/$dataset_name/predictions
nnUNetv2_predict -i input_path_to_test_dir -o $output_path_to_predictions -d $dataset_number -c 3d_fullres -f $fold -chk checkpoint_best.pth --verbose


