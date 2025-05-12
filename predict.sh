#!/bin/bash

# Set the required environment variables for nnUNet
export nnUNet_raw="nnunet_raw"  # Base directory for raw dataset
export nnUNet_preprocessed="nnUNet_preprocessed"     # Directory for preprocessed data
export nnUNet_results="nnunet_results"          # Directory to store trained model results
export dataset_name="Dataset001_test"       # Name of the dataset
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

# Predictions for test set
nnUNetv2_predict -i $input_path_to_test_dir -o $output_path_to_predictions -d $dataset_number -c 3d_fullres -f $fold -chk checkpoint_best.pth --verbose

# evaluate predictions and compute metrics
python evaluation/evaluation.py $dataset_name
