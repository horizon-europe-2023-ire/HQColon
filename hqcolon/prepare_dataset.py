"""
This file is used to create the nnunet dataset. As required by nnunet the data is saved in the nnunet_raw folder
and we rename our files as colon_xxx_0000.mha. xxx is a new id number while 0000 specifies the modality. As
we are working only with ct scans this is always the same. The binary labels are named colon_xxx.mha (we don't have to
specify the modality).

To run this file we need to pass the dataset name as followed: DatasetXXX_<name>. There can't be two datasets with
the same identification number xxx.

run this file with:
python nnunet_dataset.py DatasetXXX_<name>

or let it run using the shellscript
"""
import shutil
import SimpleITK as sitk
import os
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def copy_file(source_path, destination_path):
    """Move a file from source to destination."""
    try:
        shutil.copy(source_path, destination_path)
        print(f"File moved to {destination_path}")
    except Exception as e:
        print(f"Error during moving the file: {e}")


def mask_image(image_path, mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Ensure the image and mask are of the same size and type
    if image.GetSize() != mask.GetSize():
        raise ValueError("Image and mask must have the same dimensions")

    # Ensure the mask has the same physical properties as the image
    mask = sitk.Cast(mask, image.GetPixelID())
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Resample the mask to the image grid to align them
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(image.GetSize())  # Match the image size
    resampler.SetOutputSpacing(image.GetSpacing())  # Match the image spacing
    resampler.SetOutputOrigin(image.GetOrigin())  # Match the image origin
    resampler.SetOutputDirection(image.GetDirection())  # Match the image direction
    resampler.SetTransform(sitk.Transform())  # Identity transform (no rotation, scaling, etc.)
    resampler.SetDefaultPixelValue(0)  # Default value for outside areas of the mask
    mask_resampled = resampler.Execute(mask)

    # Apply the mask to the image (use element-wise multiplication)
    masked_image = sitk.Mask(image, mask_resampled)

    # Save the masked image
    sitk.WriteImage(masked_image, image_path)
    return True


def mask_dataset(df, img_dir):
    for index, row in df.iterrows():
        image_file = row['nnunet_image_file']
        mask_name = image_file.replace('0000.mha', 'mask_dilated.mha')
        image_path = os.path.join(img_dir, image_file)
        mask_path = os.path.join(BASE_DIR, 'data', 'Masks Total-Segmentator', mask_name)
        mask_image(image_path, mask_path)


def move_files(df, dest_dir_img, dest_dir_labels):
    source_dir_img = os.path.join(BASE_DIR, 'data', 'CTC Scans')
    source_dir_label = os.path.join(BASE_DIR, 'data', 'Segmentation Air')
    if FLUID:
        source_dir_label = os.path.join(BASE_DIR, 'data', 'Segmentation Air and Fluid')

    for index, row in df.iterrows():
        image_file = row['nnunet_image_file']
        copy_file(os.path.join(source_dir_img, image_file), os.path.join(dest_dir_img, image_file))
        label_file = row['nnunet_label_file']
        copy_file(os.path.join(source_dir_label, label_file), os.path.join(dest_dir_labels, label_file))


def main():

    os.makedirs(os.path.join(BASE_DIR, 'nnunet_raw'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'nnunet_raw', DATASET_NAME), exist_ok=True)

    df_train = pd.read_json(os.path.join(BASE_DIR, 'data', 'train_mapping.json'), lines=True)
    df_test = pd.read_json(os.path.join(BASE_DIR, 'data', 'test_mapping.json'), lines=True)

    destination_path_labels_tr = os.path.join(BASE_DIR, 'nnunet_raw', DATASET_NAME, f"labelsTr")
    destination_path_labels_ts = os.path.join(BASE_DIR, 'nnunet_raw', DATASET_NAME, f"labelsTs")
    destination_path_img_tr = os.path.join(BASE_DIR, 'nnunet_raw', DATASET_NAME, f"imagesTr")
    destination_path_img_ts = os.path.join(BASE_DIR, 'nnunet_raw', DATASET_NAME, f"imagesTs")
    os.makedirs(destination_path_labels_tr, exist_ok=True)
    os.makedirs(destination_path_labels_ts, exist_ok=True)
    os.makedirs(destination_path_img_tr, exist_ok=True)
    os.makedirs(destination_path_img_ts, exist_ok=True)

    move_files(df_train, destination_path_img_tr, destination_path_labels_tr)
    move_files(df_test, destination_path_img_ts, destination_path_labels_ts)

    if MASKED:
        print("Mask the existing images using total segmentator colon segmentations")
        mask_dataset(df_train, destination_path_img_tr)
        mask_dataset(df_test, destination_path_img_ts)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Update dataset JSON with training and testing files.")

parser.add_argument("dataset", type=str, help="Name of the dataset directory (e.g., Dataset003_region-growing)")

# Add a boolean argument for masking
parser.add_argument("--masked", action="store_true",
                    help="Indicate whether the images should be masked using total segmentator segmentations")

# Add a boolean argument to indicate if fluid should be included in labels
parser.add_argument("--fluid", action="store_true", help="Indicate whether the labels should include fluid or not")

args = parser.parse_args()
DATASET_NAME = args.dataset
MASKED = args.masked
FLUID = args.fluid

print("Start to create dataset")
print(f"Arguments: {DATASET_NAME}, {MASKED}, {FLUID}")
main()
print("Done creating the dataset")

