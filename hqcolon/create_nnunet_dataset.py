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
import tempfile
import argparse
import pandas as pd
import numpy as np
from helper import decompress_gzip, move_file, add_surrounding
from pathlib import Path

# ERDA_DIR = 'Z:\IRE-DATA\CT'  # Ronja windows
ERDA_DIR = '/home/amin/ucph-erda-home/IRE-DATA/CT'  # Ronja linux path
# ERDA_DIR = '/home/bgn595/data/'  # ronja cluster


def get_files_in_subfolders(parent_path):
    # List all files in subfolders of the provided directory
    files = []
    current_path = os.path.join(ERDA_DIR, parent_path)
    for subject in os.listdir(current_path):
        current_folder = os.path.join(current_path, subject)
        for file in os.listdir(current_folder):
            files.append((current_folder, file))  # Print full path of each file
    return files


def copy_and_decompress_file_from_erda(source_path, dest_path, dest_filename):
    """Main function to decompress and move file.
    """
    # erda_total_seg_path, dest_dir, mha_source_file
    assert os.path.isfile(source_path)
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the path for the decompressed file
        decompressed_file_path = os.path.join(temp_dir, dest_filename)
        # Decompress the gzipped file into the temporary directory
        decompress_gzip(source_path, decompressed_file_path)
        new_file = os.path.join(dest_path, dest_filename)
        # Move the decompressed file to the new destination
        move_file(decompressed_file_path, new_file)
        print(f"Moved {decompressed_file_path}, to {new_file}")
    return new_file


def mask_overlaps_label(mask, image_path):
    label_path = str(image_path).replace('_0000', '').replace('images', 'labels')
    if os.path.exists(label_path):
        label = sitk.ReadImage(str(label_path), sitk.sitkUInt8)
        label_array = sitk.GetArrayFromImage(label)
        mask_array = sitk.GetArrayFromImage(mask)
        if np.any((label_array == 1) & (mask_array == 0)):
            return label
    else:
        print(f"Error label file could not be found: {label_path}")
    return mask


def mask_image(image_path, mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # if not mask_overlaps_label(mask, image_path):
    #     return False

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


def convert_to_totalseg_path(converted_name):
    folder_name = converted_name.replace('.mha.gz', '')
    subject = folder_name.split('_')[0]
    total_seg_filename = converted_name.replace('.mha.gz', '_totalseg-colon.mha.gz')
    path = os.path.join(subject, folder_name, total_seg_filename)
    return path


def copy_files():
    dataset_to_copy_path = os.path.join('nnunet_raw', DATASET_TO_COPY)
    dataset_path = os.path.join('nnunet_raw', DATASET_NAME)
    if not os.path.exists(dataset_to_copy_path):
        raise ValueError(f"Dataset {DATASET_TO_COPY} does not exist")

    for folder in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']:
        data_to_copy = os.listdir(os.path.join(dataset_to_copy_path, folder))
        for file in data_to_copy:
            scr = os.path.join(dataset_to_copy_path, folder, file)
            if os.path.isfile(scr):
                dst = os.path.join(dataset_path, folder, file)
                try:
                    shutil.copy(scr, dst)
                except Exception as e:
                    print(f"Error copying {scr} to {dst}: {e}")
            else:
                print(f"Skipping {file} because it does not exist in {DATASET_TO_COPY}/{folder}")


def get_images(df, dest_data, dest_labels):
    for index, row in df.iterrows():
        source_file_image = row['converted_file']
        source_file_label = row['regional_growing_file']

        destination_file_image = row['nnunet_image_file']
        destination_file_label = row['nnunet_label_file']
        if not os.path.isfile(os.path.join(dest_data, destination_file_image)):
            copy_and_decompress_file_from_erda(source_file_image, dest_data, destination_file_image)

        if not os.path.isfile(os.path.join(dest_labels, destination_file_label)):
            copy_and_decompress_file_from_erda(source_file_label, dest_labels, destination_file_label)


def find_colon_files(directory):
    files = []
    for file_path in Path(directory).rglob('*'):  # Recursively find all files and folders
        if file_path.is_file() and 'colon' in file_path.name:  # Check if it is a file and contains a colon
            files.append(file_path)
    print(f"Found {len(files)} colon segmentation files for totalsegmentator")
    return files


def create_image_mask(nnunet_file_path, dest_dir, erda_total_seg_path):
    mask_name = nnunet_file_path.replace('0000.mha', 'mask.mha')
    filename = os.path.basename(erda_total_seg_path)
    mask_dir = os.path.join('nnunet_raw', 'masks')
    if os.path.isfile(os.path.join(mask_dir, mask_name)):
        mask_file = os.path.join(mask_dir, mask_name)
    else:
        if os.path.isfile(erda_total_seg_path):
            mask_file = copy_and_decompress_file_from_erda(erda_total_seg_path, mask_dir, mask_name)
            add_surrounding(mask_file)
        else:
            print(f"Skipped {filename} because it was not found in total segmentator")
    mask_image(os.path.join(dest_dir, nnunet_file_path), mask_file)


def mask_dataset(df, dest_data):
    total_seg_dir = os.path.join(ERDA_DIR, 'segmentations/segmentations-totalsegmentator')

    for index, row in df.iterrows():
        source_file_image = row['converted_file']
        nnunet_file_image = row['nnunet_image_file']

        if os.path.isfile(os.path.join(dest_data, nnunet_file_image)):

            filename = os.path.basename(source_file_image)
            erda_total_seg_path = convert_to_totalseg_path(filename)
            erda_total_seg_path = os.path.join(total_seg_dir, erda_total_seg_path)

            create_image_mask(nnunet_file_image, dest_data, erda_total_seg_path)
        else:
            print(f"{nnunet_file_image} does not exist")



def check_label_values(label_image):
    label_array = sitk.GetArrayFromImage(label_image)
    if np.unique(label_array).size != 2:
        print("Unique labels found:", np.unique(label_array))
        return False
    if np.unique(label_array)[0] != 0 or np.unique(label_array)[1] != 1:
        # Print unique labels in the file
        print("Unique labels found:", np.unique(label_array))
        return False
    return True


def get_fluid(label_file, split):
    if split == 'test':
        base_folder = os.path.join(HOME, 'fluid_test')
        fluid_file = label_file.replace('.mha', '_0000_fluidmask_cleaned_attached_v2.mha')
        fluid_path = os.path.join(base_folder, 'modified', fluid_file)
        if os.path.exists(fluid_path):
            print("Modified fluid found for: ", fluid_file)
            return sitk.ReadImage(fluid_path)
        fluid_file = fluid_file.replace('_v2.mha', '.mha')
        fluid_path = os.path.join(base_folder, fluid_file)
        if os.path.exists(fluid_path):
            print("Fluid found for: ", fluid_file)
            return sitk.ReadImage(fluid_path)
    elif split == 'train':
        base_folder = os.path.join(HOME, 'fluid_training')
        fluid_file = label_file.replace('.mha', '_0000_fluidmask_cleaned_attached_v3.mha')
        fluid_path = os.path.join(base_folder, 'modified', 'modified_for_dim', fluid_file)
        if os.path.exists(fluid_path):
            print("Corrected fluid found for: ", fluid_file)
            return sitk.ReadImage(fluid_path)
        fluid_file = label_file.replace('.mha', '_0000_fluidmask_cleaned_attached_v2.mha')
        fluid_path = os.path.join(base_folder, 'modified', fluid_file)
        if os.path.exists(fluid_path):
            print("Modified fluid found for: ", fluid_file)
            return sitk.ReadImage(fluid_path)
        fluid_file = fluid_file.replace('_v2.mha', '.mha')
        fluid_path = os.path.join(base_folder, fluid_file)
        if os.path.exists(fluid_path):
            print("Fluid found for: ", fluid_file)
            return sitk.ReadImage(fluid_path)
    return None


def add_fluid(label, fluid):
    # Perform logical XOR to ensure no overlaps (1 where only one of the images has 1)
    label = sitk.Or(sitk.Cast(label, sitk.sitkUInt8), sitk.Cast(fluid, sitk.sitkUInt8))
    return label


def add_fluid_to_split(label_folder, dataset, split):
    label_dir = os.path.join('nnunet_raw', dataset, label_folder)
    if not os.path.isdir(label_dir):
        raise ValueError("Label folders do not exist. Ensure the dataset already exists.")

    files = os.listdir(label_dir)

    for label_file in files:
        label_path = os.path.join(label_dir, label_file)
        label = sitk.ReadImage(label_path)

        if not check_label_values(label):
            print(f"Error: {label_file} label has the wrong values")
            continue

        fluid = get_fluid(label_file, split)

        # check if fluid file exists
        if fluid is None:
            print(f"Error: No fluid could be found for: {label_file}")
            continue

        # # check if fluid and label have the same direction
        if fluid.GetSize() != label.GetSize():
            print(f"Error: Image and label are have not the same direction or size: {label_file}")
            print(fluid.GetDirection(), label.GetDirection(), fluid.GetSize(), label.GetSize())
            continue

        label = add_fluid(label, fluid)
        sitk.WriteImage(label, label_path)


def main():

    os.makedirs(os.path.join('nnunet_raw'), exist_ok=True)
    os.makedirs(os.path.join('nnunet_raw', DATASET_NAME), exist_ok=True)

    df_train = pd.read_json(os.path.join('nnunet_raw', 'train_mapping.json'), lines=True)
    df_test = pd.read_json(os.path.join('nnunet_raw', 'test_mapping.json'), lines=True)

    destination_path_labels_tr = os.path.join('nnunet_raw', DATASET_NAME, f"labelsTr")
    destination_path_labels_ts = os.path.join('nnunet_raw', DATASET_NAME, f"labelsTs")
    destination_path_data_tr = os.path.join('nnunet_raw', DATASET_NAME, f"imagesTr")
    destination_path_data_ts = os.path.join('nnunet_raw', DATASET_NAME, f"imagesTs")
    os.makedirs(destination_path_labels_tr, exist_ok=True)
    os.makedirs(destination_path_labels_ts, exist_ok=True)
    os.makedirs(destination_path_data_tr, exist_ok=True)
    os.makedirs(destination_path_data_ts, exist_ok=True)

    if COPY:
        copy_files()
    else:
        get_images(df_train, destination_path_data_tr, destination_path_labels_tr)
        get_images(df_test, destination_path_data_ts, destination_path_labels_ts)

    if MASKED:
        os.makedirs(os.path.join('nnunet_raw', 'masks'), exist_ok=True)
        print("Mask the existing images using total segmentator")
        mask_dataset(df_train, destination_path_data_tr)
        mask_dataset(df_test, destination_path_data_ts)

    if ADD_FLUID:
        os.makedirs(os.path.join('nnunet_raw', DATASET_NAME), exist_ok=True)
        add_fluid_to_split('labelsTr', DATASET_NAME, 'train')
        add_fluid_to_split('labelsTs', DATASET_NAME, 'test')



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Update dataset JSON with training and testing files.")
parser.add_argument("dataset", type=str, help="Name of the dataset directory (e.g., Dataset003_region-growing)")
parser.add_argument("dataset_to_copy", type=str, help="Name of the dataset which we want to copy")

# Add a boolean argument for copying
parser.add_argument("--copy", action="store_true",
                    help="Indicate whether the dataset should be copied or not.")

# Add a boolean argument for masking
parser.add_argument("--masked", action="store_true",
                    help="Indicate whether the images should be masked using total segmentator segmentations")

# Add a boolean argument for masking
parser.add_argument("--fluid", action="store_true",
                    help="Indicate that the labels should include additional to air also the fluid")

# Add a boolean argument to indicate if fluid should be included in labels
parser.add_argument("--fluid", action="store_true", help="Indicate whether the labels should include fluid or not")

args = parser.parse_args()

# Assign dataset and masked values based on the command-line arguments
DATASET_NAME = args.dataset
DATASET_TO_COPY = args.dataset_to_copy
MASKED = args.masked
COPY = args.copy
ADD_FLUID = args.fluid

print("Start to create dataset")
print(f"Arguments: {DATASET_NAME}, {DATASET_TO_COPY}, {MASKED}, {COPY}")
main()
print("Done creating the dataset")

