import gzip
import shutil
import os
import tempfile
import random
import argparse
import pandas as pd
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np


def convert_to_totalseg_path(converted_name):
    folder_name = converted_name.replace('.mha.gz', '')
    subject = folder_name.split('_')[0]
    total_seg_filename = converted_name.replace('.mha.gz', '_totalseg-colon.mha.gz')
    path = os.path.join(subject, folder_name, total_seg_filename)
    return path


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


def add_surrounding(filepath):
    """
    Dilates a binary mask and ensures any holes in the mask are closed.

    Parameters:
    - dilation_factor: int, the radius for dilation.
    - filepath: str, the path to the binary mask image (input and output).
    """
    dilation_factor = 35
    # Read the binary mask image
    image = sitk.ReadImage(filepath)
    img_data = sitk.GetArrayFromImage(image)

    # Ensure the image is binary (only two unique values: 0 and 1)
    unique_values = np.unique(img_data)
    assert len(unique_values) == 2 and set(unique_values).issubset({0, 1}), "Input must be a binary mask."

    # Step 1: Dilate to add surrounding areas
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(dilation_factor)
    dilate_filter.SetKernelType(sitk.sitkBall)  # Use a spherical kernel
    dilate_filter.SetForegroundValue(1)
    dilated_image = dilate_filter.Execute(image)

    # Step 2: Fill holes to make the mask completely solid
    hole_filled_image = sitk.BinaryFillhole(dilated_image)

    # Step 3: Perform extensive morphological closing for large-scale smoothing
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius(dilation_factor // 2)  # Use a smaller kernel for closing
    closing_filter.SetForegroundValue(1)
    closed_image = closing_filter.Execute(hole_filled_image)

    # Step 4: Apply Gaussian smoothing to soften jagged edges (very aggressive smoothing)
    smoothing_radius = dilation_factor * 0.5  # Proportional to dilation factor
    smoothed_image = sitk.SmoothingRecursiveGaussian(closed_image, smoothing_radius)

    # Step 5: Threshold back to binary (ensures output is 0 or 1)
    binary_image = sitk.BinaryThreshold(smoothed_image, lowerThreshold=0.5, upperThreshold=255, insideValue=1,
                                        outsideValue=0)

    # Save the processed binary mask back to the filepath
    sitk.WriteImage(binary_image, filepath)


def show_plot(ax, data, title="", points=None, cmap="tab20b"):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    return ax


def visualize_image(data, file_path, title='', initial_point=None, end_point=None, cmap="tab20b", show=False):
    # flip the z axis so the lungs are shown on top and the rectum on the bottom.
    data = np.flip(data, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0] = show_plot(axes[0], data[-70, :, :], title="Axial", cmap=cmap)
    axes[1] = show_plot(axes[1], data[:, data.shape[1] // 2, :], title="Coronal", cmap=cmap)
    axes[2] = show_plot(axes[2], data[:, :, data.shape[2] // 2], title="Sagital", cmap=cmap)

    # Convert SimpleITK image to a NumPy array for visualization
    if initial_point is not None:
        i = data.shape[0] - initial_point[0]
        axes[0].scatter(initial_point[1], initial_point[2], c='red')
        axes[1].scatter(initial_point[2], i, c='red')
        axes[2].scatter(initial_point[1], i, c='red')
    if end_point is not None:
        i = data.shape[0] - end_point[0]
        axes[0].scatter(end_point[1], end_point[2], c='white')
        axes[1].scatter(end_point[2], i, c='white')
        axes[2].scatter(end_point[1], i, c='white')

    plt.suptitle(title)
    if file_path is not None:
        plt.savefig(f"{file_path}.png", format='png')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_mha():
    # Load the .mha file
    file_path = 'nnunet_predictions/test/colon_002.mha'  # Replace with the actual file path
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)  # Convert the SimpleITK image to a NumPy array

    # Print the shape of the 3D array
    print("Shape of the image array:", image_array.shape)  # Output should be (depth, height, width)

    # Display a slice from each dimension
    def show_slices(slices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice, cmap="gray")
            axes[i].axis('off')
        plt.show()

    # Extract the middle slices from each dimension
    depth_mid = image_array.shape[0] // 2
    height_mid = image_array.shape[1] // 2
    width_mid = image_array.shape[2] // 2

    slices = [
        image_array[depth_mid, :, :],  # Axial slice (along z-axis)
        image_array[:, height_mid, :],  # Coronal slice (along y-axis)
        image_array[:, :, width_mid]  # Sagittal slice (along x-axis)
    ]

    show_slices(slices)


def decompress_gzip(compressed_path, target_path):
    """Decompress a gzipped file."""
    try:
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File decompressed and saved as {target_path}")
    except Exception as e:
        print(f"Error during decompression: {e}")


def move_file(source_path, destination_path):
    """Move a file from source to destination."""
    try:
        shutil.copy(source_path, destination_path)
        print(f"File moved to {destination_path}")
    except Exception as e:
        print(f"Error during moving the file: {e}")
