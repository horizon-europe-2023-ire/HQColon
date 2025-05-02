import shutil
import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def check_label_values(label_path):
    label_image = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_image)
    print("Unique labels found:", np.unique(label_array))
    if np.unique(label_array).size != 2:
        return False
    if np.unique(label_array)[0] != 0 or np.unique(label_array)[1] != 1:
        return False
    return True


def iterate_fluid_labels():
    fluid_dir = os.path.join(BASE_DIR, 'data', 'Segmentation Air and Fluid')
    files = os.listdir(fluid_dir)
    for file in files:
        if not check_label_values(os.path.join(fluid_dir, file)):
            print(file)


# colon_185.mha has unique labels [0, 3]
if __name__ == '__main__':
    # iterate_fluid_labels()
    fluid_dir = os.path.join(BASE_DIR, 'data', 'Segmentation Air and Fluid')
    filename = "colon_185.mha"
    filepath = os.path.join(fluid_dir, filename)
    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    print("Unique labels found:", np.unique(array))
    array[array == 3] = 1
    new_image = sitk.GetImageFromArray(array)
    new_image.CopyInformation(image)  # Copy spacing, origin, direction from original image
    sitk.WriteImage(new_image, filepath)

    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    print("Unique labels found:", np.unique(array))

    print("Done")
