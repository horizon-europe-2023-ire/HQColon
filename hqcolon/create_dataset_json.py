"""
This file created automatically the required dataset_json file for a given dataset.
In this file we need to keep track of dataset properties and additionally of each filename
of training and test files as their number.

To run this we need to pass the dataset name we want to create the file for:

python create_dataset_json.py <datasetname>
"""

import os
import json
import argparse

BASE = 'nnunet_raw'  # ronja local
# BASE = '/home/bgn595/nnunet/nnunet_raw'  # ronja cluster path -> adapt this to your path


def get_files(dataset, split, labels):
    path_to_files = os.path.join(BASE, f'{dataset}/{split}')
    files = os.listdir(path_to_files)
    file_names = []
    for file in files:
        if labels:
            case = {'image': file, 'label': file.replace('_0000', '')}
        else:
            case = {'image': file}
        file_names.append(case)
    return file_names


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Update dataset JSON with training and testing files.")
parser.add_argument("dataset", type=str, help="Name of the dataset directory (e.g., Dataset003_region-growing)")
args = parser.parse_args()
dataset = args.dataset
print(f"Create dataset.json file for: {dataset}")

split = "imagesTr"
current_path = os.getcwd()
train_files = get_files(dataset, split, True)

split = "imagesTs"
test_files = get_files(dataset, split, False)

json_file_path = os.path.join(BASE, f'{dataset}/dataset.json')

data = {
    "name": dataset,
    "channel_names": {
        "0": "CT"
    },
    "modality": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "colon": 1
    },
    "file_ending": ".mha",
    "dataset_name": dataset,
    "license": "",
    "description": "",
    "reference": ""
}

data['training'] = train_files
data['numTraining'] = len(train_files)
#data['test'] = test_files
#data['numTest'] = len(test_files)

with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file)
