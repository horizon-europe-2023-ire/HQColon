"""
In this file we load the meta data table of all files in our dataset.

We collect the InstanceUID the subject_id (e.g. sub_001) the mha_file_name the
 label file name and the newly created id number for the nnunet.

We drop all those files we have no ground truth for = have no file in the non-collapsed
folder from the regional growing method.

We add nnunet file names for the image and the label
"""

import pandas as pd
import os
import re

HOME_ERDA = '/home/amin/ucph-erda-home/IRE-DATA/CT'  # Path to ERDA


def filter_substring(substring, files):
    pattern = re.compile(substring)
    filtered_strings = [s for s in files if pattern.search(s)]
    return filtered_strings


def get_files_in_subfolders(parent_folder):
    subfolders = os.listdir(parent_folder)
    if "README.txt" in subfolders:
        subfolders.remove("README.txt")
    file_list = []
    for sub in subfolders:
        files = os.listdir(os.path.join(parent_folder, sub))
        for file in files:
            file_path = os.path.join(file)
            file_list.append(file_path)
    return file_list


def process_row_mha(row):
    number = row['number']
    nnunet_mha_file = f"colon_{number}_0000.mha"
    nnunet_label_file = f"colon_{number}.mha"
    row['nnunet_image_file'] = nnunet_mha_file
    row['nnunet_label_file'] = nnunet_label_file
    return row


def create_mapping():
    meta_file_path = 'metadata/meta_data_df.json'  # file containing all kind of metadata
    meta_data_df = pd.read_json(os.path.join(HOME_ERDA, meta_file_path), lines=True)

    iuid = meta_data_df['InstanceUID'].tolist()
    genders = meta_data_df['Sex'].tolist()
    subject_ids = meta_data_df['new_subject_id'].tolist()
    mha_file_list = meta_data_df['mha_path'].tolist()
    # mha_file_list = [os.path.join(HOME_ERDA, mha_file) for mha_file in mha_file_list]
    label_files = []
    positions = []

    label_folder = 'segmentations/segmentations-regionalgrowing-qc/good-quality'
    label_file_names = get_files_in_subfolders(
        os.path.join(HOME_ERDA, label_folder))

    for mha_file in mha_file_list:
        mha_file_name = os.path.basename(mha_file)
        splits = mha_file_name.split('_')
        subject = splits[0]
        position = splits[1]
        positions.append(position.replace('pos-', ''))
        scan = splits[2]
        filtered_files = filter_substring(subject, label_file_names)
        filtered_files = filter_substring(position, filtered_files)
        filtered_files = filter_substring(scan, filtered_files)
        if len(filtered_files) == 1:
            label_filename = filtered_files[0]
            label_file = os.path.join(HOME_ERDA, label_folder, subject, label_filename)
            label_files.append(label_file)
        else:
            label_files.append(None)

    df = pd.DataFrame({
        'InstanceUID': iuid,
        'Sex': genders,
        'Position': positions,
        'subject_id': subject_ids,
        'converted_file': mha_file_list,
        'regional_growing_file': label_files,
    })

    df = df.dropna(subset=["regional_growing_file"])

    # add new image and label file names

    numbers = [str(i).zfill(3) for i in range(1, len(df) + 1)]
    df['number'] = numbers

    df['nnunet_image_file'] = ''
    df['nnunet_label_file'] = ''
    df = df.apply(process_row_mha, axis=1)

    print(f"There are {len(df)} filename mappings")
    df.to_json('name_mapping.json', orient='records', lines=True)


create_mapping()
