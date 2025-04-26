# This script is used to create a dataset from training and prediction in the expected nnUNet hierarchy.

# Note than any dataset name needs to start with DatasetXXX_<name> where XXX are 3 unique digits and name can be replaced by anything
dataset="Dataset002_fluid " # change this to the name of your new dataset

# here we split the created name mappings into test and train resulting in two seperated files.
# for this file crucial are properties used for splitting like: Position, Sex or subject_id
# by default the dataset is stratified split by Sex and Position, this can be changed in the file
# python dataset_split_creator.py

# here we create the local dataset the following flags can be set:
# --masked: mask images, else we use the original images using the dilated totalsegmentator segmentation
# --fluid: add fluid labels to existing air labels
python prepare_dataset.py $dataset --fluid

# this creates an additional file in the dataset folder, needed by the nnunet to configure the dataset
python create_dataset_json.py $dataset
