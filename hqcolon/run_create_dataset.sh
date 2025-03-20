# This script is used to create a dataset from training and prediction in the expected nnUNet hierarchy.

# Instead of creating the data from scratch, we can use an existing dataset an adapt it instead.
# If the dataset is created from scratch the dataset_to_copy can be set to ""
# Note than any dataset name needs to start with DatasetXXX_<name> where XXX are 3 unique digits and name can be replaced by anything
dataset_to_copy="" # change this to the name of the dataset you want to copy and adapt
dataset="Dataset001_test" # change this to the name of your new dataset

# this creates the mapping from future nnunet colon names to erda.
# In case of using a different dataset, write your own mapping file. To use the code below, ensure the mapping files have the expected properties.
# the created dataframe includes the following columns:
  #        'InstanceUID':           the unique identifier for each scan
  #        'Sex'                    sex of the subject
  #        'Position'               position of the subject for this scan
  #        'subject_id'             unique subject id
  #        'converted_file'         path to the scan (.mha file)
  #        'regional_growing_file'  path to the segmentation (.mha file)
  #        'number'                 nnunet unique number for naming
  #        'nnunet_image_file'      the future nnunet filename for the input data / scan (e.g. "colon_313_0000.mha")
  #        'nnunet_label_file'      the future nnunet filename for the label data / segmentation (e.g. "colon_313.mha"
python get_dataset_mapping.py

# here we split the created name mappings into test and train resulting in two seperated files.
# for this file crucial are properties used for splitting like: Position, Sex or subject_id
# by default the dataset is stratified split by Sex and Position, this can be changed in the file
python dataset_split_creator.py

# here we create the local dataset the following flags can be set:
# --masked: mask images, else we use the original images using the dilated totalsegmentator segmentation
# --copy: copy dataset from existing local dataset instead of downloading it from erda
# --fluid: add fluid labels to existing air labels
python create_nnunet_dataset.py $dataset $dataset_to_copy --fluid

# this creates an additional file in the dataset folder, needed by the nnunet to configure the dataset
python create_dataset_json.py $dataset
