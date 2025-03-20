How to use nnUnet: [see documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

1. Load the cuda modul you want to use e.g. module load cuda/11.8
2. Load the conda modul: module load anaconda3/2024.06-py3.12.4
3. Create Conda Environment 
4. Install Pytorch according to what cuda module you loaded: [pytorch]https://pytorch.org/get-started/locally/
5. pip install nnunetv2

to create a new file:
touch <filename>

to reformat the file:
sed -i 's/\r$//' <filename>


