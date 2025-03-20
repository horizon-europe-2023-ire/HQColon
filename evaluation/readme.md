# Evaluation

In order to use the evaluation script MetricsRedload has to be installed.

1. Check out the website: https://github.com/Project-MONAI/MetricsReloaded

2. Clone the GitHub repository

3. On top of the evaluation script add the path to your local MetricsReload repository:
```
import sys
sys.path.append('/path/to/MetricsReloaded')
```

## evaluation.py

This script needs the nnunet dataset (labelsTs) and the model's predictions (stored in nnunet_results/DatasetName/predictions). 
Running the script will create the results_detail.json and the results_island_detail.json files. Additionally in results.json and results_island.json an overview of the average scores can be found for this dataset.



