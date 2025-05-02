"""
This script is used to evaluate a trained model
Predictions of a model will be compared with the ground truth.
If the results already exist, evaluation will be skipped to avoid unnecessary computation.
"""

import sys
import pandas as pd
sys.path.append('/home/amin/MetricsReloaded')

import numpy as np
from scipy.ndimage import binary_dilation
import os
import SimpleITK as sitk
from MetricsReloaded.processes.mixed_measures_processes import MultiLabelPairwiseMeasures as MLPM
from MetricsReloaded.processes.overall_process import ProcessEvaluation as PE
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as PM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MPM
import json
import argparse
from scipy.stats import bootstrap


def get_border_img(img):
    # Apply dilation
    dilated_img = binary_dilation(img).astype(img.dtype)

    # Extract the borders by subtracting the original from the dilated
    borders = dilated_img - img
    return borders


def load_mha(file_path, biggest_island, prediction=False):
    img = sitk.ReadImage(file_path)

    if biggest_island and prediction:
        array = get_biggest_island(img)
        return array
    arr = sitk.GetArrayFromImage(img)
    return arr


def get_biggest_island(img):
    # Compute connected components
    components = sitk.ConnectedComponent(img)
    components_array = sitk.GetArrayFromImage(components)

    # Get unique labels (excluding background 0)
    labels = np.unique(components_array)
    labels = labels[labels != 0]  # Gets all the labels removing the background

    if len(labels) == 0:
        raise ValueError(f"No labels detected.")

    # Compute sizes of each component
    sizes = {label: np.sum(components_array == label) for label in labels}

    # Find the label corresponding to the largest component
    # biggest_label = max(sizes, key=sizes.get)
    # Minimum voxel threshold for inclusion
    min_voxel_threshold = 2000
    included_labels = [label for label, size in sizes.items() if size > min_voxel_threshold]

    # Create a binary mask for the largest component
    cleaned_array = np.isin(components_array, included_labels).astype(np.uint8)
    #cleaned_array = (components_array == biggest_label).astype(np.uint8)

    return cleaned_array


def get_images(filename, biggest_island, pred_dir, gt_dir):
    # Generate paths
    prediction_file = os.path.join(pred_dir, filename)
    ground_truth_file = os.path.join(gt_dir, filename)

    if os.path.exists(prediction_file) and os.path.exists(ground_truth_file):
        # Load images
        prediction = load_mha(prediction_file, biggest_island, True)
        ground_truth = load_mha(ground_truth_file, biggest_island, False)

        # Ensure I and J are binary
        prediction = (prediction > 0).astype(int)
        ground_truth = (ground_truth > 0).astype(int)

        return prediction, ground_truth

    return None, None


def calc_metrics_for_file(file, biggest_island, pred_dir, gt_dir):
    results = {'file': file}
    p_pred, p_ref = get_images(file, biggest_island, pred_dir, gt_dir)
    if p_pred is not None and p_ref is not None:
        pm = PM(p_pred, p_ref, dict_args={"hd_perc": 95})
        #
        # # confusion matrix
        # mpm = MPM(np.reshape(p_pred, [-1]), np.reshape(p_ref, [-1]), [0, 1])
        # cm = mpm.confusion_matrix()
        # TP = cm[0,0]
        # FN = cm[0,1]
        # FP = cm[1,0]
        # TN = cm[1,1]
        # # print(f"Confusion Matrix: {cm}")
        # results['TP'] = TP
        # results['FN'] = FN
        # results['FP'] = FP
        # results['TN'] = TN
        #
        # # accuracy
        # accuracy = pm.accuracy()
        # # print(f"Accuracy: {accuracy}")
        # results['accuracy'] = accuracy
        #
        # # Matthews correlation coefficient
        # mcc = mpm.matthews_correlation_coefficient()
        # # print(f"Matthews Correlation Coefficient: {mcc}")
        # results['mcc'] = mcc

        assd = pm.measured_average_distance()
        # print("ASSD: ", ASSD)
        results['ASSD'] = assd

        masd = pm.measured_masd()
        # print("MASD: ", masd)
        results['MASS'] = masd

        # bpm = PM(p_pred, p_ref, dict_args={"nsd": 1})
        # nsd = bpm.normalised_surface_distance()
        # # print("NSD ", nsd)
        # results['NSD'] = nsd
        #
        # iou = pm.intersection_over_union()
        # # print("IoU ", iou)
        # results['IOU'] = iou

        hausdorff = pm.measured_hausdorff_distance()
        hausdorff_distance_perc = pm.measured_hausdorff_distance_perc()
        # print(f"Hausdorff distance: {hausdorff}")
        # print(f"Hausdorff Perc distance: {hausdorff_distance_perc}")
        # results['HD'] = hausdorff
        results['HD_perc'] = hausdorff_distance_perc

        dice = pm.dsc()
        # print(f"Dice: {dice}")
        results['DICE'] = dice

        return results
    print(f"Skipping {file} because it doesn't exist")
    return None


def read_existing_results(biggest_island, results_dir):
    filename = 'results_details.jsonl'
    if biggest_island:
        filename = 'island_results_details.jsonl'
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data = [json.loads(line) for line in f]
                if len(data) == 0:
                    return pd.DataFrame(), filepath
            except Exception as e:
                print(f"Error: {e}")
                return pd.DataFrame(), filepath

        df = pd.read_json(filepath, lines=True)
        return df, filepath
    return pd.DataFrame(), filepath


def evaluate(dataset, biggest_island, gt_dir, pred_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    files = [file for file in os.listdir(pred_dir) if file.endswith(".mha")]

    df, filepath = read_existing_results(biggest_island, results_dir)
    num_files = len(files)
    i = 0
    if biggest_island:
        print(f"Evaluation of {dataset}")
    else:
        print(f"Evaluation of {dataset} using Biggest Islands")

    for file in files:
        if i % 10 == 0:
            print(f"Processing file {i}/{num_files}")
        i += 1
        if len(df) > 0:
            if file in df['file'].tolist():
                continue
        results = calc_metrics_for_file(file, biggest_island, pred_dir, gt_dir)
        if results is not None:
            row = pd.DataFrame([results])
            df = pd.concat([df, row])
            # Check if file exists, create if not
            if not os.path.exists(filepath):
                open(filepath, 'a').close()  # Create an empty file
            with open(filepath, 'a') as f:
                row.to_json(f, orient='records', lines=True)

    if 'file' in df.columns:
        df = df.drop(columns=["file"])

    # Calculate the mean, median, and standard deviation using numpy
    means = np.mean(df.to_numpy(), axis=0)
    medians = np.median(df.to_numpy(), axis=0)
    stds = np.std(df.to_numpy(), axis=0)

    ci = bootstrap((df.to_numpy(),), np.median, confidence_level=0.95, n_resamples=10000, method='percentile')
    ci_low = ci.confidence_interval.low
    ci_high = ci.confidence_interval.high

    # Create the metrics summary DataFrame
    metrics_summary = pd.DataFrame({
        "Metric": df.columns,
        "Median": medians,
        "CI Low": ci_low,
        "CI High": ci_high,
        "Mean": means,
        "Std": stds,
    }).reset_index(drop=True)

    with pd.option_context('display.float_format', '{:.4f}'.format):
        print("Summary (Mean and Std):")
        print(metrics_summary)

    filename = 'results.jsonl'
    if biggest_island:
        filename = 'island_results.jsonl'
    metrics_summary.to_json(os.path.join(results_dir, filename), orient='records', lines=True)
    return True


def evaluate_dataset(dataset):
    biggest_island = False
    # gt_dir = os.path.join('..', 'nnunet_raw', dataset, 'labelsTs')
    gt_dir = os.path.join('..', 'nnunet_raw', 'Dataset002_fluid', 'labelsTs')
    pred_dir = os.path.join('..', 'nnunet_results', dataset, 'predictions')
    results_dir = os.path.join('..', 'nnunet_results', dataset)
    evaluate(dataset, biggest_island, gt_dir, pred_dir, results_dir)

    biggest_island = True
    evaluate(dataset, biggest_island, gt_dir, pred_dir, results_dir)


parser = argparse.ArgumentParser(description="Update dataset JSON with training and testing files.")
parser.add_argument("dataset", type=str, help="Name of the dataset directory (e.g., Dataset003_fluid)")
args = parser.parse_args()
dataset = args.dataset
print(f"Create dataset.json file for: {dataset}")

evaluate_dataset(dataset)

