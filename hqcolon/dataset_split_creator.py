"""
This file is used to create balanced splits from the extracted existing dataset mapping.

We aim to find a balanced split accroding to supine / prone balance and according to male/female balance.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

HOME_ERDA = '/home/amin/ucph-erda-home/IRE-DATA/CT'  # Path to ERDA


def plot_histogram(df, column_to_plot, show=False):
    # Create a histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column_to_plot], bins=10, color='blue', edgecolor='black')

    # Add labels and title
    plt.title(f'Histogram of {column_to_plot}', fontsize=16)
    plt.xlabel(column_to_plot, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    if show:
        plt.show()


def load_data():
    mapping_df = pd.read_json('name_mapping.json', lines=True)
    plot_histogram(mapping_df, 'Sex')
    plot_histogram(mapping_df, 'Position')
    mapping_df['Sex'] = mapping_df['Sex'].replace({'U': 'Not available', 'O': 'Not available'})
    plot_histogram(mapping_df, 'Sex')
    ffdr = mapping_df[mapping_df['Position'] == 'ffdr']
    print(f'There are {len(ffdr)} ffdr cases (added to prone)')
    hfdr = mapping_df[mapping_df['Position'] == 'hfdr']
    print(f'There are {len(hfdr)} hfdr cases (added to prone)')
    mapping_df['Position'] = mapping_df['Position'].replace({'ffdr': 'prone', 'hfdr': 'prone'})
    return mapping_df


def split_df(df, strat_columns=['Sex', 'Position']):
    df['Stratify'] = df[strat_columns].agg('-'.join, axis=1)

    # Split into train + temp, and then split temp into validation and test
    train_df, test_df = train_test_split(df, test_size=0.333, stratify=df['Stratify'], random_state=42)

    # Drop the helper column
    for split_df in [train_df, test_df]:
        split_df.drop(columns='Stratify', inplace=True)

    print(f"Test: {len(test_df)}, Train: {len(train_df)}, Ratio should be around 1:2")

    train_df.to_json('train_mapping.json', orient='records', lines=True)
    test_df.to_json('test_mapping.json', orient='records', lines=True)


df = load_data()
###############################################################################################################
# Adapt the following columns according to what characteristics you want to split your dataset ex. ['Sex', 'Position']
strat_columns = ['Sex', 'Position']
split_df(df, strat_columns=strat_columns)
