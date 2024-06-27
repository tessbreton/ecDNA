# OTHER UTILS -----------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from tqdm import tqdm
import pandas as pd
from math import ceil 
from scipy.stats import wasserstein_distance

def dict_to_values(d):
    values = []
    for k, v in d.items():
        values.extend([k] * v)
    return values

def normalize_distribution(distribution:dict):
    total_count = sum(distribution.values())
    return {k: v / total_count for k, v in distribution.items()}

def filter_distribution(distribution, threshold):
    return {k: v for k, v in distribution.items() if k >= threshold}

def filter_and_normalize(distribution, threshold=10):
    return normalize_distribution(filter_distribution(distribution, threshold))

def standard_score(distances):
    mean = float(np.mean(distances))
    std = float(np.std(distances))
    return [(d - mean) / std for d in distances]

def get_best_indices(distances, top_percent):
    '''Find indices corresponding to {top_percent}% smallest distances'''
    num_points = len(distances)
    num_smallest_points = ceil(0.01 * top_percent * num_points)
    smallest_indices = np.argsort(distances)[:num_smallest_points]
    smallest_indices = smallest_indices.tolist()
    return smallest_indices

def calculate_distance(data, reference):
    '''weighted sum of Wasserstein distances at P4 and P15'''
    total_distance = 0
    weights = {'P4':2, 'P15':0.5}
    
    for key in reference.keys():
        reference_normalized = normalize_distribution(reference[key])
        data_normalized = normalize_distribution(data[key])
        values1, weights1 = zip(*reference_normalized.items())
        values2, weights2 = zip(*data_normalized.items())
        distance = float(wasserstein_distance(values1, values2, u_weights=weights1, v_weights=weights2))
        total_distance += weights[key] * distance

    return total_distance

def generate_intervals(s_range, step=0.01):
    start = s_range[0]
    stop = s_range[1]
    intervals = []

    while start < stop:
        next_value = round(start + step, 2)
        if next_value > stop:
            break
        intervals.append([round(start, 2), next_value])
        start = next_value

    return intervals

def get_boundaries(group_limits, df):
    boundaries = []

    for i in range(len(group_limits) + 1):

        if i == 0: lower_bound = 1
        else: lower_bound = group_limits[i - 1] + 1

        if i == len(group_limits): upper_bound = df.shape[1]
        else: upper_bound = group_limits[i] + 1

        boundaries.append((lower_bound, upper_bound))

    return boundaries

def get_labels(boundaries):
    labels = []
    for i,boundary in enumerate(boundaries):
        lower_bound, upper_bound = boundary
        if i==len(boundaries)-1:
            labels.append(f'{lower_bound}+ copies')
        else: 
            if lower_bound == upper_bound - 1 :
                if lower_bound==1:
                    labels.append('1 copy')
                else:
                    labels.append(f'{lower_bound} copies')
                
            else:
                labels.append(f'{lower_bound}-{upper_bound-1} copies')
    return labels + ['no copy']

def to_dataframe(times, cell_counts):
    max_copies = max(list(cell_counts[-1].keys()))

    # Create an empty DataFrame with columns for each copy count
    df = pd.DataFrame(0, columns=[i for i in range(max_copies + 1)], index=times)

    # Fill the DataFrame with counts from cell_counts
    for time, subdict in tqdm(zip(times, cell_counts)):
        for copies, count in subdict.items():
            df.at[time, copies] = count

    return df