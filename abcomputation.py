from scipy.stats import wasserstein_distance
from ecDNA import *
from utils import *
import numpy as np
from math import ceil

# ABC -----------------------------------------------------------------------------------------------------------------------------------------------

def normalize_distribution(dist):
    total_count = sum(dist.values())
    return {k: v / total_count for k, v in dist.items()}


def compute_distance(d1, d2, distance_type='L1'):
    # Normalize distributions
    d1_normalized = normalize_distribution(d1)
    d2_normalized = normalize_distribution(d2)
    
    # Ensure both distributions have the same set of keys
    all_keys = set(d1_normalized.keys()) | set(d2_normalized.keys())
    
    # Compute distance based on distance_type
    if distance_type == 'L1':
        distance = sum(abs(d1_normalized.get(k, 0) - d2_normalized.get(k, 0)) for k in all_keys)
    elif distance_type == 'L2':
        distance = np.linalg.norm(np.array([d1_normalized.get(k, 0) for k in all_keys]) - np.array([d2_normalized.get(k, 0) for k in all_keys]))
    elif distance_type == 'Wasserstein':
        # Extract values and weights for each distribution
        values1, weights1 = zip(*d1_normalized.items())
        values2, weights2 = zip(*d2_normalized.items())

        # Compute Wasserstein distance
        distance = wasserstein_distance(values1, values2, u_weights=weights1, v_weights=weights2)

    else:
        raise ValueError("Unsupported distance type. Choose from 'L1' or 'L2' or 'Wasserstein'.")
    
    return distance


# Define the function to simulate data
def simulate_data(n_events, fitness, s, size=1000):
    population = Population(fitness=fitness, s=s)
    population.simulate_moran(size=size, n_events=n_events, verbose=False)
    return population.cell_counts[-1]

# Define the ABC algorithm
def abc(target_data, n_events, fitness, metric='L1', range_s=(0, 1), num_samples=3, num_simulations=1, epsilon=10, plot_all=False):
    best_s = None
    best_distance = float('inf')
    best_simulated_data = None
    s_values = []
    distances = []
    simulated_data_all = []

    for i in tqdm(range(num_samples)):
        # Sample parameter value
        s = np.random.uniform(range_s[0], range_s[1])

        # Store distances for each simulation
        simulation_distances = []

        # Perform multiple simulations for each sampled value of s
        if num_simulations > 1:
            for _ in range(num_simulations):
                # Simulate data
                simulated_data = simulate_data(n_events, fitness, s)

                # Compute distance
                distance = compute_distance(simulated_data, target_data, metric)
                simulation_distances.append(distance)

            # Choose the best distance among the simulations
            distance = min(simulation_distances)

        else:
            # Simulate data
            simulated_data = simulate_data(n_events, fitness, s)

            # Compute distance
            distance = compute_distance(simulated_data, target_data, metric)
        
        if plot_all:
            plot_histograms_dict_overlay(dictionaries=[simulated_data],
                                         true_data=target_data,
                                         plot_true=True,
                                        bin_size=10,
                                        labels=(f'Simulation {i+1}, distance {distance:.2f}',''),
                                        colors=('deepskyblue', ''),
                                        title=f'Normalized distributions of ecDNA counts at Passage 4 and simulation {i+1} with {fitness} fitness (s={s:.2f})')
            
        s_values.append(s)
        distances.append(distance)
        simulated_data_all.append(simulated_data)

        if distance < best_distance:
            best_distance = distance
            best_s = s
            best_simulated_data = simulated_data
        


    return best_s, best_distance, best_simulated_data, s_values, distances, simulated_data_all


# Define a function to run simulation for each best s value
def run_simulations(best_s_values, n_events, fitness, num_simulations=1):
    simulated_data_list = []
    for s in tqdm(best_s_values):
        # Perform multiple simulations for each best s value
        for _ in range(num_simulations):
            # Simulate data
            simulated_data = simulate_data(n_events, fitness, s)
            simulated_data_list.append(simulated_data)
    return simulated_data_list


def get_best_indices(distances, top_percent):
    '''Find indices corresponding to {top_percent}% smallest distances'''
    top_percent = 5
    num_points = len(distances)
    num_smallest_points = ceil(0.01 * top_percent * num_points)
    smallest_indices = np.argsort(distances)[:num_smallest_points]
    smallest_indices = smallest_indices.tolist()
    return smallest_indices


# PLOTS -----------------------------------------------------------------------------------------------------------------------------------------------

def plot_best_points(s_values, 
                     distances, 
                     top_percent=5, 
                     best_color='turquoise',
                     save_fig=False, 
                     width=700, 
                     height=400, 
                     filepath='best_points.png', 
                     scale=3):
    fig = go.Figure()  # Initialize the figure

    # Get indices of the best points (smallest distances)
    smallest_indices = get_best_indices(distances, top_percent)
    
    # Plot all points in black
    fig.add_trace(go.Scatter(
        x=s_values,
        y=distances,
        mode='markers',
        marker=dict(color='black', size=5),
        name='All points'
    ))

    # Plot points corresponding to 5% smallest distances in blue
    fig.add_trace(go.Scatter(
        x=np.array(s_values)[smallest_indices],
        y=np.array(distances)[smallest_indices],
        mode='markers',
        marker=dict(color=best_color, size=8),
        name='5% smallest distances'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='s',
        yaxis_title='Wasserstein Distance',
        plot_bgcolor='white'
    )

    # Show the plot
    fig.show()


    if save_fig:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)