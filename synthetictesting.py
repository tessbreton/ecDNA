print('imports')
import numpy as np
from model import Population
from utils import generate_intervals, load_yaml, normalize_distribution, get_best_indices, calculate_distance
from tqdm import tqdm 
import os 
from utils import save_yaml
import time

start_time = time.time()
# PARAMETERS
s_range = [0.01, 0.11]
start_range = [-9, 1]
num_simulations = 10
s_intervals = generate_intervals(s_range)
runs_path = 'experiments/synthetic1double/runs'
num_samples = 0
sampled_s, sampled_start = [], []
abc_simulations_P4, abc_simulations_P15 = [], []

for run_folder in tqdm(os.listdir(runs_path)):
        run_path = os.path.join(runs_path, run_folder)
        
        results_file_path = os.path.join(run_path, 'results.yaml')
        
        if os.path.isfile(results_file_path):

            print('loading results from', results_file_path)
            results = load_yaml(results_file_path)
            simulations = load_yaml(os.path.join(run_path, 'simulations.yaml'))

            num_samples += results['num_samples']

            sampled_s.extend(results['sampled_s'])
            sampled_start.extend(results['sampled_start'])

            abc_simulations_P4.extend(simulations['P4'])
            abc_simulations_P15.extend(simulations['P15'])

print('loading complete.')

# GENERATE SYNTHETIC DATA
def synthetic_data(s, start, size=1000, fitness='log'):
    EVENTS_PER_PASSAGE = 9
    N_PASSAGES = 15 - (start)
    n_events = EVENTS_PER_PASSAGE * N_PASSAGES * size
    n_eventsP4 = EVENTS_PER_PASSAGE * (4-start) * size

    population = Population(fitness=fitness, s=s)
    population.simulate_moran(size=size, n_events=n_events, verbose=False)

    return {'P4': population.cell_counts[n_eventsP4], 'P15':population.cell_counts[-1]}


def abc_estimation(reference, abc_simulations_P4, abc_simulations_P15, top_percent=5):
    print('starting abc estimation of parameters')
    distances = [0]*num_samples
    for i in range(num_samples):
        try:
            data = {'P4': abc_simulations_P4[i], 'P15': abc_simulations_P15[i]}
            distances[i] = calculate_distance(data, reference)
        except Exception as e:
            pass
            # print(f"Erreur avec l'index {i} : {e}")
            # print(f"Référence : {reference}")
            # print(f"Data : { {'P4': abc_simulations_P4[i], 'P15': abc_simulations_P15[i]}}")
    smallest_indices = get_best_indices(distances, top_percent)

    top_distances, top_s, top_start = [], [], []

    for i in smallest_indices:
        top_s.append(sampled_s[i])
        top_start.append(sampled_start[i])
        top_distances.append(distances[i])

    estimated_s = np.mean(top_s)
    estimated_start = np.mean(top_start)

    return estimated_s, estimated_start

array_shape = (len(s_intervals), len(range(*start_range)))
results_s = np.zeros(array_shape)
results_start = np.zeros(array_shape)

for a, s_interval in enumerate(s_intervals):

    for b, start in enumerate(list(range(*start_range))):
        print(s_interval,start)

        errors_s = [0]*num_simulations
        errors_start = [0]*num_simulations
        simulations = []

        for i in range(num_simulations):
            s = np.random.uniform(*s_interval)
            synthetic = synthetic_data(s, start)
            simulations.append(synthetic)
            estimated_s, estimated_start = abc_estimation(synthetic, abc_simulations_P4, abc_simulations_P15)
            print(f'true = {start}, estimated = {estimated_start}')
            print(f'true = {s}, estimated = {estimated_s}')
            errors_s[i] = np.abs(estimated_s-s)
            errors_start[i] = np.abs(estimated_start-start) # can't take relative error for start at 0, and not relevant for 1 either... so just absolute OK

        results_s[a,b] = np.mean(errors_s)
        results_start[a,b] = np.mean(errors_start)

        save_yaml(simulations, f'synthetic testing/data/{a}{b}/simulations.yaml')

print(results_start)
print(results_s)
np.save('errors_start.npy', results_start)
np.save('errors_s.npy', results_s)

start_values = list(range(*start_range))
save_yaml(
    dictionary={'s_intervals': s_intervals,'start_values': start_values},
    file_path='synthetic testing/synthetic.yaml'
    )

print('runtime:', time.time()-start_time)