### ABC SYNTHETIC TESTING FOR SELECTION PARAMETER ONLY
import numpy as np
from utils import calculate_distance, get_best_indices, load_yaml, save_yaml
import time, os
import multiprocessing as mp
import datetime
from tqdm import tqdm
from functools import partial

start_time = time.time()

# PARAMETERS
s_range = [0.03, 0.8]
start = -5
runs_path = 'runs/P-5'

def worker_load(path):
    sampled = load_yaml(os.path.join(path, 'sampled.yaml'))
    simulations = load_yaml(os.path.join(path, 'simulations.yaml'))
    return sampled, simulations

def load_parallel():
    sampled_s = []
    abc_simulations_P4, abc_simulations_P15 = [], []    
    start_time = time.time()
    num_cores = 32

    print('Loading simulations in parallel...')
    with mp.Pool(processes=num_cores) as pool:
        split = [os.path.join(runs_path, run) for run in os.listdir(runs_path)]
        print(split)
        results = pool.map(worker_load, split)
    print('Parallel simulations loaded.')

    num_samples = 0
    for sampled, simulations in results:
        num_samples += sampled['num_samples']
        sampled_s.extend(sampled['sampled_s'])
        abc_simulations_P4.extend(simulations['P4'])
        abc_simulations_P15.extend(simulations['P15'])

    end_time = time.time()
    total_runtime = end_time - start_time

    print("Loading complete.\n")
    print(f"Total samples collected: {len(results)}")
    print(f"Total runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}\n")

    return sampled_s, abc_simulations_P4, abc_simulations_P15

# Load simulations to perform ABC inference
sampled_s, abc_simulations_P4, abc_simulations_P15 = load_parallel()
num_samples = len(sampled_s)

def abc_estimation(reference, abc_simulations_P4, abc_simulations_P15, num_samples, sampled_s, top_percent=5):
    distances = [0]*num_samples
    for i in range(num_samples):
        try:
            data = {'P4': abc_simulations_P4[i], 'P15': abc_simulations_P15[i]}
            distances[i] = calculate_distance(data, reference)
        except Exception as e:
            # might be an error if no cell has more than 10 copies at P4
            pass
    smallest_indices = get_best_indices(distances, top_percent)

    top_distances, top_s = [], []

    for i in smallest_indices:
        top_s.append(sampled_s[i])
        top_distances.append(distances[i])

    estimated_s = np.mean(top_s)

    return estimated_s

def worker_interval(i, data):
    args = data.copy()
    args['reference']={'P4':data['reference']['P4'][i],
                       'P4':data['reference']['P4'][i]}
    estimated_s = abc_estimation(**args)
    return float(estimated_s)

average_errors = []

for interval_folder in tqdm(os.listdir('synthetic/P-5')):
    s_path = os.path.join('synthetic/P-5', interval_folder)
    print(s_path)
    params = load_yaml(os.path.join(s_path, 'params.yaml'))
    sampled = load_yaml(os.path.join(s_path, 'sampled.yaml'))
    simulations = load_yaml(os.path.join(s_path, 'simulations.yaml'))
    print('loaded dictionaries')
    s_range_interval = params['s_range']
    sampled_s_interval = sampled['sampled_s']
    num_samples_interval = sampled['num_samples']
    simsP15, simsP4 = simulations['P15'], simulations['P4']

    data = {'reference':{'P4':simsP4, 'P15':simsP15},
            'abc_simulations_P4':abc_simulations_P4,
            'abc_simulations_P15':abc_simulations_P15,
            'num_samples':num_samples, 
            'sampled_s':sampled_s}
    
    partial_worker = partial(worker_interval, data=data)
    
    num_cores = 32
    with mp.Pool(processes=num_cores) as pool:
        split = list(range(num_samples_interval))
        estimated_s_values = pool.map(partial_worker, split)

    for true, estimated in zip(sampled_s_interval, estimated_s_values):
        print(f'true: {true:.3f}, estimation: {estimated:.3f}')

    errors_interval = [float(np.abs(estimated_s_values[i]-sampled_s_interval[i])) for i in range(num_samples_interval)]
    average_error_interval = np.mean(errors_interval)
    average_errors.append(float(average_error_interval))

    print(average_errors)

    save_yaml(errors_interval, file_path=f'results/P-5/{interval_folder}/errors.yaml')
    save_yaml(sampled_s_interval, file_path=f'results/P-5/{interval_folder}/sampled.yaml')
    save_yaml(estimated_s_values, file_path=f'results/P-5/{interval_folder}/estimated.yaml')


values = ['0.03-0.04', '0.04-0.05', '0.05-0.06', '0.06-0.07', '0.07-0.08']
save_yaml({values[i]: average_errors[i] for i in range(len(values))}, file_path=f'results/P-5/average_errors.yaml')
