### ABC SYNTHETIC TESTING FOR SELECTION PARAMETER ONLY
import numpy as np
import time, os
import multiprocessing as mp
import datetime
from tqdm import tqdm
from functools import partial

from utils.other import calculate_distance, get_best_indices
from utils.load import load_yaml, save_yaml

# PARAMETERS
start = -4
runs_path = f'runs/P{start}' # simulations used for prior distribution

def worker_load(path):
    sampled = load_yaml(os.path.join(path, 'sampled.yaml'))
    simulations = load_yaml(os.path.join(path, 'simulations.yaml'))
    return sampled, simulations

def load_parallel():
    sampled_s = []
    abc_simulations_P4, abc_simulations_P15 = [], []    
    start_time = time.time()
    num_cores = 32

    print(f'Loading simulations for ABC prior... (parallel with {num_cores} CPU cores, should take less than 1min)')
    with mp.Pool(processes=num_cores) as pool:
        split = [os.path.join(runs_path, run) for run in os.listdir(runs_path)]
        results = pool.map(worker_load, split)

    num_samples = 0
    for sampled, simulations in results:
        num_samples += sampled['num_samples']
        sampled_s.extend(sampled['sampled_s'])
        abc_simulations_P4.extend(simulations['P4'])
        abc_simulations_P15.extend(simulations['P15'])

    end_time = time.time()
    total_runtime = end_time - start_time

    print(f"Total samples collected: {num_samples}")
    print(f"Loading complete. Runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}\n")

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

print('Starting inference... (parallel with 32 CPU cores, should take less than 1min per interval of 100 samples)')
for run in tqdm(os.listdir(f'synthetic/simulations/P{start}')):
    s_path = os.path.join(f'synthetic/simulations/P{start}', run)
    print('\nRunning ABC inference with references from path', s_path)
    params = load_yaml(os.path.join(s_path, 'params.yaml'))
    sampled = load_yaml(os.path.join(s_path, 'sampled.yaml'))
    simulations = load_yaml(os.path.join(s_path, 'simulations.yaml'))
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

    errors_interval = [float(np.abs(estimated_s_values[i]-sampled_s_interval[i])) for i in range(num_samples_interval)]
    average_error_interval = np.mean(errors_interval)
    average_errors.append(float(average_error_interval))

    interval = params['s_range'][0]
    main_path = f'synthetic/results/P{start}/s{interval}/'
    save_yaml(errors_interval, file_path=main_path+'errors.yaml')
    save_yaml({'sampled': sampled_s_interval, 'estimated': estimated_s_values}, file_path=main_path+'s.yaml')

values = ['0.03-0.04', '0.04-0.05', '0.05-0.06', '0.06-0.07', '0.07-0.08']
errors_dict = {values[i]: average_errors[i] for i in range(len(values))}
print(errors_dict)
save_yaml(errors_dict, file_path=f'synthetic/results/P{start}/average_errors.yaml')
