import multiprocessing as mp
from scipy.stats import wasserstein_distance
import os, time, datetime, argparse
import numpy as np
from utils import normalize_distribution, filter_distribution, load_yaml, save_yaml, get_save_folder, get_best_indices
from model import *

class ABCInference:
    def __init__(self, num_samples, expname, fitness='log', size=1000, top_percent=5, synthetic=False):
        self.num_samples = num_samples
        print('Number of samples:', num_samples)
        self.fitness = fitness
        self.size = size
        self.top_percent = top_percent

        self.sampled_params = []
        self.simulated_data_list = []
        self.distances = []
        self.distancesP4 = []
        self.distancesP15 = []

        self.threshold = 10
        self.synthetic = synthetic

        self.basepath = f'experiments/{expname}/'
        self.run_folder = get_save_folder(self.basepath + 'runs/')

    def load_reference(self):
        reference_files = {'P4': 'cell_counts_p4.yaml', 'P15': 'cell_counts_p15.yaml'}
        self.reference_data = {key: filter_distribution(load_yaml(self.basepath + 'data/' + file), self.threshold)
                               for key, file in reference_files.items()}
        print(f'\nReference data filtered with threshold {self.threshold}.')

        for key, file in reference_files.items(): print(key,':', self.basepath + 'data/' + file)

    def sample_parameters(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def run_simulation(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def calculate_distance(self, simulated_data):
        '''weighted sum of Wasserstein distances at P4 and P15'''
        distances = {}
        total_distance = 0
        weights = {'P4':2, 'P15':0.5}
        
        for key in self.reference_data.keys():
            reference_normalized = normalize_distribution(self.reference_data[key])
            simulated_normalized = normalize_distribution(simulated_data[key])
            values1, weights1 = zip(*reference_normalized.items())
            values2, weights2 = zip(*simulated_normalized.items())
            distance = float(wasserstein_distance(values1, values2, u_weights=weights1, v_weights=weights2))
            distances[key] = distance
            total_distance += weights[key] * distance

        return total_distance, distances['P4'], distances['P15']
    
    def worker(self, num_samples):
        np.random.seed()  # Set a unique seed for each worker process
        results = []
        for _ in range(num_samples):
            params = self.sample_parameters()
            simulated_data = self.run_simulation(**params)
            distance = self.calculate_distance(simulated_data)
            results.append((params, distance, simulated_data))
        return results

    def perform_inference(self):
        start_time = time.time()
        num_cores = 32

        print('\nRunning simulations in parallel...')
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.worker, [1]*self.num_samples)

        for sublist in results:
            for params, distances, simulated_data in sublist:

                distance, distanceP4, distanceP15 = distances

                self.sampled_params.append(params)
                self.simulated_data_list_P4.append(filter_distribution(simulated_data['P4'], self.threshold))
                self.simulated_data_list_P15.append(filter_distribution(simulated_data['P15'], self.threshold))
                self.distances.append(distance)
                self.distancesP4.append(distanceP4)
                self.distancesP15.append(distanceP15)

        self.simulated_data = {'P4': self.simulated_data_list_P4, 'P15': self.simulated_data_list_P15}
        
        end_time = time.time()
        total_runtime = end_time - start_time

        print("Run complete.\n")
        print(f"Total samples collected: {len(self.sampled_params)}")
        print(f"Total runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}\n")
    
    def save_results(self, filename='results.yaml'):
        results = {'sampled_params': self.sampled_params, 'simulated_data': self.simulated_data_list,'distances': self.distances}
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, filename))


class SelectionInference(ABCInference):

    def __init__(self, s_range, num_samples, expname, start=-5, fitness='log', size=1000, synthetic=False):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size, synthetic=synthetic, expname=expname)
        self.s_range = s_range
        self.start = start
        self.sampled_s = []
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []

    def sample_parameters(self):
        s = np.random.uniform(*self.s_range)
        return {'s': s}
    
    def run_simulation(self, s):
        events_per_passage = 9
        n_passages = 15 - self.start
        n_events = events_per_passage * n_passages * self.size

        population = Population(fitness=self.fitness, s=s)
        population.simulate_moran(size=self.size, n_events=n_events, verbose=False, disable=True)

        # Retrieve cell counts at P4 and P15
        population_P4 = population.cell_counts[events_per_passage * (4 - self.start) * self.size]
        population_P15 = population.cell_counts[-1]

        return {'P4': population_P4, 'P15': population_P15}

    def save_results(self):
        self.sampled_s = [params['s'] for params in self.sampled_params]

        simulations = {'P4': self.simulated_data_list_P4, 'P15': self.simulated_data_list_P15}
        save_yaml(dictionary=simulations, file_path=os.path.join(self.run_folder, 'simulations.yaml'))

        results = {
            'sampled_s': self.sampled_s,
            'distances': self.distances,
            'distancesP4': self.distancesP4,
            'distancesP15': self.distancesP15,
            'num_samples': self.num_samples,
        }
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, 'results.yaml'))
    

class DoubleInference(ABCInference):

    def __init__(self, s_range, start_range, num_samples, expname, fitness='log', size=1000, synthetic=False):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size, synthetic=synthetic, expname=expname)
        self.s_range = s_range
        self.start_range = start_range
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []

    def run_simulation(self, s:float, start:int):
        events_per_passage = 9
        n_passages = 15 - start
        n_events = events_per_passage * n_passages * self.size

        population = Population(fitness=self.fitness, s=s)
        population.simulate_moran(size=self.size, n_events=n_events, verbose=False, disable=True)

        # Retrieve cell counts at P4 and P15
        population_P4 = population.cell_counts[events_per_passage * (4 - start) * self.size]
        population_P15 = population.cell_counts[-1]

        return {'P4': population_P4, 'P15': population_P15}

    def sample_parameters(self):
        s = np.random.uniform(*self.s_range)
        start = np.random.randint(*self.start_range)
        return {'s': s, 'start': start}

    def save_results(self):
        self.sampled_s = [params['s'] for params in self.sampled_params]
        self.sampled_start = [params['start'] for params in self.sampled_params]

        simulations = {'P4': self.simulated_data_list_P4, 'P15': self.simulated_data_list_P15}
        save_yaml(dictionary=simulations, file_path=os.path.join(self.run_folder, 'simulations.yaml'))

        results = {
            'sampled_s': self.sampled_s,
            'sampled_start': self.sampled_start,
            'distances': self.distances,
            'distancesP4': self.distancesP4,
            'distancesP15': self.distancesP15,
            'num_samples': self.num_samples,
        }
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, 'results.yaml'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ABC simulations')
    parser.add_argument('--expname', type=str, required=True, help='Name of the experiment. There must be a folder with this name in experiments/.')
    parser.add_argument('--inference', type=str, required=True, choices=['selection', 'double'], 
                        help='Type of inference. double to infer both the selection parameter and the starting passage, simple to infer only the selection parameter given the starting passage.')
    parser.add_argument('--start', type=int, required=False, help='Starting passage when we sample only the selection parameter.')

    args = parser.parse_args()
    expname = args.expname
    inference = args.inference

    s_range = [0.01, 0.11]
    start_range = [-9, 1]
    num_samples = 1000
    if expname!='CAM277':
        params = load_yaml(f'experiments/{expname}/data/params.yaml')
        start, sref = params['start'], params['s']

    if inference == 'selection':
        abc_inference = SelectionInference(s_range=s_range, start=start, num_samples=num_samples, synthetic=True, expname=expname)
    elif inference == 'double':
        abc_inference = DoubleInference(s_range=s_range, start_range=start_range, num_samples=num_samples, synthetic=True, expname=expname)

    abc_inference.load_reference()
    abc_inference.perform_inference()
    abc_inference.save_results()