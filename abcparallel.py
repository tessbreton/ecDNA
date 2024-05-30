import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import os
import numpy as np
from utils import normalize_distribution, load_yaml, save_yaml, get_save_folder, plot_histograms_avg
from ecDNA import *
from abcomputation import get_best_indices
import time, datetime

class ABCInference:
    def __init__(self, num_samples, fitness='log', size=1000, top_percent=5):
        self.num_samples = num_samples
        self.fitness = fitness
        self.size = size
        self.run_folder = None
        self.top_percent = top_percent

        self.sampled_params = []
        self.simulated_data_list = []
        self.distances = []

    def load_reference(self):        
        referenceP4 = load_yaml('data/cell_counts_p4.yaml')
        referenceP15 = load_yaml('data/cell_counts_p15.yaml')
        self.reference_data = {'P4': referenceP4, 'P15': referenceP15}

    def run_simulation(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def calculate_distance(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def sample_parameters(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
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

        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.worker, [1]*self.num_samples)

        for sublist in results:
            for params, distance, simulated_data in sublist:

                self.sampled_params.append(params)
                self.simulated_data_list_P4.append(simulated_data['P4'])
                self.simulated_data_list_P15.append(simulated_data['P15'])
                self.distances.append(distance)

        self.simulated_data = {'P4': self.simulated_data_list_P4, 'P15': self.simulated_data_list_P15}
        
        end_time = time.time()
        total_runtime = end_time - start_time

        print("Inference complete.")
        print(f"Total samples collected: {len(self.sampled_params)}")
        print(f"Total runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}")
    
    def save_results(self, filename='results.yaml'):
        results = {
            'sampled_params': self.sampled_params,
            'simulated_data': self.simulated_data_list,
            'distances': self.distances,
        }
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, filename))

    def get_top_simulations(self):
        smallest_indices = get_best_indices(self.distances, self.top_percent)

        top_simulated_data_P4, top_simulated_data_P15 = [], []
        top_s, top_start = [], []
        top_distances = []

        # use one single for loop
        for i in smallest_indices:
            top_simulated_data_P4.append(self.simulated_data_list_P4[i])
            top_simulated_data_P15.append(self.simulated_data_list_P15[i])
            top_s.append(self.sampled_s[i])
            top_start.append(self.sampled_start[i])
            top_distances.append(self.distances[i])

        self.top_simulated_data = {'P4': top_simulated_data_P4, 'P15': top_simulated_data_P15}
        self.top_s, self.top_start = top_s, top_start
        self.top_distances = top_distances

        
    def plot_results(self):
        raise NotImplementedError("This method should be implemented in a subclass")


class SelectionInference(ABCInference):

    def __init__(self, s_range, num_samples, start=-5, fitness='log', size=1000):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size)
        self.s_range = s_range
        self.start = start
        self.run_folder = get_save_folder(base_path='runs/selection/parallel')
        self.distances = []
        self.sampled_s = []
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []
        self.distances_split = {'P4':[], 'P15':[]}

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

    def save_results(self, filename='results.yaml'):
        results = {
            'sampled_s': [params['s'] for params in self.sampled_params],
            'simulated_data': self.simulated_data_list,
            'distances': self.distances,
        }
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, filename))
    

class DoubleInference(ABCInference):

    def __init__(self, s_range, start_range, num_samples, fitness='log', size=1000):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size)
        self.s_range = s_range
        self.start_range = start_range
        self.run_folder = get_save_folder(base_path='runs/double/parallel')
        self.distances_split = {'P4':[], 'P15':[]}
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []

    def run_simulation(self, s, start):
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
    
    def calculate_distance(self, simulated_data):
        '''weighted sum of Wasserstein distances at P4 and P15'''
        total_distance = 0
        weights = {'P4':2, 'P15':0.5}
        
        for key in self.reference_data.keys():
            reference_normalized = normalize_distribution(self.reference_data[key])
            simulated_normalized = normalize_distribution(simulated_data[key])
            values1, weights1 = zip(*reference_normalized.items())
            values2, weights2 = zip(*simulated_normalized.items())
            distance = float(wasserstein_distance(values1, values2, u_weights=weights1, v_weights=weights2))
            self.distances_split[key].append(distance)

            total_distance += weights[key]*distance
        return total_distance

    def save_results(self):
        results = {
            'sampled_s': [params['s'] for params in self.sampled_params],
            'sampled_start': [params['start'] for params in self.sampled_params],
            'distances': self.distances,
            'distances_split': self.distances_split,
            'num_samples': self.num_samples,
        }
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, 'results.yaml'))

        top_results = {'top_simulated_data': self.top_simulated_data,
                                'top_s': self.top_s,
                                'top_start': self.top_start,
                                'num_samples': self.num_samples,
                                'top_distances': self.top_distances}
        save_yaml(dictionary=top_results,
                  file_path=os.path.join(self.run_folder, 'topresults.yaml'))

    def plot_results(self):
        # plot average of {top_percent}% best simulations
        for P in self.reference_data.keys():
            title = f'ecDNA counts at {P} and {self.top_percent}% best simulations (over {self.num_samples} samples)'
            reference_label = f'Reference data ({P})'
            plot_histograms_avg(self.top_simulated_data[P], self.reference_data[P], reference_label, title, filepath=os.path.join(self.run_folder, 'plots', f'average{P}.png'))


# Sampling parameters for two parameters
s_range = [0.015, 0.11]
start_range = [-8, 1]
num_samples = 10

abc_inference = DoubleInference(s_range, start_range, num_samples)
abc_inference.load_reference()
abc_inference.perform_inference()
abc_inference.get_top_simulations()
abc_inference.save_results()
abc_inference.plot_results()
