import multiprocessing as mp
import os, time, datetime, argparse
import numpy as np
from model import *

from utils.load import save_yaml, get_save_folder
from utils.other import filter_distribution

class Run:
    def __init__(self, num_samples, expname, fitness='log', size=1000, top_percent=5):
        self.num_samples = num_samples
        print('Number of samples:', num_samples)
        self.fitness = fitness
        self.size = size
        self.top_percent = top_percent

        self.sampled_params = []
        self.simulated_data_list = []

        self.threshold = 10

        self.basepath = f'{expname}'
        self.run_folder = get_save_folder(self.basepath)

    def sample_parameters(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def run_one_simulation(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def worker(self, num_samples):
        np.random.seed()  # Set a unique seed for each worker process
        results = []
        for _ in range(num_samples):
            params = self.sample_parameters()
            simulated_data = self.run_one_simulation(**params)
            results.append((params, simulated_data))
        return results

    def run_simulations(self):
        start_time = time.time()
        num_cores = 32

        print(f'\nRunning simulations... (parallel with {num_cores} CPU cores, should take less than 5min for 100, less than 50min for 1000)')
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.worker, [1]*self.num_samples)
        print('Parallel simulations complete.')
        for sublist in results:
            for params, simulated_data in sublist:

                self.sampled_params.append(params)
                self.simulated_data_list_P4.append(filter_distribution(simulated_data['P4'], self.threshold))
                self.simulated_data_list_P15.append(filter_distribution(simulated_data['P15'], self.threshold))

        self.simulated_data = {'P4': self.simulated_data_list_P4, 'P15': self.simulated_data_list_P15}
        
        end_time = time.time()
        total_runtime = end_time - start_time

        print("Run complete.\n")
        print(f"Total samples collected: {len(self.sampled_params)}")
        print(f"Total runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}\n")
    
    def save_results(self, filename='results.yaml'):
        results = {'sampled_params': self.sampled_params, 'simulated_data': self.simulated_data_list}
        save_yaml(dictionary=results, file_path=os.path.join(self.run_folder, filename))


class SelectionRun(Run):

    def __init__(self, s_range, num_samples, expname, start, fitness='log', size=1000):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size, expname=expname)
        self.s_range = s_range
        print('s range:', s_range)
        self.start = start
        self.sampled_s = []
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []

    def sample_parameters(self):
        s = np.random.uniform(*self.s_range)
        return {'s': s}
    
    def run_one_simulation(self, s):
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

        simulations = {
            'P4': self.simulated_data_list_P4, 
            'P15': self.simulated_data_list_P15
        }
        save_yaml(dictionary=simulations, file_path=os.path.join(self.run_folder, 'simulations.yaml'))

        sampled = {
            'sampled_s': self.sampled_s,
            'num_samples': self.num_samples,
        }
        save_yaml(dictionary=sampled, file_path=os.path.join(self.run_folder, 'sampled.yaml'))

        save_yaml(dictionary={'start': self.start,'s_range': self.s_range}, 
                  file_path=os.path.join(self.run_folder, 'params.yaml'))


class DoubleRun(Run):

    def __init__(self, s_range, start_range, num_samples, expname, fitness='log', size=1000):
        super().__init__(num_samples=num_samples, fitness=fitness, size=size, expname=expname)
        self.s_range = s_range
        self.start_range = start_range
        self.simulated_data_list_P4 = []
        self.simulated_data_list_P15 = []

    def run_one_simulation(self, s:float, start:int):
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

        simulations = {
            'P4': self.simulated_data_list_P4, 
            'P15': self.simulated_data_list_P15
        }
        save_yaml(dictionary=simulations, file_path=os.path.join(self.run_folder, 'simulations.yaml'))

        sampled = {
            'sampled_s': self.sampled_s,
            'sampled_start': self.sampled_start,
            'num_samples': self.num_samples,
        }
        save_yaml(dictionary=sampled, file_path=os.path.join(self.run_folder, 'sampled.yaml'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ABC simulations')
    parser.add_argument('--expname', type=str, required=True, help='Name of the experiment. There must be a folder with this name in runs/.')
    parser.add_argument('--sample', type=str, required=True, choices=['selection', 'double'], 
                        help='double to sample both the selection parameter and the starting passage, simple to sample only the selection parameter given the starting passage.')
    parser.add_argument('--start', type=int, required=False, help='Starting passage when we sample only the selection parameter.')

    args = parser.parse_args()
    expname = args.expname
    sample = args.sample
    s_range = [0.01, 0.1]
    # s_range = [0.07, 0.08]
    # num_samples = 100
    num_samples = 1000

    if sample == 'selection':
        start = args.start
        run = SelectionRun(s_range=s_range, start=start, num_samples=num_samples, expname=expname)
    elif sample == 'double':
        start_range = [-9, 1]
        run = DoubleRun(s_range=s_range, start_range=start_range, num_samples=num_samples, expname=expname)

    run.run_simulations()
    run.save_results()