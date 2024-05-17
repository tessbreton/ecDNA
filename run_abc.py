import argparse
from ecDNA import *
from utils import *
from abcomputation import *
import os

class RunABC:
    def __init__(self, num_samples=2, fitness='log', top_percent=5, metric='Wasserstein'):
        
        # Run params
        self.num_samples = num_samples
        self.fitness = fitness
        self.top_percent = top_percent
        self.metric = metric

        self.run_folder = None

        # Run results
        self.s_values = None
        self.distances = None
        self.true_data = None
        self.best_s = None
        self.best_simulated_data = None

    def load_true_data(self):
        file_path_p4 = 'data/cell_counts_p4.yaml'
        self.true_data = load_yaml(file_path_p4)

    def run_simulation(self):
        # Parameters
        EVENTS_PER_PASSAGE = 9
        N_PASSAGES = 9
        N_CELLS = 1000
        range_s = [0.015,0.12]
        n_events = EVENTS_PER_PASSAGE * N_PASSAGES * N_CELLS

        self.run_folder = get_save_folder(base_path='runs')

        # CALL ABC ALGORITHM
        best_s, _, self.best_simulated_data, self.s_values, self.distances, self.simulated_data_all = abc(
            target_data=self.true_data, 
            n_events=n_events, 
            metric=self.metric, 
            fitness=self.fitness, 
            range_s=range_s, 
            num_samples=self.num_samples, 
            num_simulations=1, 
            save_results=True,
            run_folder=self.run_folder
        )

        self.best_s = best_s

    def get_top_simulations(self):
        smallest_indices = get_best_indices(self.distances, self.top_percent)
        top_simulations = np.array(self.simulated_data_all)[smallest_indices].tolist()
        return top_simulations
    
    def plot_best_simulation(self):
        plot_histograms_dict_overlay(
            dictionaries=[self.best_simulated_data], true_data=self.true_data, 
            plot_ref=True, labels=['Best simulation'], colors=['turquoise'],
            title=f'Normalized distributions of ecDNA counts at Passage 4 and best simulation with {self.fitness} fitness (s={self.best_s:.3f})',
            show=False, save=True, filepath=os.path.join(self.run_folder, 'plots', 'bestsimulation.png')
        )

    def plot_best_points(self):
        plot_best_points_util(
            self.s_values, self.distances, self.metric, top_percent=self.top_percent, 
            save=True, show=False, filepath=os.path.join(self.run_folder, 'plots', 'bestpoints.png')
        )

    def plot_posterior(self):
        plot_posterior_util(
            self.s_values, self.distances, top_percent=self.top_percent, 
            save_fig=True, plot_color='coral', show=False, 
            filepath=os.path.join(self.run_folder, 'plots', 'posterior.png')
        )

    def plot_top_simulations(self):
        plot_histograms_dict_overlay(
            dictionaries=self.get_top_simulations(), true_data=self.true_data, bin_size=10,
            plot_ref=True, true_label='Reference data (passage 4)', title=f'ecDNA counts at Passage 4 and {self.top_percent}% best simulations (over {self.num_samples} samples) with {self.fitness} fitness',
            plot_avg=True, confidence_level=5, show=False, save=True, 
            filepath=os.path.join(self.run_folder, 'plots', 'topsimulationsavg.png')
        )

    def plot_results(self):
        self.plot_best_simulation()
        self.plot_best_points()
        self.plot_posterior()
        self.plot_top_simulations()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Moran process simulations of ecDNA count evolution, and save simulation results.')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of s sampled for ABC')
    parser.add_argument('--fitness', type=str, default='log', help='Fitness function used in Moran process')
    parser.add_argument('--top_percent', type=float, default=5, help='Top percent best simulations to keep in ABC algorithm')
    parser.add_argument('--metric', type=str, default='Wasserstein', help='Metric used in ABC to compare simulations to reference data')
    args = parser.parse_args()

    # Instantiate and run the simulation
    run = RunABC(num_samples=args.num_samples, fitness=args.fitness, top_percent=args.top_percent, metric=args.metric)
    run.load_true_data()
    run.run_simulation()
    run.plot_results()
