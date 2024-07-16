import multiprocessing as mp
from scipy.stats import wasserstein_distance
import os, time, datetime, argparse
from model import *

from utils.load import load_yaml, save_yaml
from utils.plot import plot_histograms_dict_overlay, plot_best_points_util, plot_posterior, qqplot, plot_ecdfs
from utils.other import normalize_distribution, filter_distribution, get_best_indices

class ABCInference:
    def __init__(self, expname, fitness='log', size=1000, top_percent=5):
        self.fitness = fitness
        self.size = size
        self.top_percent = top_percent

        self.sampled_params = []
        self.simulated_data_list = []
        self.distances = []
        self.distancesP4 = []
        self.distancesP15 = []

        self.threshold = 10

        self.basepath = f'experiments/{expname}/'
        self.runs_path = 'runs/P-5'

    def load_reference(self):
        reference_files = {'P4': 'cell_counts_p4.yaml', 'P15': 'cell_counts_p15.yaml'}
        self.reference_data = {key: filter_distribution(load_yaml(self.basepath + 'data/' + file), self.threshold)
                               for key, file in reference_files.items()}
        print(f'\nReference data filtered with threshold {self.threshold}.')

        for key, file in reference_files.items(): print(key,':', self.basepath + 'data/' + file)

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
    
    def worker_distance(self, i):
        results = []
        try:
            simulated_data = {'P4': self.simulationsP4[i], 'P15': self.simulationsP15[i]}
            distances = self.calculate_distance(simulated_data)
            results.append(distances)
        except:
            pass
        return results

    def perform_inference(self):
        print('Starting ABC inference...')
        start_time = time.time()
        num_cores = 32

        # COMPUTE DISTANCES ---------------------------------------------------------------------------------------------------------------------------------------------

        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(self.worker_distance, list(range(self.num_samples)))

        for distances in results:
            distance, distanceP4, distanceP15 = distances[0]
            self.distances.append(distance)
            self.distancesP4.append(distanceP4)
            self.distancesP15.append(distanceP15)

        self.simulated_data = {'P4': self.simulationsP4, 'P15': self.simulationsP15}
        
        end_time = time.time()
        total_runtime = end_time - start_time

        print(f"Inference complete. Runtime: {str(datetime.timedelta(seconds=int(total_runtime)))}\n")

        # FIND TOP RESULTS ---------------------------------------------------------------------------------------------------------------------------------------------

        top_percent = 5
        smallest_indices = get_best_indices(self.distances, top_percent)

        top_simulationsP4, top_simulationsP15 = [], []
        top_distances, top_distancesP4, top_distancesP15 = [], [], []
        top_s = []

        # use one single for loop
        for i in smallest_indices:
            top_simulationsP4.append(self.simulationsP4[i])
            top_simulationsP15.append(self.simulationsP15[i])
            top_s.append(self.sampled_s[i])
            top_distances.append(self.distances[i])
            top_distancesP4.append(self.distancesP4[i])
            top_distancesP15.append(self.distancesP15[i])

        # PLOT RESULTS -----------------------------------------------------------------------------------------------------------------------------------------------------

        fitness = 'log'    
        exp_dir = f'experiments/{expname}/'
        dirs = ['runs', 'results', 'plots', 'data']
        paths = {d: exp_dir+d+'/' for d in dirs}
        os.makedirs(paths['plots'], exist_ok=True)
        os.makedirs(paths['results'], exist_ok=True)

        plot_histograms_dict_overlay(
            dictionaries=[top_simulationsP4[0]], plot_ref=True, true_data=self.reference_data['P4'], ref_color='turquoise',
            labels=['Simulation'], colors=['#FF005E'],
            title=f'Reference and best simulation at P4 with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, distance={top_distancesP4[0]:.1f})',
            save=True, show=False, filepath=paths['plots']+'bestsimulationp4.png')

        plot_histograms_dict_overlay(
            dictionaries=[top_simulationsP15[0]], plot_ref=True, true_data=self.reference_data['P15'], ref_color='turquoise',
            labels=['Simulation'], colors=['#FF005E'],
            title=f'Reference and best simulation at P15 with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, distance={top_distancesP15[0]:.1f})',
            save=True, show=False, filepath=paths['plots']+'bestsimulationp15.png')

        plot_histograms_dict_overlay(
            dictionaries=top_simulationsP4, true_data=self.reference_data['P4'], bin_size=10, plot_ref=True, true_label='Reference at P4', ref_color='turquoise',
            title=f'ecDNA counts at passage 4 and {top_percent}% best simulations (over {self.num_samples} samples) with {fitness} fitness',
            plot_avg=True, plot_all=False, show=False, save=True, filepath=paths['plots']+'topsimulationsP4.png', CI_fillcolor='rgba(181,255,209,0.2)',
            labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s))

        plot_histograms_dict_overlay(
            dictionaries=top_simulationsP15, true_data=self.reference_data['P15'], bin_size=10, plot_ref=True, true_label='Reference at P15', ref_color='turquoise',
            title=f'ecDNA counts at Passage 15 and {top_percent}% best simulations (over {self.num_samples} samples) with {fitness} fitness',
            plot_avg=True, plot_all=False, show=False, save=True, filepath=paths['plots']+'topsimulationsP15.png', CI_fillcolor='rgba(181,255,209,0.2)',
            labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s))
        
        if expname=='CAM277':
            plot_posterior(
                s_values=self.sampled_s, distances=self.distances, top_percent=top_percent, save=True, show=False, 
                add_ref=False, filepath=paths['plots']+'posterior.png', posterior_color='coral')
        else: 
            plot_posterior(
                s_values=self.sampled_s, distances=self.distances, top_percent=top_percent, save=True, show=False, 
                add_ref=True, sref=sref, filepath=paths['plots']+'posterior.png', posterior_color='coral')

        plot_best_points_util(self.sampled_s, self.distances, top_percent=top_percent, save=True, show=False, filepath=paths['plots']+'bestpoints.png')
        
        qqplot(simulation=top_simulationsP4[0], reference=self.reference_data['P4'], title='Quantile-Quantile plot at passage 4', show=False, save=True, filepath=paths['plots']+'qqp4.png')
        qqplot(simulation=top_simulationsP15[0], reference=self.reference_data['P15'], title='Quantile-Quantile plot at passage 15', show=False, save=True, filepath=paths['plots']+'qqp15.png')

        plot_ecdfs(distributions=[self.reference_data['P4'], top_simulationsP4[0]], labels=['Reference','Best simulation'], colors=['turquoise','#FF005E'], title='ECDFs at passage 4', show=False, save=True, filepath=paths['plots']+'ecdfp4.png')
        plot_ecdfs(distributions=[self.reference_data['P15'], top_simulationsP15[0]], labels=['Reference','Best simulation'], colors=['turquoise','#FF005E'], title='ECDFs at passage 15', show=False, save=True, filepath=paths['plots']+'ecdfp15.png')
        
        # SAVE RESULTS -----------------------------------------------------------------------------------------------------------------------------------------------------

        top_results = {
            'top_distancesP4': top_distancesP4,
            'top_distancesP15': top_distancesP15,
            'top_distances': top_distances,
            'top_s': top_s,
            'top_percent': top_percent,
            'num_samples': self.num_samples}

        print('\nSaving top results...')
        save_yaml(dictionary=top_results, file_path=paths['results']+'topresults.yaml')

        print('\nSaving top simulations...')
        save_yaml(dictionary={'P4': top_simulationsP4, 'P15': top_simulationsP15}, file_path=paths['results']+'topsimulations.yaml')

        print('\nSaving sampled s values...')
        save_yaml(dictionary=self.sampled_s, file_path=paths['results']+'sampled_s.yaml')


class SelectionInference(ABCInference):

    def __init__(self, s_range, expname, start=-5, fitness='log', size=1000):
        super().__init__(fitness=fitness, size=size, expname=expname)
        self.s_range = s_range
        self.start = start
        self.runs_path = f'runs/P{start}'


    def worker_load(self, path):
        sampled = load_yaml(os.path.join(path, 'sampled.yaml'))
        simulations = load_yaml(os.path.join(path, 'simulations.yaml'))
        return sampled, simulations

    def load_simulations(self):
        start_time = time.time()
        num_samples = 0
        sampled_s = []
        simulationsP4, simulationsP15 = [], []

        num_cores = 32

        print('Loading simulations in parallel...')
        with mp.Pool(processes=num_cores) as pool:
            split = [os.path.join(self.runs_path, run_folder) for run_folder in os.listdir(self.runs_path)]
            results = pool.map(self.worker_load, split)
        print(f'Loading complete. Runtime: {str(datetime.timedelta(seconds=int(time.time()-start_time)))}\n')

        num_samples = 0
        for sampled, simulations in results:
            num_samples += sampled['num_samples']

            sampled_s.extend(sampled['sampled_s'])
            simulationsP4.extend(simulations['P4'])
            simulationsP15.extend(simulations['P15'])
        
        self.num_samples = num_samples
        self.sampled_s = sampled_s
        self.simulationsP4 = simulationsP4
        self.simulationsP15 = simulationsP15
        self.sampled_params = {'sampled_s': sampled_s}


class DoubleInference(ABCInference):

    def __init__(self, s_range, start_range, expname, fitness='log', size=1000):
        super().__init__(fitness=fitness, size=size, expname=expname)
        self.s_range = s_range
        self.start_range = start_range
        self.simulationsP4 = []
        self.simulationsP15 = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ABC simulations')
    parser.add_argument('--expname', type=str, required=True, help='Name of the experiment. There must be a folder with this name in experiments/.')
    parser.add_argument('--inference', type=str, required=True, choices=['selection', 'double'], 
                        help='Type of inference. double to infer both the selection parameter and the starting passage, simple to infer only the selection parameter given the starting passage.')

    args = parser.parse_args()
    expname = args.expname
    inference = args.inference

    s_range = [0.01, 0.1]
    start_range = [-9, 1]

    if expname=='CAM277':
        start = -5
    elif expname!='CAM277double':
        params = load_yaml(f'experiments/{expname}/data/params.yaml')
        start, sref = params['start'], params['s']

    if inference == 'selection':
        abc_inference = SelectionInference(s_range=s_range, start=start, expname=expname)
    elif inference == 'double':
        abc_inference = DoubleInference(s_range=s_range, start_range=start_range, expname=expname)

    abc_inference.load_simulations()
    abc_inference.load_reference()
    abc_inference.perform_inference()