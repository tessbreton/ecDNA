import os
from utils import *

# Chemin du dossier 'runs'
expname = 'synthetic1'
exp_dir = f'experiments/{expname}/'
runs_dir = exp_dir + 'runs/'
results_dir = exp_dir + 'results/'
plots_dir = exp_dir + 'plots/'

fitness = 'log'
params = load_yaml(f'experiments/{expname}/data/params.yaml')
start, sref = params['start'], params['s']

# Listes pour stocker les valeurs concaténées
distances = []
distancesP4 = []
distancesP15 = []

sampled_s = []
sampled_start = []

simulated_data_P4 = []
simulated_data_P15 = []

num_samples = 0

# Parcourir les sous-dossiers dans le dossier 'runs'
for run_folder in tqdm(os.listdir(runs_dir)):
    run_path = os.path.join(runs_dir, run_folder)
    
    # Vérifier si c'est un dossier
    if os.path.isdir(run_path):
        results_file_path = os.path.join(run_path, 'results.yaml')
        
        # Vérifier si le fichier results.yaml existe
        if os.path.isfile(results_file_path):
            # Charger le fichier YAML
            print('loading results')
            print(results_file_path)
            results = load_yaml(results_file_path)
            print('results loaded')
            simulations = load_yaml(os.path.join(run_path, 'simulations.yaml'))
            print('simulations loaded')

            # Concaténer les valeurs pour chaque champ
            num_samples += results['num_samples']

            sampled_s.extend(results['sampled_s'])
            # sampled_start.extend(results['sampled_start'])

            distances.extend(results['distances'])
            distancesP4.extend(results['distancesP4'])
            distancesP15.extend(results['distancesP15'])

            simulated_data_P4.extend(simulations['P4'])
            simulated_data_P15.extend(simulations['P15'])

referenceP4 = filter_distribution(load_yaml(exp_dir+'data/cell_counts_p4.yaml'), 10)
referenceP15 = filter_distribution(load_yaml(exp_dir+'data/cell_counts_p15.yaml'), 10)

top_percent = 5
smallest_indices = get_best_indices(distances, top_percent)

top_simulated_data_P4, top_simulated_data_P15 = [], []
top_distancesP4, top_distancesP15 = [], []
top_s, top_start = [], []
top_distances = []

# use one single for loop
for i in smallest_indices:
    top_simulated_data_P4.append(simulated_data_P4[i])
    top_simulated_data_P15.append(simulated_data_P15[i])
    top_s.append(sampled_s[i])
    # top_start.append(sampled_start[i])
    top_distances.append(distances[i])
    top_distancesP4.append(distancesP4[i])
    top_distancesP15.append(distancesP15[i])


plot_histograms_dict_overlay(dictionaries=[top_simulated_data_P4[0]],
                                true_data=referenceP4,
                                plot_ref=True,
                                labels=['Simulation'],
                                colors=['turquoise'],
                                title=f'Normalized distributions of ecDNA counts at passage 4 and simulation with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, score={top_distances[0]:.1f})',
                                save=True, show=False,
                                filepath=plots_dir+'bestsimulationp4.png')

plot_histograms_dict_overlay(dictionaries=[top_simulated_data_P15[0]],
                                true_data=referenceP15,
                                plot_ref=True,
                                labels=['Simulation'],
                                colors=['turquoise'],
                                title=f'Normalized distributions of ecDNA counts at passage 15 and simulation with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, score={top_distances[0]:.1f})',
                                save=True, show=False,
                                filepath=plots_dir+'bestsimulationp15.png')


plot_histograms_dict_overlay(dictionaries=top_simulated_data_P4,
                             true_data=referenceP4, bin_size=10,
                             plot_ref=True, true_label='Reference at P4',
                             title=f'ecDNA counts at passage 4 and {top_percent}% best simulations (over {num_samples} samples) with {fitness} fitness',
                             plot_avg=True, plot_all=False, show=False,
                             labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s),
                             save=True, filepath=plots_dir+'topsimulationsP4.png')

plot_histograms_dict_overlay(dictionaries=top_simulated_data_P15,
                             true_data=referenceP15, bin_size=10,
                             plot_ref=True, true_label='Reference at P15',
                             title=f'ecDNA counts at Passage 15 and {top_percent}% best simulations (over {num_samples} samples) with {fitness} fitness',
                             plot_avg=True, plot_all=False, show=False,
                             labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s),
                             save=True, filepath=plots_dir+'topsimulationsP15.png')


plot_posterior_util(s_values=sampled_s, 
                    distances=distances, 
                    top_percent=top_percent, 
                    save=True, show=False,
                    add_ref=True, sref=sref,
                    plot_color='paleturquoise', 
                    filepath=plots_dir+'posterior.png')


top_results = {'top_distancesP4': top_distancesP4,
               'top_distancesP15': top_distancesP15,
               'top_distances': top_distances,
               'top_s': top_s,
            #    'top_start': top_start,
               'top_simulated_data_P4': top_simulated_data_P4,
               'top_simulated_data_P15': top_simulated_data_P15,
               'top_percent': top_percent,
               'num_samples': num_samples}

print('\nSaving top results...')
save_yaml(dictionary=top_results, file_path=results_dir+'topresults.yaml')

print('\nSaving all simulations...')
save_yaml(dictionary={'P4': simulated_data_P4, 'P15': simulated_data_P15}, file_path=results_dir+'simulations.yaml')

print('\nSaving all results...')
save_yaml(dictionary={'distances': distances,
           'distancesP4': distancesP4,
           'distancesP15': distancesP15,
           'sampled_s': sampled_s,
        #    'sampled_start': sampled_start,
           'num_samples': num_samples,
           }, file_path=results_dir+'results.yaml')