import os
from utils import *
import argparse

def main(expname, inference):
    exp_dir = f'experiments/{expname}/'
    dirs = ['runs', 'results', 'plots', 'data']
    paths = {d: exp_dir+d+'/' for d in dirs}
    os.makedirs(paths['plots'], exist_ok=True)
    os.makedirs(paths['results'], exist_ok=True)

    fitness = 'log'
    params = load_yaml(paths['data']+'params.yaml')
    start, sref = params['start'], params['s']

    distances = []
    distancesP4 = []
    distancesP15 = []

    sampled_s = []
    if inference=='double': sampled_start = []

    simulated_data_P4 = []
    simulated_data_P15 = []

    num_samples = 0

    for run_folder in tqdm(os.listdir(paths['runs'])):
        run_path = os.path.join(paths['runs'], run_folder)
        
        if os.path.isdir(run_path):
            results_file_path = os.path.join(run_path, 'results.yaml')
            
            if os.path.isfile(results_file_path):

                print('loading results')
                print(results_file_path)
                results = load_yaml(results_file_path)
                print('results loaded')
                simulations = load_yaml(os.path.join(run_path, 'simulations.yaml'))
                print('simulations loaded')

                num_samples += results['num_samples']

                sampled_s.extend(results['sampled_s'])
                if inference=='double': sampled_start.extend(results['sampled_start'])

                distances.extend(results['distances'])
                distancesP4.extend(results['distancesP4'])
                distancesP15.extend(results['distancesP15'])

                simulated_data_P4.extend(simulations['P4'])
                simulated_data_P15.extend(simulations['P15'])

    referenceP4 = filter_distribution(load_yaml(exp_dir+'data/cell_counts_p4.yaml'), 10)
    referenceP15 = filter_distribution(load_yaml(exp_dir+'data/cell_counts_p15.yaml'), 10)

    # PLOT REFERENCE DATA -------------------------------------------------------------------------------------------------------------------------------------------------------

    plot_histograms_dict_overlay(
        dictionaries=[referenceP4, referenceP15], colors=('navy', 'turquoise'), opacity=(1, 0.6), 
        labels=('Passage 4', 'Passage 15'), bin_size=10, save=True, filepath=paths['data']+'data.png', show=False,
        width=1000, height=400, scale=10, xaxis_title='Number of copies of ecDNA per cell', yaxis_title='Counts', 
        title='Histograms of ecDNA counts', histnorm='')

    plot_histograms_dict_overlay(
        dictionaries=[referenceP4], colors=['navy'], opacity=[1], labels=['Passage 4'],
        bin_size=10, save=True, filepath=paths['data']+'dataP4.png', width=1000, height=400, scale=10, show=False,
        xaxis_title='Number of copies of ecDNA per cell', yaxis_title='Counts', 
        title='Histogram of ecDNA counts at passage 4', histnorm='')

    plot_histograms_dict_overlay(
        dictionaries=[referenceP15], colors=['turquoise'], opacity=[0.6], labels=['Passage 15'], show=False,
        bin_size=10, save=True, filepath=paths['data']+'dataP15.png', width=1000, height=400, scale=10,
        xaxis_title='Number of copies of ecDNA per cell', yaxis_title='Counts', 
        title='Histogram of ecDNA counts at passage 15', histnorm='')

    # FIND TOP RESULTS ---------------------------------------------------------------------------------------------------------------------------------------------

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
        if inference=='double': top_start.append(sampled_start[i])
        top_distances.append(distances[i])
        top_distancesP4.append(distancesP4[i])
        top_distancesP15.append(distancesP15[i])

    # PLOT RESULTS -----------------------------------------------------------------------------------------------------------------------------------------------------

    plot_histograms_dict_overlay(
        dictionaries=[top_simulated_data_P4[0]], plot_ref=True, true_data=referenceP4,
        labels=['Simulation'], colors=['turquoise'],
        title=f'ecDNA counts at passage 4 and simulation with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, score={top_distances[0]:.1f})',
        save=True, show=False, filepath=paths['plots']+'bestsimulationp4.png')

    plot_histograms_dict_overlay(
        dictionaries=[top_simulated_data_P15[0]], plot_ref=True, true_data=referenceP15,
        labels=['Simulation'], colors=['turquoise'],
        title=f'ecDNA counts at passage 15 and simulation with {fitness} fitness (s={top_s[0]:.3f}, start P{start}, distance={top_distances[0]:.1f})',
        save=True, show=False, filepath=paths['plots']+'bestsimulationp15.png')

    plot_histograms_dict_overlay(
        dictionaries=top_simulated_data_P4, true_data=referenceP4, bin_size=10, plot_ref=True, true_label='Reference at P4',
        title=f'ecDNA counts at passage 4 and {top_percent}% best simulations (over {num_samples} samples) with {fitness} fitness',
        plot_avg=True, plot_all=False, show=False, save=True, filepath=paths['plots']+'topsimulationsP4.png',
        labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s))

    plot_histograms_dict_overlay(
        dictionaries=top_simulated_data_P15, true_data=referenceP15, bin_size=10, plot_ref=True, true_label='Reference at P15',
        title=f'ecDNA counts at Passage 15 and {top_percent}% best simulations (over {num_samples} samples) with {fitness} fitness',
        plot_avg=True, plot_all=False, show=False, save=True, filepath=paths['plots']+'topsimulationsP15.png',
        labels=[""]*len(top_s), colors=['gold']*len(top_s), opacity=[0.3]*len(top_s))

    plot_posterior_util(
        s_values=sampled_s, distances=distances, top_percent=top_percent, save=True, show=False, 
        add_ref=True, sref=sref, plot_color='paleturquoise', filepath=paths['plots']+'posterior.png')

    plot_best_points_util(sampled_s, distances, top_percent=top_percent, save=True, filepath=paths['plots']+'bestpoints.png')
    
    if inference=='double':
        plot_histograms_grouped(top_s, top_start, 's', 'start', show=False,
                        title=f'Posterior distributions of s grouped by starting passage, {top_percent}% best over {num_samples} samples with {fitness} fitness', 
                        save=True, width=800, height=1300, filepath=paths['plots']+'histogramsgrouped.png')
        
        scatter_joint_marginal(sampled_s, sampled_start, 's', 'start', 
                            title='Scatter plot of sampled parameters with marginals', show=False,
                            width=700, height=700, save=True, filepath=paths['plots']+'jointmarginalsampled.png')
        
        scatter_joint_marginal(top_s, top_start, 's', 'start', 
                            title=f'Scatter plot of {top_percent}% best sampled parameters with marginals (over {num_samples} samples)', 
                            info=None, infolabel='distance', add_ref=True, xref=sref, yref=start, show=False,
                            width=700, height=700, save=True, filepath=paths['plots']+'jointmarginal.png')

    # SAVE RESULTS -----------------------------------------------------------------------------------------------------------------------------------------------------

    top_results = {
        'top_distancesP4': top_distancesP4,
        'top_distancesP15': top_distancesP15,
        'top_distances': top_distances,
        'top_s': top_s,
        'top_percent': top_percent,
        'num_samples': num_samples}
    if inference=='double': top_results['top_start'] = top_start

    results = {
        'distances': distances,
        'distancesP4': distancesP4,
        'distancesP15': distancesP15,
        'sampled_s': sampled_s,
        'num_samples': num_samples}
    if inference=='double': results['sampled_start'] = sampled_start


    print('\nSaving top results...')
    save_yaml(dictionary=top_results, file_path=paths['results']+'topresults.yaml')

    print('\nSaving top simulations...')
    save_yaml(dictionary={'P4': top_simulated_data_P4, 'P15': top_simulated_data_P15}, file_path=paths['results']+'topsimulations.yaml')

    print('\nSaving all simulations...')
    save_yaml(dictionary={'P4': simulated_data_P4, 'P15': simulated_data_P15}, file_path=paths['results']+'simulations.yaml')

    print('\nSaving all results...')
    save_yaml(dictionary=results, file_path=paths['results']+'results.yaml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process abc runs.')
    parser.add_argument('--expname', type=str, required=True, help='Name of the experiment. There must be a folder with this name in experiments/, and at least one run in runs/.')
    parser.add_argument('--inference', type=str, required=True, choices=['selection', 'double'], 
                        help='Type of inference. double to infer both the selection parameter and the starting passage, simple to infer only the selection parameter given the starting passage.')

    args = parser.parse_args()
    expname = args.expname
    inference = args.inference

    main(expname=expname, inference=inference)