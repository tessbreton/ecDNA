import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import Population


# Plotting functions ---------------------------------------------------------------------------------------------------

def plot_histograms_overlay(dfs:list, 
                            column_name='ecCounts', 
                            title='Histograms of ecDNA counts', 
                            xaxis_title='Number of copies of ecDNA per cell', 
                            yaxis_title='Counts',
                            labels=('Passage 4', 'Passage 15'),
                            colors=('violet', 'deepskyblue'),
                            opacity=(1, 0.5),
                            plot_bgcolor='white',
                            bin_size=10,
                            save_fig=False,
                            file_path=None,
                            width=1000,
                            height=500,
                            scale=5):  

    fig = go.Figure()

    for df, label, color, op in zip(dfs, labels, colors, opacity):
        fig.add_trace(go.Histogram(x=df[column_name], name=label, marker_color=color, opacity=op, 
                                    xbins=dict(size=bin_size)))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode='overlay',
        bargap=0.1,
        plot_bgcolor=plot_bgcolor
    )

    fig.show()

    if save_fig:
        fig.update_layout(width=width, height=height)
        fig.write_image(file_path, scale=scale)
        
        
def plot_histograms(data:list, binsize=10, opacity=None, colors=None, labels=None, title='', xaxis_title=''):
    if opacity==None: opacity = [0.5]*len(data)
    if colors==None: colors = ['blue']*len(data)
    if labels==None: labels = ['']*len(data)

    fig = go.Figure()
    for x, op, color, label in zip(data, opacity, colors, labels):
        hist = go.Histogram(
            x=x,
            xbins=dict(size=binsize),
            opacity=op,
            name=label,
            marker=dict(color=color)
        )
        fig.add_trace(hist)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title='Count',
        barmode='overlay',
        bargap=0.2,
        bargroupgap=0.1,
        plot_bgcolor='white',
        width=800,
        height=500
    )

    fig.show()


def trace_ecdf(data,
               label='',
               color='blue'):
    
    sorted_data = np.sort(data)
    ecdf_attr = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return go.Scatter(x=sorted_data, y=ecdf_attr, mode='lines', line=dict(color=color, width=2.5), name=label)


def plot_ecdfs(dfs:list,
               column_name='ecCounts',
               labels=('Passage 4', 'Passage 15'),
               title='ECDFs of ecDNA counts at different passages',
               xaxis_title='ecDNA counts',
               yaxis_title='Probability',
               colors=('violet', 'deepskyblue')):
    
    fig = go.Figure()
    
    for df, label, color in zip(dfs, labels, colors):
        trace = trace_ecdf(df[column_name], label=label, color=color)
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#F1F6FA'
    )

    fig.show()


# DATAFRAME

def to_dataframe(times, cell_counts):
    max_copies = max(list(cell_counts[-1].keys()))

    # Create an empty DataFrame with columns for each copy count
    df = pd.DataFrame(0, columns=[i for i in range(max_copies + 1)], index=times)

    # Fill the DataFrame with counts from cell_counts
    for time, subdict in tqdm(zip(times, cell_counts)):
        for copies, count in subdict.items():
            df.at[time, copies] = count

    return df

# plot utils with dataframe as input 

def plot_histogram_dict(dictionary,bin_size=5,
                   plot_color='deepskyblue', 
                   plot_bgcolor='white',
                   title='Histogram of ecDNA copy number'):

    data = [key for key, value in dictionary.items() for _ in range(value)]
    
    histogram_trace = go.Histogram(x=data, marker=dict(color=plot_color), xbins=dict(start=0,size=bin_size))

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Number of copies'),
        bargap=0.1,
        plot_bgcolor=plot_bgcolor
    )

    fig = go.Figure(data=[histogram_trace], layout=layout)

    fig.show()


def plot_histograms_dict_overlay(dictionaries:list[dict], 
                                 true_data=None,
                                 bin_size=10,
                                 title='', 
                                 xaxis_title='ecDNA counts', yaxis_title='Frequency',
                                 labels=[''],
                                 colors=[''],
                                 opacity=[1],
                                 plot_avg=False, avg_color='red',
                                 confidence_level=5, CI_linecolor='orangered', CI_fillcolor='rgba(206, 255, 79, 0.3)',
                                 plot_all=True,
                                 plot_ref=False,
                                 true_label='Reference data',
                                 plot_bgcolor='white',
                                 threshold=10,
                                 show=True,
                                 save=False,filepath='histograms_overlay.png', save_html=False,
                                 width=1000, height=500, scale=5,
                                 histnorm='probability density'
                                 ):  

    fig = go.Figure()


    data_list = [[key for key, value in dictionary.items() for _ in range(value)] for dictionary in dictionaries]
    max_data = max(max(data) for data in data_list)

    if plot_avg: 
        histograms = []
        bins = np.arange(0, max_data + bin_size, bin_size)

    if plot_all or plot_avg:
        for i, data in enumerate(data_list):
            if plot_avg:
                hist, bins = np.histogram(data, bins=np.arange(threshold, max_data + bin_size, bin_size), density=True)
                histograms.append(hist)
            if plot_all:
                label, color, op = labels[i], colors[i], opacity[i]
                fig.add_trace(go.Histogram(x=data, name=label, marker_color=color, opacity=op, 
                                            xbins=dict(size=bin_size), histnorm=histnorm,marker_pattern_shape=""))
        
    if plot_ref:
        true_data = [key for key, value in true_data.items() for _ in range(value)]
        fig.add_trace(go.Histogram(x=true_data, name=true_label, marker_color='palegreen', opacity=0.7, 
                                    xbins=dict(size=bin_size), histnorm=histnorm, marker_pattern_shape=""))
    
    if plot_avg:
        avg_hist = np.mean(histograms, axis=0)
        
        # Compute percentiles for confidence interval
        lower_percentile = confidence_level
        upper_percentile = 100 - confidence_level
        lower_bound = np.percentile(histograms, lower_percentile, axis=0)
        upper_bound = np.percentile(histograms, upper_percentile, axis=0)

        # Add confidence interval shaded region
        fig.add_trace(go.Scatter(
            x=np.concatenate([bins[:-1], bins[:-1][::-1]]),
            y=avg_hist,
            line=dict(color=avg_color, width=2),
            name=f'Mean simulations histogram'
        ))

        # Add confidence interval shaded region
        fig.add_trace(go.Scatter(
            x=np.concatenate([bins[:-1], bins[:-1][::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor=CI_fillcolor,
            line=dict(color=CI_linecolor, width=1, dash='dash'),
            name=f'{upper_percentile}% Confidence Interval'
        ))



    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode='overlay',
        bargap=0.1,
        plot_bgcolor=plot_bgcolor
    )

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)

    if save_html:
        html_filepath = filepath.replace('.png', '.html')
        fig.write_html(html_filepath)


def plot_histograms_avg(dictionaries, reference_data, reference_label, title, filepath):
    
    plot_histograms_dict_overlay(
            dictionaries=dictionaries, true_data=reference_data, bin_size=10,
            plot_ref=True, true_label=reference_label, title=title,
            plot_avg=True, confidence_level=5, show=False, save=True, plot_all=False,
            filepath=filepath
        )


def plot_histogramm(cell_counts, 
                   time_index=-1, 
                   bin_size=5,
                   plot_color='deepskyblue', 
                   plot_bgcolor='white',
                   title='Histogram of ecDNA copy number',
                   save_fig=False):

    timestep_ecDNA_counts = cell_counts[time_index]
    data = [key for key, value in timestep_ecDNA_counts.items() for _ in range(value)]
    
    histogram_trace = go.Histogram(x=data, marker=dict(color=plot_color), xbins=dict(start=0,size=bin_size))

    layout = go.Layout(
        title=title+f" on XX cells",
        xaxis=dict(title='Number of copies'),
        bargap=0.1,
        plot_bgcolor=plot_bgcolor
    )

    fig = go.Figure(data=[histogram_trace], layout=layout)

    fig.show()

    if save_fig:
        fig.update_layout(width=1000, height=500)
        fig.write_image(f"ecDNA_histogram_timestep[{time_index}].png", scale=3)

def plot_histogram(df, 
                   time_index=-1, 
                   bin_size=5,
                   plot_color='deepskyblue', 
                   plot_bgcolor='white',
                   title='Histogram of ecDNA copy number',
                   save_fig=False):

    timestep_ecDNA_counts = df.iloc[time_index]
    data = [key for key, value in timestep_ecDNA_counts.items() for _ in range(value)]
    
    histogram_trace = go.Histogram(x=data, marker=dict(color=plot_color), xbins=dict(start=0,size=bin_size))

    layout = go.Layout(
        title=title+f" on XX cells",
        xaxis=dict(title='Number of copies'),
        bargap=0.1,
        plot_bgcolor=plot_bgcolor
    )

    fig = go.Figure(data=[histogram_trace], layout=layout)

    fig.show()

    if save_fig:
        fig.update_layout(width=1000, height=500)
        fig.write_image(f"ecDNA_histogram_timestep[{time_index}].png", scale=3)



def plot_ecdf(df, 
              title='ECDF of ecDNA counts',
              xaxis_title='ecDNA counts',
              yaxis_title='Probability',
              color='deepskyblue',
              plot_bgcolor='#F1F6FA'):
    """
    Plot the ECDF of the ecDNA counts at the last time point.
    """
    
    counts_at_last_time = df.iloc[-1]
    l = [key for key, value in counts_at_last_time.items() for _ in range(value)]

    fig = go.Figure()

    trace = trace_ecdf(l, label='label', color=color)
    fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(orientation='h'),
        plot_bgcolor=plot_bgcolor,
    )

    fig.show()

def scatter_joint_marginal(x, y, xlabel, ylabel, title='', info=None, infolabel=None,
                           show=True, save=False, filepath=None, width=600, height=600, scale=3, add_ref=False, xref=None, yref=None):
    if info is not None:
        hover_text = [f"{xlabel}: {x[i]:.3f}, {ylabel}: {y[i]}, {infolabel}: {info[i]:.2f}" for i in range(len(x))]
    else: 
        hover_text = [f"{xlabel}: {x[i]:.3f}, {ylabel}: {y[i]}" for i in range(len(x))]

    # Create the scatter plot with updated marker and labels
    scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=7, color='deepskyblue'),
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
    )

    # Create the x histogram
    hist_x = go.Histogram(
        x=x,
        nbinsx=40,
        marker=dict(color='navy'),
        showlegend=False,
        yaxis='y2'
    )

    # Create the y histogram
    hist_y = go.Histogram(
        y=y,
        nbinsy=40,
        marker=dict(color='coral'),
        showlegend=False,
        xaxis='x2',
    )

    if add_ref:
        # Create the big red point
        big_red_point = go.Scatter(
            x=[xref],
            y=[yref],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Reference Parameters',
            showlegend=True,
        )

    # Create the layout with grid specs and updated background color
    layout = go.Layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(
            title=xlabel,  # Label for the x-axis
            domain=[0.0, 0.85],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=ylabel,  # Label for the y-axis
            domain=[0.0, 0.85],
            showgrid=False,
            zeroline=False
        ),
        xaxis2=dict(
            domain=[0.85, 1.0],
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.85, 1.0],
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white',  # Change background color
        paper_bgcolor='white',  # Change the outer background color
        bargap=0,
        hovermode='closest'
    )

    # Combine the plots
    if add_ref: fig = go.Figure(data=[scatter, hist_x, hist_y, big_red_point], layout=layout)
    else : fig = go.Figure(data=[scatter, hist_x, hist_y], layout=layout)

    # Show figure
    if show: 
        fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_histograms_grouped(x, y, xlabel, ylabel, title='', show=True, save=False, width=500, height=700, scale=3, filepath=None):
    unique_y_values = sorted(set(y))  # Get unique values of y
    num_y_values = len(unique_y_values)

    # Calculate total height for the figure
    total_height = 200 + 100 * num_y_values

    # Create subplot grid with shared y axes
    fig = make_subplots(
        rows=num_y_values, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"starting at P{val}" for val in unique_y_values],
        row_heights=[100] * num_y_values  # Adjust height of each subplot
    )

    # Loop through unique y values
    for i, val in enumerate(unique_y_values, start=1):
        # Filter x values corresponding to current y value
        filtered_x = [x[j] for j in range(len(y)) if y[j] == val]

        # Create histogram for filtered x values
        hist_x = go.Histogram(
            x=filtered_x,
            name=f'{ylabel} = {val}',
            showlegend=False
        )

        # Add histogram to subplot
        fig.add_trace(hist_x, row=i, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        bargap=0.1,
        height=total_height  # Set total height of the figure
    )

    # Share y axes
    for i in range(2, num_y_values + 1):
        fig.update_yaxes(matches='y', row=i, col=1)

    # Show x axis on each histogram
    for i in range(1, num_y_values + 1):
        fig.update_xaxes(showticklabels=True, row=i, col=1)

    # Set x axis label on the last subplot
    fig.update_xaxes(title=xlabel, row=num_y_values, col=1)

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=total_height)
        fig.write_image(filepath, scale=scale)


# plot evolution


def get_boundaries(group_limits, df):
    boundaries = []

    for i in range(len(group_limits) + 1):

        if i == 0: lower_bound = 1
        else: lower_bound = group_limits[i - 1] + 1

        if i == len(group_limits): upper_bound = df.shape[1]
        else: upper_bound = group_limits[i] + 1

        boundaries.append((lower_bound, upper_bound))

    return boundaries

def get_labels(boundaries):

    labels = []
    for i,boundary in enumerate(boundaries):
        lower_bound, upper_bound = boundary
        if i==len(boundaries)-1:
            labels.append(f'{lower_bound}+ copies')
        else: 
            if lower_bound == upper_bound - 1 :
                if lower_bound==1:
                    labels.append('1 copy')
                else:
                    labels.append(f'{lower_bound} copies')
                
            else:
                labels.append(f'{lower_bound}-{upper_bound-1} copies')
    return labels + ['no copy']

def plot_evolution_area_multiple(df, 
                                 group_limits=[10,20,30,40,50,100], 
                                 colors = ['gold', 'orange', 'darkorange', 'orangered', 'red', 'firebrick', 'darkred', 'palegreen'], 
                                 plot_bgcolor='white'):
    times = df.index
    
    boundaries = get_boundaries(group_limits, df)
    
    names = get_labels(boundaries)
    # colors = generate_cmap(len(names)-1) + ['palegreen']

    boundaries +=  [(0,0)]
    groups = []

    for boundary in boundaries:
        if boundary == (0,0): groups.append(df[0])
        else: groups.append(df.iloc[:, boundary[0]:boundary[1]].sum(axis=1))

    fig = go.Figure()

    for group, color, name in zip(groups, colors, names):
        fig.add_trace(go.Scatter(x=times, y=group, mode='none', stackgroup='one',
                                 fillcolor=color, name=name))

    # Update layout
    fig.update_layout(title='Evolution of cell count',
                      xaxis_title='Time',
                      yaxis_title='Count',
                      plot_bgcolor=plot_bgcolor)

    fig.show()


def plot_mean_CN_multi_simulations(n_simulations, 
                                   filepath='meanmultiple.png',
                                   max_time=100, 
                                   fitness='log', 
                                   s=0.1, 
                                   size=1000, 
                                   plot_bgcolor='#F1F6FA',
                                   save_fig=False,
                                   width=1000,
                                   height=400,
                                   scale=10):
    fig = go.Figure()

    for i in tqdm(range(n_simulations)):
        # Run simulation
        population = Population(max_time=max_time, fitness=fitness, s=s)
        population.simulate_moran(size=size, verbose=False)

        # Get mean ecDNA copy numbers for each time
        mean_ecDNA_copy_numbers = [population.get_mean(time_index=i) for i in range(len(population.times))]

        # Plot mean ecDNA copy number for the current simulation
        fig.add_trace(go.Scatter(x=population.times, y=mean_ecDNA_copy_numbers,
                                 mode='lines',
                                 #marker=dict(color=colors[i]),
                                 name=f'Simulation {i+1}'))

    fig.update_layout(
        title=f'Mean ecDNA copy number over time on {n_simulations} simulations'+ f" with {population.model} model and {fitness} fitness function (s={s})",
        xaxis=dict(title='Time'),
        yaxis=dict(title='Mean ecDNA copy number'),
        plot_bgcolor=plot_bgcolor,
        showlegend=True,
        hovermode='closest',
    )

    fig.show()

    if save_fig:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)




# LOADING AND SAVING DATA -----------------------------------------------------------------------------------------------------------------------------------------------

import yaml
import os

def save_yaml(dictionary, file_path):
    # Crée les répertoires nécessaires si ce n'est pas déjà fait
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Sauvegarde le dictionnaire dans un fichier YAML
    with open(file_path, "w") as yaml_file:
        yaml.dump(dictionary, yaml_file)
    
    print('Dictionary saved as yaml file to', file_path)

def get_save_folder(path='runs'):
    index = 1
    while True:
        result_folder = os.path.join(path, f"run{index}")
        
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            return result_folder
        
        index += 1


def load_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return loaded_dict




def normalize_distribution(distribution:dict):
    total_count = sum(distribution.values())
    return {k: v / total_count for k, v in distribution.items()}

def filter_distribution(distribution, threshold):
    return {k: v for k, v in distribution.items() if k >= threshold}

def filter_and_normalize(distribution, threshold=10):
    return normalize_distribution(filter_distribution(distribution, threshold))

def standard_score(distances):
    mean = float(np.mean(distances))
    std = float(np.std(distances))
    return [(d - mean) / std for d in distances]


from math import ceil 

def get_best_indices(distances, top_percent):
    '''Find indices corresponding to {top_percent}% smallest distances'''
    num_points = len(distances)
    num_smallest_points = ceil(0.01 * top_percent * num_points)
    smallest_indices = np.argsort(distances)[:num_smallest_points]
    smallest_indices = smallest_indices.tolist()
    return smallest_indices




def plot_best_points_util(s_values, 
                     distances, 
                     metric,
                     filepath=None,
                     top_percent=5, 
                     best_color='turquoise',
                     show=True,
                     save=False, width=700, height=400, scale=3):
    
    fig = go.Figure()  # Initialize the figure

    # Get indices of the best points (smallest distances)
    smallest_indices = get_best_indices(distances, top_percent)
    
    fig.add_trace(go.Scatter(
        x=s_values,
        y=distances,
        mode='markers',
        marker=dict(color='black', size=5),
        name='All points'
    ))

    # Plot points corresponding to 5% smallest distances in blue
    fig.add_trace(go.Scatter(
        x=np.array(s_values)[smallest_indices],
        y=np.array(distances)[smallest_indices],
        mode='markers',
        marker=dict(color=best_color, size=8),
        name=f'{top_percent}% smallest distances'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='s',
        yaxis_title='Distance',
        plot_bgcolor='white'
    )
    fig.update_traces(hovertemplate='s: %{x:.3f}<br>'+metric+' distance: %{y:.1f}')  # Utilisez '.2f' pour afficher 2 décimales

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)


def plot_posterior_util(s_values, 
                   distances, 
                   top_percent=5, 
                   plot_color='lightskyblue',
                   show=True,
                   save=False, filepath=None,
                   add_ref=False, sref=None, 
                   width=700, height=400, scale=5):   
    
    smallest_indices = get_best_indices(distances, top_percent)
    top_s_values = np.array(s_values)[smallest_indices].tolist()
    
    mean_s_value = np.mean(top_s_values)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s_values, name=f'sampled', marker_color=plot_color, opacity=0.5, 
                                        xbins=dict(size=0.005), histnorm='probability density',marker_pattern_shape=""))
    fig.add_trace(go.Histogram(x=top_s_values, name=f'top {top_percent}%', marker_color='#1ADCBE', opacity=0.5, 
                                        xbins=dict(size=0.005), histnorm='probability density',marker_pattern_shape=""))
    if add_ref:
        # fig.add_vline(x=sref, line=dict(color='blue', width=2), 
        #               annotation_text=f'Reference: {sref:.5f}',
        #               annotation_position="top", annotation=dict(font=dict(color='blue')))
        fig.add_vline(x=sref, line=dict(color='blue', width=2))     
    # Add a vertical line at the mean value
    # fig.add_vline(x=mean_s_value, line=dict(color='red', width=2, dash='dash'), annotation_text=f'Mean: {mean_s_value:.5f}', 
    #               annotation_position="top", annotation=dict(font=dict(color='red')))
    fig.add_vline(x=mean_s_value, line=dict(color='red', width=2, dash='dash'))
    
    fig.update_layout(
        title=f'Posterior distribution of selection parameter s ({top_percent}% best over {len(s_values)} samples)',
        xaxis_title='s values',
        yaxis_title='Count',
        barmode='overlay',
        bargap=0,
        plot_bgcolor='white'
    )
    
    if show: 
        fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)