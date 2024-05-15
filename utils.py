import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ecDNA import Population


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
        fig.write_image("data.png", scale=scale)


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
                                 title='Normalized distributions of ecDNA counts', 
                                 xaxis_title='ecDNA counts', 
                                 yaxis_title='Frequency',
                                 labels=('Passage 4', 'Passage 15'),
                                 colors=('violet', 'deepskyblue'),
                                 opacity=(1, 0.5),
                                 plot_avg=False,
                                 plot_true=False,
                                 avg_color='white',
                                 plot_bgcolor='white',
                                 bin_size=10,
                                 save_fig=False,
                                 width=1000,
                                 height=500,
                                 scale=5,
                                 filepath='histograms_overlay.png'):  

    fig = go.Figure()


    data_list = [[key for key, value in dictionary.items() for _ in range(value)] for dictionary in dictionaries]
    max_data = max(max(data) for data in data_list)

    if plot_avg: avg_hist = np.zeros((max_data + bin_size - 1) // bin_size)

    for data, label, color, op in zip(data_list, labels, colors, opacity):
        if plot_avg:
            hist, bins = np.histogram(data, bins=np.arange(0, max_data + bin_size, bin_size), density=True)
            avg_hist += hist
        fig.add_trace(go.Histogram(x=data, name=label, marker_color=color, opacity=op, 
                                    xbins=dict(size=bin_size), histnorm='probability density',marker_pattern_shape=""))
    
    if plot_true:
        true_data = [key for key, value in true_data.items() for _ in range(value)]
        fig.add_trace(go.Histogram(x=true_data, name='True data', marker_color='palegreen', opacity=0.7, 
                                    xbins=dict(size=bin_size), histnorm='probability density',marker_pattern_shape=""))
    
    if plot_avg:
        avg_hist /= len(dictionaries)
        bin_centers = (bins[:-1] + bins[1:]) / 2 - 0.5
        # Define hover text with bin ranges
        hover_text = [f'({bins[i]:.0f} - {bins[i+1]-1:.0f})' for i in range(len(bins) - 1)]
        hoverinfo = 'text+y'
        fig.add_trace(go.Bar(x=bin_centers, y=avg_hist, name='Average Histogram', marker_color=avg_color, opacity=0.3, marker_pattern_shape='x',hoverinfo=hoverinfo, hovertext=hover_text))



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
        fig.write_image(filepath, scale=scale)


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




# DATA -----------------------------------------------------------------------------------------------------------------------------------------------

import yaml

def save_yaml(dictionary, file_path):
    with open(file_path, "w") as yaml_file:
        yaml.dump(dictionary, yaml_file)
    print('Dictionary saved as yaml file to', file_path)


def load_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return loaded_dict


