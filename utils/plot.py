# PLOTS -----------------------------------------------------------------------------------------------------------------------------------------------

import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
from model import Population
from plotly.subplots import make_subplots

from utils.other import dict_to_values, get_boundaries, get_labels, get_best_indices

def plot_histograms_dict_overlay(dictionaries:list[dict], bin_size=10, 
                                 title='', xaxis_title='ecDNA counts', yaxis_title='Frequency',
                                 labels=[''], colors=[''], opacity=[1],
                                 plot_avg=False, avg_color='red', threshold=10,
                                 confidence_level=5, CI_linecolor='orangered', CI_fillcolor='rgba(206, 255, 79, 0.3)',
                                 plot_all=True, plot_ref=False, true_data=None, true_label='Reference data',
                                 show=True, save=False,filepath='histograms_overlay.png', save_html=False,
                                 plot_bgcolor='white', width=1000, height=500, scale=5,
                                 histnorm='probability density'):  
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
        fig.add_trace(go.Scatter(x=np.concatenate([bins[:-1], bins[:-1][::-1]]), y=avg_hist, 
                                 line=dict(color=avg_color, width=2), name=f'Mean simulations histogram'))

        # Add confidence interval shaded region
        fig.add_trace(go.Scatter( x=np.concatenate([bins[:-1], bins[:-1][::-1]]), y=np.concatenate([upper_bound, lower_bound[::-1]]),
                                 fill='toself', fillcolor=CI_fillcolor, line=dict(color=CI_linecolor, width=1, dash='dash'),
                                 name=f'{upper_percentile}% Confidence Interval'))

    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                      barmode='overlay', bargap=0.1, plot_bgcolor=plot_bgcolor)

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)

    if save_html:
        html_filepath = filepath.replace('.png', '.html')
        fig.write_html(html_filepath)


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

# Function to calculate ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def plot_ecdfs(distributions:list,
               labels, 
               title='Empirical Cumulative Distribution Functions (ECDF)', 
               colors=('#60EE95','navy'),
               show=True, save=False, filepath=None):
    
    fig = go.Figure()

    for distribution, label, color in zip(distributions, labels, colors):
        values = dict_to_values(distribution)
        x, y = ecdf(values)
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            mode='lines', 
            name=label,
            line=dict(color=color),
        ))

    # Add titles and axis labels
    fig.update_layout(
        title=title,
        xaxis_title='ecDNA copy number',
        yaxis_title='probability',
        width=500,
        height=500,
        plot_bgcolor='white'
    )

    if show: fig.show()
    if save: fig.write_image(filepath, scale=5)


def plot_histograms_avg(dictionaries, reference_data, reference_label, title, filepath):
    
    plot_histograms_dict_overlay(
            dictionaries=dictionaries, true_data=reference_data, bin_size=10,
            plot_ref=True, true_label=reference_label, title=title,
            plot_avg=True, confidence_level=5, show=False, save=True, plot_all=False,
            filepath=filepath
        )


def plot_histograms_grouped(x, y, xlabel, ylabel, title='', show=True, save=False, width=500, height=700, scale=3, filepath=None):
    unique_y_values = sorted(set(y))  # Get unique values of y
    num_y_values = len(unique_y_values)

    # Calculate total height for the figure
    total_height = 200 + 100 * num_y_values

    # Create subplot grid with shared y axes
    fig = make_subplots(rows=num_y_values, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"starting at P{val}" for val in unique_y_values],
                        row_heights=[100] * num_y_values)

    # Loop through unique y values
    for i, val in enumerate(unique_y_values, start=1):
        filtered_x = [x[j] for j in range(len(y)) if y[j] == val]
        hist_x = go.Histogram(x=filtered_x, name=f'{ylabel} = {val}', showlegend=False)
        fig.add_trace(hist_x, row=i, col=1)
    fig.update_layout(title=title, bargap=0.1, height=total_height)

    # Share y axes
    for i in range(2, num_y_values + 1): fig.update_yaxes(matches='y', row=i, col=1)
    # Show x axis on each histogram
    for i in range(1, num_y_values + 1): fig.update_xaxes(showticklabels=True, row=i, col=1)

    # Set x axis label on the last subplot
    fig.update_xaxes(title=xlabel, row=num_y_values, col=1)

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=total_height)
        fig.write_image(filepath, scale=scale)


def plot_evolution_area_multiple(df, group_limits=[10,20,30,40,50,100], plot_bgcolor='white',
                                 colors = ['gold', 'orange', 'darkorange', 'orangered', 'red', 'firebrick', 'darkred', 'palegreen'],):
    times = df.index
    boundaries = get_boundaries(group_limits, df)
    names = get_labels(boundaries)
    boundaries +=  [(0,0)]
    groups = []

    for boundary in boundaries:
        if boundary == (0,0): groups.append(df[0])
        else: groups.append(df.iloc[:, boundary[0]:boundary[1]].sum(axis=1))

    fig = go.Figure()

    for group, color, name in zip(groups, colors, names):
        fig.add_trace(go.Scatter(x=times, y=group, mode='none', stackgroup='one',
                                 fillcolor=color, name=name))

    fig.update_layout(title='Evolution of cell count',
                      xaxis_title='Time',
                      yaxis_title='Count',
                      plot_bgcolor=plot_bgcolor)
    fig.show()


def plot_mean_CN_multi_simulations(n_simulations, 
                                   filepath='meanmultiple.png',
                                   n_events=1000, 
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
        population = Population(fitness=fitness, s=s)
        population.simulate_moran(size=size, n_events=n_events, verbose=False)

        # Get mean ecDNA copy numbers for each time
        mean_ecDNA_copy_numbers = [population.get_mean(time_index=i) for i in range(len(population.times))]

        # Plot mean ecDNA copy number for the current simulation
        fig.add_trace(go.Scatter(x=population.times, y=mean_ecDNA_copy_numbers,
                                 mode='lines',
                                 #marker=dict(color=colors[i]),
                                 name=f'Simulation {i+1}'))

    fig.update_layout(
        title=f'Mean ecDNA copy number on {n_simulations} simulations'+ f" with {population.model} model and {fitness} fitness function (s={s})",
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




def plot_posterior(s_values, distances, top_percent=5, plot_color='lightskyblue',
                        show=True, save=False, filepath=None,
                        add_ref=False, sref=None, 
                        width=700, height=400, scale=5):   
    
    smallest_indices = get_best_indices(distances, top_percent)
    top_s_values = np.array(s_values)[smallest_indices].tolist()
    
    mean_s_value = np.mean(top_s_values)
    median_s_value = np.median(top_s_values)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s_values, name='Sampled', marker_color=plot_color, opacity=0.5, 
                               xbins=dict(size=0.005), histnorm='probability density', marker_pattern_shape=""))
    fig.add_trace(go.Histogram(x=top_s_values, name=f'Top {top_percent}%', marker_color='#1ADCBE', opacity=0.5, 
                               xbins=dict(size=0.005), histnorm='probability density', marker_pattern_shape=""))
    
    if add_ref:
        # Add a phantom legend for the reference vertical line
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='black', width=2), name=f'reference: {sref:.3f}'))
        # Add the reference vertical line
        fig.add_vline(x=sref, line=dict(color='black', width=2))
    
    # Same for mean vertical line
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=2, dash='dash'), name=f'average: {mean_s_value:.3f}'))
    fig.add_vline(x=mean_s_value, line=dict(color='red', width=2, dash='dash'))

    # Same for median vertical line
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', width=2, dash='dot'), name=f'median: {median_s_value:.3f}'))
    fig.add_vline(x=median_s_value, line=dict(color='blue', width=2, dash='dot'))
    
    fig.update_layout(title=f'Posterior distribution of selection parameter s ({top_percent}% best over {len(s_values)} samples)',
                      xaxis_title='s values', yaxis_title='Frequency', barmode='overlay', bargap=0, plot_bgcolor='white')
    
    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)




def error_s_heatmap(error_s, s_intervals, start_values):
    
    hover_text = []
    for j in range(len(start_values)):
        hover_text.append([f's interval: {s_intervals[i][0]} - {s_intervals[i][1]}<br>starting passage: {start_values[j]}<br>relative error: {100 * error_s[j][i]:.2f}%' for i in range(len(s_intervals))])

    fig = go.Figure(data=go.Heatmap(
        z=100 * error_s,
        x=[f'{s_intervals[i][0]} - {s_intervals[i][1]}' for i in range(len(s_intervals))],
        y=start_values,
        #colorscale='Viridis',
        text=hover_text,
        hoverinfo='text',
        # zmin=0, zmax=100
    ))

    fig.update_layout(
        title='Error on selection parameter estimations',
        xaxis_title='Selection parameter interval',
        yaxis_title='Starting passage',
        width=600, height=600,
        yaxis=dict(
            tickvals=start_values,
            ticktext=start_values
        )
    )

    # Afficher la heatmap
    fig.show()


def qqplot(simulation, reference, title, show=True, save=False, filepath=None):
    values1 = dict_to_values(simulation)
    values2 = dict_to_values(reference)

    # Sort the values
    values1.sort()
    values2.sort()

    # Ensure both distributions have the same number of points for comparison
    min_len = min(len(values1), len(values2))
    values1 = np.array(values1[:min_len])
    values2 = np.array(values2[:min_len])

    # Create QQ plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=values1,
        y=values2,
        mode='markers',
        name='QQ Plot',
        marker=dict(size=5, color='blue')
    ))

    # Add diagonal line y=x for reference
    min_val = min(min(values1), min(values2))
    max_val = max(max(values1), max(values2))

    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color='red', dash='dash')
    )
    # Add titles and axis labels
    fig.update_layout(
        title=title,
        xaxis_title='Quantiles of best simulation',
        yaxis_title='Quantiles of reference data',
        width=500,
        height=500,
        plot_bgcolor='white'
    )

    # Show the plot
    if show: fig.show()

    # Save the plot as an image (optional)
    if save: fig.write_image(filepath, scale=5)


def plot_best_points_util(s_values, distances, metric='Wasserstein', top_percent=5, best_color='turquoise',
                          show=True, save=False, width=700, height=400, scale=3, filepath=None,):
    fig = go.Figure()

    # Get indices of the best points (smallest distances)
    smallest_indices = get_best_indices(distances, top_percent)
    
    fig.add_trace(go.Scatter(x=s_values, y=distances, mode='markers', marker=dict(color='black', size=5), name='All points'))
    fig.add_trace(go.Scatter(x=np.array(s_values)[smallest_indices], y=np.array(distances)[smallest_indices], mode='markers',
                             marker=dict(color=best_color, size=8), name=f'{top_percent}% smallest distances'))

    fig.update_layout(xaxis_title='s', yaxis_title='Distance', plot_bgcolor='white')
    fig.update_traces(hovertemplate='s: %{x:.3f}<br>'+metric+' distance: %{y:.1f}')  # Utilisez '.2f' pour afficher 2 décimales

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)


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


def scatter_joint_marginal(x, y, xlabel, ylabel, title='', info=None, infolabel=None,
                           show=True, save=False, filepath=None, width=600, height=600, scale=3, add_ref=False, xref=None, yref=None):
    
    if info is not None: hover_text = [f"{xlabel}: {x[i]:.3f}, {ylabel}: {y[i]}, {infolabel}: {info[i]:.2f}" for i in range(len(x))]
    else: hover_text = [f"{xlabel}: {x[i]:.3f}, {ylabel}: {y[i]}" for i in range(len(x))]

    scatter = go.Scatter(x=x, y=y, mode='markers', marker=dict(size=7, color='deepskyblue'),
                         text=hover_text, hoverinfo='text', showlegend=False,)
    hist_x = go.Histogram(x=x, nbinsx=40, marker=dict(color='navy'), showlegend=False, yaxis='y2')
    hist_y = go.Histogram(y=y, nbinsy=40, marker=dict(color='coral'), showlegend=False, xaxis='x2')

    if add_ref:
        big_red_point = go.Scatter(x=[xref], y=[yref], mode='markers', marker=dict(size=12, color='red'), 
                                   name='Reference Parameters', showlegend=True,)

    layout = go.Layout(title=title, width=width, height=height, 
                       xaxis=dict(title=xlabel, domain=[0.0, 0.85], showgrid=False, zeroline=False),
                       yaxis=dict(title=ylabel, domain=[0.0, 0.85], showgrid=False, zeroline=False),
                       xaxis2=dict(domain=[0.85, 1.0], showgrid=False, zeroline=False),
                       yaxis2=dict(domain=[0.85, 1.0], showgrid=False, zeroline=False),
                       plot_bgcolor='white', paper_bgcolor='white', bargap=0, hovermode='closest')

    if add_ref: fig = go.Figure(data=[scatter, hist_x, hist_y, big_red_point], layout=layout)
    else : fig = go.Figure(data=[scatter, hist_x, hist_y], layout=layout)

    if show: fig.show()

    if save:
        fig.update_layout(width=width, height=height)
        fig.write_image(filepath, scale=scale)