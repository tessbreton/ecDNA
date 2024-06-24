import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import argparse


# UTILS --------------------------------------------------------------------------------------------------------------------------------------------------------

def binomial_segregation(n, p=0.5):
    """Randomly segregate ecDNA copies to daughter cells using binomial distribution."""
    k = np.random.binomial(n,p)
    return k, n-k

def compute_time_to_next_event(rate):
    """Calculate time to next event based on the rate for Gillespie algorithm."""
    return -1 / rate * np.log(np.random.uniform(0, 1))


# CLASS Population TO TRACK EVOLUTION ------------------------------------------------------------------------------------------------------------------------

class Population:

    def __init__(self, 
                 initial_nb_without=0, 
                 turnover=1, 
                 max_cells=1e5, 
                 max_time=30,
                 s=1,
                 growth='constant',
                 target_size=1000,
                 fitness='linear'):

        self.population = {0:initial_nb_without, 1:1}
        # self.population = {k: number of cells with k copies of ecDNA}

        self.turnover = turnover
        self.max_cells = max_cells
        self.max_time = max_time
        self.initial_nb_without = initial_nb_without

        self.growth = growth
        if self.growth == 'constant': self.target_size = target_size

        self.s = s
        self.fitness = fitness
        self.fitness_by_key = {}


    # SIMULATION UTILS ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    def cell_division(self, key):
        if key == 0: 
            self.population[0] += 1
        else: 
            n = 2 * key
            k = np.random.binomial(n, 0.5)
            if k not in self.population.keys(): self.fitness_by_key[k] = self.cell_fitness(k)
            if n-k not in self.population.keys(): self.fitness_by_key[n-k] = self.cell_fitness(n-k)

            self.population[k] = self.population.get(k, 0) + 1
            self.population[n-k] = self.population.get(n-k, 0) + 1
            self.population[key] -= 1

    def cell_death(self, key):
        self.population[key] -= 1

    def random_cell_division(self, cell_division_type):
        if cell_division_type == 'zero':
            self.cell_division(0)

        elif cell_division_type == 'with_ecDNA':
            ecDNA_keys = [key for key in self.population.keys() if key != 0 and self.population[key] != 0]

            # Pick a random cell uniformly among all cells with ecDNA
            counts = [self.population[key] for key in ecDNA_keys]
            key = np.random.choice(ecDNA_keys, p=counts/np.sum(counts))

            self.cell_division(key)

        else:
            raise ValueError("Invalid cell division type. Supported types are 'zero' and 'with_ecDNA'.")

    def logistic_growth(self, t, x0=3, k=0.1):
        L = self.max_cells
        return L / (1 + np.exp(-k * (t-x0)))
    
    def constant_growth(self, t):
        return self.target_size
    
    def feedback(self, t, input_dynamic):
        """Feedback loop for cell division probability. Formula from CINner supplementary."""
        p_bar = input_dynamic(t)
        p = sum(self.population.values())
        g = p_bar / (p_bar + p)
        return g
    
    def cell_fitness(self, key):
        if self.fitness == 'power': return (1+self.s)**key
        elif self.fitness == 'linear': return 1+ key*self.s
        elif self.fitness == 'log': return 1 + self.s * np.log(1+key)
        elif self.fitness == 'loglog': return 1 + self.s * np.log(1+np.log(1+key))
        elif self.fitness == 'constant': return 1
        else: print('Fitness function not defined.')
    
    def relative_fitness(self, key):
        average_fitness = sum(self.cell_fitness(key) * self.population[key] for key in self.population.keys()) / sum(self.population.values())
        cell_fitness = self.cell_fitness(key)
        return cell_fitness / average_fitness
    
    def division_probability(self, key, growth, current_time=0):
        if growth=='constant':
            g = self.feedback(t=current_time, input_dynamic=self.constant_growth)
            f = self.relative_fitness(key)
            return g * f

        elif growth=='logistic':
            g = self.feedback(t=current_time, input_dynamic=self.logistic_growth)
            f = self.relative_fitness(key)
            return g * f

        else:
            print('Growth model not implemented yet.')

    def get_next_event(self):
        neutral_cells_count = self.population[0]
        ecDNA_cells_count = sum([self.population[key] for key in self.population.keys() if key > 0])
        
        if neutral_cells_count == 0:
            rate = ecDNA_cells_count * self.s * 1 / self.turnover
            time_to_next_event = compute_time_to_next_event(rate)
            cell_division_type = 'with_ecDNA'
        
        elif ecDNA_cells_count == 0:
            rate = self.population[0] * 1 / self.turnover
            time_to_next_event = compute_time_to_next_event(rate)
            cell_division_type = 'zero'
        
        else:
            rate_zero = self.population[0] * 1 / self.turnover
            rate_with_ecDNA = ecDNA_cells_count * self.s * 1 / self.turnover

            time_to_next_event_zero = compute_time_to_next_event(rate_zero)
            time_to_next_event_with_ecDNA = compute_time_to_next_event(rate_with_ecDNA)

            if time_to_next_event_zero <= time_to_next_event_with_ecDNA:
                cell_division_type = 'zero'
                time_to_next_event = time_to_next_event_zero
            else:
                cell_division_type = 'with_ecDNA'
                time_to_next_event = time_to_next_event_with_ecDNA

        return time_to_next_event, cell_division_type


    # SIMULATION ALGORITHMS ------------------------------------------------------------------------------------------------------------------------------------------------------------

    def simulate_feedback(self, verbose=True):
        """
        Simulate population dynamics using CINner feedback loop to follow input growth dynamic.
        """
        if verbose: print(f"Simulating cell population evolution starting with {self.population[0]} cells without ecDNA, and 1 cell with 1 copy of ecDNA.")
        if verbose: print(f"Growth model for global cell population: {self.growth}")
        failed_simulations = 0
        self.model = 'CINner feedback'
        
        while True:

            while True:
                self.population = {0: self.initial_nb_without, 1: 1}
                current_time = 0
                cell_counts = [self.population.copy()]
                times = [current_time]

                with tqdm(total=self.max_time, leave=False) as pbar:
                    pbar.set_postfix_str(f"Simulation {failed_simulations + 1}")

                    while (sum(self.population.values())) > 0 and current_time < self.max_time:
                        # Compute time to next event
                        total_cell_count = sum(self.population.values())
                        rate = total_cell_count * 1 / self.turnover
                        time_to_next_event = compute_time_to_next_event(rate)

                        # Choose a random cell in the population to undergo the event
                        ecDNA_counts = list(self.population.keys())
                        weights = list(self.population.values())
                        random_key = np.random.choice(ecDNA_counts, p=weights/np.sum(weights))

                        prob = self.division_probability(key=random_key, growth=self.growth, current_time=current_time)

                        if np.random.uniform(0,1) < prob: self.cell_division(random_key)
                        else: self.cell_death(random_key)
                    
                        current_time += time_to_next_event
                        cell_counts.append(self.population.copy())
                        times.append(current_time)

                        pbar.update(current_time - pbar.n)

                        if sum(self.population.values())==self.population[0]: break

                # Restart conditions
                population_size = sum(self.population.values())
                restart_simulation = False
                if population_size == self.population[0]: restart_simulation = True
                elif population_size == 0: restart_simulation = True

                if restart_simulation:
                    failed_simulations += 1
                    self.population = {0: self.initial_nb_without, 1: 1}
                    break
                else:
                    if verbose: print(f'Simulation {failed_simulations+1} complete.')
                    if verbose: print('Failed simulations:', failed_simulations)
                    if verbose: 
                        if self.population[0] == 0: print('NB: all cells have ecDNA.')
                        elif population_size == self.population[0]: print('NB: no cell has ecDNA.')
                    
                    self.cell_counts = cell_counts
                    self.times = times
                    return times, cell_counts

    def simulate_moran(self, 
                       size=1000, 
                       n_events=5e4, 
                       verbose=True, 
                       from_start=True, 
                       initial_cell_counts=None, 
                       disable=False):
        """
        Simulate population evolution using Moran process
        If from_start, the simulation starts with 1 cell with ecDNA and size-1 cells without ecDNA.
        Otherwise, the simulation starts with the initial_cell_counts dictionary.
        """
        if verbose: print(f"Simulating cell population evolution using Moran process. Total population size: {size} cells.")
        if verbose: print(f"Fitness function: {self.fitness}\n")

        failed_simulations = 0
        self.model = 'Moran'
        
        while True:

            while True:
                if from_start: self.population = {0: size-1, 1: 1}
                else: self.population = initial_cell_counts.copy()
                current_time = 0
                cell_counts = [self.population.copy()]
                times = [current_time]
                events_count = 0

                self.fitness_by_key = {key: self.cell_fitness(key) for key in self.population.keys()}  
                # self.fitness_by_key = {k: fitness of a cell with k copies of ecDNA}

                with tqdm(total=n_events, leave=False, disable=disable) as pbar:
                    pbar.set_postfix_str(f"Simulation {failed_simulations + 1}")

                    for _ in range(n_events):
                        
                        # COMPUTE TIME TO NEXT EVENT
                        keys = list(self.population.keys())
                        fitness_grouped = {key: self.population[key]*self.fitness_by_key[key] for key in keys}

                        total_fitness = sum(fitness_grouped.values())
                        if current_time==0: self.initial_rate = total_fitness
                        time_to_next_event = compute_time_to_next_event(total_fitness)

                        # CELL DEATH
                        random_key_death = int(np.random.choice(keys, p=[self.population[key]/(size) for key in keys])) # uniform 
                        self.cell_death(random_key_death)

                        # Update population fitness after cell death
                        fitness_grouped[random_key_death] = self.population[random_key_death] * self.fitness_by_key[random_key_death]
                        total_fitness = sum(fitness_grouped.values())
                        probabilities_by_clone = [fitness/ total_fitness for fitness in list(fitness_grouped.values())] # probabilities_by_clone = {k: probability to chose the clone with k ecDNA copies}
                        
                        # CELL DIVISION
                        random_key_division = int(np.random.choice(keys, p=probabilities_by_clone))
                        self.cell_division(random_key_division)

                        # UPDATE CURRENT POPULATION
                        current_time += time_to_next_event
                        cell_counts.append(self.population.copy())
                        times.append(current_time)
                        events_count += 1

                        pbar.update(events_count - pbar.n)

                        if size==self.population[0]: 
                            break

                # Restart conditions
                population_size = sum(self.population.values())
                restart_simulation = False
                if population_size == self.population[0]: restart_simulation = True
                elif population_size == 0: restart_simulation = True
                # elif sum([self.population[i] for i in range(11)])==0: restart_simulation = True # restart if no cell has more than 10 copies

                if restart_simulation:
                    failed_simulations += 1
                    break
                else:
                    if verbose: print(f'Simulation {failed_simulations+1} complete.')
                    if verbose: print('Failed simulations:', failed_simulations)
                    if verbose: 
                        if self.population[0] == 0: print('NB: all cells have ecDNA.')
                        elif population_size == self.population[0]: print('NB: no cell has ecDNA.')
                    
                    self.cell_counts = cell_counts
                    self.times = times
                    return

    def simulate_huang(self, verbose=True):
        """
        Simulate population dynamics using model from Huang paper.
        """
        if verbose: print(f"Simulating cell population evolution starting with {self.population[0]} cells without ecDNA, and 1 cell with 1 copy of ecDNA.")
        if verbose: print(f"Cell count target: {self.max_cells}")
        if verbose: print(f"Fitness: {self.s}")

        current_time = 0
        cell_counts = [self.population.copy()]
        times = [current_time]
        self.model = 'Huang'

        with tqdm(total=self.max_cells) as pbar:
            while (sum(self.population.values()) < self.max_cells and sum(self.population.values())) > 0:
                
                time_to_next_event, cell_division_type = self.get_next_event()
                self.random_cell_division(cell_division_type)                

                current_time += time_to_next_event
                cell_counts.append(self.population.copy())
                times.append(current_time)

                pbar.update(sum(self.population.values()) - pbar.n)

                if sum(self.population.values())==self.population[0]: 
                    if verbose: print('All ecDNA copies were lost.')
                    if verbose: print('Ending simulation.')
                    self.times = times
                    self.cell_counts = cell_counts
                    return cell_counts
            
        if verbose: 
            if sum(self.population.values())==0: print('Population went extinct.')
            elif sum(self.population.values()) >= self.max_cells: 
                print('Population reached input size.')
                if self.population[0]==0: print('All cells have ecDNA.')
                elif sum(self.population.values())==self.population[0]: print('No cell has ecDNA.')

        self.cell_counts = cell_counts
        self.times = times
        return times, cell_counts
    
    def get_mean(self, time_index=-1):
        weighted_sum = sum(key * value for key, value in self.cell_counts[time_index].items())
        total_sum = sum(self.cell_counts[time_index].values())
        weighted_average = weighted_sum / total_sum
        return weighted_average

    # PLOT UTILS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def plot_histogram(self, 
                       time_index=-1, 
                       bin_size=5,
                       plot_color='#DD0054', 
                       plot_bgcolor='#F1F6FA',
                       title='Histogram of ecDNA copy number',
                       save_fig=False):

        timestep_ecDNA_counts = self.cell_counts[time_index]
        data = [key for key, value in timestep_ecDNA_counts.items() for _ in range(value)]
        
        histogram_trace = go.Histogram(x=data, xbins=dict(start=0,size=bin_size), marker=dict(color=plot_color))

        layout = go.Layout(
            title=title+f" on {sum(timestep_ecDNA_counts.values()):,.0f} cells"+f" with {self.model} model and {self.fitness} fitness function (s={self.s:.3f})",
            xaxis=dict(title='Number of copies'),
            bargap=0.1,
            plot_bgcolor=plot_bgcolor
        )

        fig = go.Figure(data=[histogram_trace], layout=layout)

        fig.show()

        if save_fig:
            fig.update_layout(width=1000, height=500)
            fig.write_image(f"ecDNA_histogram_timestep[{time_index}].png", scale=3)

    def plot_evolution_area(self, 
                            colors={}, 
                            plot_bgcolor='#F1F6FA'):
        

        ecDNA_counts = [sum(self.cell_counts[i][key] for key in self.cell_counts[i] if key > 0) for i in range(len(self.cell_counts))]
        neutral_counts = [self.cell_counts[i][0] for i in range(len(self.cell_counts))]

        fig = go.Figure()

        data = {'time': self.times, 'ecDNA_count': ecDNA_counts, 'neutral_count': neutral_counts}
        df = pd.DataFrame(data)

        df_melted = pd.melt(df, id_vars=['time'], var_name='Type', value_name='Count')

        fig = px.area(df_melted, x='time', y='Count', color='Type', title='Evolution of cell count', labels={'Count': 'Count', 'time': 'Time'},
                      color_discrete_map={'ecDNA_count': colors.get('ecDNA','tomato'), 'neutral_count': colors.get('neutral', 'lightsteelblue')})
        
        fig.update_layout(plot_bgcolor=plot_bgcolor,)

        name_mapping = {'ecDNA_count': 'with ecDNA', 'neutral_count': 'without ecDNA'}
        for trace in fig.data:
            if trace.name in name_mapping: trace.name = name_mapping[trace.name]
        
        fig.show()

    def plot_mean_CN(self, plot_color='navy', plot_bgcolor='#F1F6FA'):
        times = self.times
        mean_ecDNA_copy_numbers = [self.get_mean(time_index=i) for i in range(len(self.times))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=mean_ecDNA_copy_numbers, mode='lines', marker=dict(color=plot_color), name='Mean ecDNA copy number'))
        fig.update_layout(title='Mean ecDNA copy number over time', xaxis=dict(title='Time'), yaxis=dict(title='Mean ecDNA copy number'),
                          plot_bgcolor=plot_bgcolor, showlegend=True, hovermode='closest',)
        fig.show()

    def plot_histograms_slider(self,
                               plot_color = 'deepskyblue',
                               title = 'Evolution of ecDNA copy number distribution over time',
                               plot_bgcolor = '#F1F6FA',
                               xlabel = 'Number of copies of ecDNA',
                               leap = 1000,
                               bin_size = 10,
                               upper_xaxis = 500):
        
        print('Generating slider histograms...')
        fig = go.Figure()

        for time_index in tqdm(range(0, len(self.times), leap)):
            timestep_ecDNA_counts = self.cell_counts[time_index]
            data = [key for key, value in timestep_ecDNA_counts.items() for _ in range(value)]
            fig.add_trace(go.Histogram(x=data, xbins=dict(size=bin_size), name="time = " + str(round(self.times[time_index], 1)), marker=dict(color=plot_color)))

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(method="update", args=[{"visible": [False] * len(fig.data)}],)
            step["args"][0]["visible"][i] = True
            steps.append(step)

        sliders = [dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)]

        fig.update_layout(sliders=sliders, title=title+ f" with {self.model} model and {self.fitness} fitness function (s={self.s})",
                          xaxis=dict(title=xlabel, range=[-0.5,upper_xaxis]), bargap=0.1, plot_bgcolor=plot_bgcolor)
        fig.show()

    def plot_event_times(self, plot_color='turquoise', plot_bgcolor='#F1F6FA', expected_color='navy'):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.times, y=list(range(len(self.times))), mode='markers', marker=dict(color=plot_color, size=5), 
                                 line=dict(width=3), name='Simulated event times'))
        
        x_line = np.array([min(self.times), max(self.times)])
        y_line = self.initial_rate * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Expected event times if the total fitness was constant',
                                 line=dict(color=expected_color, dash='dash'), hoverinfo='none'))
        
        fig.update_layout(title='Event times in cell population' + f" with {self.model} model and {self.fitness} fitness function (s={self.s})",
                          xaxis=dict(title='Time'), yaxis=dict(title='Event index'), plot_bgcolor=plot_bgcolor, showlegend=True, hovermode='closest',)
        fig.show()



# MAIN --------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate Gillespie process and plot histogram of cell counts.')

    parser.add_argument('--initial_nb_without', type=int, default=0, help='Initial count of neutral cells')
    parser.add_argument('--turnover', type=float, default=1, help='Turnover rate of cells')
    parser.add_argument('--max_cells', type=int, default=1e5, help='Maximum number of cells')
    parser.add_argument('--growth', type=str, default='constant', help='growth type : neutral or positive (ecDNA improves fitness)')
    parser.add_argument('--time_index', type=int, default=-1, help='Time index for the ecDNA copy number histogram - must be in range(0,len(times))')
    parser.add_argument('--plot_histogram', action='store_true', help='If the argument is present, then the histogram will be plotted. Otherwise it will not.')
    parser.add_argument('--plot_histograms_slider', action='store_true', help='If the argument is present, then the figure with histograms and slider will be plotted. Otherwise it will not.')    
    parser.add_argument('--plot_evolution', action='store_true', help='If the argument is present, then the evolution of cell counts will be plotted. Otherwise it will not.')

    args = parser.parse_args()

    # INITIALIZE POPULATION AND SIMULATE EVOLUTION
    population = Population(initial_nb_without=args.initial_nb_without, 
                            turnover=args.turnover, 
                            max_cells=args.max_cells, 
                            growth=args.growth)
    times, cell_counts = population.simulate_feedback()

    # PLOTS
    if args.plot_evolution:
        colors = {'ecDNA': 'tomato', 'neutral': 'lightsteelblue'}
        population.plot_evolution_area(colors=colors)

    if args.plot_histogram: population.plot_histogram(time_index=args.time_index, plot_color='deepskyblue', plot_bgcolor='#F1F6FA')

    if args.plot_histograms_slider: population.plot_histograms_slider()