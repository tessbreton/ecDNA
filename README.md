# Evolutionary dynamics of ecDNA counts in cancer cells

This repository was used during my internship at the IICD in April-July 2024. I worked on modeling ecDNA dynamics in cancer cells populations. We tried to build a model that would 'explain' the reference data, which comes from [this paper](https://www.nature.com/articles/s41467-024-47619-4) and can be found in ```experiments/CAM277/data```. I used this code to run simulations an to run the parameter inference using an ABC algorithm. 

Most of the code was run on a distant server via ```.sh``` files, but they can all be run locally using the command lines given in these files.

## Package requirements

### Create a Python environment
I recommend using a virtual environment specific to this project to avoid conflicting packages. It can be done easily using venv:
```bash
python -m venv ecDNA
source venv/bin/activate
```

### Install packages
```bash
pip install -r requirements.txt
```

If you encounter the error `ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed` while rendering a Plotly figure, then upgrade nbformat by running the following command in your terminal or command prompt
```bash
pip install --upgrade nbformat
```

and then **restart VSCode** completely. This should solve the issue.


## Code overview
The easiest way to get started with the code is through the notebook ```moran.ipynb```, which shows how to run a simulation and visualize the final distributions.

The simulation algorithms are implemented in ```model.py```, in the class ```Population```. The model we decided to use throughout the whole project is the one implemented in the method ```simulate_moran```.

The ```utils``` folder is a package of functions that are useful for plots, laoding or others, sorted accordingly in separate files.

## ABC inference

### Run random simulations for ABC inference

```bash
sbatch runsimulations.sh
```
Specify parameters in the ```runsimulations.sh``` file in the following command line. Use ```--sample selection``` to sample only the selection parameter, and ```--sample double``` for both the selection parameter and the starting passage. 

```bash
python runsimulations.py --expname runs/P-5 --sample selection --start -5 --num_samples 1000
```


### Run the ABC inference
```bash
sbatch abcinference.sh
```
Parameters must be specified in the ```abcinference.sh``` file in the command line:
```bash
python abcinference.py --expname CAM277 --inference selection
```

### Run synthetic testing
```bash
sbatch abctesting.sh
```