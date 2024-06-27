# LOADING AND SAVING DATA -----------------------------------------------------------------------------------------------------------------------------------------------

import yaml
import os

def save_yaml(dictionary, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
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