import os 
from pathlib import Path

fpath = os.path.dirname(os.path.realpath(__file__))

year_base = 1979
year_last = 2022

path_models = Path( fpath, '../../models/')
path_results = Path( fpath, '../../results/')
path_data = Path( fpath, '../../data/')

