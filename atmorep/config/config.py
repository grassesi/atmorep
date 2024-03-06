import os 
from pathlib import Path

FPATH = os.path.dirname(os.path.realpath(__file__))

YEAR_BASE = 1979
YEAR_LAST = 2022

PATH_MODELS = Path( FPATH, '../../models/')
PATH_RESULTS = Path( FPATH, '../../results/')
PATH_DATA = Path( FPATH, '../../data/')

