'''
Compute correlations on PIV and stack

TO DO:
- add local density
- add height
'''

import os
import sys
import json
import pickle
import argparse
import subprocess

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.PlottingUtils import bin_by_density
from src.FormatConversions import import_holomonitor_stack, import_txt_with_NaN
from src.HolomonitorFunctions    import get_pixel_size
from src.Correlations            import general_spatial_autocorrelation
from src.MaskedArrayCorrelations import general_temporal_correlation

# parse input
parser = argparse.ArgumentParser(description='Compute correlations of data from cell tracks')
parser.add_argument('path',             type=str,   help="path to folcer")
args = parser.parse_args()


# import binned data
with open(f"{args.path}/hAV_distributions.pkl", 'rb') as handle:
    data_tmp = pickle.load(handle)