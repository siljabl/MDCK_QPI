'''

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
parser.add_argument('datasets',         nargs='*',  help="path to folcer")
args = parser.parse_args()

# make input arguments
well = args.datasets[0].split("-")[0]
print(f"Computing average of technical replicates in well {well}")


# import data from all techincal replicates
dicts = []
for data in args.datasets:
    with open(f"{args.path}{data}/hAV_distributions.pkl", 'rb') as handle:
        dicts.append(pickle.load(handle))


# get unique range of densities 
density = np.unique(np.concatenate([dicts[i]['density'] for i in range(len(dicts))]))


# sort by density
area   = []
height = []
volume = []
shape  = []

i = 0
for rho in density[:-1]:
    area_tmp   = []
    shape_tmp  = []
    height_tmp = []
    volume_tmp = []

    for data in range(len(args.datasets)):
        if rho in dicts[data]['density'][:-1]:

            mask = np.where(rho == dicts[data]['density'])[0][0]

            area_tmp.append(dicts[data]['area'][mask].compressed())
            shape_tmp.append(dicts[data]['shape'][mask].compressed())
            height_tmp.append(dicts[data]['height'][mask].compressed())
            volume_tmp.append(dicts[data]['volume'][mask].compressed())

    area.append(np.concatenate(area_tmp))
    shape.append(np.concatenate(shape_tmp))
    height.append(np.concatenate(height_tmp))
    volume.append(np.concatenate(volume_tmp))

    i += 1


out_dict = {'density': density,
            'area':    area,
            'shape':   shape,
            'height':  height,
            'volume':  volume
            }


# save as pickle
with open(f"{args.path}/hAV_distributions_well_{well}.pkl", 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


