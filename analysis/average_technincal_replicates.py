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
parser.add_argument('path',             type=str,   help="path to folder")
parser.add_argument('datasets',         nargs='*',  help="path to folder")
args = parser.parse_args()

# get well name
well = args.datasets[0].split("-")[0]
print(f"Computing average of technical replicates in well {well}")




# import data from all techincal replicates
dicts_hAV = []
dicts_track_corr = []
dicts_cont_corr  = []

for data in args.datasets:
    with open(f"{args.path}{data}/hAV_distributions.pkl", 'rb') as handle:
        dicts_hAV.append(pickle.load(handle))

    with open(f"{args.path}{data}/track_correlations.pkl", 'rb') as handle:
        dicts_track_corr.append(pickle.load(handle))

# get unique range of densities 
density = np.unique(np.concatenate([dicts_hAV[i]['density'] for i in range(len(dicts_hAV))]))



# sort by density
area   = []
height = []
volume = []
shape  = []
Cr_hh  = [] 
Cr_AA  = []
Cr_VV  = []



i = 0
for rho in density[:-1]:
    area_tmp   = []
    shape_tmp  = []
    height_tmp = []
    volume_tmp = []
    Cr_hh_tmp  = []
    Cr_AA_tmp  = []
    Cr_VV_tmp  = []

    for data in range(len(args.datasets)):
        if rho in dicts_hAV[data]['density'][:-1]:

            mask = np.where(rho == dicts_hAV[data]['density'])[0][0]

            area_tmp.append(dicts_hAV[data]['area'][mask].compressed())
            shape_tmp.append(dicts_hAV[data]['shape'][mask].compressed())
            height_tmp.append(dicts_hAV[data]['height'][mask].compressed())
            volume_tmp.append(dicts_hAV[data]['volume'][mask].compressed())

            Cr_hh_tmp.append(dicts_track_corr[data]['C_r'][0][mask].compressed())
            Cr_AA_tmp.append(dicts_track_corr[data]['C_r'][1][mask].compressed())
            Cr_VV_tmp.append(dicts_track_corr[data]['C_r'][2][mask].compressed())

    area.append(np.concatenate(area_tmp))
    shape.append(np.concatenate(shape_tmp))
    height.append(np.concatenate(height_tmp))
    volume.append(np.concatenate(volume_tmp))

    Cr_hh.append(np.concatenate(Cr_hh_tmp))
    Cr_AA.append(np.concatenate(Cr_AA_tmp))
    Cr_VV.append(np.concatenate(Cr_VV_tmp))

    i += 1


out_dict_hAV = {'density': density,
                'area':    area,
                'shape':   shape,
                'height':  height,
                'volume':  volume
                }

out_dict_track_corr = {'Cr_hh': Cr_hh,
                       'Cr_AA': Cr_AA,
                       'Cr_VV': Cr_VV,
                       'r_arr': dicts_track_corr[0]['r']}

# create out folder
try:
    os.mkdir(f"{args.path}/Well_{well}")
except:
    None



# save as pickle
with open(f"{args.path}/Well_{well}/hAV_distributions.pkl", 'wb') as handle:
    pickle.dump(out_dict_hAV, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(f"{args.path}/Well_{well}/track_correlations.pkl", 'wb') as handle:
    pickle.dump(out_dict_track_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)


# add continous correlations ...
