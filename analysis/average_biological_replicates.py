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

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)


from tqdm import tqdm
from cmcrameri import cm
from src.PlottingUtils import hist_to_curve


# parse input
parser = argparse.ArgumentParser(description='Compute correlations of data from cell tracks')
#parser.add_argument('path',             type=str,   help="path to folder")
parser.add_argument('datasets',         nargs='*',  help="path to folder")
args = parser.parse_args()



# import data from all biological replicates
dicts_hAV = []
dicts_track_corr = []
dicts_cont_corr  = []

for data in args.datasets:
    with open(f"{data}/hAV_distributions.pkl", 'rb') as handle:
        dicts_hAV.append(pickle.load(handle))

    with open(f"{data}/track_correlations.pkl", 'rb') as handle:
        dicts_track_corr.append(pickle.load(handle))


# get unique range of densities 
density = np.unique(np.concatenate([dicts_hAV[i]['density'] for i in range(len(dicts_hAV))]))
Nbins   = len(density) - 1

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

            area_tmp.append(dicts_hAV[data]['area'][mask])
            shape_tmp.append(dicts_hAV[data]['shape'][mask])
            height_tmp.append(dicts_hAV[data]['height'][mask])
            volume_tmp.append(dicts_hAV[data]['volume'][mask])

            Cr_hh_tmp.append(dicts_track_corr[data]['Cr_hh'][mask])
            Cr_AA_tmp.append(dicts_track_corr[data]['Cr_AA'][mask])
            Cr_VV_tmp.append(dicts_track_corr[data]['Cr_VV'][mask])

    area.append(np.concatenate(area_tmp))
    shape.append(np.concatenate(shape_tmp))
    height.append(np.concatenate(height_tmp))
    volume.append(np.concatenate(volume_tmp))

    Cr_hh.append(np.concatenate(Cr_hh_tmp))
    Cr_AA.append(np.concatenate(Cr_AA_tmp))
    Cr_VV.append(np.concatenate(Cr_VV_tmp))

    i += 1



######################
# plot distributions #
######################
colors  = cm.roma_r(np.linspace(0, 1, Nbins))

fig, ax = plt.subplots(2,2, figsize=(8,6))

for i in range(Nbins):
    h_x, h_y, bins = hist_to_curve(height[i], bins=22)
    A_x, A_y, bins = hist_to_curve(area[i],   bins=22)
    V_x, V_y, bins = hist_to_curve(volume[i], bins=22)
    p_x, p_y, bins = hist_to_curve(shape[i],  bins=22)

    ax[0,0].plot(h_x, h_y, '-', color=colors[i])
    ax[0,1].plot(A_x, A_y, '-', color=colors[i])
    ax[1,0].plot(V_x, V_y, '-', color=colors[i])
    ax[1,1].plot(p_x, p_y, '-', color=colors[i])

   

ax[0,0].set(xlabel=r"$h ~[µm]$",   ylabel="PDF")
ax[0,1].set(xlabel=r"$A ~[µm^2]$")
ax[1,0].set(xlabel=r"$V ~[µm^3]$", ylabel="PDF")
ax[1,1].set(xlabel=r"shape index", xlim=(0.9,3.1))
ax[1,0].set(xlim=(990,6000))

fig.tight_layout()
fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
sm      = plt.cm.ScalarMappable(cmap=cm.roma_r, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))

fig.colorbar(sm, cax=cbar_ax, label=r"$\rho_{cell}(t) ~[mm^{-2}]$")
fig.savefig(f"../figs/height_area_volume_distributions_average.png", dpi=300)



################
# spatial auto #
################

r_arr = dicts_track_corr[0]['r_arr']
fig, ax = plt.subplots(1,3, figsize=(10,2.5), sharey=True)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

# loop over variables
for i in range(3):

    #ax[i].hlines(0, 0, dicts_track_corr[0]['r_arr'].max(), linestyles="dashed", color="gray")
    #ax[i].set(xlabel=r"$r ~[µm]$",  title=rf"$C_{variable[i]}(r)$")

    # bin by density
    #C_r_binned, _ = bin_by_density(data['C_r'][i], density, bin_size=args.bin_size)
    #C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)

    # loop over density
    for j in range(Nbins):
        ax[i].plot(Cr_hh[j])

fig.subplots_adjust(right=0.85)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/spatial_autocorrelations_height_area_volume.png", dpi=300, bbox_inches='tight')


