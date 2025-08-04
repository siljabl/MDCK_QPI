import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar

from src.HolomonitorFunctions import get_pixel_size
from src.FormatConversions import import_holomonitor_stack

parser = argparse.ArgumentParser(description='Transfer dataframe to pickle of masked arrays')
parser.add_argument('path',          type=str, help="path to datasett")
#parser.add_argument('dataset', type=int, help="data set as listed in config")
args = parser.parse_args()


# folder and path settings
file = args.path.split("/")[-2]
dir  = args.path.split(file)[0]

config = json.load(open(f"{args.path}/config.txt"))
fmin = config["fmin"]
fmax = config["fmax"]


try:
    os.mkdir(f"{args.path}/instant_velocities")
except:
    None

# read data
stack     = import_holomonitor_stack(dir, file, f_min=fmin, f_max=fmax)
tracks    = pd.read_csv(f"{args.path}/cell_tracks.csv")
pix_to_um = get_pixel_size()


# sort data frame to masked arrays
fMax = np.max(tracks.frame) + 1
pMax = np.max(tracks.particle) + 1

x_arr = -np.ones([fMax, pMax])
y_arr = -np.ones([fMax, pMax])
A_arr = np.zeros([fMax, pMax])
h_arr = np.zeros([fMax, pMax])
a_max = np.zeros([fMax, pMax])
a_min = np.zeros([fMax, pMax])
cell_density = np.zeros(fMax)

for f in range(fMax):
    p = tracks[tracks.frame==f].particle.values
    
    cell_density[f] = 10 ** 6 * len(p) / np.sum(tracks[tracks.frame==f].A*pix_to_um[-1]**2)

    # position
    x_arr[f, p] = tracks[tracks.frame==f].x.values
    y_arr[f, p] = tracks[tracks.frame==f].y.values

    # shape
    A_arr[f, p] = tracks[tracks.frame==f].A.values
    h_arr[f, p] = tracks[tracks.frame==f].h_avrg.values
    a_max[f, p] = tracks[tracks.frame==f].a_max.values
    a_min[f, p] = tracks[tracks.frame==f].a_min.values

x_position  = np.ma.masked_array(x_arr, mask = (x_arr < 0) * (y_arr < 0))
y_position  = np.ma.masked_array(y_arr, mask = (x_arr < 0) * (y_arr < 0))
cell_area   = np.ma.masked_array(A_arr, mask = (x_arr < 0) * (y_arr < 0))
mean_height = np.ma.masked_array(h_arr, mask = (x_arr < 0) * (y_arr < 0))
minor_axis  = np.ma.masked_array(a_min, mask = (x_arr < 0) * (y_arr < 0))
major_axis  = np.ma.masked_array(a_max, mask = (x_arr < 0) * (y_arr < 0))

# compute instantaneous velocity
x_displacement = np.ma.diff(x_position, axis=0)
y_displacement = np.ma.diff(y_position, axis=0)

out_dict = {'cell_density': cell_density,
             'x_position': x_position, 
             'y_position': y_position,
             'x_displacement': x_displacement,
             'y_displacement': y_displacement,
             'cell_area': cell_area,
             'mean_height': mean_height,
             'minor_axis': minor_axis,
             'major_axis': major_axis}

# save as pickle
with open(f"{args.path}/masked_arrays.pkl", 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# plot velocity field
for f in tqdm(range(len(x_displacement))):
    Ncells = np.sum(x_position)

    # plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(stack[f].T, ax=ax, square=True, cmap="gray", vmin=0, vmax=20, 
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label': 'h [µm]'})
    
    ax.invert_yaxis()
    ax.quiver(x_position[f], y_position[f], x_displacement[f], y_displacement[f], scale=75/pix_to_um[-1], color="c")


    # add scalebar
    sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
    sb.location = 'lower left'
    ax.add_artist(sb)


    # save
    fig.tight_layout()
    fig.savefig(f"{args.path}/instant_velocities/frame_{f+1:03d}.png", dpi=300);
    plt.close()

