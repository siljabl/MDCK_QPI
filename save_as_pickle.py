import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.HolomonitorFunctions import get_pixel_size
from src.FormatConversions import import_holomonitor_stack

parser = argparse.ArgumentParser(description='Transfer dataframe to pickle of masked arrays')
parser.add_argument('dataset', type=int, help="data set as listed in holo_dict")
args = parser.parse_args()


# folder and path settings
holo_dict = json.load(open("../data/Holomonitor/settings.txt"))

path = "../" + holo_dict["files"][args.dataset].split("../../")[-1]
fmin = holo_dict["fmin"][args.dataset]
fmax = holo_dict["fmax"][args.dataset]
file = path.split("/")[-1]
dir  = path.split(file)[0]

try:
    os.mkdir(f"{dir}{file}/instant_velocities")
except:
    None

# read data
print(dir)
stack     = import_holomonitor_stack(dir, file, f_min=fmin, f_max=fmax)
tracks    = pd.read_csv(f"{path}/cell_tracks.csv")
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
    
    cell_density[f] = 10 ** 6 * len(p) / np.sum(tracks[tracks.frame==f].A)

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

# save as pickle
with open(f"{path}/masked_arrays.pkl", 'wb') as handle:
    pickle.dump(x_position,     handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_position,     handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(x_displacement, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(x_displacement, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(cell_area,      handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(mean_height,    handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(minor_axis,     handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(major_axis,     handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot velocity field
for f in range(len(x_displacement)):
    Ncells = np.sum(x_position)

    # plot
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set(title=f"{file}, frame: {f+1}, #cells: {cell_density[f]}")
    ax.imshow(stack[f].T, origin="lower")
    ax.quiver(x_position[f], y_position[f], x_displacement[f], y_displacement[f], scale=75/pix_to_um[0])
    
    fig.tight_layout()
    fig.savefig(f"{dir}{file}/instant_velocities/frame_{f+1}.png");
    plt.close()

