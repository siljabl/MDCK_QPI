import json
import pickle
import argparse
import numpy as np
import pandas as pd

from src.FormatConversions import import_holomonitor_stack

parser = argparse.ArgumentParser(description='Transfer dataframe to pickle of masked arrays')
parser.add_argument('dataset', type=int, help="data set as listed in holo_dict")
args = parser.parse_args()

# impoert data
holo_dict = json.load(open("../../data/Holomonitor/settings.txt"))

path = holo_dict["files"][args.dataset]
fmin = holo_dict["fmin"][args.dataset]
fmax = holo_dict["fmax"][args.dataset]
file = path.split("/")[-1]
dir  = path.split(file)[0]

tracks = pd.read_csv(f"{path}/area_volume_filtered.csv") # change to cell_tracks

fMax = np.max(tracks.frame) + 1
pMax = np.max(tracks.particle) + 1

x_arr = -np.ones([fMax, pMax])
y_arr = -np.ones([fMax, pMax])
A_arr = -np.ones([fMax, pMax])
h_arr = -np.ones([fMax, pMax])
a_max = -np.ones([fMax, pMax])
a_min = -np.ones([fMax, pMax])

for f in range(fMax):
    p = tracks[tracks.frame==f].particle.values

    # position
    x_arr[f, p] = tracks[tracks.frame==f].x.values
    y_arr[f, p] = tracks[tracks.frame==f].y.values

    # shape
    A_arr[f, p] = tracks[tracks.frame==f].A.values
    h_arr[f, p] = tracks[tracks.frame==f].h_mean.values
    a_max[f, p] = tracks[tracks.frame==f].a_max.values
    a_min[f, p] = tracks[tracks.frame==f].a_min.values

x_position  = np.ma.masked_array(x_arr, mask = (x_arr < 0) * (y_arr < 0))
y_position  = np.ma.masked_array(y_arr, mask = (x_arr < 0) * (y_arr < 0))
cell_area   = np.ma.masked_array(A_arr, mask = (x_arr < 0) * (y_arr < 0))
mean_height = np.ma.masked_array(h_arr, mask = (x_arr < 0) * (y_arr < 0))
minor_axis  = np.ma.masked_array(a_min, mask = (x_arr < 0) * (y_arr < 0))
major_axis  = np.ma.masked_array(a_max, mask = (x_arr < 0) * (y_arr < 0))

