import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:

    sys.path.append(module_path)

import json
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from cmcrameri import cm
from src.HolomonitorFunctions import get_pixel_size
from src.PlottingUtils import hist_to_curve

parser = argparse.ArgumentParser(description='Plot data set')
parser.add_argument('in_path',          type=str, help="data set as listed in holo_dict")
parser.add_argument('-out_path',        type=str, help="data set as listed in holo_dict", default=None)
parser.add_argument('-frames_per_hour', type=int, help="Number of frames in an hour",     default=12)
args = parser.parse_args()

# folder and path settings
config = json.load(open(f"{args.in_path}/config.txt"))

fmin = config["fmin"]
fmax = config["fmax"]
file = args.in_path.split("/")[-1]
dir  = args.in_path.split(file)[0]


try:
    os.mkdir(f"{args.in_path}/figs")
except:
    None

# unit conversion
pix_to_um = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour


# read data
with open(f"{args.in_path}/masked_arrays.pkl", 'rb') as handle:
    data = pickle.load(handle)

density = data['cell_density']

x  = data['x_position']
y  = data['y_position']
vx = data['x_displacement'] * pix_to_um[1] / frame_to_hour
vy = data['y_displacement'] * pix_to_um[1] / frame_to_hour


#####################
# Plotting velocity #
#####################
mean_xvelocity = np.ma.mean(vx, axis=1)
mean_yvelocity = np.ma.mean(vy, axis=1)

fig, ax = plt.subplots(2,2, figsize=(7,7))
ax[0,0].hlines(0, density.min(), density.max(), ls="dashed", color="gray")
ax[0,0].plot(density[:-1], mean_xvelocity, '.', label=r"$\dot{x}$")
ax[0,0].plot(density[:-1], mean_yvelocity, '.', label=r"$\dot{y}$")

# plot velocity distribution at various
# for i in range(len(h_binned)):
#     v_x, v_y, bins = hist_to_curve(v_binned[i], bins=11, hist_range=[-0.5, 10.5])
#     ax[1].plot(v_x * pix_to_um[0] / frame_to_hour, v_y, '-', color=colors[i])

ax[0].set(xlabel=r"$\rho_{cell} ~[mm^{-2}]$", ylabel="velocity [µm/h]")
ax[1].set(xlabel=r"$speed ~[µm/h]$", yscale="log")

ax[0].legend()
fig.tight_layout()
fig.savefig(f"{args.in_path}/figs/average_velocity.png", dpi=300)
# check if range of speed is correct
