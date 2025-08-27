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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from cmcrameri import cm
from src.PlottingUtils import bin_by_density
from src.FormatConversions import import_holomonitor_stack
from src.HolomonitorFunctions    import get_pixel_size
from src.Correlations            import velocity_spatial_autocorrelation
from src.MaskedArrayCorrelations import general_temporal_correlation

# parse input
parser = argparse.ArgumentParser(description='Compute correlations of data from cell tracks')
parser.add_argument('in_path',          type=str,   help="path to dataset")
parser.add_argument('-out_path',        type=str,   help="path to output. Using in_path if set to None",               default=None)
parser.add_argument('-dr',              type=int,   help="size of radial bin for spatial correlation [pix]",           default=40)
parser.add_argument('-r_max',           type=int,   help="max radial distance for spatial correlation [pix]",          default=500)
parser.add_argument('-t_max',           type=float, help="max fraction of timeinterval used in temporal correlation",  default=0.5)
parser.add_argument('-frames_per_hour', type=int,   help="number of frames in an hour",                                default=12)
parser.add_argument('-fmax',            type=int,   default=None)
parser.add_argument('-plot_PIV',        type=bool,  help="plot and save PIV velocity field",  default=False)
parser.add_argument('-vlim',            type=float, help="plot and save PIV velocity field", default=None)
args = parser.parse_args()

# if not given, use input folder for output also
if args.out_path == None:
    args.out_path = f"{args.in_path}/figs"


# Folder for plotting velocity fields
try:
    os.mkdir(f"{args.in_path}/PIV/velocity_fields")
except:
    None


# folder and path settings
config = json.load(open(f"{args.in_path}/config.txt"))

fmin = config["fmin"]
fmax = config["fmax"]
if args.fmax != None:
    fmax = args.fmax

file = args.in_path.split("/")[-1]
dir  = args.in_path.split(file)[0]



###############
# import data #
###############

# unit conversion
pix_to_um = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour


# import stack
stack = import_holomonitor_stack(dir, file, fmin, fmax)
dims  = np.shape(stack) 

# import PIV velocity field
data_PIV = np.loadtxt(f"{args.in_path}/PIV/velocities/PIVlab_0001.txt", delimiter=",", skiprows=3)
x = np.array(data_PIV[:, 0], dtype=int)
y = np.array(data_PIV[:, 1], dtype=int)



###########################
# transform data to image #
###########################

# transform PIV position to matrix entries
dx   = y[1] - y[0]
x0   = y[0] - dx
xmax = int((np.max(y) - x0) / dx) + 1

x_tmp = np.array((x - x0) / dx, dtype=int)
y_tmp = np.array((y - x0) / dx, dtype=int)

#PIV_position_x = np.zeros((fmax, xmax, xmax), dtype=np.float64)
#PIV_position_y = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_velocity_x = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_velocity_y = np.zeros((fmax, xmax, xmax), dtype=np.float64)

# Fill arrays
for frame in range(fmax):

    # Load data
    data_PIV = np.loadtxt(f"{args.in_path}/PIV/velocities/PIVlab_{frame+1:04d}.txt", delimiter=",", skiprows=3)

    # Extract values
    u = np.array(data_PIV[:, 2], dtype=np.float64) * pix_to_um[0] / frame_to_hour
    v = np.array(data_PIV[:, 3], dtype=np.float64) * pix_to_um[0] / frame_to_hour

    # masking didn't work properly, so probably mean is affected by outside data points
    #PIV_position_x[frame-1, x_tmp, y_tmp] = x
    #PIV_position_y[frame-1, x_tmp, y_tmp] = y
    PIV_velocity_x[frame-1, x_tmp, y_tmp] = u - np.mean(u)
    PIV_velocity_y[frame-1, x_tmp, y_tmp] = v - np.mean(v)


# center position
idx_c  = int(xmax / 2)
if args.vlim == None:
    vlim_x = np.max(abs(PIV_velocity_x))
    vlim_y = np.max(abs(PIV_velocity_y))
    vlim   = np.max([vlim_x, vlim_y])
else:
    vlim = args.vlim

fig, ax = plt.subplots(2,3, figsize=(7.5,5.5), sharex=True, sharey='col')

for i in range(2):
    ax[i,0].imshow(stack[0], cmap="gray")
    ax[i,0].axis('off')

    ax[i,1].set(title="x-velocity [µm/h]", ylabel="t [h]")
    ax[i,2].set(title="y-velocity [µm/h]")

    ax[0,i+1].set(xlabel="x [µm]")
    ax[1,i+1].set(xlabel="y [µm]")

ax[0,0].hlines(idx_c*dx, 0, xmax*dx, linestyle="dashed", color="c")
ax[1,0].vlines(idx_c*dx, 0, xmax*dx, linestyle="dashed", color="c")

im01 = ax[0,1].imshow(PIV_velocity_x[:,idx_c], aspect="auto", extent=[0, dims[1], 0, fmax*frame_to_hour], cmap=cm.vik, vmin=-vlim, vmax=vlim)
im02 = ax[0,2].imshow(PIV_velocity_y[:,idx_c], aspect="auto", extent=[0, dims[1], 0, fmax*frame_to_hour], cmap=cm.vik, vmin=-vlim, vmax=vlim)

im11 = ax[1,1].imshow(PIV_velocity_x[:,:,idx_c], aspect="auto", extent=[0, dims[1], 0, fmax*frame_to_hour], cmap=cm.vik, vmin=-vlim, vmax=vlim)
im12 = ax[1,2].imshow(PIV_velocity_y[:,:,idx_c], aspect="auto", extent=[0, dims[1], 0, fmax*frame_to_hour], cmap=cm.vik, vmin=-vlim, vmax=vlim)

plt.colorbar(im01, ax=ax[0,1])
plt.colorbar(im02, ax=ax[0,2])
plt.colorbar(im11, ax=ax[1,1])
plt.colorbar(im12, ax=ax[1,2])

ax[0,0].set(title=file)
fig.tight_layout()
fig.savefig(f"{args.out_path}/PIV_kymograph.png", dpi=300);
plt.close()

