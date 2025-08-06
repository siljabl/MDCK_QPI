'''
Compute correlations on PIV and stack

TO DO:
-
'''

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from cmcrameri import cm
from src.PlottingUtils import bin_by_density
from src.HolomonitorFunctions import get_pixel_size


# parse input
parser = argparse.ArgumentParser(description='Plot correlations based on data from cell tracks')
parser.add_argument('in_path',          type=str,  help="path to dataset")
parser.add_argument('-out_path',        type=str,  help="path to output. Using in_path if set to None", default=None)
parser.add_argument('-bin_size',        type=int,  help="size of density bins used to bin data",        default=100)
parser.add_argument('-frames_per_hour', type=int,  help="number of frames in an hour",                  default=12)
parser.add_argument('-average',         type=bool, help="plot time average correlation",                default=False)
args = parser.parse_args()

# if not given, use input folder for output also
if args.out_path == None:
    args.out_path = f"{args.in_path}/figs"

# make output directory if it doesn't exist
try:
    os.mkdir(f"{args.out_path}")
except:
    None



###############
# import data #
###############

# load correlations
with open(f"{args.in_path}/continuous_correlations.pkl", 'rb') as handle:
    data = pickle.load(handle)

f_max = len(data['C_r_vv'])

density = data['density'][:f_max]

t_arr = data['t_vv']
r_arr = data['r_vv']

# unit conversion
pix_to_um = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour

f_max = len(data['C_r_vv'])


##############################
# plot velocity correlations #
##############################

# bin correlations by density
C_r_binned, density_bins = bin_by_density(data['C_r_vv'], density, bin_size=args.bin_size)
C_t_binned, _            = bin_by_density(data['C_t_vv'], density, bin_size=args.bin_size)
C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)
print(data['C_t_vv'])

# define colormap
Nbins  = len(density_bins) - 1
colors = cm.roma_r(np.linspace(0, 1, Nbins))
sm     = plt.cm.ScalarMappable(cmap=cm.roma_r, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))

fig, ax = plt.subplots(1,2, figsize=(7,2.5))
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])

ax[0].set(xlabel=r"$t ~(h)$",  title=rf"$C_{{vv}}(t)$")
ax[1].set(xlabel=r"$r ~(µm)$",  title=rf"$C_{{vv}}(r)$")

# highlight y=0
ax[0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
ax[1].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")

# loop over density
for i in range(Nbins):
    ax[0].plot(t_arr, C_t_binned[i], color=colors[i])
    ax[1].plot(r_arr, C_r_binned[i], color=colors[i])


fig.subplots_adjust(right=0.8)
fig.colorbar(sm, label=r"$\rho_{cell} ~(mm^{-2})$", cax=cbar_ax)
fig.savefig(f"{args.in_path}/figs/autocorrelations_PIV.png", dpi=300, bbox_inches='tight')




############################
# plot height correlations #
############################

# bin correlations by density
C_r_binned, density_bins = bin_by_density(data['C_r_hh'], density, bin_size=args.bin_size)
#C_t_binned, _            = bin_by_density(data['C_t_hh'], density, bin_size=args.bin_size)
#C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)

r_hh = np.arange(len(C_r_binned[0])) * np.max(r_arr) / len(C_r_binned[0])

fig, ax = plt.subplots(1,2, figsize=(7,2.5))
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])

ax[0].set(xlabel=r"$t ~[h]$",  title=rf"$C_{{hh}}(t)$")
ax[1].set(xlabel=r"$r ~[µm]$",  title=rf"$C_{{hh}}(r)$")

# highlight y=0
ax[0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
ax[1].hlines(0, 0, r_hh.max(),  linestyles="dashed", color="gray")

# loop over density
for i in range(Nbins):
    #ax[0].plot(t_arr, C_t_binned[i], color=colors[i])
    ax[1].plot(r_hh,  C_r_binned[i], color=colors[i])


fig.subplots_adjust(right=0.8)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.in_path}/figs/autocorrelations_height_continous.png", dpi=300, bbox_inches='tight')




###########################
# plot cross correlations #
###########################

# bin correlations by density
C_r_binned, density_bins = bin_by_density(data['C_r_hv'], density, bin_size=args.bin_size)
#C_t_binned, _            = bin_by_density(data['C_t_hv'], density, bin_size=args.bin_size)#
#C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)


# define colormap
Nbins  = len(density_bins) - 1
colors = cm.roma_r(np.linspace(0, 1, Nbins))
sm     = plt.cm.ScalarMappable(cmap=cm.roma_r, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))

fig, ax = plt.subplots(1,2, figsize=(7,2.5))
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])

ax[0].set(xlabel=r"$t ~[h]$",  title=rf"$C_{{hv}}(t)$")
ax[1].set(xlabel=r"$r ~[µm]$",  title=rf"$C_{{hv}}(r)$")

# highlight y=0
ax[0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
ax[1].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")

# loop over density
for i in range(Nbins):
    #ax[0].plot(t_arr, C_t_binned[i], color=colors[i])
    ax[1].plot(r_arr, C_r_binned[i], color=colors[i])


fig.subplots_adjust(right=0.8)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.in_path}/figs/correlations_PIV_height_velocity.png", dpi=300, bbox_inches='tight')