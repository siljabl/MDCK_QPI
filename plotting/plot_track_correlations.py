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
with open(f"{args.in_path}/track_correlations.pkl", 'rb') as handle:
    data = pickle.load(handle)

variable = data['variable']
density  = data['density']

t_arr = data['t']
r_arr = data['r']

# unit conversion
pix_to_um     = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour



#####################
# plot correlations #
#####################

# dummy binning to get density bins
test, density_bins = bin_by_density(data['C_t'][0], density, bin_size=args.bin_size)

# define colormap
Nbins  = len(density_bins)
colors = cm.roma_r(np.linspace(0, 1, Nbins))
sm     = plt.cm.ScalarMappable(cmap=cm.roma_r, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))


#################
# temporal auto #
#################

fig, ax = plt.subplots(1,3, figsize=(10,2.5), sharey=True)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

# loop over variables
for i in range(3):

    # highlight y=0
    ax[i].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
    ax[i].set(xlabel=r"$t ~[h]$",   title=rf"$C_{variable[i]}(t)$")

    # bin by density
    C_t_binned, _ = bin_by_density(data['C_t'][i], density, bin_size=args.bin_size)
    C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)

    # loop over density
    for j in range(Nbins):
        ax[i].plot(t_arr, C_t_binned[j], color=colors[j])

fig.subplots_adjust(right=0.85)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/temporal_autocorrelations_height_area_volume.png", dpi=300, bbox_inches='tight')



################
# spatial auto #
################

fig, ax = plt.subplots(1,3, figsize=(10,2.5), sharey=True)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

# loop over variables
for i in range(3):

    ax[i].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")
    ax[i].set(xlabel=r"$r ~[µm]$",  title=rf"$C_{variable[i]}(r)$")

    # bin by density
    C_r_binned, _ = bin_by_density(data['C_r'][i], density, bin_size=args.bin_size)
    C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)

    # loop over density
    for j in range(Nbins-1):
        ax[i].plot(r_arr, C_r_binned[j], color=colors[j])

fig.subplots_adjust(right=0.85)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/spatial_autocorrelations_height_area_volume.png", dpi=300, bbox_inches='tight')



###############################
# temporal cross-correlations #
###############################

fig, ax = plt.subplots(1,3, figsize=(10,2.5), sharey=True)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

# loop over variables
for i in range(3):

    ax[i].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
    ax[i].set(xlabel=r"$t ~[h]$", title=rf"$C_{variable[3+i]}(t)$")
    
    # bin by density
    C_t_binned, _ = bin_by_density(data['C_t'][3+i], density, bin_size=args.bin_size)
    C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)

    # loop over density
    for j in range(Nbins-1):
        ax[i].plot(t_arr, C_t_binned[j], color=colors[j])

fig.subplots_adjust(right=0.85)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/temporal_crosscorrelations_height_area_volume.png", dpi=300, bbox_inches='tight')



##############################
# spatial cross-correlations #
##############################
fig, ax = plt.subplots(1,3, figsize=(10,2.5), sharey=True)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

# loop over variables
for i in range(3):

    ax[i].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")
    ax[i].set(xlabel=r"$r ~[µm]$", title=rf"$C_{variable[3+i]}(r)$")

    # bin by density
    C_r_binned, _ = bin_by_density(data['C_r'][3+i], density, bin_size=args.bin_size)
    C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)

    # loop over densities
    for j in range(Nbins-1):
        ax[i].plot(r_arr, C_r_binned[j], color=colors[j])

fig.subplots_adjust(right=0.85)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/spatial_crosscorrelations_height_area_volume.png", dpi=300, bbox_inches='tight')




#################
# plot averages #
#################
if args.average:
    fig, ax = plt.subplots(2,2, figsize=(6,5), sharey=True)

    for i in range(2):

        ax[i,0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
        ax[i,1].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")

        ax[i,0].set(xlabel=r"$t ~[h]$")
        ax[i,1].set(xlabel=r"$r ~[µm]$")

        for j in range(3):

            # bin by density
            C_t_binned, _ = bin_by_density(data['C_t'][3*i+j], density, bin_size=args.bin_size)
            C_r_binned, _ = bin_by_density(data['C_r'][3*i+j], density, bin_size=args.bin_size)
            C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)
            C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)


            C_t_mean = np.ma.mean(C_t_binned, axis=0)
            C_r_mean = np.ma.mean(C_r_binned, axis=0)

            ax[i,0].plot(t_arr, C_t_mean, '-', label=rf"$C_{variable[3*i+j]}$")
            ax[i,1].plot(r_arr, C_r_mean, '-', label=rf"$C_{variable[3*i+j]}$")

        ax[i,0].legend()
        ax[i,1].legend()

    fig.savefig(f"{args.out_path}/average_correlations_height_area_volume.png", dpi=300, bbox_inches='tight')



###################
# plot velocities #
###################
C_t_binned, density_bins = bin_by_density(data['C_t_vv'], density[:-1], bin_size=args.bin_size)
C_r_binned, _            = bin_by_density(data['C_r_vv'], density[:-1], bin_size=args.bin_size)

C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)
C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)


fig, ax = plt.subplots(1,2, figsize=(7,2.5), sharey=True)
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])

ax[0].set(xlabel=r"$t ~[h]$",  title=rf"$C_{{vv}}(t)$")
ax[1].set(xlabel=r"$r ~[µm]$", title=rf"$C_{{vv}}(r)$")

# highlight y=0
ax[0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
ax[1].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")

# loop over density
for j in range(Nbins-1):
    ax[0].plot(t_arr, C_t_binned[j], color=colors[j])
    ax[1].plot(r_arr, C_r_binned[j], color=colors[j])

fig.subplots_adjust(right=0.8)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/autocorrelations_velocity.png", dpi=300, bbox_inches='tight')



#############################
# plot velocity heigh cross #
#############################
C_t_binned, density_bins = bin_by_density(data['C_t'][-1], density[:-1], bin_size=args.bin_size)
C_r_binned, _            = bin_by_density(data['C_r'][-1], density[:-1], bin_size=args.bin_size)

C_t_binned = np.ma.array(C_t_binned, mask=C_t_binned==0)
C_r_binned = np.ma.array(C_r_binned, mask=C_r_binned==0)


fig, ax = plt.subplots(1,2, figsize=(7,2.5), sharey=True)
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])

ax[0].set(xlabel=r"$t ~[h]$",  title=rf"$C_{{dVdV}}(t)$")
ax[1].set(xlabel=r"$h ~[h]$",  title=rf"$C_{{dVdV}}(r)$")

# highlight y=0
ax[0].hlines(0, 0, t_arr.max(), linestyles="dashed", color="gray")
ax[1].hlines(0, 0, r_arr.max(), linestyles="dashed", color="gray")

# loop over density
for j in range(Nbins-1):
    ax[0].plot(t_arr, C_t_binned[j], color=colors[j])
    ax[1].plot(r_arr, C_r_binned[j], color=colors[j])

fig.subplots_adjust(right=0.8)
fig.colorbar(sm, label=r"$\rho_{cell} ~[mm^{-2}]$", cax=cbar_ax)
fig.savefig(f"{args.out_path}/autocorrelations_volume_change.png", dpi=300, bbox_inches='tight')

