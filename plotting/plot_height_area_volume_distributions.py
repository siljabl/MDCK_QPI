import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# add source files to path
module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from cmcrameri import cm
from src.PlottingUtils import hist_to_curve
from src.HolomonitorFunctions import get_pixel_size


parser = argparse.ArgumentParser(description='Plot data set')
parser.add_argument('in_path',   type=str, help="data set as listed in holo_dict")
parser.add_argument('-out_path', type=str, help="data set as listed in holo_dict", default=None)
parser.add_argument('-bin_size', type=int, help="data set as listed in holo_dict", default=100)
args = parser.parse_args()


# if not given, use input folder for outpur also
if args.out_path == None:
    args.out_path = args.in_path

# create figure folder
try:
    os.mkdir(f"{args.out_path}/figs")
except:
    None



#######################
# import and bin data #
#######################
with open(f"{args.in_path}/masked_arrays.pkl", 'rb') as handle:
    data = pickle.load(handle)


# format conversion
pix_to_um = get_pixel_size()

# round density range to nearest 100
density = data['cell_density']
min_density = int(data['cell_density'].min() / 100) *100
max_density = int(data['cell_density'].max() / 100) *100 + args.bin_size

density_bins = np.arange(min_density, max_density, args.bin_size)

# number of bins
Nbins = len(density_bins) - 1

binned_height = []
binned_area   = []
binned_volume = []
binned_shape  = []


for i in range(Nbins):

    # define low and high boundary on bin
    low_lim  = density_bins[i]   #min_density + args.bin_size * i
    high_lim = density_bins[i+1] #min_density + args.bin_size *(i+1)
    
    # mask relevant densities
    density_mask = (density >= low_lim) * (density < high_lim)

    heights_in_bin = data['mean_height'][density_mask]
    areas_in_bin   = data['cell_area']  [density_mask] * pix_to_um[1]**2
    volumes_in_bin = heights_in_bin * areas_in_bin
    shapes_in_bin  = ((data['major_axis'] / data['minor_axis']))[density_mask]

    binned_height.append(heights_in_bin.ravel())
    binned_area.  append(areas_in_bin.ravel())
    binned_volume.append(volumes_in_bin.ravel())
    binned_shape. append(shapes_in_bin.ravel())
        



######################
# plot distributions #
######################
colors  = cm.roma_r(np.linspace(0, 1, Nbins))

fig, ax = plt.subplots(2,2, figsize=(8,6))

for i in range(len(binned_height)):
    h_x, h_y, bins = hist_to_curve(binned_height[i], bins=22)
    A_x, A_y, bins = hist_to_curve(binned_area[i],   bins=30)
    V_x, V_y, bins = hist_to_curve(binned_volume[i], bins=40)
    p_x, p_y, bins = hist_to_curve(binned_shape[i],  bins=30)

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
fig.savefig(f"{args.out_path}/figs/height_area_volume_distributions.png", dpi=300)



##################
# Save as pickle #
##################
out_dict = {'area':    binned_area,
            'height':  binned_height,
            'volume':  binned_volume,
            'shape':   binned_shape,
            'density': density_bins}



# save as pickle
with open(f"{args.out_path}/hAV_distributions.pkl", 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

