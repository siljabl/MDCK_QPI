import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.morphology as morph

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar

from src.FormatConversions import import_holomonitor_stack
from src.HolomonitorFunctions import get_pixel_size

parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
parser.add_argument('path',          type=str,   help="path to dataset")
parser.add_argument("-edges",   type=bool, help="Plot edges", default=False)
args = parser.parse_args()


# create out folder
try:
    os.mkdir(f"{args.path}/cell_height_and_volume")
except:
    None

file = args.path.split("/")[-2]
dir  = args.path.split(file)[0]


# import data
config = json.load(open(f"{args.path}/config.txt"))
h_im = import_holomonitor_stack(dir, file, f_min=config['fmin'], f_max=config['fmax'])
A_im = np.load(f"{args.path}/cell_areas.npy")
df   = pd.read_csv(f"{args.path}/cell_tracks.csv")



# conversion factor
pix_to_um = get_pixel_size()


# set value range
hmin = 0
hmax = 14
Vmin = 1000
Vmax = 10_000

# define colormap
h_cmap = sns.color_palette("Blues",   as_cmap=True)
V_cmap = sns.color_palette("Oranges", as_cmap=True)
e_cmap = mpl.colors.ListedColormap(['none', 'w'])


# loop through frames
for frame in tqdm(np.unique(df.frame)):

    # mask data frame
    df_frame = df[df.frame == frame]

    h_mean = np.zeros_like(A_im[frame], dtype=np.float64)
    V_mean = np.zeros_like(A_im[frame], dtype=np.float64)
    e_im   = np.zeros_like(A_im[frame], dtype=int)

    # create ims of h_avrg and V
    for A_idx in np.unique(A_im[frame]):

        # isolate cell
        cell_mask    = (A_im[frame] == A_idx)
        df_cell_mask = (df_frame.label==A_idx)

        # skip if label doesn't belong to cell
        if np.sum(df_cell_mask) == 0:
            continue

        cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
        edge = cell_mask ^ cell_interior

        # cells_with_edges   += cell_interior*h_im[0]
        # heights_with_edges += cell_interior*cellwise_heigts
        e_im += edge

        h_mean[cell_mask] = df_frame[df_cell_mask].h_avrg.values[0] 
        V_mean[cell_mask] = df_frame[df_cell_mask].V.values[0] * pix_to_um[0] ** 2

    e_im += (h_mean == 0)


    
    # plot
    fig_h, ax_h = plt.subplots(1,1, figsize=(10,8))
    fig_V, ax_V = plt.subplots(1,1, figsize=(10,8))

    sns.heatmap(h_mean.T, ax=ax_h, square=True, cmap=h_cmap, vmin=hmin, vmax=hmax, xticklabels=False, yticklabels=False, cbar=True)#, cbar_kws={'label': 'h [µm]'})
    sns.heatmap(V_mean.T, ax=ax_V, square=True, cmap=V_cmap, vmin=Vmin, vmax=Vmax, xticklabels=False, yticklabels=False, cbar=True)#, cbar_kws={'label': 'h [µm]'})

    sns.heatmap(e_im.T, ax=ax_h, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
    sns.heatmap(e_im.T, ax=ax_V, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)

    ax_h.set(title=f"average cell height (µm)")
    ax_V.set(title=f"cell volume (µm³)")


    # # add scalebar
    # sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
    # sb.location = 'lower left'
    # ax.add_artist(sb)

    # save
    fig_h.tight_layout()
    fig_V.tight_layout()

    fig_h.savefig(f"{args.path}/cell_height_and_volume/height_frame_{frame+1}.png", dpi=300);
    fig_V.savefig(f"{args.path}/cell_height_and_volume/volume_frame_{frame+1}.png", dpi=300);

    plt.close(fig_h)
    plt.close(fig_V)

