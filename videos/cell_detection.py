import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.morphology as morph

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from tqdm import tqdm
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar

from src.FormatConversions import import_holomonitor_stack
#from CellSegmentation import *
from src.HolomonitorFunctions import get_pixel_size

parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
parser.add_argument("dir",      type=str,  help="Path to data folder")
parser.add_argument("file",     type=str,  help="Name of data series")
parser.add_argument("-edges",   type=bool, help="Plot edges", default=False)
args = parser.parse_args()


# create out folder
try:
    os.mkdir(f"{args.dir}{args.file}/cell_detection")
except:
    None


# import data
config = json.load(open(f"{args.dir}{args.file}/config.txt"))
h_im = import_holomonitor_stack(args.dir, args.file, f_min=config['fmin'], f_max=config['fmax'])
A_im = np.load(f"{args.dir}{args.file}/cell_areas.npy")
df   = pd.read_csv(f"{args.dir}{args.file}/area_volume_unfiltered.csv")

# conversion factor
pix_to_um = get_pixel_size()

vmin = 0
vmax = h_im.max()

# loop through frames
for frame in tqdm(np.unique(df.frame)):

    # mask data frame
    tmp_df = df[df.frame == frame]

    # add edges to height field
    if args.edges:
        h_im_edges = np.zeros_like(A_im[frame], dtype=np.float64)

        for A_idx in np.unique(A_im[frame]):
            cell = (A_im[frame] == A_idx)
            cell_interior = morph.erosion(cell, footprint=morph.disk(2))

            h_im_edges += cell_interior*h_im[frame]

        h_im[frame] = h_im_edges


    
    # plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))

    sns.heatmap(h_im[frame].T, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label': 'h [µm]'})
    ax.plot(tmp_df.x, tmp_df.y, 'r.', ms=5)
    ax.set(title=f"#cells: {len(tmp_df)}")

    # add scalebar
    sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
    sb.location = 'lower left'
    ax.add_artist(sb)

    fig.tight_layout()
    plt.savefig(f"{args.dir}{args.file}/cell_detection/frame_{frame+1}.png", dpi=300);
    plt.close()

