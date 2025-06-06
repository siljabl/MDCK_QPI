import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from src.FormatConversions import import_tomocube_stack
from src.Segmentation3D import *
from src.CellSegmentation import *

parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Tomocube.py dir file")
parser.add_argument("dir",      type=str,   help="Path to data folder")
parser.add_argument("file",     type=str,   help="Name of data series")
parser.add_argument("-H",       type=float, help="value to add to imextendedmin",  default=0.015)
parser.add_argument("-s_low",   type=int,   help="kernel size for low pass Gaussian filter",  default=30)
parser.add_argument("-s_high",  type=int,   help="kernel size for high pass Gaussian filter", default=33)
parser.add_argument("-fmin",    type=int, help="First useful frame", default=1)
parser.add_argument("-fmax",    type=int, help="First useful frame", default=40)
args = parser.parse_args()


vox_to_um = get_voxel_size_35mm()
n_im, h_im = import_tomocube_stack(args.dir, args.file, h_scaling=vox_to_um[0], f_min=args.fmin, f_max=args.fmax)

try:
    os.mkdir(f"{args.dir}/cell_detection")
except:
    None


cells_df = pd.DataFrame()
im_areas = []
im_edges = []

H_arr = np.linspace(args.H, args.H-0.005, len(n_im), endpoint=True)

for i in tqdm(range(len(n_im))):
    # identify cells
    n_norm = smoothen_normalize_im(n_im[i], args.s_low, args.s_high)
    pos = find_cell_pos(-n_norm, H_arr[i])

    # segment cell areas using watershed
    # using less smoothed input to avoid too smooth cell areas
    n_norm = smoothen_normalize_im(n_im[i], 10, 15)
    areas, edges = get_cell_areas(-n_norm, pos, h_im[i], clear_edge=True)
    pos = update_pos(pos, areas)

    # compute cell properties
    tmp_df = compute_cell_props(areas, pos, h_im[i], n_im[i], vox_to_um)
    tmp_df['frame'] = i

    # save to df and list
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)
    im_areas.append(areas)
    im_edges.append(edges)

    # plot
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    fig.suptitle(f"frame: {i+1}, #cells: {len(tmp_df)}")
    ax[0].imshow(n_im[i].T, origin="lower", vmin=1.35)
    ax[1].imshow(n_norm.T,  origin="lower")

    ax[0].plot(tmp_df.x, tmp_df.y, 'r.', ms=5)
    ax[1].plot(tmp_df.x, tmp_df.y, 'r.', ms=5)

    ax[0].set(title="original image")
    ax[1].set(title="image fed to immax")
    fig.tight_layout()
    plt.savefig(f"{args.dir}/cell_detection/frame_{i+1}_sigma_{args.s_low}-{args.s_high}_H{args.H}.png");
    plt.close()

cells_df.to_csv(f"{args.dir}/area_volume_unfiltered.csv", index=False)
np.save(f"{args.dir}/cell_areas.npy", im_areas)
np.save(f"{args.dir}/cell_edges.npy", im_edges)

