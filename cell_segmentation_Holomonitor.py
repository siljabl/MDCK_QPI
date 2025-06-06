import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from tifffile import tifffile

from src.FormatConversions import import_holomonitor_stack
from src.CellSegmentation import *
from src.HolomonitorFunctions import *

parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
parser.add_argument("dir",      type=str, help="Path to data folder")
parser.add_argument("file",     type=str, help="Name of data series")
parser.add_argument("-Hmax",   type=float, help="value to add to imextendedmin",  default=0.03)
parser.add_argument("-Hmin",   type=float, help="value to add to imextendedmin",  default=0.01)
parser.add_argument("-s_low",   type=int, help="kernel size for low pass Gaussian filter",  default=5)
parser.add_argument("-s_high",  type=int, help="kernel size for high pass Gaussian filter", default=10)
parser.add_argument("-scaling", type=int, help="Holomonitor scaling to µm", default=100)
parser.add_argument("-fmin",    type=int, help="First useful frame", default=1)
parser.add_argument("-fmax",    type=int, help="First useful frame", default=181)
args = parser.parse_args()


# Import data
h_dir = f"{args.dir}{args.file}"
path = Path(h_dir)
try:
    os.mkdir(f"{h_dir}/cell_detection")
except:
    None

# # Check filenaming convension
h_im = import_holomonitor_stack(args.dir, args.file, f_min=args.fmin, f_max=args.fmax)

cells_df = pd.DataFrame()
im_areas = []
im_edges = []

pix_to_um = get_pixel_size()
H_arr = np.linspace(args.Hmax, args.Hmin, len(h_im), endpoint=True)

for i in tqdm(range(len(h_im))):
    # identify cells
    n_norm = smoothen_normalize_im(h_im[i], args.s_low, args.s_high)
    pos = find_cell_pos(-n_norm, H_arr[i])

    # segment cell areas using watershed
    areas, edges = get_cell_areas(-n_norm, pos, h_im[i], clear_edge=True)
    pos = update_pos(pos, areas)

    # compute cell properties
    tmp_df = compute_cell_props(areas, pos, h_im[i], h_im[i], pix_to_um)
    tmp_df['frame'] = i

    # save to df and list
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)
    im_areas.append(areas)
    im_edges.append(edges)

    # plot
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    fig.suptitle(f"{args.file}, frame: {i+1}, #cells: {len(tmp_df)}")
    ax[0].imshow(h_im[i].T, origin="lower")
    ax[1].imshow(n_norm.T,  origin="lower")

    ax[0].plot(tmp_df.x, tmp_df.y, 'r.', ms=5)
    ax[1].plot(tmp_df.x, tmp_df.y, 'r.', ms=5)

    ax[0].set(title="original image")
    ax[1].set(title="image fed to immax")
    fig.tight_layout()
    plt.savefig(f"{h_dir}/cell_detection/frame_{i+1}_sigma_{args.s_low}_{args.s_high}_H{args.Hmax}.png");
    plt.close()


cells_df.to_csv(f"{h_dir}/area_volume_unfiltered.csv", index=False)
np.save(f"{h_dir}/cell_areas.npy", im_areas)
np.save(f"{h_dir}/cell_edges.npy", im_edges)
