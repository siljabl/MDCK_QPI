import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from src.FormatConversions import import_tomocube_stack
from src.Segmentation3D import *
from src.CellSegmentation import *

parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Tomocube.py dir file")
parser.add_argument("dir",      type=str,   help="Path to data folder")
parser.add_argument("file",     type=str,   help="Name of data series")
parser.add_argument("-Hmax",    type=float, help="value to add to imextendedmin",  default=0.015)
parser.add_argument("-Hmin",    type=float, help="value to add to imextendedmin",  default=0.01)
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

H_arr = np.linspace(args.Hmax, args.Hmin, len(n_im), endpoint=True)

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
    tmp_df = compute_cell_props(areas, pos, h_im[i], n_im[i], type='tomo')
    tmp_df['frame'] = i

    # save to df and list
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)
    im_areas.append(areas)
    im_edges.append(edges)


cells_df.to_csv(f"{args.dir}/area_volume_unfiltered.csv", index=False)
np.save(f"{args.dir}/cell_areas.npy", im_areas)
np.save(f"{args.dir}/cell_edges.npy", im_edges)


# save input
config = {'fmin': args.fmin,
          'fmax': args.fmax,
          's_low': args.s_low,
          's_high': args.s_high,
          'Hmax': args.Hmax,
          'Hmin': args.Hmin}

json.dump(config, open(f"{args.dir}{args.file}/config.txt", "w"))
