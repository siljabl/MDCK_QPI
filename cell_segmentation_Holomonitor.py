import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from src.FormatConversions import import_holomonitor_stack
from src.CellSegmentation import *
from src.HolomonitorFunctions import *


parser = argparse.ArgumentParser(
    description="Usage: python cell_segmentation_Holomonitor.py dir file"
    )
parser.add_argument("path",          type=str,    help="Path to data series")
parser.add_argument("-Hmax",        type=float,  help="value to add to imextendedmin",                           default=0.03)
parser.add_argument("-Hmin",        type=float,  help="value to add to imextendedmin",                           default=0.01)
parser.add_argument("-s_high",      type=int,    help="kernel size for Gaussian filter applied to data",         default=5)
parser.add_argument("-s_low",       type=int,    help="kernel size for Gaussian filter subtracting from data",   default=10)
parser.add_argument("-scaling",     type=int,    help="Holomonitor scaling to µm",                               default=100)
parser.add_argument("-fmin",        type=int,    help="First useful frame",                                      default=1)
parser.add_argument("-fmax",        type=int,    help="First useful frame",                                      default=181)
parser.add_argument("-clear_edge",  type=bool,   help="Should be True if monolayer is larger than FOV, otherwise False")

args = parser.parse_args()



# create folders for output
# try:    os.mkdir(f"{args.path}/frames/cell_detection")
# except: None
try:    os.mkdir(f"{args.path}/processed_data")
except: None
try:    os.mkdir(f"{args.path}/figs")
except: None
try:    os.mkdir(f"{args.path}/videos")
except: None



# import data
path_to_tiff = f'{args.path}frames/raw/'
filename     = args.path.split("/")[-2]
h_im = import_holomonitor_stack(path_to_tiff, filename, f_min=args.fmin, f_max=args.fmax)


# empty arrays for storing data
cells_df = pd.DataFrame()
im_areas = []
im_edges = []

# conversion factor
pix_to_um = get_pixel_size()

# linear array of H for imextendmax
H_arr = np.linspace(args.Hmax, args.Hmin, len(h_im), endpoint=True)


# segment cells in each frame
for i in tqdm(range(len(h_im))):
    
    # identify cells
    n_norm = smoothen_normalize_im(h_im[i], args.s_high, args.s_low)
    pos = find_cell_pos(-n_norm, H_arr[i])

    # segment cell areas using watershed
    areas, edges = get_cell_areas(-n_norm, pos, h_im[i], clear_edge=True)
    pos = update_pos(pos, areas)

    # compute cell properties
    tmp_df = compute_cell_props(areas, pos, h_im[i], h_im[i], type='holo')
    tmp_df['frame'] = i

    # save to df and list
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)
    im_areas.append(areas)
    im_edges.append(edges)


# filter out small cells
cells_df = cells_df[cells_df.A*pix_to_um[0]**2 >= 100]
cells_df.to_csv(f"{args.path}/processed_data/dataframe_unfiltered.csv", index=False)

np.save(f"{args.path}/processed_data/im_cell_areas.npy", im_areas)
#np.save(f"{args.path}/processed_data/cell_edges.npy", im_edges)


# save input
config = {'date':   datetime.today().strftime('%Y-%m-%d'),
          'fmin':   args.fmin,
          'fmax':   args.fmax,
          's_high': args.s_high,
          's_low':  args.s_low,
          'Hmax':   args.Hmax,
          'Hmin':   args.Hmin}

json.dump(config, open(f"{args.path}/config.txt", "w"))
