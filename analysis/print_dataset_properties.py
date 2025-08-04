
import os
import sys
import argparse
import numpy as np

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

import json
from scipy.stats import linregress
from scipy.optimize import curve_fit

from src.Segmentation3D import get_voxel_size_35mm
from src.HolomonitorFunctions import get_pixel_size
from src.FormatConversions import import_holomonitor_stack, import_tomocube_stack
from src.CellSegmentation import *



parser = argparse.ArgumentParser(description='Plot data set')
parser.add_argument('path',          type=str, help="data set as listed in holo_dict")
parser.add_argument('-frames_per_hour', type=int, help="Number of frames in an hour",     default=12)
args = parser.parse_args()


# Image resolution
pix_to_um = get_pixel_size()
vox_to_um = get_voxel_size_35mm()

print(f"Image resolution\nHolomonitor: {pix_to_um[0]:0.6f} x {pix_to_um[1]:0.6f}\nTomocube:    {vox_to_um[1]} x {vox_to_um[2]}\n")
print(f"Ratio H to T:  {vox_to_um[1]/pix_to_um[0]:0.6f}")
print(f"Ratio z to xy: {vox_to_um[0] / vox_to_um[1]:0.6f}\n")


# Import data
file = args.path.split("/")[-2]
dir  = args.path.split(file)[0]
config = json.load(open(f"{args.path}/config.txt"))
fmin = config['fmin']
fmax = config['fmax']

microscope = dir.split("/")[-3]

if microscope == "Holomonitor":
    h_stack = import_holomonitor_stack(dir, file, f_min=fmin, f_max=fmax)
    df = pd.read_csv(f"{dir}{file}/cell_tracks.csv")
    conversion = pix_to_um

# # # Tomocube
# # tomo_dir  = "../../data/Tomocube/MDCK_10.02.2025/A2P1"
# # tomo_file = "250210.113448.MDCK dynamics.001.MDCK B.A2"
# # n_stack, h_stack = import_tomocube_stack(tomo_dir, tomo_file, vox_to_um[0], f_min=1)

# # df = pd.read_csv(f"{tomo_dir}/area_volume_filtered.csv")
# # conversion = vox_to_um



# Size of FOV
x_FOV = len(h_stack[0])   * conversion[1]
y_FOV = len(h_stack[0,0]) * conversion[1]
A_FOV = x_FOV * y_FOV * 1e-6

print(f"FOV:\n{x_FOV:2.0f}x{y_FOV:2.0f} µm² = {A_FOV:0.3f} mm²")
print(f"{len(h_stack[0]):2.0f}x{len(h_stack[0,0]):2.0f} pixels\n")


frame = 0
N0_cells = np.sum(df.frame==frame)
A0_cells = np.sum(df[df.frame==frame].A) * 1e-6

frame = fmax-fmin-1
Nf_cells = np.sum(df.frame==frame)
Af_cells = np.sum(df[df.frame==frame].A) * 1e-6

print(f"Number of cells:\n{N0_cells:0.0f}-{Nf_cells:0.0f} cells\n")
print(f"Cell density:\n{N0_cells / A0_cells:0.0f}-{Nf_cells / Af_cells:0.0f} cells/mm² ")