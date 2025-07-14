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

from scipy.optimize import curve_fit
from src.CellFunctions import cell_growth
from src.HolomonitorFunctions import get_pixel_size

parser = argparse.ArgumentParser(description='Plot data set')
parser.add_argument('in_path',          type=str, help="data set as listed in holo_dict")
parser.add_argument('-out_path',        type=str, help="data set as listed in holo_dict", default=None)
parser.add_argument('-frames_per_hour', type=int, help="Number of frames in an hour",     default=12)
args = parser.parse_args()


# if not given, use input folder for outpur also
if args.out_path == None:
    args.out_path = args.in_path

# create figure folder
try:
    os.mkdir(f"{args.out_path}/figs")
except:
    None



###############
# import data #
###############
with open(f"{args.in_path}/masked_arrays.pkl", 'rb') as handle:
    data = pickle.load(handle)


# unit conversion
pix_to_um     = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour


# fit growth curve to cell density
density   = data['cell_density']
time      = np.arange(len(data['cell_density'])) / 12
params, _ = curve_fit(cell_growth, time, density)



############
# plotting #
############
fig, ax = plt.subplots(1,1, figsize=(5,3.5))

ax.plot(time, density, '.', ms=4)
ax.plot(time, cell_growth(time, *params), 'r-', label=rf"$\propto2^{{~t~/~{params[0]:0.0f}}}$")

ax.set(xlabel="time [h]",
       ylabel=r"$\rho_{cell} ~[mm^{-2}]$")
ax.legend()

fig.tight_layout()
fig.savefig(f"{args.out_path}/figs/cell_density_per_frame.png", dpi=300)
