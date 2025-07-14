'''
Compute correlations on cell tracks

TO DO:
- smoothen velocities
- why is C_vv suddenly noisy? Issue is not related to rewriting of code. Sander's code reproduces noice
'''

import os
import sys
import pickle
import argparse
import numpy as np

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.HolomonitorFunctions    import get_pixel_size
from src.MaskedArrayCorrelations import general_temporal_correlation, general_spatial_correlation

# parse input
parser = argparse.ArgumentParser(description='Compute correlations of data from cell tracks')
parser.add_argument('in_path',          type=str,   help="path to dataset")
parser.add_argument('-out_path',        type=str,   help="path to output. Using in_path if set to None",               default=None)
parser.add_argument('-dr',              type=int,   help="size of radial bin for spatial correlation [pix]",           default=40)
parser.add_argument('-r_max',           type=int,   help="max radial distance for spatial correlation [pix]",          default=500)
parser.add_argument('-t_max',           type=float, help="max fraction of timeinterval used in temporal correlation",  default=0.5)
parser.add_argument('-frames_per_hour', type=int,   help="number of frames in an hour",                                default=12)
args = parser.parse_args()

# if not given, use input folder for output also
if args.out_path == None:
    args.out_path = args.in_path



###############
# import data #
###############

# unit conversion
pix_to_um     = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour


# load masked arrays
with open(f"{args.in_path}/masked_arrays.pkl", 'rb') as handle:
    data = pickle.load(handle)

x_pos = data['x_position']
y_pos = data['y_position']

h = data['mean_height']
A = data['cell_area'] * pix_to_um[1]**2
V = h*A
dV = np.ma.diff(V, axis=0)


x_mean_pos = (x_pos[:-1] + x_pos[1:]) / 2
y_mean_pos = (y_pos[:-1] + y_pos[1:]) / 2
#h_mean_pos = (h[:-1] + h[1:]) / 2

vx = data['x_displacement'] * pix_to_um[1] / frame_to_hour
vy = data['y_displacement'] * pix_to_um[1] / frame_to_hour

density = data['cell_density']


# remove mean
hfluct =  h - np.ma.mean(h,  axis=1)[:,np.newaxis]
Afluct =  A - np.ma.mean(A,  axis=1)[:,np.newaxis]
Vfluct =  V - np.ma.mean(V,  axis=1)[:,np.newaxis]
Vdiff  = dV - np.ma.mean(dV, axis=1)[:,np.newaxis]


# velocity should have mean 0, but subtract mean to be sure
vx = vx - np.ma.mean(vx, axis=1)[:,np.newaxis]
vy = vy - np.ma.mean(vy, axis=1)[:,np.newaxis]

# apply smoothing to velocities



################
# correlations #
################

# convert t_max to frames
t_max = int(args.t_max * len(density))

# compute temporal correlation
Ct_hh_dict   = general_temporal_correlation(hfluct, t_max=t_max)
Ct_AA_dict   = general_temporal_correlation(Afluct, t_max=t_max)
Ct_VV_dict   = general_temporal_correlation(Vfluct, t_max=t_max)
Ct_dVdV_dict = general_temporal_correlation(Vdiff,  t_max=t_max)

Ct_hA_dict  = general_temporal_correlation(hfluct, Afluct, t_max=t_max)
Ct_hV_dict  = general_temporal_correlation(hfluct, Vfluct, t_max=t_max)
Ct_AV_dict  = general_temporal_correlation(Afluct, Vfluct, t_max=t_max)
#Ct_hdV_dict = general_temporal_correlation(hfluct, Vdiff,  t_max=t_max)
#Ct_AdV_dict = general_temporal_correlation(Afluct, Vdiff,  t_max=t_max)
#Ct_VdV_dict = general_temporal_correlation(Vfluct, Vdiff,  t_max=t_max)


# compute spatial correlation
Cr_hh_dict   = general_spatial_correlation(x_pos, y_pos, hfluct, hfluct, dr=args.dr, r_max=args.r_max)
Cr_AA_dict   = general_spatial_correlation(x_pos, y_pos, Afluct, Afluct, dr=args.dr, r_max=args.r_max)
Cr_VV_dict   = general_spatial_correlation(x_pos, y_pos, Vfluct, Vfluct, dr=args.dr, r_max=args.r_max)
Cr_dVdV_dict = general_spatial_correlation(x_mean_pos, y_mean_pos, Vdiff,  Vdiff,  dr=args.dr, r_max=args.r_max)

Cr_hA_dict = general_spatial_correlation(x_pos, y_pos, hfluct, Afluct, dr=args.dr, r_max=args.r_max)
Cr_hV_dict = general_spatial_correlation(x_pos, y_pos, hfluct, Vfluct, dr=args.dr, r_max=args.r_max)
Cr_AV_dict = general_spatial_correlation(x_pos, y_pos, Afluct, Vfluct, dr=args.dr, r_max=args.r_max)
#Cr_hdV_dict = general_spatial_correlation(x_pos, y_pos, hfluct, Vdiff, dr=args.dr, r_max=args.r_max)
#Cr_AdV_dict = general_spatial_correlation(x_pos, y_pos, Afluct, Vdiff, dr=args.dr, r_max=args.r_max)
#Cr_VdV_dict = general_spatial_correlation(x_pos, y_pos, Vfluct, Vdiff, dr=args.dr, r_max=args.r_max)


# compute velocity correlations
C_t_vv_dict = general_temporal_correlation([vx, vy], t_max=t_max)
C_r_vv_dict = general_spatial_correlation(x_mean_pos, y_mean_pos, [vx, vy], dr=args.dr, r_max=args.r_max)




# bin by density
Ct = [Ct_hh_dict['C_norm'], 
      Ct_AA_dict['C_norm'], 
      Ct_VV_dict['C_norm'], 
      Ct_hA_dict['C_norm'], 
      Ct_hV_dict['C_norm'], 
      Ct_AV_dict['C_norm'], 
      Ct_dVdV_dict['C_norm']]

Cr = [Cr_hh_dict['C_norm'], 
      Cr_AA_dict['C_norm'], 
      Cr_VV_dict['C_norm'],
      Cr_hA_dict['C_norm'], 
      Cr_hV_dict['C_norm'], 
      Cr_AV_dict['C_norm'], 
      Cr_dVdV_dict['C_norm']]



r = Cr_hh_dict['r_bin_centers'] * pix_to_um[1]

t = np.arange(t_max) * frame_to_hour

var = ["{hh}", "{AA}", "{VV}", 
       "{hA}", "{hV}", "{AV}",
       "{dVdV}"]


##################
# Save as pickle #
##################
out_dict = {'C_t': Ct,
            'C_r': Cr,
            'C_t_vv': C_t_vv_dict['C_norm'],
            'C_r_vv': C_r_vv_dict['C_norm'],
            't':   t,
            'r':   r,
            'variable': var, 
            'density':  density}

# save as pickle
with open(f"{args.out_path}/track_correlations.pkl", 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

