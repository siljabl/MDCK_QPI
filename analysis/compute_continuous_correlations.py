'''
Compute correlations on PIV and stack

TO DO:
- add local density
- add height
'''

import os
import sys
import json
import pickle
import argparse
import subprocess

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.PlottingUtils import bin_by_density
from src.FormatConversions import import_holomonitor_stack
from src.HolomonitorFunctions    import get_pixel_size
from src.Correlations            import general_spatial_autocorrelation
from src.MaskedArrayCorrelations import general_temporal_correlation

# parse input
parser = argparse.ArgumentParser(description='Compute correlations of data from cell tracks')
parser.add_argument('in_path',          type=str,   help="path to dataset")
parser.add_argument('-out_path',        type=str,   help="path to output. Using in_path if set to None",               default=None)
parser.add_argument('-dr',              type=int,   help="size of radial bin for spatial correlation [pix]",           default=40)
parser.add_argument('-r_max',           type=int,   help="max radial distance for spatial correlation [pix]",          default=500)
parser.add_argument('-t_max',           type=float, help="max fraction of timeinterval used in temporal correlation",  default=0.5)
parser.add_argument('-frames_per_hour', type=int,   help="number of frames in an hour",                                default=12)
parser.add_argument('-fmax',            type=int,   default=None)
parser.add_argument('-plot_PIV',        type=bool,  help="plot and save PIV velocity field", default=False)
args = parser.parse_args()

# if not given, use input folder for outpur also
if args.out_path == None:
    args.out_path = args.in_path


# Folder for plotting velocity fields
try:
    os.mkdir(f"{args.in_path}/PIV_velocity_fields")
except:
    None


# folder and path settings
config = json.load(open(f"{args.in_path}/config.txt"))

fmin = config["fmin"]
fmax = config["fmax"]-1 #-1 because PIV missing last frame
if args.fmax != None:
    fmax = args.fmax

file = args.in_path.split("/")[-2]
dir  = args.in_path.split(file)[0]



###############
# import data #
###############

# unit conversion
pix_to_um = get_pixel_size()
frame_to_hour = 1 / args.frames_per_hour


# import cell density
with open(f"{args.in_path}/masked_arrays.pkl", 'rb') as handle:
    data_tmp = pickle.load(handle)

density = data_tmp['cell_density']

# import stack
stack = import_holomonitor_stack(dir, file, fmin, fmax)

# import PIV velocity field
data_PIV = np.loadtxt(f"{args.in_path}/PIV/velocities/PIVlab_0001.txt", delimiter=",", skiprows=3)
x = np.array(data_PIV[:, 0], dtype=int)
y = np.array(data_PIV[:, 1], dtype=int)



###########################
# transform data to image #
###########################

# transform PIV position to matrix entries
dx   = y[1] - y[0]
x0   = y[0] - dx
xmax = int((np.max(y) - x0) / dx) + 1

x_tmp = np.array((x - x0) / dx, dtype=int)
y_tmp = np.array((y - x0) / dx, dtype=int)

PIV_position_x = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_position_y = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_velocity_x = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_velocity_y = np.zeros((fmax, xmax, xmax), dtype=np.float64)
PIV_height     = np.zeros((fmax, xmax, xmax), dtype=np.float64) # not PIV, but same size as PIV

im_height = np.copy(stack)

# Fill arrays
for frame in tqdm(range(fmax)):

    # Load data
    data_PIV = np.loadtxt(f"{args.in_path}/PIV/velocities/PIVlab_{frame+1:04d}.txt", delimiter=",", skiprows=3)

    # Extract values
    u = np.array(data_PIV[:, 2], dtype=np.float64)
    v = np.array(data_PIV[:, 3], dtype=np.float64)

    # masking didn't work properly, so probably mean is affected by outside data points
    PIV_position_x[frame-1, x_tmp, y_tmp] = x
    PIV_position_y[frame-1, x_tmp, y_tmp] = y
    PIV_velocity_x[frame-1, x_tmp, y_tmp] = u - np.mean(u)
    PIV_velocity_y[frame-1, x_tmp, y_tmp] = v - np.mean(v)

    frame_mean_height = np.mean(stack[frame][stack[frame] > 0])

    # Probably better to take mean
    PIV_height[frame, x_tmp, y_tmp]           = stack[frame, x, y]
    PIV_height[frame][PIV_height[frame] > 0] -= frame_mean_height
    
    # subtrack mean
    im_height[frame]                        = stack[frame]
    im_height[frame][im_height[frame] > 0] -= frame_mean_height


    # plot
    if args.plot_PIV:

        # plot
        fig, ax = plt.subplots(1,1, figsize=(10,8))
        sns.heatmap(stack[frame].T, ax=ax, square=True, cmap="gray", vmin=0, vmax=14, 
                    xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label': 'h [µm]'})
        
        ax.invert_yaxis()
        ax.quiver((y_tmp*dx + x0), (x_tmp*dx + x0), v, u, scale=75/pix_to_um[-1], color="c")


        # add scalebar
        sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
        sb.location = 'lower left'
        ax.add_artist(sb)


        # save
        fig.tight_layout()
        fig.savefig(f"{args.in_path}/PIV_velocity_fields/frame_{frame:03d}.png", dpi=300);
        plt.close()

        # Make video
        out_video = '../videos/PIV_velocities.mp4'

        # Change to the image directory
        os.chdir(f"{args.path}/PIV_velocity_fields")

        # Define the FFmpeg command to convert images to video
        ffmpeg_command = ['ffmpeg',
                        '-framerate', '10',           # Set frame rate
                        '-i', 'frame_%03d.png',       # Input format
                        '-c:v', 'libx264',            # Video codec
                        '-pix_fmt', 'yuv420p',        # Pixel format
                        out_video                  # Output video file name
                        ]

        # Run the FFmpeg command
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f'Video saved as {ffmpeg_command}')
        except subprocess.CalledProcessError as e:
            print(f'Error generating video: {e}')



########################
# Spatial correlations #
########################

PIV_velocity = [PIV_velocity_x, PIV_velocity_y]

# compute correlations
C_r_vv = general_spatial_autocorrelation(PIV_velocity, PIV_velocity, vox_to_um=pix_to_um * dx, r_max=args.r_max*pix_to_um[1])
C_r_hh = general_spatial_autocorrelation(im_height,    im_height,    vox_to_um=pix_to_um     , r_max=args.r_max*pix_to_um[1])
C_r_hv = general_spatial_autocorrelation(PIV_height,   PIV_velocity, vox_to_um=pix_to_um * dx, r_max=args.r_max*pix_to_um[1])

# Transform to masked arrays
dims  = np.shape(stack)
x_pos = np.arange(dims[1])
y_pos = np.arange(dims[2])

X, Y = np.meshgrid(x_pos, y_pos)
X, Y = np.repeat(X, dims[0]), np.repeat(Y, dims[0])

# reshape
height = np.reshape(im_height, (dims[0], dims[1]*dims[2]))
x_pos  = np.reshape(X,         (dims[0], dims[1]*dims[2]))
y_pos  = np.reshape(Y,         (dims[0], dims[1]*dims[2]))

height = np.ma.array(height, mask=height==0)
x_pos  = np.ma.array(x_pos,  mask=height==0)
y_pos  = np.ma.array(y_pos,  mask=height==0)


PIV_dims = np.shape(PIV_position_x)
PIV_position_x = np.reshape(PIV_position_x, (PIV_dims[0], PIV_dims[1]*PIV_dims[2]))
PIV_position_y = np.reshape(PIV_position_y, (PIV_dims[0], PIV_dims[1]*PIV_dims[2]))
PIV_velocity_x = np.reshape(PIV_velocity_x, (PIV_dims[0], PIV_dims[1]*PIV_dims[2]))
PIV_velocity_y = np.reshape(PIV_velocity_y, (PIV_dims[0], PIV_dims[1]*PIV_dims[2]))
PIV_height     = np.reshape(PIV_height,     (PIV_dims[0], PIV_dims[1]*PIV_dims[2]))


# Mask, not correct because error in PIV
PIV_position_x = np.ma.array(PIV_position_x, mask=PIV_velocity_x==0)
PIV_position_y = np.ma.array(PIV_position_y, mask=PIV_velocity_y==0)
PIV_velocity_x = np.ma.array(PIV_velocity_x, mask=PIV_velocity_x==0)
PIV_velocity_y = np.ma.array(PIV_velocity_y, mask=PIV_velocity_y==0)
PIV_height     = np.ma.array(PIV_height,     mask=PIV_velocity_y==0)



#########################
# temporal correlations #
#########################

# convert t_max to frames
t_max = int(args.t_max * len(density))

PIV_velocity = [PIV_velocity_x, PIV_velocity_y]

# compute correlation and bin
C_t_vv = general_temporal_correlation(PIV_velocity, PIV_velocity, t_max=t_max)
C_t_hh = general_temporal_correlation(height,       height,       t_max=t_max)
C_t_hv = general_temporal_correlation(PIV_height,   PIV_velocity, t_max=t_max)

t_arr = np.arange(t_max) * frame_to_hour
r_arr  = C_r_vv['r_bin']



##################
# Save as pickle #
##################
out_dict = {'C_t_vv': C_t_vv['C_norm'],
            'C_r_vv': C_r_vv['C_norm'],
            'C_t_hh': C_t_hh['C_norm'],
            'C_r_hh': C_r_hh['C_norm'],
            'C_t_hv': C_t_hv['C_norm'],
            'C_r_hv': C_r_hv['C_norm'],
            't_vv':   t_arr,
            'r_vv':   r_arr, 
            'density':  density[:args.fmax]}

# save as pickle
with open(f"{args.out_path}/continuous_correlations.pkl", 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
