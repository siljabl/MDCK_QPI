'''
Create 3D mask out of 3D probabilities from catBioM_MlM
'''

import os
import argparse
import tifffile
import numpy as np

from pathlib import Path
from datetime import datetime

from skimage.filters import median
from skimage.morphology import disk

from src.Segmentation3D  import *
from src.PlottingFunctions import *
from src.ImUtils import commonStackReader

parser = argparse.ArgumentParser(description='Segment cell from MlM probabilities')
parser.add_argument('dir',      type=str, help="path to main dir")
parser.add_argument('-r1_min',  type=int, help="min radius of kernel 1", default='3')
parser.add_argument('-r1_max',  type=int, help="max radius of kernel 1", default='5')
parser.add_argument('-r2',      type=int, help="radius of kernel 2", default='25')
args = parser.parse_args()



# create folders
in_dir = f"{args.dir}predictions"
assert os.path.exists(in_dir), "In path does not exist, maybe because of missing '/' at the end of path"
path = Path(in_dir)
out_dir  = f"{args.dir}segmentation"
mhds_dir = f"{out_dir}{os.sep}mhds"

try:
    os.mkdir(out_dir)
    os.mkdir(mhds_dir)
except:
    pass


# get experiment specific dataseries names
experiment = []
for file in path.glob("*HT3D_0_prob.npy"):
    experiment.append(file.stem.split("HT3D_0_prob")[0])


# create arrays for prediction and filtering
thresholds = np.linspace(0.5, 1, 20, endpoint=False)
kernel_1 = generate_kernel(args.r1_min, args.r1_max)
kernel_2 = np.array([disk(args.r2)])




# saving MlM threshold of each file
new_log = 1
logfile = f"{mhds_dir}{os.sep}log.txt"
if os.path.exists(logfile):
    new_log = 0

with open(logfile, "a") as log:
    if new_log:
        log.write("# file, zero-level, MlM threshold, r1_min, r1_max, r2\n")
    log.write(f"date: {datetime.today().strftime('%Y-%m-%d')}\n")

    # sort by experiment
    for exp in experiment:
        print(exp)
        ri_z_list     = []
        prob_z_list   = []
        dri_dz_list   = []
        dprob_dz_list = []

        sum_above = np.zeros_like(thresholds)
        sum_below = np.zeros_like(thresholds)

        # compute list for determination of zero level
        print(f"\nDetermining zero-level for experiment {exp} ...")
        for file in path.glob(f"{exp}*_prob.npy"):

            stack_name = f"{path.parent}{os.sep}{file.name.split('_prob.npy')[0]}.tiff"

            # load stacks
            stack = commonStackReader(stack_name)
            #MlM_probabilities = commonMultiChannelStackReader(file)
            MlM_probabilities = np.load(file)
            cell_prob = MlM_probabilities[:,:,:,1]

            # compute mean and derivative of mean along z
            ri_z   = np.mean(stack,     axis=(1,2))
            prob_z = np.mean(cell_prob, axis=(1,2))
            dri_dz   = np.diff(ri_z)   + 1
            dprob_dz = np.diff(prob_z) + 1

            # add to list for experiment
            ri_z_list.append(ri_z)
            prob_z_list.append(prob_z)
            dri_dz_list.append(dri_dz)
            dprob_dz_list.append(dprob_dz)
        
        # compute zero level. same for entire experiment
        z_0 = estimate_cell_bottom(dri_dz_list)
        fig = plot_z_profile([ri_z_list, dri_dz_list], [prob_z_list, dprob_dz_list], stack, cell_prob, z_0)
        fig.savefig(f"{mhds_dir}{os.sep}{Path(exp).name}_zero_level.png", dpi=300)
        print(f"zero-level: {z_0}\n")


        # determine threshold
        print(f"Creating masks for:")
        for file in path.glob(f"{exp}*_prob.npy"):
            out_mask = f"{out_dir}{os.sep}{file.name.split('_prob.npy')[0]}_mask.tiff"
            if os.path.exists(out_mask):
                continue
            print(file)

            # load probabilities
            MlM_probabilities = np.load(file)
            cell_prob = MlM_probabilities[:,:,:,1]

            # compute array for determination of MlM threshold
            for i in range(len(thresholds)):
                mask = (cell_prob > thresholds[i])
                sum_above[i] = np.sum(mask[z_0:])
                sum_below[i] = np.sum(mask[:z_0])
            
            # apply threshold
            threshold = determine_threshold(thresholds, sum_above)
            cell_pred = (cell_prob > threshold)
            log.write(f'{file.name}, {z_0}, {threshold}, {args.r1_min}, {args.r1_max}, {args.r2}\n')

            # filter mask
            tmp_mask = median(cell_pred[z_0:], kernel_1)
            tmp_mask = median(tmp_mask,  kernel_2)
            cell_mask = np.zeros_like(cell_pred)
            cell_mask[z_0:] = tmp_mask
            
            # save mask
            basename = file.stem.split('_prob')[0]
            tifffile.imwrite(out_mask, np.array(cell_mask, dtype=np.uint8), bigtiff=True)


        # plot illustration of MlM threshold and final mask
        fig = plot_threshold(thresholds, [sum_above, sum_below], cell_prob.shape, z_0)
        fig.savefig(f"{mhds_dir}{os.sep}{file.name.split('_prob.npy')[0]}_threshold.png", dpi=300)



