'''
Transform images to tiff and fit mean intensity to growth curve
'''
import os
import json
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize        import curve_fit
from scipy.ndimage         import gaussian_filter
from src.FormatConversions import import_holomonitor_stack

parser = argparse.ArgumentParser(description='Compute cell properties from segmented data')
parser.add_argument('dataset',    type=int, help="path to main dir")
args = parser.parse_args()


# folder and path settings
holo_dict = json.load(open("../data/Holomonitor/settings.txt"))

path = "../" + holo_dict["files"][args.dataset].split("../../")[-1]
fmin = holo_dict["fmin"][args.dataset]
fmax = holo_dict["fmax"][args.dataset]
file = path.split("/")[-1]
dir  = path.split(file)[0]

try:
    os.mkdir(f"{path}/PIV")
except:
    None

stack = import_holomonitor_stack(dir, file, fmin, fmax)
stack = np.ma.array(stack, mask = stack == 0)

def func(x, a, b):
    return a*x + b
    #return a * b ** (x/c)

mean_height = np.ma.mean(stack, axis=(1,2))

frame = np.arange(len(mean_height)) / 12

corrected_mean = gaussian_filter(mean_height, sigma=10)

plt.plot(frame, mean_height, 'k', lw=2)
plt.plot(frame, corrected_mean, 'r--')
plt.xlabel("Time [h]")
plt.ylabel("Mean height [µm]")
plt.savefig(f"{path}/PIV/mean_intensity.png")


PIV_stack = stack.data * (corrected_mean / mean_height)[:,np.newaxis, np.newaxis]
PIV_stack = np.array(PIV_stack.data * (2**8 / np.max(PIV_stack)), dtype=np.uint8)

for frame in range(len(PIV_stack.data)):
    im = PIV_stack[frame]
    imageio.imwrite(f"{path}/PIV/frame_{frame}.tiff", im, dtype=np.uint8)
