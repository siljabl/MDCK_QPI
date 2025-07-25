import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from src.Segmentation3D import *
from src.HolomonitorFunctions import get_pixel_size


def subregion(im, c, size, vox_to_um):
    # subregion size from µm to pixel
    size = int(size / vox_to_um[-1])

    xmin = int(c[0] - size/2)
    xmax = int(c[0] + size/2)
    ymin = int(c[1] - size/2)
    ymax = int(c[1] + size/2)

    if len(np.shape(im)) == 3:
        im = im[:,xmin:xmax, ymin:ymax]
    
    elif len(np.shape(im)) == 2:
        im = im[xmin:xmax, ymin:ymax]

    else:
        print("Wrong size on im!")

    return im


# normalizing
def normalize(mean, std):
        '''
        Normalize mean and std
        '''
        new_mean = (np.array(mean) - np.min(mean)) / (np.max(mean) - np.min(mean))
        new_std  = std / (np.max(mean) - np.min(mean))

        return new_mean, new_std


# def hist_to_curve(arr, type):
#     '''
#     Returns histogram as curve. Number of bins is equal to max height.
#     '''
#     if type == 'height':
#         range = (0,np.max(arr))
#         bins = int(np.max(arr)) + 1
#     elif type =='n_z':
#         range = (1.33,1.38)
#         bins = 50
#     elif type =='holomonitor':
#         range = (0,12*100)
#         bins = 25

#     y, x = np.histogram(arr.flatten(), bins=bins, range=range, density=True)
#     x = 0.5*(x[1:] + x[:-1])

#     return y, x

def hist_to_curve(arr, bins=0, hist_range=None):
    if hist_range == None:
        hist_range = (np.ma.min(arr), np.ma.max(arr))

    if bins == 0:
        bins  = int(np.max(arr))

    y, x = np.histogram(arr, bins=bins, range=hist_range, density=True)

    return 0.5*(x[1:] + x[:-1]), y, bins



def mean_dist(arr, bins=0, hist_range=0):
    print(f"Array lenght is: {len(arr)}")
    hist = []
    if bins == 0:
        hist_range = (0, np.max(arr))
        bins  = int(np.max(arr) + 1)

    for f in range(len(arr)):
        tmp_arr = arr[f].ravel()
        y, x = np.histogram(tmp_arr[tmp_arr > 0], bins=bins, range=hist_range, density=True)
        hist.append(y)

    mean = np.mean(hist, axis=0)
    std  = np.std(hist, axis=0)

    return 0.5*(x[1:] + x[:-1]), mean, std




def bin_by_density(arr, density, bin_size=100):

    # round density to nearest 100
    min_density = int(density.min()/100)*100
    max_density = int(density.max()/100)*100 + bin_size

    # range of densities coveres
    density_bins = np.arange(min_density, max_density, bin_size)
    Nbins = len(density_bins) - 1

    binned_arr = []

    for i in range(Nbins):

        # define low and high boundary on bin
        low_lim  = density_bins[i]
        high_lim = density_bins[i+1]

        density_mask = (density >= low_lim) * (density < high_lim)
        
        binned_arr.append(np.ma.mean(arr[density_mask], axis=0))
        
    return np.array(binned_arr), density_bins[:-1]


