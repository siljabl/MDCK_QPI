import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

#from src.MDCKSegmentation import *

def get_pixel_size():
    ''' 
    From Nigar's thesis 
    '''

    return np.array([567 / 1024, 567 / 1024])


def plot_distribution_Holomonitor(h_counts, im_heights):
    # prepare plotting arrays
    h_bins = len(h_counts[0])

    # compute average
    mean_h_counts = np.mean(h_counts, axis=0)
    std_h_counts  = np.std(h_counts,  axis=0)
    heights       = (np.arange(h_bins) + 1) / 2

    # dimensions of inset
    scalebar_h = ScaleBar(1, 'um', box_alpha=0, color="w")
    scalebar_h.location = 'lower left'

    fig, ax = plt.subplots(1,1 ,figsize=(6,3))

    ax.errorbar(heights[1:], mean_h_counts[1:], yerr=std_h_counts[1:], fmt="o", ms=5, lw=1, color="k", capsize=3, capthick=1)
    ax.set(xlabel="Cell height [µm]", ylabel="Density")
    ax.set(yscale="linear")

    fig.tight_layout()
    
    left, bottom, width, height = [0.6, 0.5, 0.3, 0.3]
    imh = fig.add_axes([left, bottom, width, height])
    img=imh.imshow(im_heights)
    imh.add_artist(scalebar_h)
    imh.set_axis_off()
    fig.colorbar(img, ax=imh)

    return fig
