import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from skimage import measure, draw
from skimage import morphology as morph
from skimage.segmentation import watershed, clear_border

n0 = 1.33
n_cell = 1.38

def smoothen_normalize_im(n_im, s_low, s_high, fig=False):
    '''
    Smoothens image using Gaussian blur and then normalizes it with respect to lowest refractive index within cells.
    I.e. avoids being dominated by empty areas.
    '''
    n_copy = np.copy(n_im)
    n_copy[n_copy == 0] = np.mean(n_copy)
    n_low_pass  = sc.ndimage.gaussian_filter(n_copy, s_low)
    n_high_pass = sc.ndimage.gaussian_filter(n_copy, s_high)

    n_norm = n_low_pass - n_high_pass
    n_norm = ((n_norm - n_norm.min()) / (n_norm.max() - n_norm.min()))
    n_norm[n_im == 0] = 0

    # Plots illustration of original image and blurred image
    if fig:
        fig, ax = plt.subplots(1,4, figsize=(12, 4))
        ax[0].imshow(n_im.T, origin="lower", vmin=1.33)
        ax[1].imshow(n_low_pass.T, origin="lower")
        ax[2].imshow(n_high_pass.T, origin="lower")
        ax[3].imshow(n_norm.T, origin="lower")

        ax[0].set(title="raw image")
        ax[1].set(title=f"sigma = {s_low}")
        ax[2].set(title=f"sigma = {s_high}")
        ax[3].set(title="low - high")
        fig.tight_layout()
        #fig.savefig("normalization.png")

    return n_norm


def extendedmin(im, H):
    '''
    Inverse of MATLABs imextendedmax: https://se.mathworks.com/help/images/ref/imextendedmax.html
    '''
    mask   = im.copy() 
    marker = mask + H  
    hmin   =  morph.reconstruction(marker, mask, method='erosion')

    return morph.local_minima(hmin)


def find_cell_pos(im, H):
    '''
    Finds cell positions as centroid of extendedmin
    '''
    
    im_min = extendedmin(im, H)
    im_label = measure.label(im_min)
    reg_prop = measure.regionprops(im_label, im)
    pos = np.array([[int(r.centroid[0]), int(r.centroid[1])] for r in reg_prop])

    return pos


def generate_seed_mask(pos, _shape):
    '''
    Turns list of cell positions into matrix with cell labels.
    Used as seeds for watershed.
    '''
    seeds = np.zeros(_shape)
    i = 1
    for x, y in pos:
        seeds[x,y] = i
        i += 1

    return seeds


def update_pos(pos, labels):
    n = np.max([np.max(labels)+1, len(pos)])
    new_pos = np.zeros([n, 2])

    for x, y in pos:
        l = labels[x,y]
        new_pos[l] = [x,y]
    
    x = new_pos.T[0] #[new_pos.T[0] > 0]
    y = new_pos.T[1] #[new_pos.T[1] > 0]

    return np.array([x[1:],y[1:]], dtype=int).T


def get_cell_areas(im, pos, h_im, clear_edge=True):
    '''
    Uses watershed to obtain mask of labeled cell areas
    '''

    # get cell areas with watershed
    seeds = generate_seed_mask(pos, im.shape)
    areas = watershed(im, seeds, watershed_line=False, connectivity=2)
    edges = watershed(im, seeds, watershed_line=True,  connectivity=2)

    # remove empty areas
    cell_mask = (h_im > 0)
    cell_areas = areas*cell_mask

    # remove small holes and areas
    #cell_areas = morph.remove_small_holes(cell_areas, area_threshold=100)

    if clear_edge:
        cell_areas = clear_border(cell_areas)

    cell_edges = (edges == 0)

    return cell_areas, cell_edges


def compute_polarization(reg_props):
    a_major = [reg.axis_major_length for reg in reg_props]
    a_minor = [reg.axis_minor_length for reg in reg_props]
    theta  = [reg.orientation for reg in reg_props]
    radius = [(a_max-a_min) / np.sqrt(a_max**2+a_min**2) for a_max, a_min in zip(a_major, a_minor)]

    return radius, theta


def compute_cell_props(label_im, pos, h_im, n_im, vox_to_um):
    '''
    Uses mask of labeled areas to compute cell position, area, volume and mass
    '''
    # Tomocube data
    if len(vox_to_um) == 3:
        vox_h    = vox_to_um[0]
        vox_area = vox_to_um[1]*vox_to_um[2]
        vox_vol  = vox_to_um[1]*vox_to_um[2]*vox_to_um[0]
    # Holomonitor data
    elif len(vox_to_um) == 2:
        vox_h    = 1
        vox_area = vox_to_um[0]*vox_to_um[1]
        vox_vol  = vox_to_um[0]*vox_to_um[1]

    area = []
    mass = []
    volume = []
    labels = []
    h_mean, h_max = [], []
    n_mean, h_2D  = [], []

    reg_prop = measure.regionprops(label_im, n_im)
    magnitude, angle = compute_polarization(reg_prop)

    for l in range(label_im.max()):
        label = l+1
        mask = (label_im == label)
        # skip labels associated with empty areas 
        if np.sum(mask) == 0:
            continue

        # update position
        if np.all(pos[l]) == 0:
            x, y = reg_prop[l].centroid_weighted
            pos[l] = int(x), int(y)

        labels.append(label)
        area.append(vox_area* np.sum(mask))
        volume.append(vox_vol * np.sum(mask * h_im))
        h_mean.append(vox_h * np.sum(mask*h_im) / np.sum(mask))
        h_max.append(vox_h * np.max(mask*h_im))
        h_2D.append(vox_h * np.sum(mask*h_im*(n_im-n0)/(n_cell-n0)) / np.sum(mask))

        # Tomocube data
        if len(vox_to_um) == 3:
            mass.append(vox_vol * np.sum(mask * h_im * n_im))   # replace with n-n0/(n_d-n_0)
            n_mean.append(np.sum(mask*n_im) / np.sum(mask))

    mask = (np.all(pos, axis=1) > 0)
    cells_tmp = pd.DataFrame({'x': pos.T[0][mask],
                              'y': pos.T[1][mask],
                              'A': area, 
                              'V': volume,
                              'h_avrg': h_mean,
                              'h_max': h_max,
                              'angle': angle,
                              'magnitude': magnitude,
                              'label': labels})
    # Tomocube data
    if len(vox_to_um) == 3:
        cells_tmp['m'] = mass
        cells_tmp['h_2D'] = h_2D
        cells_tmp['n_avrg'] = n_mean
    
    return cells_tmp




# def get_cell_areas(im, pos, h_im, w_edge=2, area_threshold=200, clear_edge=True):
#     '''
#     Uses watershed to obtain mask of labeled cell areas
#     area_threshold: in pixels
#     '''
#     seeds = generate_seed_mask(pos, im.shape)
    
#     # get cell areas with watershed
#     raw_areas = watershed(im, seeds, watershed_line=True)
#     if clear_edge:
#         raw_areas = clear_border(raw_areas)
#     raw_areas = morph.remove_small_holes((raw_areas>0), area_threshold=10, connectivity=2)

#     # make edges thicker
#     raw_edges = (raw_areas == 0)
#     edges = morph.dilation(raw_edges, morph.disk(w_edge))
#     areas = (edges == 0)

#     # remove empty areas
#     cell_mask = (h_im > 0)
#     cell_areas = areas*cell_mask
#     cell_areas = morph.remove_small_holes(cell_areas, area_threshold=100)

#     # remove small areas not corresponding to cell pos
#     cell_areas = morph.remove_small_objects(cell_areas, min_size=area_threshold)
#     cell_areas = measure.label(cell_areas)

#     return cell_areas



# def add_edges(areas):
#     new_areas = np.zeros_like(areas)
#     for l in np.unique(areas):
#         new_areas += morph.erosion(areas*(areas == l), morph.disk(2))
#     return new_areas


def get_voronoi_ridges(pos):
    '''
    Copied from scipy.spatial.plot_voronoi_2d source code: 
    https://github.com/scipy/scipy/blob/v1.15.1/scipy/spatial/_plotutils.py#L0-L1
    '''
    vor = Voronoi(pos)
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    voronoi_ridges = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            voronoi_ridges.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            voronoi_ridges.append([vor.vertices[i], far_point])

    return voronoi_ridges


def get_voronoi_areas(labels, pos, h_im, w_edge=4, area_threshold=200, clear_edge=True):
    dims = np.shape(h_im)
    voronoi_ridges = get_voronoi_ridges(pos)
    raw_areas = np.ones_like(h_im, dtype=np.uint8)

    for ridge in voronoi_ridges:
        ridge = np.array(ridge, dtype=int)# + w_pad
        row, col = draw.line(*ridge.ravel())
        mask = (row >= 0) * (row < dims[0]) * (col >= 0) * (col < dims[1])
        raw_areas[row[mask], col[mask]] = 0
    
    # make edges thicker
    raw_edges = (raw_areas == 0)
    edges = morph.dilation(raw_edges, morph.disk(w_edge))
    areas = (edges == 0)

    # remove empty areas
    cell_mask = (h_im > 0)
    cell_areas = areas*cell_mask

    # remove small areas not corresponding to cell pos
    cell_areas = morph.remove_small_objects(cell_areas, min_size=area_threshold)
    if clear_edge: 
        cell_areas = clear_border(cell_areas)

    # label areas, taking same labels as watershed areas
    labels_tmp = measure.label(cell_areas)
    labels = morph.dilation(labels, morph.disk(2*w_edge))*(labels_tmp > 0)
    areas = np.zeros_like(labels_tmp)

    for x, y in pos:
        x = int(np.clip(x, 0, dims[0]-1))
        y = int(np.clip(y, 0, dims[1]-1))

        l_new = labels[x,y]
        l_old = labels_tmp[x,y]
        areas[labels_tmp == l_old] = int(l_new)

    return areas


def compute_volume_change(df):
    '''
    Compute change in volume of tracked cells
    '''
    df = df.sort_values(by=['particle', 'frame'])
    df['dV'] = df.groupby('particle')['V'].diff()
    df['dV'] = df['dV'].fillna(0)

    df['dV'] /= df['V']

    return df



