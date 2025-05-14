# HTH-TomocubeAnalysis

This contains scripts to analyse Tomocube images. This is not a library but more a bunch of useful scripts that all leverage on the same functions which are located in "src".


## Source code
- 'Segmentation3D.py': contains all functions needed to compute 2D heights and refractive index from 3D stacks
- 'PlottingFunctions.py': contains functions that return figures that illustrate operations
- 'HolomonitorFunctions.py': contains all functions that are specific for Holomonitor data
- 'CellSegmentation.py': functions used to obtain cell specific properties

## Image processing
- 'predict_folder_bioMlM.py': Uses model trained by Thomas to predict the probability that a voxel is a cell or not. Runs on entire folder and saves probabilities in 'prediction' folder
- 'segment_folder.py': Segments cells based on probabilities in 'predictions'. Estimates the zero-level for each petri dish and a frame-specific threshold on cell probabilities and applies median filter. Saves mask and configurations in 'segmentation'.
- 'save_2D_folder.py': Turns 3D masks from 'segmentation' into 2D tiffs of heights (um) and refractive indices. Saves tiffs in 'heights' and 'refractive_index'. Refractive index is scales to have 1.38 as mean.
- 'cell_segmentation_Holomonitor/Tomocube.py': find cells and cell properties in cell monolayers. Saves output as csv dataframe.
- 'filtering.ipynb': filter out spurious cells based on height, area and volume.

## Analysis
- 'flatness.ipynb': computes spatial variation in monolayer
- 'temporal_fluctuations.ipynb':


- 'holomonitor_pixel_fluctuations.py'/'tomocube_pixel_fluctuations.py': Plots distributions, fluctuations and correlation of h(x,y) and n_z(x,y)
- 'plot_refractive_index': Plots histogram of n and n_z for a specific tiff




