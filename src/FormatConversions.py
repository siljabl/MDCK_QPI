import imageio
import numpy as np
import skimage.io as io

def convert_to_old_format(stack):
    if (stack[0].dtype == np.uint16):
        new_stack = []
        for f in range(len(stack)):
            new_stack.append(np.array(stack[f],dtype=np.float32)/10000)
        return new_stack
    return stack

def convert_to_new_format(stack):
    if (stack[0].dtype == np.float32):
        new_stack=[]
        for f in range(len(stack)):
            new_stack.append(np.array(stack[f]*10000,dtype = np.uint16))
        return new_stack
    return stack

def is_tile(filepath):
    ### A tile does not satisfy one of the two following conditions:
    # - does not contain 61 frames
    # - is not 829 width and heigh. Maybe this is sufficient?
    im = io.imread(filepath,key=0)[0]
    return (len(im)!=829  or len(im[0])!=829)



def import_holomonitor_stack(dir, dataset, f_min=1, f_max=180, h_scaling=100):
    # mask that set area outside cells to zero
    try:
        mask = (imageio.v2.imread(f"{dir}{dataset}/mask.tiff") > 0)
    except:
        mask = np.ones_like(imageio.v2.imread(f"{dir}{dataset}/Well {dataset} _reg_Zc_{1}.tiff"))
    
    stack = []
    for f in range(f_min, f_max+1):
        try:
            frame = imageio.v2.imread(f"{dir}{dataset}/Well {dataset} _reg_Zc0fluct_{f}.tiff")
        except:
            try:
                frame = imageio.v2.imread(f"{dir}{dataset}/Well {dataset} _reg_Zc_{f}.tiff")
            except:
                frame = imageio.v2.imread(f"{dir}{dataset}/DWell {dataset} _reg_zero_corr_{f}.tiff")

        stack.append(frame * mask)

    stack = np.array(stack) / h_scaling

    return stack



def import_tomocube_stack(dir, dataset, h_scaling, f_min=0, f_max=40, n_cell=1.38):
    n_stack = []
    h_stack = []
    for f in range(f_min, f_max+1):
        h_frame = imageio.v2.imread(f"{dir}/heights/{dataset}.T001P01_HT3D_{f}_heights.tiff")
        n_frame = imageio.v2.imread(f"{dir}/refractive_index/{dataset}.T001P01_HT3D_{f}_mean_refractive.tiff")

        h_stack.append(h_frame)
        n_stack.append(n_frame)

    # scale data
    h_stack = np.array(h_stack) * h_scaling
    n_stack = np.array(n_stack)
    n_mean  = np.mean(n_stack[h_stack > 0])
    n_stack = n_stack * n_cell / n_mean

    return n_stack, h_stack