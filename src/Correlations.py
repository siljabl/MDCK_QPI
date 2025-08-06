import numpy as np
from tqdm import tqdm
from numba import njit, prange



def pairwise_autocorrelation(tracks, var, vox_to_um, step=8):
    '''
    tracks:     pandas dataframe with tracked particles
    var:        'str' refering to column in tracks
    vox_to_um:  array of conversion constants [z, x, y]
    step:       size of bins, in µm
    '''

    distance    = []
    correlation = []

    # multiply all pairs
    for t in range(np.max(tracks.frame)):
        # dataframe at time t
        df_tmp  = tracks[tracks.frame==t]
        tmp_arr = np.array([df_tmp['x'].values, df_tmp['y'].values, df_tmp[var].values])

        # number of unique combinations
        nn_max = int(np.ceil(len(tmp_arr.T) / 2))
        for nn in range(nn_max):
            roll_arr = np.roll(tmp_arr, shift=-nn, axis=1)
            tmp_var   = tmp_arr[2] * roll_arr[2]
            
            # calculate distance
            x_pos = np.array(tmp_arr[0] - roll_arr[0], dtype=int)
            y_pos = np.array(tmp_arr[1] - roll_arr[1], dtype=int) 
            tmp_r = np.sqrt(x_pos**2 + y_pos**2)*vox_to_um[1]

            distance.append(tmp_r)
            correlation.append(tmp_var)

    distance    = np.concatenate(distance)
    correlation = np.concatenate(correlation)

    # bin data
    step = 8
    r_arr = np.arange(0, np.max(distance), step)
    C_arr = np.zeros_like(r_arr)

    i = 0
    for r in r_arr:
        mask = (distance >= r - step/2) * (distance < r + step/2)
        if np.sum(mask) > 0:
            C_arr[i] = np.mean(correlation[mask])

        i += 1

    return r_arr, C_arr



def temporal_correlation(x_cell, y_cell):  
    '''
    Cellwise correlation
    '''      
    auto_corr = np.zeros_like(x_cell)
    auto_corr[0] = np.mean((x_cell*y_cell))

    for dt in range(1,len(x_cell)):
        auto_corr[dt] = np.mean((x_cell * np.roll(y_cell, axis=0, shift=-dt))[:-dt])

    return auto_corr


@njit(parallel=True)
def mean_numba(vx):
    """
    Corresponds to np.mean(vx, axis=(1,2))
    """

    res = []
    for i in prange(vx.shape[0]):
        res.append(vx[i].mean())

    return np.array(res)


@njit(parallel=True)
def velocity_temporal_correlation(vx, vy):  
    """
    Cellwise correlation
    """
    Nframes = vx.shape[0]
    Npixels = vx.shape[1] * vx.shape[2]

    auto_corr = np.zeros((Nframes, Nframes))
    mean_square = mean_numba(vx**2 + vy**2)

    auto_corr[:, 0] = 1.0

    for dt in prange(1, Nframes):
        vx_dt = vx[dt:]  # vx shifted down by dt
        vy_dt = vy[dt:]  # vy shifted down by dt

        # Calculate RMS, careful about the shift
        rms = np.sqrt(mean_square[:-dt] * mean_square[dt:])

        # Calculate the correlation for current dt
        auto_corr[:-dt, dt] = mean_numba((vx[:-dt] * vx_dt) + (vy[:-dt] * vy_dt)) / rms

    return auto_corr



def spatial_autocorrelation(im):
    # center input around zero (ensure negative correlation)
    # var = np.mean(im**2) - np.mean(im)**2
    # im  = (im - np.mean(im)) / np.sqrt(var)

    # compute FFT of image
    fft_im  = np.fft.fft2(im, norm="ortho")
    ifft_im = np.fft.ifft2(np.abs(fft_im)**2, norm="ortho")
    ifft_im = np.fft.fftshift(ifft_im)

    # normalize output
    ifft_im_norm = np.real(ifft_im) / np.max(np.real(ifft_im))

    return ifft_im_norm


def radial_distribution(im, vox_to_um, binsize):

    # reshape to odd number
    _shape = np.shape(im)
    
    if _shape[0]%2 == 0:
        im = im[:-1]

    if _shape[1]%2 == 0:
        im = im[:,:-1]
    
    # define raduis matrix
    Lx = int(len(im)/2)
    Ly = int(len(im[0])/2)

    x = np.arange(-Lx,Lx+1,1) * vox_to_um[-2]
    y = np.arange(-Ly,Ly+1,1) * vox_to_um[-1]
    xx, yy = np.meshgrid(y, x)
    r = np.sqrt(xx**2. + yy**2.)

    nbins = int((r.max() / binsize)+1)
    dist = np.zeros(nbins)
    mean = np.zeros(nbins)
    std  = np.zeros(nbins)

    for i in range(nbins):
        mask = (r >= i*binsize) * (r < (i+1)*binsize)
        if np.sum(mask) > 0:
            dist[i] = np.mean(r[mask])
            mean[i] = np.mean(im[mask])
            std[i]  = np.std(im[mask])

        else:
            dist[i] = np.nan
            mean[i] = np.nan
            std[i]  = np.nan

    dist = np.ma.array(dist, mask = dist==np.nan)
    mean = np.ma.array(mean, mask = mean==np.nan)
    std  = np.ma.array(std,  mask =  std==np.nan)
        

    return dist, mean, std


def velocity_spatial_autocorrelation(imx, imy, vox_to_um, r_max, binsize=5):
    
    Nframes = len(imx)

    C_norm = []

    for frame in tqdm(range(Nframes)):

        # compute FFT of image
        fft_imx  = np.fft.fft2(imx[frame], norm="ortho")
        fft_imy  = np.fft.fft2(imy[frame], norm="ortho")

        # inverse of power spectrum
        ifft_imx = np.fft.ifft2(np.abs(fft_imx)**2, norm="ortho")
        ifft_imy = np.fft.ifft2(np.abs(fft_imy)**2, norm="ortho")

        ifft_imx = np.real(np.fft.fftshift(ifft_imx))
        ifft_imy = np.real(np.fft.fftshift(ifft_imy))

        # normalize output
        ifft_im = ifft_imx + ifft_imy
        ifft_im_norm = ifft_im / np.max(ifft_im)

        r_bin, C_tmp, _ = radial_distribution(ifft_im_norm, vox_to_um, binsize)

        C_norm.append(np.ma.array(C_tmp, mask=C_tmp==np.nan))
    
    mask = r_bin <= r_max

    COR = {'C_norm': np.array(C_norm)[:,mask],
           'r_bin':  r_bin[mask]}

    return COR


def general_spatial_autocorrelation(im1, im2=None, vox_to_um=1, r_max=500, binsize=5):
    if np.any(im2==None):
        im2 = im1

    dim_var1 = np.shape(im1)
    dim_var2 = np.shape(im2)

    # print(len(dim_var1), len(dim_var2))
    # len=3: variable is scalar (t,x,y), len=4: variable is vector (dim, t, x, y)
    assert len(dim_var1) in [3,4] and len(dim_var2) in [3,4]

    if len(dim_var1) == 3:
        im1x = im1
        im1y = im1
    else:
        im1x, im1y = im1
    
    if len(dim_var2) == 3:
        im2x = im2
        im2y = im2
    else:
        im2x, im2y = im2
    
    Nframes = len(im1x)

    C_norm = []


    for frame in tqdm(range(Nframes)):

        # replace masked entries with mean
        im1x.data[im1x.mask] = np.ma.mean(im1x)
        im2x.data[im2x.mask] = np.ma.mean(im2x)
        im1y.data[im1y.mask] = np.ma.mean(im1y)
        im2y.data[im2y.mask] = np.ma.mean(im2y)


        # compute FFT of image
        fft_im1x = np.fft.fft2(im1x[frame], norm="ortho")
        fft_im2x = np.fft.fft2(im2x[frame], norm="ortho")

        # inverse of power spectrum
        ifft_imx = np.fft.ifft2(np.abs(fft_im2x)*np.abs(fft_im1x), norm="ortho")
        ifft_imx = np.real(np.fft.fftshift(ifft_imx))

        ifft_im = ifft_imx

        if len(dim_var1) == 4 or len(dim_var2) == 4:

            # compute FFT of image
            fft_im1y = np.fft.fft2(im1y[frame], norm="ortho")
            fft_im2y = np.fft.fft2(im2y[frame], norm="ortho")

            # inverse of power spectrum
            ifft_imy = np.fft.ifft2(np.abs(fft_im2y)*np.abs(fft_im1y), norm="ortho")
            ifft_imy = np.real(np.fft.fftshift(ifft_imy))

            ifft_im += ifft_imy


        # normalize output
        ifft_im_norm = ifft_im / np.max(ifft_im)

        r_bin, C_tmp, _ = radial_distribution(ifft_im_norm, vox_to_um, binsize)

        C_norm.append(np.ma.array(C_tmp, mask=C_tmp==np.nan))
    
    mask = r_bin <= r_max

    COR = {'C_norm': np.array(C_norm)[:,mask],
           'r_bin':  r_bin[mask]}

    return COR
