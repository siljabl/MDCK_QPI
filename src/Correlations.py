import numpy as np

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



def radial_distribution(im, vox_to_um, binsize=5):
    # raduis matrix
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
        dist[i] = np.mean(r[mask])
        mean[i] = np.mean(im[mask])
        std[i]  = np.std(im[mask])

    return dist, mean, std