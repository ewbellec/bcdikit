import os
import numpy as np

from bcdikit.utils.plot import plot_projections
from bcdikit.utils.general import apply_roi, print_warning, file_opener

### -----------------------------------------------------------------------
### Mask
### -----------------------------------------------------------------------

def mask_loading(detector, data, roi=None, mask_dir=None, plot=False):
    '''
    Load a 2D detector mask for `detector`, broadcast to `data`'s 3D shape.

    Looks for `mask_dir/mask_<detector>.npy` (optionally cropped to
    `roi`'s detector-plane extent, `roi[2:]` = `[y0, y1, x0, x1]`).
    Falls back to an all-zero ("nothing masked") mask if the file isn't
    found.

    mask_dir : str, optional
        Defaults to bcdikit's own `detector_masks/` folder. Pass your
        own directory to use a mask that isn't bundled with the package.
    '''
    if mask_dir is None:
        # That's a trick, my mask files are saved, starting from this mask_flat_diagnosis.py path in ../detector_masks
        # os.path.dirname(__file__) is the path of the current mask_flat_diagnosis.py file. That's from Claude .
        mask_dir = os.path.join(os.path.dirname(__file__), '..', 'detector_masks')

    mask_file = os.path.join(mask_dir, f'mask_{detector}.npy')

    if not os.path.exists(mask_file):
        print_warning(f'mask file not found ({mask_file}), returning an empty mask')
        mask = np.zeros(data.shape)
    else:
        mask2d = np.load(mask_file)
        if roi is not None:
            mask2d = apply_roi(mask2d, roi[2:])
        mask = np.zeros(data.shape)
        mask += mask2d[None, :, :]

    if plot:
        plot_projections(mask, log_scale=False, cmap='gray_r', max_projection=True, title='mask')
        plot_projections((1 - mask) * data, title='data with mask applied')

    return mask


### -----------------------------------------------------------------------
### Detector saturation
### -----------------------------------------------------------------------
 
def saturation_detector_check(data, detector):
    '''
    Warn if `data`'s maximum is above the detector's known linear
    dynamic range.
    '''
    levels = {'mpx1x4': 150000,
              'mpxgaas': 150000,
              'eiger2M' : 1e6}
 
    maxi = np.nanmax(data)
    print('maximum counts :', maxi)
 
    if detector in levels:
        if maxi > levels[detector]:
            print(f'{detector} detector in non-linear dynamic range')
        else:
            print('no saturation')
    else:
        print('detector saturation level unknown. Can\'t tell you, sorry.')
    return


### -----------------------------------------------------------------------
### Flatfield correction
### -----------------------------------------------------------------------

def flatfield_correction(data, roi, flatfield_file, plot=False):
    """
    Multiply `data` by a flatfield loaded from `flatfield_file` (.h5,
    .npz or .npy), cropped to `roi`'s detector-plane extent
    (`[_, _, y0, y1, x0, x1]`, only the last 4 entries are used).
    """
    flatfield = file_opener(flatfield_file)
 
    flatfield = apply_roi(flatfield,roi[2:])
    data_corrected = data * flatfield[None]
    data_corrected[np.isnan(data_corrected)] = 0  # nan's in the flatfield mask out that pixel
 
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(np.log(flatfield))
        plt.colorbar()
        plt.title('log flatfield (in ROI)', fontsize=20)
 
        plot_projections(data, title='before flatfield correction')
        plot_projections(data_corrected, title='after flatfield correction')
 
    return data_corrected