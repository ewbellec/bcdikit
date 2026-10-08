import os
import numpy as np
import h5py

from bcdikit.utils.plot import plot_projections

### -----------------------------------------------------------------------
### Mask
### -----------------------------------------------------------------------
 
# def load_mask(scan, data, roi=None, mask_dir=None, plot=False):
#     """Load a detector mask for `scan`, broadcast to `data`'s 3D shape.
 
#     Uses `scan.mask` directly if the data opener already provides one
#     (e.g. PETRA). Otherwise, if `mask_dir` is given, looks for
#     `mask_dir/mask_<scan.detector>.npy` (optionally cropped to `roi`,
#     `[y0, y1, x0, x1]`, detector pixel coordinates). Falls back to an
#     all-zero ("nothing masked") mask if neither is available.
#     """
#     if getattr(scan, 'mask', None) is not None:
#         mask = np.zeros(data.shape)
#         mask += scan.mask[None, :, :]
#     elif mask_dir is not None:
#         mask_path = os.path.join(mask_dir, f'mask_{scan.detector}.npy')
#         if os.path.isfile(mask_path):
#             mask2d = np.load(mask_path)
#             if roi is not None:
#                 mask2d = mask2d[roi[0]:roi[1], roi[2]:roi[3]]
#             mask = np.zeros(data.shape)
#             mask += mask2d[None]
#         else:
#             mask = np.zeros(data.shape)
#     else:
#         mask = np.zeros(data.shape)
 
#     if plot:
#         plot_projections(mask, log_scale=False, cmap='gray_r', max_projection=True, title='mask')
#         plot_projections((1 - mask) * data, title='data with mask applied')
 
#     return mask

import os
import numpy as np

from bcdikit.utils.plot import plot_projections
from bcdikit.utils.general import apply_roi, print_warning

### -----------------------------------------------------------------------
### Mask
### -----------------------------------------------------------------------

def load_mask(detector, data, roi=None, mask_dir=None, plot=False):
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
 
def check_detector_saturation(data, scan, saturation_levels=None):
    '''
    Warn if `data`'s maximum is above the detector's known linear
    dynamic range.
 
    `saturation_levels` is a `{detector_name: counts}` dict, merged over
    the built-in defaults (currently just `{'mpx1x4': 150000}`) - pass
    your own entry for a detector that isn't listed here.
    '''
    levels = {'mpx1x4': 150000,
              'mpxgaas': 150000,
              'eiger2M' : 1e6}
 
    maxi = np.nanmax(data)
    print('maximum counts :', maxi)
 
    if scan.detector in levels:
        if maxi > levels[scan.detector]:
            print(f'{scan.detector} detector in non-linear dynamic range')
        else:
            print('no saturation')
    else:
        print('detector saturation level unknown. Can\'t tell you, sorry.')
    return