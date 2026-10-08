"""General-purpose helper functions: array/data handling, ROI/center-of-mass
utilities, small math helpers, and file-listing helpers.

Plotting-only functions that used to live alongside these (color bars, ROI
overlays, gradient plots, detector-sum plots...) have been moved to
`bcdikit.utils.plot` instead - this file sticks to functions that return
data, not figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


### -----------------------------------------------------------------------
### File listing
### -----------------------------------------------------------------------

def check_path_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def get_npz_files(path):
    files_all = os.listdir(path)
    files = []
    for f in files_all:
        if '.npz' in f:
            files.append(path + f)
    return files


def get_numpy_files(path):
    files_all = os.listdir(path)
    files = []
    for f in files_all:
        if ('.npz' in f) or ('.npy' in f):
            files.append(path + f)
    return files


### -----------------------------------------------------------------------
### Center of mass / centering
### -----------------------------------------------------------------------

def center_of_mass(array, pos=None):
    """
    Center of mass of an n-dimensional array.
    `pos` lets you pass a pre-computed `np.indices(array.shape)` when
    calling this repeatedly on arrays of the same shape, to avoid
    recomputing it every time.
    """
    cen = np.zeros(array.ndim)
    if pos is None:
        pos = np.indices(array.shape)
    proba = array / np.nansum(array)
    for n in range(array.ndim):
        cen[n] += np.nansum(pos[n] * proba)
    return cen


def center_of_mass_calculation_two_steps(data,
                                          crop=50,
                                          return_int=False,
                                          plot=False):
    """
    Two-step center of mass: find the max first, crop a window of size
    `crop` around it, then compute the center of mass within that window
    only (much less sensitive to background/noise far from the peak than a
    center of mass over the whole array).

    return_int=False (default) returns the sub-pixel (float) position.
    Pass return_int=True if you need integer array indices instead.
    """
    center = np.unravel_index(np.nanargmax(data), data.shape)

    cropping_dim = []
    for n in range(data.ndim):
        cropping_dim.append([
            max([0, int(center[n] - crop / 2)]),
            min(int(center[n] + crop // 2), data.shape[n] - 1),
        ])

    s = [slice(cropping_dim[n][0], cropping_dim[n][1]) for n in range(data.ndim)]
    center2 = center_of_mass(data[tuple(s)])

    if return_int:
        center = [int(round(cropping_dim[n][0] + center2[n])) for n in range(data.ndim)]
    else:
        center = [cropping_dim[n][0] + center2[n] for n in range(data.ndim)]

    if plot:
        if data.ndim == 3:
            # local import: bcdikit.utils.plot imports from this module, so
            # importing it back at module level here would be circular -
            # deferring it to call time (only needed for this optional
            # plot) avoids that.
            from bcdikit.utils.plot import plot_projections
            fig, ax = plot_projections(data, fig=None, ax=None, return_fig_ax=True)
            # bug fix: the original had center[2] twice here (so the 'w'
            # marker on this panel was never actually at the found center)
            ax[0].scatter(center[2], center[1], color='w')
            ax[1].scatter(center[2], center[0], color='w')
            ax[2].scatter(center[1], center[0], color='w')
        if data.ndim == 2:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(np.log(data), cmap='plasma', vmin=1)
            plt.colorbar()
            plt.scatter(center[1], center[0], color='w')
    return center


def center_the_center_of_mass(data,
                               qx=None, qy=None, qz=None,
                               standard_com=False,
                               plot=False, vmin=None,
                               return_offsets=False,
                               cmap='plasma', norm=None,
                               scatter_color='g', scatter_size=10):
    """
    Center the center of mass of a 3D matrix.
    Used to re-center the Bragg peak after a small random shift (see
    "Createqxqyqz").
    """
    shape = data.shape

    data[~np.isfinite(data)] = 0

    if standard_com:
        com = center_of_mass(data)
    else:
        com = center_of_mass_calculation_two_steps(data)

    offset = [int(np.rint(shape[n] / 2.0 - com[n])) for n in range(len(shape))]

    data_cen = np.roll(data, offset, axis=range(len(shape)))
    if qx is not None:
        qx = np.roll(qx, offset, axis=range(len(shape)))
        qy = np.roll(qy, offset, axis=range(len(shape)))
        qz = np.roll(qz, offset, axis=range(len(shape)))

    if plot:
        if len(shape) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(data, cmap=cmap, vmin=vmin, norm=norm)
            ax[0].scatter(com[1], com[0], c=scatter_color, s=scatter_size)
            ax[1].imshow(data_cen, cmap=cmap, vmin=vmin, norm=norm)

    if return_offsets:
        return data_cen, offset
    else:
        if qx is not None:
            return data_cen, qx, qy, qz
        else:
            return data_cen


### -----------------------------------------------------------------------
### Cropping / ROI
### -----------------------------------------------------------------------

def crop_array_symmetric(array, crop_array, inverse_crop=False):
    s = []
    for n in range(array.ndim):
        if inverse_crop:
            s.append(slice(array.shape[n] // 2 - crop_array[n] // 2,
                            array.shape[n] // 2 + crop_array[n] // 2))
        else:
            if crop_array[n] != 0:
                s.append(slice(crop_array[n], -crop_array[n]))
            else:
                s.append(slice(None))
    array_crop = array[tuple(s)]
    return array_crop


def crop_given_cropped_shape(array, cropped_shape):
    """
    Like crop_array_symmetric, but you give the target shape you want
    instead of how much to crop off each side. Pass None for any axis you
    don't want cropped.
    """
    cropped_shape = np.where(np.array(cropped_shape) == None, np.array(array.shape), cropped_shape)
    crop_array = (np.array(array.shape) - np.array(cropped_shape)) // 2
    return crop_array_symmetric(array, crop_array)


def apply_roi(array, roi, verbose=True):
    """
    roi = [start0, end0, start1, end1, ...]. roi=None returns the array
    unchanged. If `array` has more axes than `roi` covers, the roi is
    applied to the last axes only (e.g. a 2D roi on a 3D stack of images).
    """
    if roi is None:
        return array

    roi_dim = len(roi) // 2
    if array.ndim > roi_dim:
        dim_diff = array.ndim - roi_dim
        roi = [None] * (2 * dim_diff) + list(roi)
        if verbose:
            print("WARNING : ROI dimension is smaller than the array. ROI is applied along last axes only")

    s = [slice(roi[2 * n], roi[2 * n + 1]) for n in range(array.ndim)]
    return array[tuple(s)]


def crop_around_position(array, position, output_shape):
    """
    Crop `array` to `output_shape`, centered on `position`.

    `output_shape` can contain None for any axis: that axis is cropped to
    the largest size that keeps the crop symmetric about `position` along
    it - i.e. the biggest region still truly centered there, bounded by
    whichever side of `position` is closer to the array's edge. This is
    different from the "shift to preserve the exact requested size"
    behaviour used for an axis you do give an explicit size for (below):
    a None axis never gets shifted off-center to reach a target size,
    since it has no target size to reach.

    An explicit (non-None) `output_shape[axis]` that is `>=
    array.shape[axis]` is clamped to the whole axis - you get everything
    along that axis rather than an error or an out-of-bounds index.

    Otherwise, the crop window is shifted (not clipped) to stay inside
    the array when `position` is close to an edge, so you still get
    exactly `output_shape[axis]` voxels whenever that's geometrically
    possible (e.g. a peak 2 voxels from the edge with a requested size of
    16 still gets a full 16-voxel crop, shifted inward, rather than a
    truncated one).

    Parameters
    ----------
    array : np.ndarray
    position : array-like of int
        Same length as `array.ndim`.
    output_shape : array-like of (int or None)
        Same length as `array.ndim`.

    Returns
    -------
    cropped_array : np.ndarray
    roi : list of int
        The roi used (see `apply_roi`).
    """
    shape = array.shape
    roi = []
    for axis, size in enumerate(shape):
        pos = position[axis]
        target = output_shape[axis]

        if target is None:
            half = min(pos, size - 1 - pos)
            start, end = pos - half, pos + half + 1
        elif target >= size:
            start, end = 0, size
        else:
            half_before = target // 2
            half_after = target - half_before  # 1 more than half_before for odd sizes

            start, end = pos - half_before, pos + half_after
            if end > size:
                shift = end - size
                start -= shift
                end -= shift
            if start < 0:
                shift = -start
                start += shift
                end += shift

        roi.append(max(start, 0))
        roi.append(min(end, size))

    return apply_roi(array, roi, verbose=False), roi


def roi_automatic_peak(data, roi_size, plot=False):
    """
    Build an roi (see apply_roi) of size `roi_size` centered on the data's
    peak (found via center_of_mass_calculation_two_steps). Clipped to the
    array bounds if the requested size doesn't fit.

    Parameters
    ----------
    data : np.ndarray
        Input array (any dimension).
    roi_size : int or array-like
        Size of the crop, one value per axis (or a single int for all axes).
    """
    center = np.round(center_of_mass_calculation_two_steps(data)).astype('int')

    roi_size = np.array([roi_size] * data.ndim) if np.isscalar(roi_size) else np.array(roi_size)

    shape = np.asarray(data.shape)
    half = roi_size // 2

    start = center - half
    end = start + roi_size

    start = np.maximum(start, 0)
    end = np.minimum(end, shape)

    roi = np.empty(data.ndim * 2, dtype=int)
    roi[0::2] = start
    roi[1::2] = end

    if plot:
        from bcdikit.utils.plot import plot_detector_sum
        plot_detector_sum(apply_roi(data, roi))
    return roi


### -----------------------------------------------------------------------
### Small math / array helpers
### -----------------------------------------------------------------------

def pearson_coef(img1, img2):
    x = img1 - np.nanmean(img1)
    y = img2 - np.nanmean(img2)
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x ** 2.) * np.sum(y ** 2.))
    return numerator / denominator


def hot_pixel_filter(data, threshold=1e2):
    """Remove hot pixels (via a median filter) that can mess up preprocessing."""
    from scipy.ndimage import median_filter
    data_median = median_filter(data, size=2)
    mask = (data < threshold * (data_median + 1))
    data_clean = data * mask
    return data_clean


def normalize_array(array, new_min=0, new_max=1):
    """Rescale `array` linearly so its values span [new_min, new_max]."""
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    if array_max == array_min:
        # constant array - nothing to rescale, avoid a divide-by-zero
        return np.full_like(array, new_min, dtype=float)
    return new_min + (array - array_min) * (new_max - new_min) / (array_max - array_min)


def slice_middle_array_along_axis(array, axis):
    s = [slice(None, None, None) for ii in range(array.ndim)]
    s[axis] = array.shape[axis] // 2
    return tuple(s)


def even_dimension_slices(shape):
    """Per-axis slices that trim `shape` down to even sizes (dropping
    index 0 of any odd axis). An fftshift/ifftshift pair - bcdikit's FFT
    convention, see `create_diffracted_amplitude`/`create_object` below -
    assumes an even-sized array along every axis; this is what
    `force_even_dimension_one_array` uses to get there.
    """
    return tuple(slice(None) if s % 2 == 0 else slice(1, None) for s in shape)


def force_even_dimension_one_array(array, verbose=True):
    array_even = array[even_dimension_slices(array.shape)]

    if verbose:
        print('shape changed :')
        print('array {} to {}'.format(array.shape, array_even.shape))
    return array_even


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2.

    Parameters
    ----------
    vec1 : 3-vector, the "source" vector
    vec2 : 3-vector, the "destination" vector

    Returns
    -------
    A (3, 3) transform matrix which, applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def create_radius_array(array):
    r = np.indices(array.shape)
    r = r - np.mean(r, axis=tuple(range(1, array.ndim + 1)))[:, None, None]
    r = np.sqrt(np.sum(r ** 2., axis=0))
    return r


def wavelength_from_energy(energy):
    """X-ray wavelength (in meters) from photon energy (in eV)."""
    planck = 6.62607015e-34
    light_speed = 299792458
    eV_unit = 1.602176634e-19
    return planck * light_speed / (energy * eV_unit)


### -----------------------------------------------------------------------
### Fourier transforms (BCDI convention: shift to/from the array corner
### before/after the FFT)
### -----------------------------------------------------------------------

from scipy.fft import fftshift, ifftshift, fftn, ifftn


def create_diffracted_amplitude(obj):
    return ifftshift(fftn(fftshift(obj)))


def create_object(fexp):
    return ifftshift(ifftn(fftshift(fexp)))


### -----------------------------------------------------------------------
### Orthogonalization / gridding (optional heavy dependency: xrayutilities)
### -----------------------------------------------------------------------

def interpolation_xrayutilities_gridder(x, y, z, data, fuzzy_gridder=False, maxbins=None):
    """Grid `data` (given at the possibly-irregular 3D coordinates `x`,
    `y`, `z`) onto a regular 3D grid via xrayutilities.

    `maxbins`, if given, skips the automatic bin-count estimation (the
    number of bins along each axis, picked from the finest step in `x`/
    `y`/`z`) - pass the same `maxbins` across several calls (e.g. data and
    a mask) to make sure they land on the exact same grid.
    """
    import xrayutilities as xu
    if maxbins is None:
        maxbins = []
        for dim in (x, y, z):
            maxstep = max((abs(np.diff(dim, axis=j)).max() for j in range(3)))
            maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))

    if fuzzy_gridder:
        gridder = xu.FuzzyGridder3D(*maxbins)
    else:
        gridder = xu.Gridder3D(*maxbins)

    gridder(x, y, z, data)
    data_ortho = gridder.data
    x1d, y1d, z1d = [gridder.xaxis, gridder.yaxis, gridder.zaxis]

    return data_ortho, x1d, y1d, z1d


def rotate_3D_array_vector_to_vector(array3D, vector_start, vector_end,
                                      voxel_sizes=None,
                                      padding=None):
    """
    Rotate a 3D array so that `vector_start` aligns with `vector_end`.

    NOTE: fixed a typo bug carried over from the original file (`nP.array`
    -> `np.array`, which raised NameError on the default voxel_sizes=None
    path). Left alone: the `padding` this function computes is never
    actually used - the array is always padded by a hardcoded (20, 20, 20)
    instead. Not sure if that's intentional or a leftover bug - let me know
    which it should be.
    """
    rotation_matrix = rotation_matrix_from_vectors(vector_start, vector_end)

    if voxel_sizes is None:
        voxel_sizes = np.array([1, 1, 1])

    if padding is None:
        size = np.array(array3D.shape) * voxel_sizes
        max_size = max(size)
        padding = - (np.array(array3D.shape) * voxel_sizes - max_size) / voxel_sizes
        padding = padding // 2 + padding % 2
        # `padding` computed here is currently unused - see docstring note.

    array3D = np.pad(array3D, ((20, 20), (20, 20), (20, 20)), mode='constant', constant_values=(np.nan,))

    pos = np.indices(array3D.shape)
    pos = pos - np.mean(pos, axis=(1, 2, 3))[:, None, None, None]
    pos = np.rollaxis(pos, 0, pos.ndim)  # place the vector axis last
    pos = pos * voxel_sizes[None, None, None]

    pos_rot = np.dot(pos, rotation_matrix.T)

    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator(
        (pos[..., 0][:, 0, 0], pos[..., 1][0, :, 0], pos[..., 2][0, 0, :]),
        array3D,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    array3D_rot = rgi((pos_rot[..., 0], pos_rot[..., 1], pos_rot[..., 2]), method='linear')

    return array3D_rot


### -----------------------------------------------------------------------
### Misc
### -----------------------------------------------------------------------

def print_dic(dico):
    string = '{'
    for n, key in enumerate(dico.keys()):
        if n != 0:
            string += '\n'
        string += f'\'{key}\': {dico[key]},'
    string = string[:-1] + '}'
    print(string)
    return


def print_warning(text, color='r'):
    colors_map = {
        'r': '\033[91m',  # red
        'g': '\033[92m',  # green
        'b': '\033[94m',  # blue
    }
    reset = '\033[0m'
    color_code = colors_map.get(color, colors_map['r'])
    print(f"{color_code}{text}{reset}")
    return