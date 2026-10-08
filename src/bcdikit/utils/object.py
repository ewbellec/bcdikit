"""Object-level utilities: module/phase extraction, centering, support
estimation, automatic ROI finding, oversampling and padding.

These work on a reconstructed BCDI object (a complex array), as opposed to
`bcdikit.utils.general`, which has broader/data-agnostic helpers.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifftshift, fftn, ifftn

from bcdikit.utils.general import center_the_center_of_mass


### -----------------------------------------------------------------------
### Centering
### -----------------------------------------------------------------------

def _crop_array_half_size(array):
    """Crop `array` to half its size along every axis, centered. Internal
    helper for `center_object`'s 2-pass centering - not meant for cropping
    your actual object (see `automatic_object_roi` +
    `bcdikit.utils.general.apply_roi` for that)."""
    shape = array.shape
    s = [slice(shape[n] // 2 - shape[n] // 4, shape[n] // 2 + shape[n] // 4) for n in range(array.ndim)]
    return array[tuple(s)]


def center_object(obj, standard_com=True, support=None):
    """Center `obj`'s center of mass on the array's center by rolling it
    (no cropping, no resizing). Centers twice - the 2nd pass on a half-size
    crop of the module around the 1st pass's result - for a more precise
    offset than a single center-of-mass pass would give."""
    module = np.abs(obj)
    shape = obj.shape

    module_cen, offset = center_the_center_of_mass(module, return_offsets=True, standard_com=standard_com)

    module_cen_crop = _crop_array_half_size(module_cen)
    _, offset2 = center_the_center_of_mass(module_cen_crop, return_offsets=True, standard_com=standard_com)

    total_offset = np.array(offset) + np.array(offset2)

    obj_cen = np.roll(obj, total_offset, axis=tuple(range(len(shape))))

    if support is not None:
        support_cen = np.roll(support, total_offset, axis=tuple(range(len(shape))))
        return obj_cen, support_cen
    return obj_cen


def center_object_list(obj_list):
    """`center_object`, applied to every object in `obj_list`."""
    obj_list_centered = np.zeros(obj_list.shape, dtype='complex128')
    for n in range(len(obj_list)):
        obj_list_centered[n] = center_object(obj_list[n])
    return obj_list_centered


### -----------------------------------------------------------------------
### Module / phase extraction
### -----------------------------------------------------------------------

def get_module_phase(obj, threshold_module=None, support=None,
                      apply_fftshift=False, unwrap=True):
    """Split a complex object `obj` into module and phase.

    The support (where the phase is kept - everywhere else comes back as
    nan) is `module >= threshold_module * module.max()` by default
    (threshold_module=.3), or pass your own boolean `support` array
    directly.

    No cropping happens here. This function used to be called
    `get_cropped_module_phase` and cropped the object to half its size by
    default - a bad default, since it silently handed back a smaller array
    than the one you passed in. Crop explicitly before or after calling
    this instead, e.g. with `automatic_object_roi` +
    `bcdikit.utils.general.apply_roi`.
    """
    if apply_fftshift:
        obj = np.fft.fftshift(obj)

    module = np.abs(obj)

    if support is None:
        if threshold_module is None:
            threshold_module = .3
        support = module >= threshold_module * np.nanmax(module)

    phase = np.angle(obj)

    if unwrap:
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            warnings.warn("scikit-image is required to unwrap the phase "
                           "(pip install scikit-image) - returning the wrapped phase instead.")
        else:
            # `phase` can contain nan (e.g. obj has nan voxels left over from
            # an earlier padding/cropping step), and skimage's unwrap_phase
            # can hang indefinitely on that - *even when that voxel is
            # masked out* (e.g. it falls outside the support, or you passed
            # your own `support` that already excludes it). Masking alone
            # isn't enough to avoid the hang: it's specifically the nan
            # sitting in the *underlying* data behind a masked position that
            # trips it up, regardless of why it's masked. So on top of the
            # normal support mask, any nan is additionally masked and its
            # underlying value swapped for a finite dummy (0) before
            # unwrapping; those voxels are set back to nan afterwards either
            # way (nan is never a meaningful phase value).
            nan_mask = np.isnan(phase)
            if np.any(nan_mask):
                warnings.warn(f"{int(np.sum(nan_mask))} voxel(s) of the object are nan - excluding them "
                               "from phase unwrapping (skimage.restoration.unwrap_phase can hang "
                               "otherwise). These voxels come back as nan in the result.")

            mask_unwrap = ~support | nan_mask
            phase_filled = np.where(nan_mask, 0.0, phase)
            phase_masked = np.ma.masked_array(phase_filled, mask=mask_unwrap)
            phase = np.ma.filled(unwrap_phase(phase_masked, wrap_around=False), np.nan)

    phase = np.where(support, phase, np.nan)

    return module, phase


def get_module_phase_high_strain(obj, threshold_module=None, apply_fftshift=False, unwrap=True):
    """Module/phase extraction tuned for objects with high local strain:
    the phase gradient is added back into the module before thresholding,
    so high-strain (= high phase-gradient) regions near the support edge
    are less likely to get cut off by a plain module threshold.

    Not extensively validated across particle types - use with care.
    """
    module, phase = get_module_phase(obj, threshold_module=0., apply_fftshift=apply_fftshift, unwrap=unwrap)

    phase_grad = np.array(np.gradient(phase))
    phase_grad = np.sqrt(np.nansum(phase_grad ** 2., axis=0))

    factor = np.copy(phase_grad)
    factor[factor > 2.] = 2.  # hardcoded cap on the phase-gradient contribution
    module_custom = module * (1. + factor * 2.)

    if threshold_module is None:
        threshold_module = .3
    support = module_custom > threshold_module * np.nanmax(module_custom)
    phase = np.where(support, phase, np.nan)

    return module, phase


### -----------------------------------------------------------------------
### Complex conjugate ("twin image")
### -----------------------------------------------------------------------

def get_complex_conjugate(obj):
    """The complex-conjugate twin of a reconstructed object (same module,
    mirrored support, opposite-sign phase - BCDI reconstructions are
    ambiguous up to this transformation)."""
    return ifftshift(ifftn(np.conj(fftn(fftshift(obj)))))


### -----------------------------------------------------------------------
### Support
### -----------------------------------------------------------------------

def create_support(obj, threshold_module, fill_support=False):
    """Boolean support of `obj` (module > threshold_module * module.max()).
    `fill_support=True` additionally fills in internal holes/concavities
    along each axis direction (so the support has no "gaps" when viewed
    along any single axis - not geometrically convex)."""
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)

    if not fill_support:
        return support

    support_filled = np.zeros(support.shape, dtype=bool)
    for axis in range(support.ndim):
        support_cum = np.cumsum(support, axis=axis)
        support_cum_inv = np.flip(np.cumsum(np.flip(support, axis=axis), axis=axis), axis=axis)
        support_filled |= (support_cum * support_cum_inv) != 0
    return support_filled


### -----------------------------------------------------------------------
### Automatic ROI
### -----------------------------------------------------------------------

def automatic_object_roi(obj, threshold=.1, factor=.4, plot=False):
    """Find an roi (see `bcdikit.utils.general.apply_roi`) around `obj`'s
    support, from its 1D projections along each axis: start from where each
    projection first crosses `threshold` (of its own max) and expand by
    `factor` times the projection's width on each side."""
    module = np.abs(obj)

    if plot:
        fig, ax = plt.subplots(1, module.ndim, figsize=(5 * module.ndim, 3))

    roi = np.zeros(2 * module.ndim, dtype='int')
    for n in range(module.ndim):
        sum_axis = tuple(a for a in range(module.ndim) if a != n)
        projection = np.nanmean(module, axis=sum_axis)
        projection = projection - np.nanmin(projection)
        projection = projection / np.nanmax(projection)

        start = np.nanmin(np.where(projection > threshold))
        end = np.nanmax(np.where(projection > threshold))

        size = end - start
        start = max(int(start - size * factor), 0)
        end = int(end + size * factor)

        roi[2 * n] = start
        roi[2 * n + 1] = end

        if plot:
            ax[n].plot(projection, '.-')
            ax[n].axvline(x=start, color='r')
            ax[n].axvline(x=end, color='r')

    return roi


def automatic_object_roi_on_support(obj, threshold_module=.3, plot=False):
    """Like `automatic_object_roi`, but computed on `obj`'s boolean support
    rather than its raw module - useful for low-photon-count data (e.g.
    BM02), where the raw module's projections are too noisy."""
    support = create_support(obj, threshold_module=threshold_module)
    return automatic_object_roi(support, plot=plot)


### -----------------------------------------------------------------------
### Oversampling
### -----------------------------------------------------------------------

def compute_oversampling_ratio(obj=None, threshold_module=.3, support=None, plot=False):
    """Oversampling ratio (array size / support size) along each axis.

    Pass `obj` (the default) to have the support thresholded from its
    module automatically (see `threshold_module`), or pass `support`
    directly to use one you already computed another way - e.g. from an
    auto-correlation, before any reconstruction even exists yet (see
    `bcdikit.preprocessing.diagnostics.oversampling_from_diffraction`).
    """
    if support is None:
        if obj is None:
            raise ValueError("compute_oversampling_ratio needs either `obj` or `support`")
        module = np.abs(obj)
        support = module > threshold_module * np.max(module)

    indices_support = np.where(support)
    size_per_dim = np.max(indices_support, axis=1) - np.min(indices_support, axis=1)
    oversampling = np.array(support.shape) / size_per_dim

    if plot:
        fig, ax = plt.subplots(1, support.ndim, figsize=(5 * support.ndim, 4))
        for n in range(support.ndim):
            # bug fix: the original hardcoded `np.arange(3)` here, which broke
            # (wrong/empty projection) for anything but a 3D support
            proj_axes = tuple(a for a in range(support.ndim) if a != n)
            proj = np.max(support, axis=proj_axes)
            ax[n].plot(proj)
            ax[n].set_title(f'oversampling along axis {n}\n{round(oversampling[n], 2)}', fontsize=15)

    return oversampling


### -----------------------------------------------------------------------
### Padding
### -----------------------------------------------------------------------

def pad_to_higher_oversampling(obj, oversampling_final):
    """Zero-pad `obj` so its oversampling ratio along each axis matches
    `oversampling_final`."""
    oversampling = compute_oversampling_ratio(obj)
    padding = []
    for axis in range(obj.ndim):
        size = obj.shape[axis]
        final_size = int(np.ceil(size * oversampling_final[axis] / oversampling[axis]))
        pad = int(np.ceil((final_size - size) / 2.))
        padding.append([pad, pad])
    return np.pad(obj, padding)


def pad_to_equal_shape(obj, target=None, verbose=True):
    """Zero-pad `obj` to a cubic/square shape (same size along every axis).
    Pads to `target` if given, otherwise to `max(obj.shape)`."""
    if target is None:
        target = max(obj.shape)

    padding = []
    for size in obj.shape:
        total_pad = target - size
        before = total_pad // 2
        after = total_pad - before
        padding.append((before, after))

    obj_padded = np.pad(obj, padding, mode='constant', constant_values=0)

    if verbose:
        print("Original shape:", obj.shape)
        print("Padded shape:", obj_padded.shape)
    return obj_padded