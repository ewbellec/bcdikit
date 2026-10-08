"""Automatic ROI / reference-position selection for raw (detector-frame)
BCDI data, to crop the data before orthogonalization.

`get_reference_position` finds a single reference voxel ("where's the
peak?") in `data`, via `'max'`, `'com'`, or your own first-guess position.
`crop_around_peak` chains a sequence of these, narrowing the crop window
at each step - similar in spirit to cdiutils'
`BcdiPipeline(voxel_reference_methods=[...])` / `CroppingHandler.chain_centring`.

A typical sequence is `["max", "com", "com"]`: a coarse peak search, then
1-2 center-of-mass refinements. A plain center of mass on the raw,
unmasked data is easily thrown off by background/noise far from the peak;
restricting it to the window found by a coarser method first (and
narrowing that window at each step) fixes that. You can also start the
sequence with your own position (a first guess from a previous run, or a
known Bragg peak position) instead of "max" - either a full 3-vector, or
just a 2-vector for axes 1 and 2 (the detector's row/column) if you don't
want to guess axis 0 (e.g. the rocking-curve/scan axis) - that axis then
defaults to its center.

`output_shape` can have a None entry for any axis - that axis is cropped
to the largest size that stays centered on the found position, instead of
a fixed size (see `bcdikit.utils.general.crop_around_position`).

Built on `bcdikit.utils.general.center_of_mass` and
`bcdikit.utils.general.crop_around_position` rather than reimplementing
them.
"""

import numpy as np
import matplotlib.patches as patches

from bcdikit.utils.general import center_of_mass, crop_around_position
from bcdikit.utils.plot import plot_projections


def get_reference_position(data, method, previous_position=None):
    """Find a single reference voxel in `data`.

    Parameters
    ----------
    data : np.ndarray
        Real or complex (the module is used if complex). Can be a
        `np.ma.MaskedArray` - masked-out voxels are excluded from the
        search either way (max: never picked; com: contribute 0).
    method : 'max' | 'com' | position vector of ints
        'max': position of the maximum value (nan-safe).
        'com': center of mass, rounded to the nearest voxel.
        a position vector: used as-is - lets you pass your own first
        guess instead of a computed one. Either a full `data.ndim`-length
        vector, or a `data.ndim - 1`-length one giving only axes 1, 2,
        ... (e.g. a (row, column) 2-vector for 3D detector data, leaving
        axis 0 - typically the rocking-curve/scan axis - unspecified):
        axis 0 is then taken from `previous_position` if given, otherwise
        defaults to that axis's center.
    previous_position : tuple of int, optional
        The previous step's reference position, used to fill in axis 0
        when `method` is a partial (`data.ndim - 1`-length) position
        vector. `crop_around_peak` passes this automatically between
        chained steps; not needed when calling this directly.

    Returns
    -------
    tuple of int
    """
    if np.iscomplexobj(data):
        data = np.abs(data)

    if isinstance(data, np.ma.MaskedArray):
        # the fill value has to be representable in the array's dtype -
        # an integer dtype (common for raw detector counts: int32/uint32)
        # can't hold -inf, so cast to float first.
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)
        data_for_max = data.filled(-np.inf)
        data_for_com = data.filled(0)
    else:
        data_for_max = data_for_com = data

    if method == 'max':
        return np.unravel_index(np.nanargmax(data_for_max), data.shape)

    if method == 'com':
        com = center_of_mass(data_for_com)
        return tuple(int(round(c)) for c in com)

    if hasattr(method, '__len__') and all(isinstance(e, (int, np.integer)) for e in method):
        method = tuple(int(e) for e in method)

        if len(method) == data.ndim:
            return method

        if len(method) == data.ndim - 1:
            axis0 = previous_position[0] if previous_position is not None else data.shape[0] // 2
            return (axis0,) + method

        raise ValueError(
            f"a position vector must have {data.ndim} entries (one per "
            f"axis), or {data.ndim - 1} to give only axes 1.. (axis 0 is "
            f"then taken from the previous step, or defaults to its "
            f"center) - got {len(method)}: {method!r}"
        )

    raise ValueError(f"method must be 'max', 'com', or a position vector, got {method!r}")


def crop_around_peak(data, output_shape=(None,None,None), methods=['max', 'com'], verbose=False, plot=False):
    """Apply a sequence of centering `methods` to `data`, refining the
    crop window at each step, and return the cropped data centered on the
    final position.

    Parameters
    ----------
    data : np.ndarray
        The (uncropped) data to center and crop.
    output_shape : array-like of (int or None)
        The shape to crop to (same length as `data.ndim`). An axis can be
        None to take the largest size that keeps that axis centered on
        the found position, rather than a fixed size (see
        `bcdikit.utils.general.crop_around_position`).
    methods : ('max' | 'com' | position vector of ints) or a list of those
        A single method, or a sequence applied one after another, each
        one refining the crop window the previous one found. E.g.
        `["max", "com", "com"]`, or `[(70, 200, 200), "com"]` to start
        from your own first guess. The first guess can also be a
        `data.ndim - 1`-length vector covering only axes 1, 2, ... (e.g.
        `[(200, 200), "com"]` for 3D data, giving just the detector
        row/column) - axis 0 then defaults to its center (see
        `get_reference_position`).
    verbose : bool
        Print each step's method, position, and the data value there.
    plot : bool
        If True, show two `bcdikit.utils.plot.plot_projections` figures:
        the full `data`, with the found `position` (a cross) and `roi` (a
        rectangle) overlaid in red (alpha=.5) on all 3 projections, and
        the cropped result on its own.

    Returns
    -------
    cropped_data : np.ndarray
    position : tuple of int
        Reference position, in `data`'s original coordinates (the center
        of the final crop window - equal to the last method's found
        position, unless the window had to be shifted to stay in bounds).
    cropped_position : tuple of int
        Reference position, in `cropped_data`'s own coordinates.
    roi : list of int
        The roi used (see `bcdikit.utils.general.apply_roi`).
    """
    if isinstance(methods, str) or not hasattr(methods, '__len__') or (
        hasattr(methods, '__len__') and all(isinstance(e, (int, np.integer)) for e in methods)
    ):
        methods = [methods]

    masked_data = data
    roi = None
    cropped_data = None
    previous_position = None
    for method in methods:
        position = get_reference_position(masked_data, method, previous_position=previous_position)
        if verbose:
            print(f"\t- {method}: {position}, value: {data[position]}")

        cropped_data, roi = crop_around_position(data, position, output_shape)

        mask = np.ones(data.shape, dtype=bool)
        mask[tuple(slice(roi[2 * n], roi[2 * n + 1]) for n in range(data.ndim))] = False
        masked_data = np.ma.array(data, mask=mask)
        previous_position = position

    position = tuple((roi[2 * n] + roi[2 * n + 1]) // 2 for n in range(data.ndim))
    cropped_position = tuple(position[n] - roi[2 * n] for n in range(data.ndim))

    if plot:
        _plot_crop_result(data, cropped_data, position, roi)

    return cropped_data.copy(), position, cropped_position, roi


### -----------------------------------------------------------------------
### Diagnostic plot
### -----------------------------------------------------------------------

def _add_cross_marker(ax, shape, position, color='red', alpha=.5, lw=1.5):
    """Overlay a small cross at `position` on each of `ax`'s 3
    `plot_projections` panels - one differently-placed cross per panel,
    since each panel shows a different pair of axes (see
    `plot_projections`'s own row/col convention: panel `n` projects out
    axis `n` and shows the other two)."""
    for axis in range(3):
        remaining = [a for a in range(3) if a != axis]
        y, x = position[remaining[0]], position[remaining[1]]
        ny, nx = shape[remaining[0]], shape[remaining[1]]

        # matshow doesn't autoscale, but a Line2D does - a cross near an
        # edge sticks out past the image's own extent, and plotting it
        # would otherwise silently zoom the view out to fit it (leaving a
        # white margin, and shrinking the image within the panel). Put
        # the limits back exactly as matshow set them once we're done.
        xlim, ylim = ax[axis].get_xlim(), ax[axis].get_ylim()
        ax[axis].plot([x, x], [y - 0.1 * ny, y + 0.1 * ny], color=color, alpha=alpha, lw=lw)
        ax[axis].plot([x - 0.1 * nx, x + 0.1 * nx], [y, y], color=color, alpha=alpha, lw=lw)
        ax[axis].set_xlim(xlim)
        ax[axis].set_ylim(ylim)
    return


def _add_roi_rectangle(ax, shape, roi, color='red', alpha=.5, lw=1.5):
    """Overlay a rectangle outlining `roi` on each of `ax`'s 3
    `plot_projections` panels - a differently-shaped rectangle per panel
    (same convention as `_add_cross_marker` above), not just one rectangle
    repeated 3 times."""
    for axis in range(3):
        remaining = [a for a in range(3) if a != axis]
        y0, y1 = roi[2 * remaining[0]], roi[2 * remaining[0] + 1]
        x0, x1 = roi[2 * remaining[1]], roi[2 * remaining[1] + 1]
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=lw, edgecolor=color, alpha=alpha, facecolor='none')

        # same reasoning as _add_cross_marker above: keep matshow's own
        # view limits, don't let the patch's extent rescale the panel.
        xlim, ylim = ax[axis].get_xlim(), ax[axis].get_ylim()
        ax[axis].add_patch(rect)
        ax[axis].set_xlim(xlim)
        ax[axis].set_ylim(ylim)
    return


def _plot_crop_result(data, cropped_data, position, roi):
    """The `plot=True` figures for `crop_around_peak`: `data`'s
    projections with `position`/`roi` overlaid, and `cropped_data`'s
    projections on their own."""
    fig, ax = plot_projections(data, title='full data', return_fig_ax=True)
    _add_cross_marker(ax, data.shape, position)
    _add_roi_rectangle(ax, data.shape, roi)

    plot_projections(cropped_data, title='cropped data')
    return