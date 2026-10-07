"""Finding a reference voxel ("where's the peak?") and cropping raw
detector data around it, chaining several methods - similar in spirit to
cdiutils' `BcdiPipeline(voxel_reference_methods=[...])` /
`CroppingHandler.chain_centring`.

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

from bcdikit.utils.general import center_of_mass, crop_around_position


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
        vector. `chain_centering` passes this automatically between
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


def chain_centering(data, output_shape, methods, verbose=False):
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

    return cropped_data.copy(), position, cropped_position, roi