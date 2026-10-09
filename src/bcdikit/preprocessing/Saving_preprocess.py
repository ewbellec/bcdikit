

### -----------------------------------------------------------------------
### Even dimensions (required before fftshift/ifftshift-based processing)
### -----------------------------------------------------------------------

def even_dimension_forcing(data, q, mask, verbose=False):
    '''
    Trim data, q and mask arrays to even sizes along each axis.
    This is to avoid possible headache with fft and strain.

    bcdikit's FFT convention (see `bcdikit.utils.general.
    create_diffracted_amplitude`/`create_object`) relies on fftshift/
    ifftshift, which assume an even-sized array.

    q : np.ndarray
        4D array, first axis the 3 q components (qx, qy, qz) and the
        other 3 the same spatial axes as `data` - that first axis is
        left as-is, only the spatial axes get trimmed.
    '''
    s = even_dimension_slices(data.shape)

    data_even = data[s]
    q_even = q[(slice(None),) + s]
    mask_even = mask[s]

    if verbose:
        print('shape changed :\n')
        print(f'data {data.shape} to {data_even.shape}')
        print(f'q {q.shape} to {q_even.shape}')
        print(f'mask {mask.shape} to {mask_even.shape}')

    return data_even, q_even, mask_even