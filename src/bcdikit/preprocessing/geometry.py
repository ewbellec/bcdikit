"""Instrument geometry: turning raw detector data + motor positions into Q
space (`create_Q_array`, `Q_space_transformation`), and a small helper to
force even array dimensions before fftshift/ifftshift-based processing
(`force_even_dimensions`).

This is the one genuinely beamline-specific part of preprocessing (ID01 /
MAXIV / PETRA motor naming, goniometer conventions...) - unlike the rest
of bcdikit, it's written against a `scan` object (from your data opener,
e.g. `bcdikit.data_openers`) rather than plain arrays.
"""

import numpy as np
import xrayutilities as xu

from bcdikit.utils.plot import plot_projections

### -----------------------------------------------------------------------
### Utilities
### -----------------------------------------------------------------------

def _fmt(value, edgeitems=3, precision=8):
    """Just a function to compact print of arrays. Nothing important."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(value)
    return np.array2string(arr, precision=precision, suppress_small=True,
                            threshold=2 * edgeitems, edgeitems=edgeitems)


### -----------------------------------------------------------------------
### Detector pixels + motor positions -> Q space
### -----------------------------------------------------------------------

def compute_Q_array(scan,
                    roi=None,
                    det_calib=None,
                    energy=None,
                    switch_x_y_direct_beam_position=False,
                    chi=0, eta=None, phi=None, delta=None, nu=None,
                    cxi_convention=True,
                    verbose=False):
    """Compute the (qx, qy, qz) Q-space arrays (detector frame, one value
    per measured voxel) for `scan`, via xrayutilities.

    Reads the goniometer motor positions (`eta`, `phi`, `delta`, `nu`) and
    the energy straight from `scan` unless you override them explicitly -
    the motor names looked up differ by beamline/`scan.data_type`
    (ID01/ESRF, MAXIV, PETRA).
    """
    if det_calib is None:
        det_calib = scan.get_det_calib_info()

    # Set default values if not calculated (these are minor corrections)
    det_calib.setdefault('detrot', 0)
    det_calib.setdefault('tiltazimuth', 0)
    det_calib.setdefault('tilt', 0)
    
    if det_calib['distance'] < 0:
        print('detector calibration has a negative distance. Forcing it positive')
        det_calib['distance'] = abs(det_calib['distance'])

    if roi is None:
        roi = [0, None, 0, scan.detector_shape[0], 0, scan.detector_shape[1]]

    if eta is None:
        eta = scan.get_motor_position('gontheta' if scan.data_type == 'MAXIV' else 'eta')
    if phi is None:
        phi = scan.get_motor_position('gonphi' if scan.data_type == 'MAXIV' else 'phi')
        
    if delta is None:
        if 'spec' in str(scan.__class__) or scan.data_type == 'PETRA':
            delta = scan.get_motor_position('del')
        else:
            delta = scan.get_motor_position('delta')
            
    if nu is None:
        if scan.data_type == 'PETRA':
            nu = -scan.get_motor_position('gam')
        elif scan.data_type == 'MAXIV':
            nu = scan.get_motor_position('gamma')
        else:
            nu = scan.get_motor_position('nu')
            
    if energy is None:
        energy = scan.energy

    beam_center_y = det_calib['beam_center_y']
    beam_center_x = det_calib['beam_center_x']
    if switch_x_y_direct_beam_position:
        beam_center_x, beam_center_y = beam_center_y, beam_center_x

    # 2S+2D goniometer (simplified ID01 goniometer: sample eta, phi;
    # detector nu, delta). Coordinate convention: x downstream, z upwards,
    # y to the "outside" (right-handed).
    qconv = xu.experiment.QConversion(['y-', 'z-', 'x+'], ['z-', 'y-'], [1, 0, 0])
    hxrd = xu.HXRD([1, 0, 0], [0, 0, 1], en=energy, qconv=qconv)

    vertical_orientation = 'z+' if scan.detector.lower() == 'citius' else 'z-'
    hxrd.Ang2Q.init_area(
        vertical_orientation, 'y+',
        cch1=beam_center_y - roi[2], cch2=beam_center_x - roi[4],
        Nch1=roi[3] - roi[2], Nch2=roi[5] - roi[4],
        pwidth1=det_calib['y_pixel_size'], pwidth2=det_calib['x_pixel_size'],
        distance=det_calib['distance'],
        detrot=det_calib['detrot'], tiltazimuth=det_calib['tiltazimuth'], tilt=det_calib['tilt'],
    )

    qx, qy, qz = hxrd.Ang2Q.area(eta, phi, chi, nu, delta)

    # apply the roi along the rocking-curve angle axis only - the
    # detector-plane roi was already applied via cch1/cch2/Nch1/Nch2 above
    qx = qx[roi[0]:roi[1]]
    qy = qy[roi[0]:roi[1]]
    qz = qz[roi[0]:roi[1]]
    
    if cxi_convention:
        qx, qy, qz = qy, qz, qx
        
        
    q = np.array([qx,qy,qz])

    if verbose:
        
        # I hate that fix, it was only for 1 beamtime. Maybe remove ?
        if switch_x_y_direct_beam_position:
            print("\x1b[31m beam_center_x and beam_center_y were switched "
                  "when loading the detector calibration parameters ! \x1b[0m\n")
            
        print(f'phi : {_fmt(phi)}')
        print(f'eta : {_fmt(eta)}')
        print(f'chi : {_fmt(chi)}')
        print(f'delta : {_fmt(delta)}')
        print(f'nu : {_fmt(nu)}')
        print(f'\nenergy (eV) : {_fmt(energy)}')
        print(f"detector_distance : {det_calib['distance']} m")
        print(f'beam_center_x : {beam_center_x} ')
        print(f'beam_center_y : {beam_center_y} ')
        print(f"detector pixel size x : {det_calib['x_pixel_size']}m")
        print(f"detector pixel size y : {det_calib['y_pixel_size']}m")
        if cxi_convention:
            print('qx, qy, qz were rotated to the CXI convention (x=outboard, y=vertical/up, z=beam)')
        print('returning a 4D array q of (qx,qy,qz) per pixel in the BCDI data array.')

    return q

### -----------------------------------------------------------------------
### Q-space center of mass 
### -----------------------------------------------------------------------

def calculate_q_center_of_mass(data, q, remove_min=True, plot=False):
    """Q-space position of `data`'s center of mass."""
    if remove_min:
        data = data - np.nanmin(data)  # bug fix: the original mutated the caller's `data` in place

    proba = data / np.nansum(data)
    qcen = np.array([np.nansum(proba * q_i) for q_i in q])

    if plot:
        plot_q_markers(data, q, qcen, title='q center of mass')
    return qcen

def plot_q_markers(data, q, q_position, title=None):
    """Diagnostic plot: `data`'s 3 projections, with the voxel closest to
    `q_position` (a 3-vector in Q space - e.g. from
    `calculate_q_center_of_mass` or `calculate_qmax`) marked.
    """

    index = np.unravel_index(
        np.argmin(np.sum((q - np.asarray(q_position)[:, None, None, None]) ** 2, axis=0)),
        data.shape,
    )

    fig, ax = plot_projections(data,return_fig_ax=True)
    # (x, y) for each panel, matching plot_projections' own row/col
    # convention (panel n projects out axis n, showing the other two)
    for a, (x, y) in zip(ax, [(index[2], index[1]), (index[2], index[0]), (index[1], index[0])]):
        a.scatter(x, y, color='w')

    if title is not None:
        fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    return fig, ax

