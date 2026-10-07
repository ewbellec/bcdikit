"""Plotting helpers for bcdikit.

Two functions cover most day-to-day notebook use:

- `plot_slices(array, ...)`      - slice(s) through the middle of a 2D/3D
  array. Complex input automatically gets module + phase shown side by
  side (2D) or stacked (3D).
- `plot_projections(array, ...)` - sum/max projections of a 3D array along
  each of its 3 axes (for a 2D array, it's just plotted as-is). Complex
  input: pick what gets projected with `component='module'|'intensity'
  |'phase'`.

Everything else here is either a small shared helper (colorbars, ROI
overlays, gradient plots, detector-sum plots) or a handful of more
specialized, more occasionally-used visualizations (histograms, surface
projections, schematic Bragg planes, interactive sliders) ported from the
old `Plot_utilities.py`.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bcdikit.utils.general import apply_roi, slice_middle_array_along_axis, create_diffracted_amplitude
from bcdikit.utils.object import get_module_phase


DEFAULT_DETECTOR_CMAP = 'turbo'


### -----------------------------------------------------------------------
### Shared helpers (colorbars, ROI overlays, gradient/detector-sum plots)
### -----------------------------------------------------------------------

def add_colorbar_subplot(fig, axes, imgs, size='5%', return_cbar=False):
    if not isinstance(imgs, list):
        imgs = [imgs]
        axes = [axes]

    cbar_list = []
    for im, ax in zip(imgs, np.array(axes).flatten()):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=size, pad=0.05)
        cbar_list.append(fig.colorbar(im, cax=cax, orientation='vertical'))
    fig.tight_layout()
    if return_cbar:
        return cbar_list
    else:
        return


def subplots_numerous_images(img_list,
                              fw=4, ncol=4,
                              extent=None,
                              log_scale=False,
                              vmin=None, vmax=None,
                              cmap=None, colorbar=True,
                              title_list=None,
                              suptitle=None,
                              return_fig_ax=False):
    nax = len(img_list)
    nrow = nax // ncol + (nax % ncol != 0)
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * fw, nrow * fw))
    axe = ax.flatten()
    mat = []

    for n in range(len(axe)):
        if n < len(img_list):
            if log_scale:
                mat.append(axe[n].matshow(img_list[n], cmap=cmap, aspect='auto',
                                           extent=extent, norm=LogNorm(vmin=vmin, vmax=vmax)))
            else:
                mat.append(axe[n].matshow(img_list[n], cmap=cmap, aspect='auto',
                                           vmin=vmin, vmax=vmax, extent=extent))
            if title_list is not None:
                axe[n].set_title(title_list[n], fontsize=15 * fw / 4.)
        else:
            fig.delaxes(axe[n])

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20 * fw / 4.)

    if colorbar:
        add_colorbar_subplot(fig, ax[:len(img_list)], mat)

    for one_axe in axe:
        one_axe.locator_params(axis='both', nbins=5)
        one_axe.xaxis.set_ticks_position('bottom')
        one_axe.tick_params(axis='both', which='major', labelsize=12 * fw / 4.)

    fig.tight_layout()

    if return_fig_ax:
        return fig, axe
    else:
        return


def check_roi(img, roi, norm=None):
    """Show `img` with a red rectangle marking `roi` (see
    bcdikit.utils.general.apply_roi for the roi format)."""
    if norm is None:
        norm = LogNorm()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(img, norm=norm, cmap='plasma')

    rect = patches.Rectangle((roi[2], roi[0]), roi[3] - roi[2], roi[1] - roi[0],
                              linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    add_colorbar_subplot(fig, ax, im)
    return


# def cnorm(data, normtype, dmin=None, dmax=None):
#     """Build a matplotlib Normalize/LogNorm for `data`, choosing a sensible
#     default dmin (smallest positive value) for log scales."""
#     if dmax is None:
#         dmax = np.nanmax(data)
#     if dmin is None:
#         ipos = data > 0
#         if ipos.any():
#             dmin = np.nanmin(data[ipos])
#         else:
#             dmin = np.nanmin(data)

#     if normtype == "log":
#         return colors.LogNorm(dmin, dmax)
#     elif normtype == 'linear' or normtype is None or normtype == '':
#         return colors.Normalize(dmin, dmax)
#     else:
#         print('unknown norm, using a linear one')
#         return colors.Normalize(dmin, dmax)


# def plot_gradient_color(x_list, y_list,
#                          colormap=cm.coolwarm,
#                          figsize=None,
#                          label_list=None,
#                          *args, **kwargs):
#     """Plot each (x_list[n], y_list[n]) curve with a color taken from
#     `colormap`, graded across the list (first curve = one end of the
#     colormap, last curve = the other end)."""
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     for n in range(len(y_list)):
#         color = colormap(n / len(y_list))
#         if label_list is not None:
#             ax.plot(x_list[n], y_list[n], '-', color=color, label=label_list[n])
#         else:
#             ax.plot(x_list[n], y_list[n], '-', color=color)
#     if label_list is not None:
#         ax.legend(fontsize=10, ncol=2, loc='upper right')
#     return fig, ax


def plot_detector_sum(data, scan=None,
                       fig=None, ax=None, fw=6,
                       cmap='plasma', scale='log',
                       mask=None, alpha_mask=.3,
                       roi=None,
                       max_proj=False,
                       title=None,
                       vmin=None, vmax=None):
    """Sum (or max-project) `data` over every axis but the last two, and
    plot the result. Pass `scan` (an opened Scan, see
    `bcdikit.data_openers`) to auto-title the plot with the scan number."""
    if max_proj:
        proj = np.max
    else:
        proj = np.sum
    det_proj = proj(data, axis=tuple(range(data.ndim)[:-2]))
    if roi is not None:
        det_proj = apply_roi(det_proj, roi)
        extent = [roi[2], roi[3], roi[1], roi[0]]
    else:
        extent = None

    if scale == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin, vmax = None, None
    else:
        norm = None

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    img = ax.matshow(det_proj, norm=norm, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    add_colorbar_subplot(fig, ax, img)
    if scan is not None and title is None:
        ax.set_title(f"{scan.h5file.split('/')[-1][:-3]} scan {scan.scan_string}", fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=15)
    if mask is not None:
        mask_plot = mask != 0
        ax.imshow(np.dstack([mask_plot, np.zeros(mask_plot.shape),
                              np.zeros(mask_plot.shape), alpha_mask * mask_plot]),
                  aspect='auto', extent=extent)
    return


def plot_detector_sum_with_roi(data, roi, fw=5, **kwargs):
    """Side-by-side: full detector sum (left) with the roi outlined, and the
    cropped roi itself (right)."""
    fig, ax = plt.subplots(1, 2, figsize=(2 * fw, fw))
    plot_detector_sum(data, fig=fig, ax=ax[0], **kwargs)
    plot_detector_sum(data, roi=roi, fig=fig, ax=ax[1], **kwargs)

    rect = patches.Rectangle(
        (roi[2], roi[0]),
        roi[3] - roi[2],
        roi[1] - roi[0],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax[0].add_patch(rect)
    return


def MIR_Colormap():
    """A custom diverging colormap (white -> blue -> green -> yellow -> red
    -> black), kept around in case old notebooks still reference it."""
    cdict = {
        'red':   ((0.0, 1.0, 1.0), (0.11, 0.0, 0.0), (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0), (0.87, 1.0, 1.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 1.0, 1.0), (0.11, 0.0, 0.0), (0.36, 1.0, 1.0),
                  (0.62, 1.0, 1.0), (0.87, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 1.0, 1.0), (0.11, 1.0, 1.0), (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0), (0.87, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }
    return mpl.colors.LinearSegmentedColormap('MIR_colormap', cdict, 256)


### -----------------------------------------------------------------------
### Module/phase extraction from a complex object
### -----------------------------------------------------------------------
### (the actual implementation lives in bcdikit.utils.object_utils -
### get_module_phase - this is just a small dispatch wrapper for
### plot_projections' `component` option)

def _component(array, component, threshold_module=None, support=None,
                unwrap=True, apply_fftshift=False):
    if component == 'phase':
        _, phase = get_module_phase(array, threshold_module=threshold_module, support=support,
                                     unwrap=unwrap, apply_fftshift=apply_fftshift)
        return phase
    module = np.abs(np.fft.fftshift(array)) if apply_fftshift else np.abs(array)
    if component == 'intensity':
        return module ** 2.
    if component == 'module':
        return module
    raise ValueError(f"component must be 'module', 'intensity' or 'phase', got {component!r}")


### -----------------------------------------------------------------------
### Voxel-size -> plot extent helpers (Angstrom in, nanometers out)
### -----------------------------------------------------------------------

def _extent_2d(shape2d, voxel_sizes2d):
    if voxel_sizes2d is None:
        return None
    vs = .1 * np.asarray(voxel_sizes2d)  # Angstrom -> nm
    return [0, shape2d[1] * vs[1], 0, shape2d[0] * vs[0]]


def _extents_3d(shape3d, voxel_sizes3d):
    """One 2D extent per axis you could slice/project along (the extent of
    the *other* two axes), in nm."""
    if voxel_sizes3d is None:
        return [None, None, None]
    vs = np.asarray(voxel_sizes3d)
    extents = []
    for axis in range(3):
        shape2d = tuple(s for i, s in enumerate(shape3d) if i != axis)
        vs2d = np.delete(vs, axis)
        extents.append(_extent_2d(shape2d, vs2d))
    return extents


### -----------------------------------------------------------------------
### Internal single-panel / 3-panel plotting primitives
### -----------------------------------------------------------------------

def _plot_2d(array, fig=None, ax=None, fw=4, cmap=None, voxel_sizes=None,
             vmin=None, vmax=None, norm=None, symmetric_colorscale=False,
             add_colorbar=True, cbar_label=None, title=None):
    if symmetric_colorscale:
        vmax = np.nanmax(np.abs(array))
        vmin = -vmax
        if cmap is None:
            cmap = 'coolwarm'

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    extent = _extent_2d(array.shape, voxel_sizes)
    img = ax.matshow(array, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, extent=extent)

    if add_colorbar:
        cbar = add_colorbar_subplot(fig, ax, img, return_cbar=True)[0]
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=270, fontsize=15 * fw / 4., labelpad=15)

    if voxel_sizes is not None:
        ax.set_xlabel('nm', fontsize=15 * fw / 4.)
        ax.set_ylabel('nm', fontsize=15 * fw / 4.)
        ax.xaxis.set_ticks_position('bottom')

    if title is not None:
        ax.set_title(title, fontsize=20 * fw / 4.)

    fig.tight_layout()
    return fig, ax


def _plot_3d_slices(array, index=None, fig=None, ax=None, fw=3, cmap='gray_r',
                     voxel_sizes=None, vmin=None, vmax=None, norm=None,
                     symmetric_colorscale=False, add_colorbar=True, cbar_label=None,
                     suptitle=None, xlabels=None, ylabels=None):
    if symmetric_colorscale:
        cmap = 'coolwarm'

    shape = array.shape
    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(3 * fw, fw))

    extents = _extents_3d(shape, voxel_sizes)

    imgs = []
    for axis in range(3):
        s = [slice(None)] * 3
        s[axis] = shape[axis] // 2 if index is None else min(index, shape[axis] - 1)
        img_data = array[tuple(s)]

        vmin_n, vmax_n = vmin, vmax
        if symmetric_colorscale:
            vmax_n = np.nanmax(np.abs(img_data))
            vmin_n = -vmax_n

        imgs.append(ax[axis].matshow(img_data, cmap=cmap, vmin=vmin_n, vmax=vmax_n,
                                      norm=norm, extent=extents[axis]))

    if add_colorbar:
        cbars = add_colorbar_subplot(fig, ax, imgs, return_cbar=True)
        if cbar_label is not None:
            for cbar in cbars:
                cbar.ax.set_ylabel(cbar_label, rotation=270, fontsize=15 * fw / 4., labelpad=10 * fw)

    if voxel_sizes is not None:
        xlabels = xlabels if xlabels is not None else ['nm', 'nm', 'nm']
        ylabels = ylabels if ylabels is not None else ['nm', 'nm', 'nm']
        for axis in range(3):
            ax[axis].set_xlabel(xlabels[axis], fontsize=15 * fw / 4.)
            ax[axis].set_ylabel(ylabels[axis], fontsize=15 * fw / 4.)
            ax[axis].xaxis.set_ticks_position('bottom')

    for axe in ax:
        axe.locator_params(nbins=4)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20 * fw / 4.)

    fig.tight_layout()
    return fig, ax


### -----------------------------------------------------------------------
### Main entry point 1/2: plot_slices
### -----------------------------------------------------------------------

def plot_slices(array,
                 index=None,
                 voxel_sizes=None,
                 threshold_module=None, support=None, unwrap=True, apply_fftshift=False,
                 symmetric_colorscale=False,
                 cmap=None, vmin=None, vmax=None,
                 fw=3, fig=None, ax=None, title=None,
                 return_fig_ax=False):
    """Plot slice(s) through the middle of `array` (or through `index` along
    each axis, if given).

    Dispatches on `array.ndim` and `np.iscomplexobj(array)`:
      - 2D real array    -> a single image
      - 2D complex array -> module + phase, side by side
      - 3D real array    -> 3 orthogonal middle slices
      - 3D complex array -> module + phase, 3 orthogonal slices each (2x3)

    `threshold_module`, `support`, `unwrap`, `apply_fftshift` only affect
    the phase of a complex array (see `bcdikit.utils.object_utils.get_module_phase`). `voxel_sizes` is in
    Angstrom, axis order matching `array`; labels/extents are shown in nm.
    `symmetric_colorscale` only applies to a real array.
    """
    is_complex = np.iscomplexobj(array)

    if array.ndim == 2:
        if is_complex:
            module, phase = get_module_phase(array, threshold_module=threshold_module,
                                           support=support, unwrap=unwrap,
                                           apply_fftshift=apply_fftshift)
            if fig is None:
                fig, ax = plt.subplots(1, 2, figsize=(2 * fw, fw))
            _plot_2d(module, fig=fig, ax=ax[0], fw=fw, cmap=cmap or 'gray_r',
                     voxel_sizes=voxel_sizes, title='module')
            _plot_2d(phase, fig=fig, ax=ax[1], fw=fw, cmap=cmap or 'hsv',
                     voxel_sizes=voxel_sizes, vmin=vmin, vmax=vmax, title='phase')
        else:
            fig, ax = _plot_2d(array, fig=fig, ax=ax, fw=fw, cmap=cmap,
                                voxel_sizes=voxel_sizes, vmin=vmin, vmax=vmax,
                                symmetric_colorscale=symmetric_colorscale)

    elif array.ndim == 3:
        if is_complex:
            module, phase = get_module_phase(array, threshold_module=threshold_module,
                                           support=support, unwrap=unwrap,
                                           apply_fftshift=apply_fftshift)
            if fig is None:
                fig, ax = plt.subplots(2, 3, figsize=(3 * fw, 2 * fw))
            _plot_3d_slices(module, index=index, fig=fig, ax=ax[0], fw=fw,
                             cmap=cmap or 'gray_r', voxel_sizes=voxel_sizes)
            _plot_3d_slices(phase, index=index, fig=fig, ax=ax[1], fw=fw,
                             cmap=cmap or 'hsv', voxel_sizes=voxel_sizes,
                             vmin=vmin, vmax=vmax)
            ax[0, 0].set_ylabel('module', fontsize=20)
            ax[1, 0].set_ylabel('phase', fontsize=20)
        else:
            fig, ax = _plot_3d_slices(array, index=index, fig=fig, ax=ax, fw=fw,
                                       cmap=cmap, voxel_sizes=voxel_sizes,
                                       vmin=vmin, vmax=vmax,
                                       symmetric_colorscale=symmetric_colorscale)
    else:
        raise ValueError(f"plot_slices only supports 2D or 3D arrays, got {array.ndim}D")

    if title is not None:
        fig.suptitle(title, fontsize=20)
    fig.tight_layout()

    if return_fig_ax:
        return fig, ax
    return


### -----------------------------------------------------------------------
### Main entry point 2/2: plot_projections
### -----------------------------------------------------------------------

def plot_projections(array,
                      component=None,
                      max_projection=False,
                      log_scale=True,
                      mask=None, alpha_mask=.3,
                      threshold_module=None, support=None, unwrap=True, apply_fftshift=False,
                      vmin=None, vmax=None, cmap=None,
                      fw=4, fig=None, ax=None, title=None,
                      colorbar=True, axes_labels=False,
                      return_fig_ax=False):
    """Plot sum (or max, if `max_projection=True`) projections of `array`
    along each of its 3 axes.

    - 3D real array    -> 3 projections, one per axis
    - 3D complex array -> pick what gets projected with `component`
      ('module' (default), 'intensity' or 'phase')
    - 2D array         -> nothing to project, just plotted as-is (complex
      input uses `component` the same way, default 'module')

    `log_scale=True` (default) uses a LogNorm on the projections; `mask`,
    if given, is overlaid in translucent red.

    Note: the old `plot_3D_projections`'s `log_threshold`/`xu.maplog`
    option is dropped here - it depended on `xrayutilities`, which isn't a
    bcdikit dependency, and `log_scale` already covers normal log-scale
    plotting.
    """
    is_complex = np.iscomplexobj(array)

    if array.ndim == 2:
        if is_complex:
            data = _component(array, component or 'module',
                                  threshold_module=threshold_module, support=support,
                                  unwrap=unwrap, apply_fftshift=apply_fftshift)
        else:
            data = array
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
        fig, ax = _plot_2d(data, fig=fig, ax=ax, fw=fw, cmap=cmap, norm=norm,
                            vmin=None if log_scale else vmin,
                            vmax=None if log_scale else vmax,
                            title=title)
        if return_fig_ax:
            return fig, ax
        return

    if array.ndim != 3:
        raise ValueError(f"plot_projections only supports 2D or 3D arrays, got {array.ndim}D")

    if is_complex:
        data = _component(array, component or 'module',
                              threshold_module=threshold_module, support=support,
                              unwrap=unwrap, apply_fftshift=apply_fftshift)
    else:
        data = array

    if cmap is None:
        cmap = DEFAULT_DETECTOR_CMAP

    if fig is None:
        if colorbar:
            fig, ax = plt.subplots(1, 3, figsize=(3.5 * fw, fw))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(3 * fw, fw))

    projector = np.nanmax if max_projection else np.nansum
    imgs = []
    for axis in range(3):
        img_data = projector(data, axis=axis)
        if log_scale:
            imgs.append(ax[axis].matshow(img_data, cmap=cmap, aspect='auto',
                                          norm=LogNorm(vmin=vmin, vmax=vmax)))
        else:
            imgs.append(ax[axis].matshow(img_data, cmap=cmap, aspect='auto',
                                          vmin=vmin, vmax=vmax))

        if mask is not None:
            mask_proj = np.nanmean(mask, axis=axis)
            mask_proj = (mask_proj != 0).astype(float)
            ax[axis].imshow(np.dstack([mask_proj, np.zeros_like(mask_proj),
                                        np.zeros_like(mask_proj), alpha_mask * mask_proj]),
                             aspect='auto')

    if axes_labels:
        labels = [('detector horizontal', 'detector vertical'),
                  ('detector horizontal', 'rocking curve'),
                  ('detector vertical', 'rocking curve')]
        for axis in range(3):
            ax[axis].set_xlabel(labels[axis][0], fontsize=15 * fw / 4)
            ax[axis].set_ylabel(labels[axis][1], fontsize=15 * fw / 4)

    if colorbar:
        add_colorbar_subplot(fig, ax, imgs)

    if title is not None:
        fig.suptitle(title, fontsize=15 * fw / 4)

    fig.tight_layout()

    if return_fig_ax:
        return fig, ax
    return


### -----------------------------------------------------------------------
### Object module histogram
### -----------------------------------------------------------------------

def plot_module_histogram(obj, bins=None, fig=None, ax=None):
    """Histogram of `obj`'s module values (pixels below 1% of the max
    module are excluded - they're background, not part of the object)."""
    module = np.abs(obj)
    module = np.where(module < .01 * np.nanmax(module), np.nan, module)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if bins is None:
        bins = 30 if obj.ndim == 2 else 50

    ax.hist(module[~np.isnan(module)].flatten(), bins=bins)
    ax.set_xlabel('object module value', fontsize=15)
    ax.set_ylabel('number of pixels', fontsize=15)
    return


def plot_strain_histo(strain, bins=50, fig=None, ax=None, title='strain histogram'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.hist(1e2 * strain.flatten(), bins=bins)
    ax.set_xlabel('strain (%)', fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=20)
    return


### -----------------------------------------------------------------------
### Compare experimental data to a reconstruction
### -----------------------------------------------------------------------

def compare_reconstruction_to_real_data(data, obj, title=None):
    """Compare measured detector intensity `data` to the intensity computed
    from a reconstructed object `obj` (|FFT(obj)|^2, using the same FFT
    convention as `bcdikit.utils.general.create_diffracted_amplitude`)."""
    I_recon = np.abs(create_diffracted_amplitude(obj)) ** 2.

    if data.ndim == 3:
        plot_projections(data, title='experimental data' if title is None else title)
        plot_projections(I_recon, title='reconstructed data')
    elif data.ndim == 2:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].matshow(np.log(data), cmap='gray')
        ax[0].set_title('experimental', fontsize=20)
        ax[1].matshow(np.log(I_recon), cmap='gray')
        ax[1].set_title('reconstructed data', fontsize=20)
        fig.tight_layout()
    else:
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    return


### -----------------------------------------------------------------------
### Support projection (for ROI selection)
### -----------------------------------------------------------------------

def plot_support_projection(obj, threshold_module=.3, fw=3, fig=None, ax=None):
    """Max-projection of a 3D object's support (module > threshold_module *
    max module) along each axis - handy to pick a crop ROI.

    bug fix: the original defined `xlabels`/`ylabels` only for ndim == 3,
    so calling it on anything else raised a confusing NameError. A 2D (or
    >3D) input can't actually work here regardless - projecting along one
    axis of an ndim-D array gives an (ndim-1)-D array, and this function
    displays each projection as a 2D image (matshow), which is only
    possible when ndim == 3. So instead of silently doing something wrong,
    it now raises a clear, explicit error for any other ndim.
    """
    if obj.ndim != 3:
        raise ValueError(f"plot_support_projection only supports 3D arrays, got {obj.ndim}D "
                          "(each projection is itself a 2D image, which only works out for a 3D input)")

    module = np.abs(obj)
    support = module > threshold_module * np.nanmax(module)

    ndim = obj.ndim
    xlabels = ['axis 2', 'axis 2', 'axis 1']
    ylabels = ['axis 1', 'axis 0', 'axis 0']

    if fig is None:
        fig, ax = plt.subplots(1, ndim, figsize=(fw * ndim, fw))

    for axis in range(ndim):
        projection = np.max(support, axis=axis)
        ax[axis].matshow(projection, cmap='gray_r', aspect='auto')
        ax[axis].set_xlabel(xlabels[axis], fontsize=15 * fw / 3.)
        ax[axis].set_ylabel(ylabels[axis], fontsize=15 * fw / 3.)
        ax[axis].set_title(f'projection along axis {axis}', fontsize=12 * fw / 3.)
    fig.tight_layout()
    return


### -----------------------------------------------------------------------
### Surface projections
### -----------------------------------------------------------------------

def one_surface_projection(strain, axis, inverse):
    strain_used = np.copy(strain)
    if inverse:
        strain_used = np.flip(strain_used, axis=axis)

    # strain is assumed nan outside the support, which lets us derive the
    # support directly from it - pass your own support upstream if that's
    # not the case for your data.
    support = 1 - np.isnan(strain_used)
    support_surface = np.cumsum(support, axis=axis)
    support_surface[support_surface > 1] = 0

    surface_strain = np.copy(strain_used)
    surface_strain[support_surface == 0] = np.nan
    surface_strain = np.nanmean(surface_strain, axis=axis)
    return surface_strain


def plot_surface_projections(strain, voxel_sizes=None, fw=3, title=None, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(1e2 * strain)
    if vmax is None:
        vmax = np.nanmax(1e2 * strain)

    fig, ax = plt.subplots(2, 3, figsize=(fw * 3.3, fw * 2.2))
    extents = _extents_3d(strain.shape, voxel_sizes)

    imgs = []
    for n1, axis in enumerate(range(3)):
        for n0, inverse in enumerate([False, True]):
            surface_strain = one_surface_projection(strain, axis, inverse)
            imgs.append(ax[n0, n1].matshow(1e2 * surface_strain, cmap='coolwarm', extent=extents[n1],
                                            vmin=vmin, vmax=vmax))

    cax = fig.add_axes([1, .1, .02, .8])
    cbar = fig.colorbar(imgs[-1], cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=20 * fw / 4.)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label('strain (%)', rotation=270, fontsize=30 * fw / 4., labelpad=15)

    for axe in ax.flatten():
        axe.locator_params(nbins=4)

    xlabel = ['Z (nm)', 'Z (nm)', 'Y (nm)']
    ylabel = ['Y (nm)', 'X (nm)', 'X (nm)']
    for n in range(3):
        for ii in range(2):
            ax[ii, n].set_xlabel(xlabel[n], fontsize=15 * fw / 4.)
            ax[ii, n].xaxis.set_ticks_position('bottom')
            ax[ii, n].set_ylabel(ylabel[n], fontsize=15 * fw / 4.)

    title_list = [['along +X', 'along +Y', 'along +Z'], ['along -X', 'along -Y', 'along -Z']]
    for n in range(3):
        for ii in range(2):
            ax[ii, n].set_title(title_list[ii][n], fontsize=17 * fw / 4.)

    if title is not None:
        fig.suptitle(title + '   surface projection', fontsize=fw * 22 / 4.)

    fig.tight_layout()
    return


### -----------------------------------------------------------------------
### Schematic Bragg-planes figures
### -----------------------------------------------------------------------

def schematic_Bragg_planes_figure(displacement, qcen, voxel_sizes,
                                   strain=None,
                                   axis=0, slice_index=None,
                                   visual_factor=50,
                                   dislo_threshold=None,
                                   close_roi=False,
                                   fig=None, ax=None, fw=6, title=None):
    """Schematic 2D cross-section of the lattice planes, displaced by
    `displacement` (and optionally colored by `strain`). Assumes `qcen` is
    close to the array's last axis (within ~10 degrees)."""
    angle_bragg_last_axis = np.rad2deg(np.arccos(np.dot(qcen, [0, 0, 1]) / np.linalg.norm(qcen)))

    if angle_bragg_last_axis > 10.:
        raise ValueError('qcen is really not along the last axis! This function is not ready for that.')
    elif angle_bragg_last_axis > 1.:
        print('Careful, this is schematic since the Bragg wavevector is not perfectly along the vertical')
        print(f'The angle between the Bragg direction and the vertical is {round(angle_bragg_last_axis, 2)} degrees')

    if axis == 2:
        raise ValueError('axis should be 0 or 1!')

    if slice_index is None:
        s = slice_middle_array_along_axis(displacement, axis=axis)
    else:
        s = [slice(None)] * displacement.ndim
        s[axis] = slice_index
        s = tuple(s)

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    if strain is not None:
        strain_color_slice = np.copy(strain[s])
        strain_color_slice = strain_color_slice - np.nanmin(strain_color_slice)
        strain_color_slice = strain_color_slice / np.nanmax(strain_color_slice)

    displacement_slice = displacement[s]
    voxel_sizes_slice = np.delete(voxel_sizes, axis)

    pos = np.indices(displacement_slice.shape)
    pos[0] = pos[0] * voxel_sizes_slice[0] * .1
    pos[1] = pos[1] * voxel_sizes_slice[1] * .1

    pos = pos + .1 * displacement_slice[None] * np.array([0, 1])[:, None, None] * visual_factor

    for n in range(displacement_slice.shape[1]):
        if dislo_threshold is not None:
            grad_amp = np.abs(np.gradient(pos[1, :, n]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                indices = grad_amp > np.nanmedian(grad_amp) * 20
            pos[:, indices, n] = np.nan
        ax.plot(pos[1, :, n], pos[0, :, n], 'k-', linewidth=1, markersize=1, alpha=1)

        if strain is not None:
            strain_line = strain_color_slice[:, n]
            color = cm.coolwarm(strain_line)
            ax.scatter(pos[1, :, n], pos[0, :, n], s=20 * fw / 10., color=color, alpha=1)

    ax.set_xlabel('nm', fontsize=15 * fw / 4.)

    if strain is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        norm = mpl.colors.Normalize(vmin=np.nanmin(1e2 * strain[s]), vmax=np.nanmax(1e2 * strain[s]))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'), cax=cax, orientation='vertical')
        cbar.set_label('strain (%)', size=20 * fw / 6)

    if not close_roi:
        ax.set_xlim(0, displacement_slice.shape[1] * voxel_sizes_slice[1] * .1)
        ax.set_ylim(0, displacement_slice.shape[0] * voxel_sizes_slice[0] * .1)

    ax.invert_yaxis()

    if title is None:
        title = 'Schematic Bragg\nplanes distortion'
    ax.set_title(title, fontsize=12 * fw / 4.)
    ax.set_aspect('equal', 'box')
    return


def final_figure_schematic_Bragg_planes(displacement, qcen, voxel_sizes,
                                         strain=None,
                                         visual_factor=200, dislo_threshold=None,
                                         close_roi=True, fw=6):
    fig, ax = plt.subplots(1, 2, figsize=(2 * fw, fw))

    for n, axis in enumerate([0, 1]):
        schematic_Bragg_planes_figure(displacement, qcen, voxel_sizes,
                                       strain=strain,
                                       axis=axis,
                                       visual_factor=visual_factor,
                                       dislo_threshold=dislo_threshold,
                                       # bug fix: the original always passed close_roi=True here,
                                       # ignoring this function's own close_roi argument
                                       close_roi=close_roi,
                                       fig=fig, ax=ax[n], title=f'slice along axis {axis}')

    fig.suptitle('Schematic Bragg planes', fontsize=20 * fw / 6.)
    fig.tight_layout()
    return


def Bragg_planes_schematic_figure_2D(displacement, u_bragg, voxel_sizes,
                                      visual_factor=50,
                                      fig=None, ax=None, title='Schematic planes',
                                      fw=4):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fw, fw))

    pos = np.indices(displacement.shape)
    pos[0] = pos[0] * voxel_sizes[0] * .1
    pos[1] = pos[1] * voxel_sizes[1] * .1

    pos = pos + displacement[None] * u_bragg[:, None, None] * visual_factor

    for n in range(displacement.shape[1]):
        ax.plot(pos[1, :, n], pos[0, :, n], 'k.-')
    ax.set_xlabel('nm', fontsize=15 * fw / 4.)

    if title is not None:
        ax.set_title(title, fontsize=15 * fw / 4.)

    ax.set_aspect('equal', 'box')
    return


### -----------------------------------------------------------------------
### Interactive sliders (Jupyter only, needs ipywidgets)
### -----------------------------------------------------------------------

def interactive_3d_object(obj, threshold_module=None, axis=0):
    """Jupyter-only: interactively scroll through slices of an object's
    module and phase along `axis`. Needs ipywidgets; tries to switch to
    `%matplotlib widget` automatically."""
    from ipywidgets import interact
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic("matplotlib widget")
    except ImportError:
        pass

    module, phase = get_module_phase(obj, threshold_module=threshold_module, unwrap=True)
    shape = module.shape

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    im0 = ax[0].matshow(module.take(indices=shape[axis] // 2, axis=axis), cmap='gray_r')
    im1 = ax[1].matshow(phase.take(indices=shape[axis] // 2, axis=axis), cmap='hsv')

    @interact(w=(0, shape[axis] - 1))
    def update(w=0):
        im0.set_data(module.take(indices=w, axis=axis))
        im0.set_clim(np.nanmin(module), np.nanmax(module))

        im1.set_data(phase.take(indices=w, axis=axis))
        im1.set_clim(np.nanmin(phase), np.nanmax(phase))

        fig.canvas.draw_idle()
        return
    return


def interactive_3d_array(array, axis=0, voxel_sizes=None, cmap='coolwarm',
                          vmin=None, vmax=None, symmetric_colorscale=False):
    """Jupyter-only: interactively scroll through slices of a real 3D array
    along `axis`. Needs ipywidgets."""
    from ipywidgets import interact
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic("matplotlib widget")
    except ImportError:
        pass

    if symmetric_colorscale:
        cmap = 'bwr'
        vmax = np.nanmax(np.abs(array))
        vmin = -vmax

    extents = _extents_3d(array.shape, voxel_sizes)
    shape = array.shape

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im0 = ax.matshow(array.take(indices=shape[axis] // 2, axis=axis), cmap=cmap,
                      vmin=vmin, vmax=vmax, extent=extents[axis])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    @interact(w=(0, shape[axis] - 1))
    def update(w=0):
        im0.set_data(array.take(indices=w, axis=axis))
        fig.canvas.draw_idle()
        return
    return