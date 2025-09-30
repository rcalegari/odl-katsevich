# coding: utf-8
# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import ResizingOperator
from odl.trafos import FourierTransform, PYFFTW_AVAILABLE
from odl.operator.operator import Operator
from scipy.interpolate import interp1d
from tqdm import tqdm
import odl

__all__ = ('fbp_op', 'fbp_filter_op', 'katsevich_filter_op', 'tam_danielson_window', 'td_window_curved',
           'parker_weighting', 'ff7_td_weighting')


def _axis_in_detector(geometry):
    """A vector in the detector plane that points along the rotation axis."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    c = np.array([np.vdot(axis, du), np.vdot(axis, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _rotation_direction_in_detector(geometry):
    """A vector in the detector plane that points in the rotation direction."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    det_normal = np.cross(dv, du)
    rot_dir = np.cross(axis, det_normal)
    c = np.array([np.vdot(rot_dir, du), np.vdot(rot_dir, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _fbp_filter(norm_freq, filter_type, frequency_scaling):
    """Create a smoothing filter for FBP.

    Parameters
    ----------
    norm_freq : `array-like`
        Frequencies normalized to lie in the interval [0, 1].
    filter_type : {'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
                   'Katsevich', callable}
        The type of filter to be used.
        If a string is given, use one of the standard filters with that name.
        A callable should take an array of values in [0, 1] and return the
        filter for these frequencies.
    frequency_scaling : float
        Scaling of the frequencies for the filter. All frequencies are scaled
        by this number, any relative frequency above ``frequency_scaling`` is
        set to 0.

    Returns
    -------
    smoothing_filter : `numpy.ndarray`

    Examples
    --------
    Create an FBP filter

    >>> norm_freq = np.linspace(0, 1, 10)
    >>> filt = _fbp_filter(norm_freq,
    ...                    filter_type='Hann',
    ...                    frequency_scaling=0.8)
    """
    filter_type, filter_type_in = str(filter_type).lower(), filter_type
    if callable(filter_type):
        filt = filter_type(norm_freq)
    elif filter_type == 'ram-lak':
        filt = np.copy(norm_freq)
    elif filter_type == 'shepp-logan':
        filt = norm_freq * np.sinc(norm_freq / (2 * frequency_scaling))
    elif filter_type == 'cosine':
        filt = norm_freq * np.cos(norm_freq * np.pi / (2 * frequency_scaling))
    elif filter_type == 'hamming':
        filt = norm_freq * (
            0.54 + 0.46 * np.cos(norm_freq * np.pi / (frequency_scaling)))
    elif filter_type == 'hann':
        filt = norm_freq * (
            np.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
    else:
        raise ValueError('unknown `filter_type` ({})'
                         ''.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt

def tam_danielson_window(ray_trafo, smoothing_width=0.05, n_pi=1):
    """Create Tam-Danielson window from a `RayTransform`.

    The Tam-Danielson window is an indicator function on the minimal set of
    data needed to reconstruct a volume from given data. It is useful in
    analytic reconstruction methods such as FBP to give a more accurate
    reconstruction.

    See [TAM1998] for more informationon the window.
    See [PKGT2000] for information on the ``n_pi`` parameter.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform for which to compute the window.
    smoothing_width : positive float, optional
        Width of the smoothing applied to the window's edges given as a
        fraction of the width of the full window.
    n_pi : odd int, optional
        Total number of half rotations to include in the window. Values larger
        than 1 should be used if the pitch is much smaller than the detector
        height.

    Returns
    -------
    tam_danielson_window : ``ray_trafo.range`` element

    See Also
    --------
    fbp_op : Filtered back-projection operator from `RayTransform`
    tam_danielson_window : Weighting for short scan data
    odl.tomo.geometry.conebeam.ConeBeamGeometry :
        Primary use case for this window function.

    References
    ----------
    [TSS1998] Tam, K C, Samarasekera, S and Sauer, F.
    *Exact cone beam CT with a spiral scan*.
    Physics in Medicine & Biology 4 (1998), p 1015.
    https://dx.doi.org/10.1088/0031-9155/43/4/028

    [PKGT2000] Proksa R, Köhler T, Grass M, Timmer J.
    *The n-PI-method for helical cone-beam CT*
    IEEE Trans Med Imaging. 2000 Sep;19(9):848-63.
    https://www.ncbi.nlm.nih.gov/pubmed/11127600
    """
    # Extract parameters
    src_radius = ray_trafo.geometry.src_radius
    det_radius = ray_trafo.geometry.det_radius
    pitch = ray_trafo.geometry.pitch

    if pitch == 0:
        raise ValueError('Tam-Danielson window is only defined with '
                         '`pitch!=0`')

    smoothing_width = float(smoothing_width)
    if smoothing_width < 0:
        raise ValueError('`smoothing_width` should be a positive float')

    if n_pi % 2 != 1:
        raise ValueError('`n_pi` must be odd, got {}'.format(n_pi))

    # Find projection of axis on detector
    axis_proj = _axis_in_detector(ray_trafo.geometry)
    rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

    # Find distance from projection of rotation axis for each pixel
    dx = (rot_dir[0] * ray_trafo.range.meshgrid[1]
          + rot_dir[1] * ray_trafo.range.meshgrid[2])

    dx_axis = dx * src_radius / (src_radius + det_radius)

    def Vn(u):
        return (pitch / (2 * np.pi)
                * (1 + (u / src_radius) ** 2)
                * (n_pi * np.pi / 2.0 - np.arctan(u / src_radius)))
    def Vn_curved(u):
        return (pitch * (src_radius + det_radius) / (2 * np.pi) 
                * (n_pi * np.pi/2 + u ) /  np.cos(u))

    # lower_proj_axis = -Vn(dx_axis)
    # upper_proj_axis = Vn(-dx_axis)
    lower_proj_axis = -Vn_curved(dx)
    upper_proj_axis = Vn_curved(-dx)

    lower_proj = lower_proj_axis * (src_radius + det_radius) / src_radius
    upper_proj = upper_proj_axis * (src_radius + det_radius) / src_radius

    # Compute a smoothed width
    interval = (upper_proj - lower_proj)
    width = interval * smoothing_width / np.sqrt(2)

    # Create window function
    def window_fcn(x):
        # Lazy import to improve `import odl` time
        import scipy.special

        x_along_axis = axis_proj[0] * x[1] + axis_proj[1] * x[2]
        if smoothing_width != 0:
            lower_wndw = 0.5 * (
                1 + scipy.special.erf((x_along_axis - lower_proj) / width))
            upper_wndw = 0.5 * (
                1 + scipy.special.erf((upper_proj - x_along_axis) / width))
        else:
            lower_wndw = (x_along_axis >= lower_proj)
            upper_wndw = (x_along_axis <= upper_proj)

        return lower_wndw * upper_wndw

    return ray_trafo.range.element(window_fcn) / n_pi

def ff7_td_weighting(ray_trafo, g5):
    # Apply Tam_Danielson window to the data. Compute the 
    # indicator function maskTD and apply it to the data.
    # see formula (57) Noo et al. 2003.
    geometry = ray_trafo.geometry
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    Rs = geometry.src_radius
    w_vals = geometry.det_partition.coord_vectors[0]
    u_vals = geometry.det_partition.coord_vectors[1]
    dw = w_vals[1] - w_vals[0]
    N_w = len(w_vals)
    N_u = len(u_vals)
    N_s = len(geometry.motion_partition.coord_vectors[0])
    a=float(0.025)

    # formula (78) Noo et al. 2003
    w_bottom = - P / (2 * np.pi * Rs * D) * (u_vals**2 + D**2) * (np.pi/2 + np.arctan(u_vals / D))
    w_top    =   P / (2 * np.pi * Rs * D) * (u_vals**2 + D**2) * (np.pi/2 - np.arctan(u_vals / D))
    
    w_bottom = np.reshape(w_bottom, (1, -1))
    w_top    = np.reshape(w_top, (1, -1))

    if P == 0:
        raise ValueError('Tam-Danielson window is only defined with ' '`pitch != 0`')
    if a < 0:
        raise ValueError('`smoothing_width` should be a positive float')

    W, U = np.meshgrid(w_vals, u_vals, indexing='ij')  # (N_w, N_u)
    mask = np.zeros(shape=(N_w, N_u), dtype=np.float32)

    # w_bottom_low = (w_bottom - a * dw).reshape(-1, 1) 
    # w_bottom_high = (w_bottom + a * dw).reshape(-1, 1)
    # w_top_low = (w_top - a * dw).reshape(-1, 1)
    # w_top_high = (w_top + a * dw).reshape(-1, 1)
    w_bottom_low = w_bottom - a * dw
    w_bottom_high = w_bottom + a * dw
    w_top_low = w_top - a * dw
    w_top_high = w_top + a * dw

    region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
    region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
    region4 = ( w_top_low < W ) & ( W <= w_top_high )

    # region1 and region5 stay at 0
    mask[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
    mask[region3] = 1
    mask[region4] = (w_top_high - W)[region4]/ (2 * a * dw)
    
    maskTD = np.broadcast_to(mask, (N_s, N_w, N_u))
    # plot the mask
    import matplotlib.pyplot as plt
    plt.imshow(maskTD[0, :, :], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title('Tam-Danielson window')
    plt.xlabel('Detector column (u)')
    plt.ylabel('Detector row (w)')
    plt.savefig('/home/rosaca/code/odl/examples/kats/phantom_simple/flat/tam_danielson_window_ff7.svg')
    plt.close()
    return g5 * maskTD

def td_window_curved(ray_trafo, g6, smoothing_width=0.025, print_tqdm=False, force_const=False):
    # Apply Tam_Danielson window to the data. Compute the 
    # indicator function maskTD and apply it to the data.
    # For non-constant pitch we have to recompute it for every angle k.
    
    # extract geometry parameters
    geometry = ray_trafo.geometry
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    R = geometry.src_radius
    alpha_vals = geometry.det_partition.coord_vectors[0]
    w_vals = geometry.det_partition.coord_vectors[1]
    N_alpha = geometry.det_partition.shape[0]
    N_w = geometry.det_partition.shape[1]
    N_s = geometry.motion_partition.shape[0]
    dw = w_vals[1] - w_vals[0]
    fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
    fov_radius = fov_dia / 2
    const_pitch = True if force_const else np.all(geometry.src_shift_func(geometry.angles[:]) == 0)

    a=float(0.025)
    if a <= 0:
        raise ValueError("`smoothing_width` must be positive")
    # 2-D grids that are the same for every source angle k
    W, A = np.meshgrid(w_vals, alpha_vals, indexing="xy")      
    maskTD = np.zeros_like(g6, dtype=np.float32)

    def nc_window_slice(k):
        s0 = geometry.angles[k]  
        s_bot  = s0 + ds_min
        s_top  = s0 + ds_plus
        p0     = geometry.src_position(s0)[2]  
        a_bot = (np.sin(ds_min) / (1 - np.cos(ds_min))) 
        a_top = (np.sin(ds_plus) / (1 - np.cos(ds_plus))) 
        try:           
            p_bot  = geometry.src_position(s_bot)[:, 2]           # (N_sampling,)
            w_bot  = D / R * (p_bot - p0) / (1 - np.cos(ds_min)) 
            exc_bot = False
        except Exception as e:
            # print(f"Error in computing source positions: bottom [{s_bot[0]}, {s_bot[-1]}] not in range of [{self.geometry.angles[0]}, {self.geometry.angles[-1]}]") 
            w_bot = w_vals[0] * np.ones_like(ds_min)  # Fallback to min w value         
            exc_bot = True
        try:
            p_top  = geometry.src_position(s_top)[:, 2]
            w_top  = D / R * (p_top - p0) / (1 - np.cos(ds_plus))
            exc_top = False
        except Exception as e:
            # print(f"Error in computing source positions: top [{s_top[0]}, {s_top[-1]}] not in range of [{geometry.angles[0]}, {geometry.angles[-1]}]") 
            w_top = w_vals[-1] * np.ones_like(ds_plus)
            exc_top = True
        if exc_bot:
            w_bot_interp = - P * D / (2 * np.pi * R) * (np.pi/2 + alpha_vals) / np.cos(alpha_vals)
        else:
            interp_bot = interp1d(a_bot, w_bot, bounds_error=False, fill_value='extrapolate')
            w_bot_interp = interp_bot(alpha_vals)
        if exc_top:
            w_top_interp =   P * D / (2 * np.pi * R) * (np.pi/2 - alpha_vals) / np.cos(alpha_vals)
        else:
            interp_top = interp1d(a_top, w_top, bounds_error=False, fill_value='extrapolate')
            w_top_interp = interp_top(alpha_vals)
        return w_bot_interp, w_top_interp

    if not const_pitch:
        delta_s_edge = np.arccos(fov_radius / R)  # R > fov!!!
        ds_plus = np.linspace(delta_s_edge, 2 * np.pi - delta_s_edge, 1000)
        ds_min  = np.linspace(delta_s_edge - 2 * np.pi, - delta_s_edge, 1000)

        if print_tqdm:
            print('non-constant pitch case')
            range_obj = tqdm(range(N_s), desc='Computing TD window')
        else:
            range_obj = range(N_s)
        for k in range_obj:
            w_bottom, w_top = nc_window_slice(k)
            # Expand to 2D so we can compare with W
            w_bottom = w_bottom[:, None]    
            w_top    = w_top[:, None]
            # Linear-ramp limits
            w_bottom_low  = w_bottom - a * dw
            w_bottom_high = w_bottom + a * dw
            w_top_low     = w_top    - a * dw
            w_top_high    = w_top    + a * dw
            # Region tests (same five regions as the constant-pitch version)
            region2 = (w_bottom_low  <= W) & (W <  w_bottom_high)
            region3 = (w_bottom_high <= W) & (W <= w_top_low)
            region4 = (w_top_low     <  W) & (W <= w_top_high)

            m = np.zeros_like(W, dtype=np.float32)
            m[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
            m[region3] = 1.0
            m[region4] = (w_top_high - W)[region4] / (2 * a * dw)
            maskTD[k] = m.astype(np.float32)

    else:
        if print_tqdm:
            print('constant pitch case')
        # constant pitch case
        w_bottom = - P * D / (2 * np.pi * R) * (np.pi/2 + alpha_vals) / np.cos(alpha_vals)
        w_top    =   P * D / (2 * np.pi * R) * (np.pi/2 - alpha_vals) / np.cos(alpha_vals)
        w_top = w_top[:, np.newaxis]      
        w_bottom = w_bottom[:, np.newaxis]

        W, A = np.meshgrid(w_vals, alpha_vals, indexing='xy')  
        mask = np.zeros(shape=(N_alpha, N_w), dtype=np.float32)

        w_bottom_low = (w_bottom - a * dw).reshape(-1, 1) 
        w_bottom_high = (w_bottom + a * dw).reshape(-1, 1)
        w_top_low = (w_top - a * dw).reshape(-1, 1)
        w_top_high = (w_top + a * dw).reshape(-1, 1)

        region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
        region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
        region4 = ( w_top_low < W ) & ( W <= w_top_high )

        # region1 and region5 stay at 0
        mask[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
        mask[region3] = 1
        mask[region4] = (w_top_high - W)[region4] / (2 * a * dw)

        maskTD = np.broadcast_to(mask, (N_s, N_alpha, N_w))
    return g6 * maskTD

def parker_weighting(ray_trafo, q=0.25):
    """Create parker weighting for a `RayTransform`.

    Parker weighting is a weighting function that ensures that oversampled
    fan/cone beam data are weighted such that each line has unit weight. It is
    useful in analytic reconstruction methods such as FBP to give a more
    accurate result and can improve convergence rates for iterative methods.

    See the article `Parker weights revisited`_ for more information.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform for which to compute the weights.
    q : float, optional
        Parameter controlling the speed of the roll-off at the edges of the
        weighting. 1.0 gives the classical Parker weighting, while smaller
        values in general lead to lower noise but stronger discretization
        artifacts.

    Returns
    -------
    parker_weighting : ``ray_trafo.range`` element

    See Also
    --------
    fbp_op : Filtered back-projection operator from `RayTransform`
    tam_danielson_window : Indicator function for helical data
    odl.tomo.geometry.conebeam.FanBeamGeometry : Use case in 2d
    odl.tomo.geometry.conebeam.ConeBeamGeometry : Use case in 3d (for pitch 0)

    References
    ----------
    .. _Parker weights revisited: https://www.ncbi.nlm.nih.gov/pubmed/11929021
    """
    # Note: Parameter names taken from WES2002

    # Extract parameters
    src_radius = ray_trafo.geometry.src_radius
    det_radius = ray_trafo.geometry.det_radius
    ndim = ray_trafo.geometry.ndim
    angles = ray_trafo.range.meshgrid[0]
    min_rot_angle = ray_trafo.geometry.motion_partition.min_pt
    alen = ray_trafo.geometry.motion_params.length

    # Parker weightings are not defined for helical geometries
    if ray_trafo.geometry.ndim != 2:
        pitch = ray_trafo.geometry.pitch
        if pitch != 0:
            raise ValueError('Parker weighting window is only defined with '
                             '`pitch==0`')

    # Find distance from projection of rotation axis for each pixel
    if ndim == 2:
        dx = ray_trafo.range.meshgrid[1]
    elif ndim == 3:
        # Find projection of axis on detector
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)
        # If axis is aligned to a coordinate axis, save some memory and time by
        # using broadcasting
        if rot_dir[0] == 0:
            dx = rot_dir[1] * ray_trafo.range.meshgrid[2]
        elif rot_dir[1] == 0:
            dx = rot_dir[0] * ray_trafo.range.meshgrid[1]
        else:
            dx = (rot_dir[0] * ray_trafo.range.meshgrid[1]
                  + rot_dir[1] * ray_trafo.range.meshgrid[2])

    # Compute parameters
    dx_abs_max = np.max(np.abs(dx))
    max_fan_angle = 2 * np.arctan2(dx_abs_max, src_radius + det_radius)
    delta = max_fan_angle / 2
    epsilon = alen - np.pi - max_fan_angle

    if epsilon < 0:
        raise Exception('data not sufficiently sampled for parker weighting')

    # Define utility functions
    def S(betap):
        return (0.5 * (1.0 + np.sin(np.pi * betap)) * (np.abs(betap) < 0.5)
                + (betap >= 0.5))

    def b(alpha):
        return q * (2 * delta - 2 * alpha + epsilon)

    # Create weighting function
    beta = np.asarray(angles - min_rot_angle,
                      dtype=ray_trafo.range.dtype)  # rotation angle
    alpha = np.asarray(np.arctan2(dx, src_radius + det_radius),
                       dtype=ray_trafo.range.dtype)

    # Compute sum in place to save memory
    S_sum = S(beta / b(alpha) - 0.5)
    S_sum += S((beta - 2 * delta + 2 * alpha - epsilon) / b(alpha) + 0.5)
    S_sum -= S((beta - np.pi + 2 * alpha) / b(-alpha) - 0.5)
    S_sum -= S((beta - np.pi - 2 * delta - epsilon) / b(-alpha) + 0.5)

    scale = 0.5 * alen / np.pi
    return ray_trafo.range.element(
        np.broadcast_to(S_sum * scale, ray_trafo.range.shape))


'''These classes are used to filter the projection data using the Katsevich filter.
   They implement the steps of the Katsevich filter as described in the paper 
   by Noo et al. (2003).
   https://www.researchgate.net/publication/8936875_Exact_helical_reconstruction_using_native_cone-beam_geometries

   KatsevichFilterCurved handles detectors of type CylindricalDetector,
   KatsevichFilterFlat handles detectors of type Flat2dDetector.

   Note that the axes order in the detector partition is different 
   for flat and curved detectors. For flat detectors, the axes are (z, x),
   while for curved detectors, the axes are (x, z). This is due to the way
   the projection geometry is created in ODL. See `astra_projection_geometry` in `astra_setup.py`.

   To validate the method, we compared the results (of the flat detector case) to the results of the implementation
   in the Pykatsevich package. 
   https://github.com/astra-toolbox/helical-kats/tree/main
'''


class KatsevichFilterCurvedConstPitch(Operator):
    def __init__(self, ray_trafo):
        self.ray_trafo = ray_trafo
        self.geometry = ray_trafo.geometry

        # extract geometry parameters
        self.D = self.geometry.src_radius + self.geometry.det_radius
        self.P = self.geometry.pitch
        self.Rs = self.geometry.src_radius

        self.alpha_vals = self.geometry.det_partition.coord_vectors[0]
        self.w_vals = self.geometry.det_partition.coord_vectors[1]
        self.dalpha = self.alpha_vals[1] - self.alpha_vals[0]
        self.dw = self.w_vals[1] - self.w_vals[0]
        self.N_alpha = self.geometry.det_partition.shape[0]
        self.N_w = self.geometry.det_partition.shape[1]
        
        # self.ds = self.geometry.motion_partition.cell_sides[0]
        self.ds = self.geometry.angles[1:] - self.geometry.angles[:-1]
        self.N_s = self.geometry.motion_partition.shape[0]

        # values for rebinning
        fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
        fov_radius = fov_dia / 2
        if fov_radius > self.Rs:
            raise ValueError('Field of view radius {} is larger than source radius {}.'.format(fov_radius, self.D))
        # alpha_m is the half fan angle
        alpha_m = np.arcsin(fov_radius / self.Rs) # Rs > fov!!!
        
        # formula (48) Noo et al. 2003
        M = int((np.pi/2 + alpha_m) * self.D * self.P / (2 * self.dw * self.Rs) * (np.cos(alpha_m) \
                + np.sin(alpha_m)*(np.tan(alpha_m) + (alpha_m + np.pi*0.5) * np.sin(alpha_m) / (np.cos(alpha_m))**2)))
        self.detector_rebin_rows = 2*M+1 # use 128 to compare to Pykatsevich 
        self.psi_vals = np.linspace(-np.pi/2 - alpha_m/2, np.pi/2 + alpha_m/2, self.detector_rebin_rows) 
        
        # forward rebinning
        self.fwd_rebin_w = np.zeros((self.N_alpha, self.detector_rebin_rows), dtype=np.float32)
        eps = 1e-6
        tan_psi = np.tan(self.psi_vals)
        tan_psi = np.where(np.abs(tan_psi) < eps, eps * np.sign(tan_psi), tan_psi)
        
        # precompute rebinning rows
        # formula (26) Noo et al. 2003
        print("Precomputing forward rebinning rows...")
        for i, a in enumerate(self.alpha_vals):
            cos_a = np.cos(a)
            sin_a = np.sin(a)
            # constant pitch
            self.fwd_rebin_w[i] =((self.D * self.P / (2 * np.pi * self.Rs)) * (
                self.psi_vals * cos_a + self.psi_vals / tan_psi * sin_a)) 
            self.fwd_rebin_w[i] = np.nan_to_num(self.fwd_rebin_w[i], nan=0.0, posinf=0.0, neginf=0.0)
            
        # reverse rebinning

        # precompute which detector row each point on the rebinning rows belongs to - map rebinning rows to detector rows
        # see formula (55) Noo et al. 2003
        self.rebin_row = np.zeros((self.N_alpha, self.N_w), dtype=np.int32)
        # self.rebin_fracs_0 = c(alpha, w, l)
        self.rebin_fracs_0 = np.zeros_like(self.rebin_row, dtype=np.float32)
        # self.rebin_fracs_1 = 1 - c(alpha, w, l)
        self.rebin_fracs_1 = np.ones_like(self.rebin_row, dtype=np.float32)

        for row in tqdm(range(self.N_w), desc='Precomputing rebinning rows -> detector rows'):

            for col in range(self.N_alpha//2, self.N_alpha):
                self.rebin_row[col, row] = 0
                for rebin in range(self.detector_rebin_rows - 1):
                    if (self.w_vals[row] >= self.fwd_rebin_w[col, rebin]) \
                        and (self.w_vals[row] <= self.fwd_rebin_w[col, rebin+1]):
                        self.rebin_row[col, row] = rebin
                        break
                self.rebin_fracs_0[col, row] = (self.w_vals[row] - self.fwd_rebin_w[col, self.rebin_row[col, row]]) \
                    / (self.fwd_rebin_w[col, self.rebin_row[col, row] + 1] - self.fwd_rebin_w[col, self.rebin_row[col, row]])

            for col in range(self.N_alpha//2):
                self.rebin_row[col, row] = 1
                for rebin in range(self.detector_rebin_rows - 1, 0, -1):
                    if (self.w_vals[row] >= self.fwd_rebin_w[col, rebin - 1]) \
                        and (self.w_vals[row] <= self.fwd_rebin_w[col, rebin]):
                            self.rebin_row[col, row] = rebin
                            break
                self.rebin_fracs_0[col, row] = (self.w_vals[row] - self.fwd_rebin_w[col, self.rebin_row[col, row] - 1]) \
                    / (self.fwd_rebin_w[col, self.rebin_row[col, row]] - self.fwd_rebin_w[col, self.rebin_row[col, row] - 1])

        self.rebin_fracs_1 -= self.rebin_fracs_0

        # initialize as ODL operator
        domain = self.ray_trafo.range
        op_range = self.ray_trafo.range
        super().__init__(domain, op_range, linear=False)
        print("KatsevichFilterCurved initialized.")

    def _call(self, g, out=None, **kwargs):
        
        diff = kwargs.get('diff', None)
        g = np.asarray(g)
        if diff == 1:
            self._g1 = self.cf1_derivative(g)
            print("Using cf1_derivative v1")
        elif diff == 2:
            self._g1 = self.cf1_derivative_v2(g)
            print("Using cf1_derivative v2")
        else:
            self._g1 = self.cf1_derivative_v2(g)
        self._g2 = self.cf2_length_weighting(self._g1)
        self._g3 = self.cf3_forward_rebin(self._g2)
        self._g4 = self.cf4_hilbert_transform(self._g3)
        self._g5 = self.cf5_backward_rebin(self._g4)
        self._g6 = self.cf6_cosine_weighting(self._g5)
        self._gF = self.cf7_td_weighting(self._g6)
        self._computed = True
        result = self.ray_trafo.range.element(self._gF)

        if out is not None:
            out[:] = result
            return out
        return result

    
    @property
    def g1(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g1.")
        return self._g1
    @property
    def g2(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g2.")
        return self._g2
    @property
    def g3(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g3.")
        return self._g3
    @property
    def g3_old(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g3_old.")
        return self._g3_old
    @property
    def g4(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g4.")
        return self._g4
    @property
    def g5(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g5.")
        return self._g5
    @property
    def g6(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g6.")
        return self._g6
    @property
    def gF(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing gF.")
        return self._gF
    
    def cf1_derivative(self, g):
        ''' expects g to be of shape (N_s, N_alpha, N_w)'''
        g1 = np.zeros((self.N_s - 1, self.N_alpha, self.N_w), dtype=g.dtype)
        # formula (46) Noo et al. 2003
        # forward difference of the projection data
        for k in tqdm(range(self.N_s - 1), desc='CF1: Derivative'):
            d_proj = (g[k + 1, :-1, :-1] - g[k, :-1, :-1] +
                      g[k + 1, 1:, :-1] - g[k, 1:, :-1]) / (2 * self.geometry.motion_partition.cell_sides[k])
            d_col = (g[k, 1:, :-1] - g[k, :-1, :-1] +
                     g[k+1, 1:, :-1] - g[k+1, :-1, :-1]) / (2 * (self.dalpha))
            g1[k, :-1, :-1] = d_proj + d_col 
        g1_full = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g1.dtype)
        g1_full[:-1] = g1
        g1_full[-1] = g1[-1]
        return g1_full
    
    def cf1_derivative_v2(self, g):
        # formula (2.4) of Katsevich 2011 using r=1
        g1us = np.zeros((self.N_s - 2, self.N_alpha-1, self.N_w), dtype=g.dtype)
        # for i in tqdm(range(self.N_alpha - 1), desc='CF1: Derivative'):
        #     g1us[:, i, :] = (g[2:, i+1, :] - g[:-2, i, :]) \
        #    + (1/self.ds - 1) * (g[2:, i, :] - g[:-2, i+1, :]) \
        #    + (1/self.ds + 2 / self.dalpha - 2) * (g[1:self.N_s-1, i+1, :] - g[1:self.N_s-1, i, :]) 
        for k in tqdm(range(1, self.N_s - 1), desc='CF1: Derivative'):
            g1us[k-1, :, :] = (g[k+1, 1:, :] - g[k-1, :-1, :]) \
            + (1/self.ds[k] - 1) * (g[k+1, :-1, :] - g[k - 1, 1:, :]) \
            + (1/self.ds[k] + 2 / self.dalpha - 2) * (g[k, 1:, :] - g[k, :-1, :])

        g1us *=0.5
        g1us_interp = 0.5 * (g1us[:, :, :-1] + g1us[:, :, 1:])  # shape: [N_s-2, N_w-1, N_u-1]

        g1 = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g.dtype)
        g1[1:self.N_s-1, :-1, :-1] = g1us_interp
        
        return g1
    
    def cf2_length_weighting(self, g1):
        # formula (24) Noo et al. 2003
        length_weight = self.D / np.sqrt(self.D*self.D + self.w_vals*self.w_vals)
        g2 = np.empty_like(g1, dtype=np.float64)
        g2 = g1 * length_weight 
        return g2
        
    def cf3_forward_rebin(self, g2):
        # use the precomputed rebinning rows and interpolate the values
        # on the detector rows to find the values on the rebinning rows.
        # see formula (25) Noo et al. 2003.
        g3 = np.zeros((self.N_s, self.N_alpha, self.detector_rebin_rows), dtype=g2.dtype)
        for i in tqdm(range(self.N_alpha), desc="CF3: Forward rebin"):
            for k in range(self.N_s):
                interp = interp1d(self.w_vals, g2[k, i, :], bounds_error=False, fill_value=0.0)
                g3[k, i, :] = interp(self.fwd_rebin_w[i])
        return g3
    
    def cf4_hilbert_transform(self, g3):
        # compute hilbert kernal and convolve with data on ribinning rows
        # see formula (50), (51) in Noo et al. 2003 for the kernel, 
        # and formula (52) in Noo et al. 2003 for the convolution.
        def discrete_hilbert_kernel(npoints, window='rect'):
            assert npoints % 2 == 1
            M = npoints // 2      
            tau_vals = np.arange(-M, M+1)
            def A(u):
                if window == 'rect':
                    return 1.
                elif window == 'hann':
                    return np.cos(np.pi * u / 2)**2
                else:
                    raise ValueError('Unknown window type') 
            H = np.zeros_like(tau_vals, dtype=np.float64)
            for idx, tau in enumerate(tau_vals):
                if tau == 0:
                    H[idx] = 0.0
                else:
                    u = np.linspace(0, 1, 100)
                    integrand = np.sin(np.pi * tau * u) * A(u)
                    integral = np.trapz(integrand, u)
                    H[idx] = (tau * self.dalpha) / (np.sin(tau * self.dalpha)) * integral
            return H 
        
        kernel = discrete_hilbert_kernel(npoints=101, window='hann')
        g4 = np.empty_like(g3)
        for k in tqdm(range(self.N_s), desc='CF4: Hilbert transform'):
            for rebin_row in range(self.detector_rebin_rows):
                conv_result = np.convolve(g3[k, :, rebin_row], kernel, mode='same')
                g4[k, :, rebin_row] = conv_result
        return g4
    
    def cf5_backward_rebin(self, g4):
        # reverse the rebinning process. Use the precomputation 
        # of the weights c, 1-c and of the map rebinning rows -> detector rows.
        # see formula (55) Noo et al. 2003.
        g5 = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g4.dtype)
        for k in tqdm(range(0, self.N_s), desc="CF5: Reverse rebin"):

            for col in range(self.N_alpha//2, self.N_alpha):
                row0 = self.rebin_row[col]
                row1 = row0 + 1
                g5[k, col, :] = (self.rebin_fracs_1[col] * g4[k, col, row0] +
                                 self.rebin_fracs_0[col] * g4[k, col, row1])
            for col in range(self.N_alpha//2):
                row0 = self.rebin_row[col] 
                rowmin1 = row0 - 1
                g5[k, col, :] = (self.rebin_fracs_1[col] * g4[k, col, rowmin1] +
                                 self.rebin_fracs_0[col] * g4[k, col, row0])
        return g5
    
    def cf6_cosine_weighting(self, g5):
        # apply cosine weighting to the data
        # see formula (30) Noo et al. 2003.
        alpha_weight = np.cos(self.alpha_vals)[np.newaxis, :, np.newaxis]
        return g5 * alpha_weight
    
    def cf7_td_weighting(self, g6):
        # Apply Tam_Danielson window to the data. Compute the 
        # indicator function maskTD and apply it to the data.
        # see formula (57) Noo et al. 2003.

        a=float(0.025) # smoothing width

        # formula (36) Noo et al. 2003
        w_bottom = - self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 + self.alpha_vals) / np.cos(self.alpha_vals)
        w_top    =   self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 - self.alpha_vals) / np.cos(self.alpha_vals)
        
        w_top = w_top[:, np.newaxis]      
        w_bottom = w_bottom[:, np.newaxis]
        if self.P == 0:
            raise ValueError('Tam-Danielson window is only defined with ' '`pitch != 0`')
        if a < 0:
            raise ValueError('`smoothing_width` should be a positive float')

        W, A = np.meshgrid(self.w_vals, self.alpha_vals, indexing='xy')  
        mask = np.zeros(shape=(self.N_alpha, self.N_w), dtype=np.float32)

        w_bottom_low = (w_bottom - a * self.dw).reshape(-1, 1) 
        w_bottom_high = (w_bottom + a * self.dw).reshape(-1, 1)
        w_top_low = (w_top - a * self.dw).reshape(-1, 1)
        w_top_high = (w_top + a * self.dw).reshape(-1, 1)

        region1 = ( W < w_bottom_low ) 
        region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
        region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
        region4 = ( w_top_low < W ) & ( W <= w_top_high )
        region5 = ( w_top_high < W )

        # region1 and region5 stay at 0
        mask[region2] = (W - w_bottom_low)[region2] / (2 * a * self.dw)
        mask[region3] = 1
        mask[region4] = (w_top_high - W)[region4] / (2 * a * self.dw)

        maskTD = np.broadcast_to(mask, (self.N_s, self.N_alpha, self.N_w))
        return g6 * maskTD


class KatsevichFilterCurved(Operator):
    def __init__(self, ray_trafo):
        self.ray_trafo = ray_trafo
        self.geometry = ray_trafo.geometry

        # extract geometry parameters
        self.D = self.geometry.src_radius + self.geometry.det_radius
        self.P = self.geometry.pitch
        self.Rs = self.geometry.src_radius
        self.const_pitch = np.all(self.geometry.src_shift_func(self.geometry.angles[:]) == 0)

        self.alpha_vals = self.geometry.det_partition.coord_vectors[0]
        self.w_vals = self.geometry.det_partition.coord_vectors[1]
        self.dalpha = self.alpha_vals[1] - self.alpha_vals[0]
        self.dw = self.w_vals[1] - self.w_vals[0]
        self.N_alpha = self.geometry.det_partition.shape[0]
        self.N_w = self.geometry.det_partition.shape[1]
        
        self.ds = self.geometry.angles[1:] - self.geometry.angles[:-1]
        self.N_s = self.geometry.motion_partition.shape[0]

        # values for rebinning
        fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
        fov_radius = fov_dia / 2
        if fov_radius > self.Rs:
            raise ValueError('Field of view radius {} is larger than source radius {}.'.format(fov_radius, self.D))
        # alpha_m is the half fan angle
        self.alpha_m = np.arcsin(fov_radius / self.Rs) # Rs > fov!!!
        alpha_m = self.alpha_m

        # formula (48) Noo et al. 2003
        M = int((np.pi/2 + alpha_m) * self.D * self.P / (2 * self.dw * self.Rs) * (np.cos(alpha_m) \
                + np.sin(alpha_m)*(np.tan(alpha_m) + (alpha_m + np.pi*0.5) * np.sin(alpha_m) / (np.cos(alpha_m))**2)))
        if M <= 0:
            raise ValueError('M should be a positive integer, got {}'.format(M))
        self.detector_rebin_rows = 2*M+1 # use 128 to compare to Pykatsevich 
        self.psi_vals = np.linspace(-np.pi/2 - alpha_m, np.pi/2 + alpha_m, self.detector_rebin_rows) 
        
        # forward rebinning
        # self.fwd_rebin_w = np.zeros((self.N_alpha, self.detector_rebin_rows), dtype=np.float32)
        eps = 1e-6
        tan_psi = np.tan(self.psi_vals)
        tan_psi = np.where(np.abs(tan_psi) < eps, eps * np.sign(tan_psi), tan_psi)
        
        # precompute rebinning rows
        # formula (26) Noo et al. 2003
        print("Precomputing forward rebinning rows...")

        if self.const_pitch:
            print('constant pitch')
            # constant pitch
            self.fwd_rebin_w = np.zeros((self.N_alpha, self.detector_rebin_rows), dtype=np.float32)
            for i, a in enumerate(self.alpha_vals):
                cos_a = np.cos(a)
                sin_a = np.sin(a)
                self.fwd_rebin_w[i] =((self.D * self.P / (2 * np.pi * self.Rs)) * (
                    self.psi_vals * cos_a + self.psi_vals / tan_psi * sin_a)) 
                self.fwd_rebin_w[i] = np.nan_to_num(self.fwd_rebin_w[i], nan=0.0, posinf=0.0, neginf=0.0)
        
        else:
            print('non constant pitch')

            self.fwd_rebin_w = np.zeros((self.N_s, self.N_alpha, self.detector_rebin_rows), dtype=np.float32)
            angles = self.geometry.angles
           
            sin_alpha = np.sin(self.alpha_vals)
            cos_alpha = np.cos(self.alpha_vals)

            for k in tqdm(range(self.N_s), desc='Precomputing forward rebinning rows (nonconst P)'):
                angle = self.geometry.angles[k]
                eu =   self.geometry.det_axes(angle)[0]
                ew =   self.geometry.det_axes(angle)[1]
                ev =   - self.geometry.det_to_src(angle, [0, 0])
                y_s = self.geometry.src_position(angle)

                for j, phi in enumerate(self.psi_vals):  
                    if angle + 2*phi > angles[-1] or angle + 2*phi < angles[0]:
                        continue
                    y_phi = self.geometry.src_position(angle + phi)
                    y_2phi = self.geometry.src_position(angle + 2 * phi)

                    cross = np.cross(y_phi - y_s, y_2phi - y_s)

                    if np.linalg.norm(cross) < 1e-8:
                        continue
                    # normal vector to the K plane
                    nu = cross / np.linalg.norm(cross) * np.sign(phi)
                    # nu = n(angle, phi)
                    eu_n = np.dot(eu, nu)
                    ew_n = np.dot(ew, nu)
                    ev_n = np.dot(ev, nu)
                    if np.abs(ew_n) < 1e-8: # avoid divide by zero
                        continue
                    self.fwd_rebin_w[k, :, j] = - self.D * (eu_n * sin_alpha + 
                                                 ev_n * cos_alpha) / ew_n

        # reverse rebinning
        
        # precompute which detector row each point on the rebinning rows belongs to - map rebinning rows to detector rows
        # see formula (55) Noo et al. 2003

        if self.const_pitch:

            self.rebin_row = np.zeros((self.N_alpha, self.N_w), dtype=np.int32)
            # self.rebin_fracs_0 = c(alpha, w, l)
            self.rebin_fracs_0 = np.zeros_like(self.rebin_row, dtype=np.float32)
            # self.rebin_fracs_1 = 1 - c(alpha, w, l)
            self.rebin_fracs_1 = np.ones_like(self.rebin_row, dtype=np.float32)

            for row in tqdm(range(self.N_w), desc='Precomputing rebinning rows -> detector rows (const P)'):

                for col in range(self.N_alpha//2, self.N_alpha):
                    self.rebin_row[col, row] = 0
                    for rebin in range(self.detector_rebin_rows - 1):
                        if (self.w_vals[row] >= self.fwd_rebin_w[col, rebin]) \
                            and (self.w_vals[row] <= self.fwd_rebin_w[col, rebin+1]):
                            self.rebin_row[col, row] = rebin
                            break
                    self.rebin_fracs_0[col, row] = (self.w_vals[row] - self.fwd_rebin_w[col, self.rebin_row[col, row]]) \
                        / (self.fwd_rebin_w[col, self.rebin_row[col, row] + 1] - self.fwd_rebin_w[col, self.rebin_row[col, row]])

                for col in range(self.N_alpha//2):
                    self.rebin_row[col, row] = 1
                    for rebin in range(self.detector_rebin_rows - 1, 0, -1):
                        if (self.w_vals[row] >= self.fwd_rebin_w[col, rebin - 1]) \
                            and (self.w_vals[row] <= self.fwd_rebin_w[col, rebin]):
                                self.rebin_row[col, row] = rebin
                                break
                    self.rebin_fracs_0[col, row] = (self.w_vals[row] - self.fwd_rebin_w[col, self.rebin_row[col, row] - 1]) \
                        / (self.fwd_rebin_w[col, self.rebin_row[col, row]] - self.fwd_rebin_w[col, self.rebin_row[col, row] - 1])

            self.rebin_fracs_1 -= self.rebin_fracs_0
        
        else: # non-constant pitch
            self.rebin_row = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=np.int32)
            # self.rebin_fracs_0 = c(alpha, w, l)
            self.rebin_fracs_0 = np.zeros_like(self.rebin_row, dtype=np.float32)
            # self.rebin_fracs_1 = 1 - c(alpha, w, l)
            self.rebin_fracs_1 = np.ones_like(self.rebin_row, dtype=np.float32)

            for k in tqdm(range(self.N_s), desc='Precomputing rebinning rows -> detector rows (nonconst P)'):
                w_vals = self.w_vals                         # shape (N_w,)
                for col in range(self.N_alpha):
                    fwd_curve = self.fwd_rebin_w[k, col, :]  # shape: (detector_rebin_rows,)
                    w0 = fwd_curve[:-1]                      # shape: (R-1,)
                    w1 = fwd_curve[1:]                       # shape: (R-1,)
                    mid = (w0[:, None] + w1[:, None]) / 2    # shape: (R-1, 1)

                    # Broadcasted bracketing condition for all detector rows
                    min_w = np.minimum(w0[:, None], w1[:, None])  # shape: (R-1, 1)
                    max_w = np.maximum(w0[:, None], w1[:, None])
                    mask = (self.w_vals[None, :] >= min_w) & (self.w_vals[None, :] <= max_w)  # shape: (R-1, N_w)

                    dist = np.where(mask, np.abs(self.w_vals[None, :] - mid), np.inf)  # shape: (R-1, N_w)
                    best_rebin = np.argmin(dist, axis=0)  # shape: (N_w,)

                    self.rebin_row[k, col, :] = best_rebin

                    # Compute interpolation fractions
                    w0_sel = w0[best_rebin]  # shape: (N_w,)
                    w1_sel = w1[best_rebin]
                    denom = w1_sel - w0_sel
                    close = np.abs(denom) < 1e-6
                    frac = np.where(close, 0.5, (self.w_vals - w0_sel) / denom)

                    self.rebin_fracs_0[k, col, :] = frac
                    self.rebin_fracs_1[k, col, :] = 1.0 - frac

        # initialize as ODL operator
        domain = self.ray_trafo.range
        op_range = self.ray_trafo.range
        super().__init__(domain, op_range, linear=False)
        print("KatsevichFilterCurved initialized.")

    def _call(self, g, out=None, **kwargs):
        '''
        parameter `diff` states which version of the differentiation scheme to use:
        - 1: cf1_derivative_NPH Noo-Pack-Heurscher scheme,
        - 2: cf1_derivative_K Katsevich scheme.
        See [1] for more details.

        [1] Faridani A., Hass. R, On Numerical Analysis of View-Dependent Derivatives in Computed Tomography. 2015.
        '''
        diff = kwargs.get('diff', None)
        print_tqdm = kwargs.get('print_tqdm', True)
        g = np.asarray(g)
        if diff == 1:
            self._g1 = self.cf1_derivative_NPH(g, print_tqdm)
            print("Using NPH derivative scheme")
        elif diff == 2:
            self._g1 = self.cf1_derivative_K(g, print_tqdm)
            print("Using K derivative scheme")
        else:
            self._g1 = self.cf1_derivative_K(g, print_tqdm)

        self._g2 = self.cf2_length_weighting(self._g1)

        if self.const_pitch:
            print('rebinning with constant pitch')
            self._g3 = self.cf3_forward_rebin(self._g2, print_tqdm)
        else:
            print('rebinning with non constant pitch')
            self._g3 = self.cf3_forward_rebin_nonconst_pitch(self._g2, print_tqdm)

        self._g4 = self.cf4_hilbert_transform(self._g3, print_tqdm)

        if self.const_pitch:
            print('cf5 with constant pitch')
            self._g5 = self.cf5_backward_rebin(self._g4, print_tqdm)
        else:
            print('cf5 with non constant pitch')
            self._g5 = self.cf5_backward_rebin_nonconst_pitch(self._g4, print_tqdm)

        self._g6 = self.cf6_cosine_weighting(self._g5)

        self._computed = True
        result = self.ray_trafo.range.element(self._g6)

        if out is not None:
            out[:] = result
            return out
        return result

    @property
    def g1(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g1.")
        return self._g1
    @property
    def g2(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g2.")
        return self._g2
    @property
    def g3(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g3.")
        return self._g3
    @property
    def g3_old(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g3_old.")
        return self._g3_old
    @property
    def g4(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g4.")
        return self._g4
    @property
    def g5(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g5.")
        return self._g5
    @property
    def g6(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g6.")
        return self._g6
    
    def cf1_derivative_NPH(self, g, print_tqdm):
        ''' expects g to be of shape (N_s, N_alpha, N_w)'''
        g1 = np.zeros((self.N_s - 1, self.N_alpha, self.N_w), dtype=g.dtype)
        # formula (46) Noo et al. 2003
        # forward difference of the projection data
        range_obj = tqdm(range(1, self.N_s-1), desc="CF1: Derivative (NPH)") if print_tqdm else range(0, self.N_s)
        for k in range_obj:
            d_proj = (g[k + 1, :-1, :-1] - g[k, :-1, :-1] +
                      g[k + 1, 1:, :-1] - g[k, 1:, :-1]) / (2 * self.geometry.motion_partition.cell_sides[0])
            d_col = (g[k, 1:, :-1] - g[k, :-1, :-1] +
                     g[k+1, 1:, :-1] - g[k+1, :-1, :-1]) / (2 * (self.dalpha))
            g1[k, :-1, :-1] = d_proj + d_col 
        g1_full = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g1.dtype)
        g1_full[:-1] = g1
        g1_full[-1] = g1[-1]
        return g1_full
    
    def cf1_derivative_K(self, g, print_tqdm):
        # formula (2.4) of Katsevich 2011 using r=1
        g1us = np.zeros((self.N_s - 2, self.N_alpha-1, self.N_w), dtype=g.dtype)
        range_obj = tqdm(range(1, self.N_s-1), desc="CF1: Derivative (K)") if print_tqdm else range(1, self.N_s-1)
        for k in range_obj:
            g1us[k-1, :, :] = (g[k+1, 1:, :] - g[k-1, :-1, :]) \
            + (1/self.ds[k] - 1) * (g[k+1, :-1, :] - g[k - 1, 1:, :]) \
            + (1/self.ds[k] + 2 / self.dalpha - 2) * (g[k, 1:, :] - g[k, :-1, :])
        g1us *=0.5
        g1us_interp = 0.5 * (g1us[:, :, :-1] + g1us[:, :, 1:])  # shape: [N_s-2, N_w-1, N_u-1]

        g1 = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g.dtype)
        g1[1:self.N_s-1, :-1, :-1] = g1us_interp
        
        return g1
    
    def cf2_length_weighting(self, g1):
        # formula (24) Noo et al. 2003
        length_weight = self.D / np.sqrt(self.D*self.D + self.w_vals*self.w_vals)
        g2 = np.empty_like(g1, dtype=np.float64)
        g2 = g1 * length_weight 
        return g2
        
    def cf3_forward_rebin(self, g2, print_tqdm):
        print('cf3 with constant pitch')
        # use the precomputed rebinning rows and interpolate the values
        # on the detector rows to find the values on the rebinning rows.
        # see formula (25) Noo et al. 2003.
        g3 = np.zeros((self.N_s, self.N_alpha, self.detector_rebin_rows), dtype=g2.dtype)
        
        range_obj = tqdm(range(0, self.N_alpha), desc="CF3: Forward rebin") if print_tqdm else range(0, self.N_alpha)
        for i in range_obj:
            for k in range(self.N_s):
                interp = interp1d(self.w_vals, g2[k, i, :], bounds_error=False, fill_value=0.0)
                g3[k, i, :] = interp(self.fwd_rebin_w[i])
        return g3
    
    def cf3_forward_rebin_nonconst_pitch(self, g2, print_tqdm):
        print('cf3 with non constant pitch')
        g3 = np.zeros((self.N_s, self.N_alpha, self.detector_rebin_rows), dtype=g2.dtype)
        
        range_obj = tqdm(range(0, self.N_alpha), desc="CF3: Forward rebin") if print_tqdm else range(0, self.N_alpha)
        for i in range_obj:
            for k in range(self.N_s):
                interp = interp1d(self.w_vals, g2[k, i, :], bounds_error=False, fill_value=0.0)
                g3[k, i, :] = interp(self.fwd_rebin_w[k, i])
        return g3
    
    def cf4_hilbert_transform(self, g3, print_tqdm):
        # compute hilbert kernal and convolve with data on ribinning rows
        # see formula (50), (51) in Noo et al. 2003 for the kernel, 
        # and formula (52) in Noo et al. 2003 for the convolution.
        def discrete_hilbert_kernel(npoints, window='rect'):
            assert npoints % 2 == 1
            M = npoints // 2      
            tau_vals = np.arange(-M, M+1)
            def A(u):
                if window == 'rect':
                    return 1.
                elif window == 'hann':
                    return np.cos(np.pi * u / 2)**2
                else:
                    raise ValueError('Unknown window type') 
            H = np.zeros_like(tau_vals, dtype=np.float64)
            for idx, tau in enumerate(tau_vals):
                if tau == 0:
                    H[idx] = 0.0
                else:
                    u = np.linspace(0, 1, 100)
                    integrand = np.sin(np.pi * tau * u) * A(u)
                    integral = np.trapz(integrand, u)
                    H[idx] = (tau * self.dalpha) / (np.sin(tau * self.dalpha)) * integral
            return H 
        if g3.shape[1] < 101:
            npoints = g3.shape[1]
        else:
            npoints = 101
        kernel = discrete_hilbert_kernel(npoints=npoints, window='hann')
        if g3.shape[1] < kernel.shape[0]:
            raise ValueError("The kernel is bigger than the column dimension. Ensure that the number of columns is at least {}.".format(kernel.shape[0]))
        g4 = np.empty_like(g3)

        range_obj = tqdm(range(0, self.N_s), desc='CF4: Hilbert transform') if print_tqdm else range(0, self.N_s)
        for k in range_obj:
            for rebin_row in range(self.detector_rebin_rows):
                conv_result = np.convolve(g3[k, :, rebin_row], kernel, mode='same')
                g4[k, :, rebin_row] = conv_result
        return g4
    
    def cf5_backward_rebin(self, g4, print_tqdm):
        # reverse the rebinning process. Use the precomputation 
        # of the weights c, 1-c and of the map rebinning rows -> detector rows.
        # see formula (55) Noo et al. 2003.
        g5 = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g4.dtype)
        
        range_obj = tqdm(range(0, self.N_s), desc="CF5: Reverse rebin") if print_tqdm else range(0, self.N_s)
        for k in range_obj:

            for col in range(self.N_alpha//2, self.N_alpha):
                row0 = self.rebin_row[col]
                row1 = row0 + 1
                g5[k, col, :] = (self.rebin_fracs_1[col] * g4[k, col, row0] +
                                 self.rebin_fracs_0[col] * g4[k, col, row1])
            for col in range(self.N_alpha//2):
                row0 = self.rebin_row[col] 
                rowmin1 = row0 - 1
                g5[k, col, :] = (self.rebin_fracs_1[col] * g4[k, col, rowmin1] +
                                 self.rebin_fracs_0[col] * g4[k, col, row0])
        return g5
    
    def cf5_backward_rebin_nonconst_pitch(self, g4, print_tqdm):
        # reverse the rebinning process. Use the precomputation 
        # of the weights c, 1-c and of the map rebinning rows -> detector rows.
        # see formula (55) Noo et al. 2003.
        g5 = np.zeros((self.N_s, self.N_alpha, self.N_w), dtype=g4.dtype)
        
        range_obj = tqdm(range(0, self.N_s), desc="CF5: Reverse rebin") if print_tqdm else range(0, self.N_s)
        for k in range_obj:

            for col in range(self.N_alpha):
                row0 = self.rebin_row[k, col]
                row1 = row0 + 1
                g5[k, col, :] = (self.rebin_fracs_1[k, col] * g4[k, col, row0] +
                                 self.rebin_fracs_0[k, col] * g4[k, col, row1])
            for col in range(self.N_alpha//2):
                row0 = self.rebin_row[k, col] 
                rowmin1 = row0 - 1
                g5[k, col, :] = (self.rebin_fracs_1[k, col] * g4[k, col, rowmin1] +
                                 self.rebin_fracs_0[k, col] * g4[k, col, row0])
        return g5
    
    def cf6_cosine_weighting(self, g5):
        # apply cosine weighting to the data
        # see formula (30) Noo et al. 2003.
        alpha_weight = np.cos(self.alpha_vals)[np.newaxis, :, np.newaxis]
        return g5 * alpha_weight
    
    def cf7_td_weighting(self, g6):
        # Apply Tam_Danielson window to the data. Compute the 
        # indicator function maskTD and apply it to the data.
        # see formula (57) Noo et al. 2003.

        a=float(0.025) # smoothing width

        # formula (36) Noo et al. 2003
        w_bottom = - self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 + self.alpha_vals) / np.cos(self.alpha_vals)
        w_top    =   self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 - self.alpha_vals) / np.cos(self.alpha_vals)
            
        w_top = w_top[:, np.newaxis]      
        w_bottom = w_bottom[:, np.newaxis]
        if self.P == 0:
            raise ValueError('Tam-Danielson window is only defined with ' '`pitch != 0`')
        if a < 0:
            raise ValueError('`smoothing_width` should be a positive float')

        W, A = np.meshgrid(self.w_vals, self.alpha_vals, indexing='xy')  
        mask = np.zeros(shape=(self.N_alpha, self.N_w), dtype=np.float32)

        w_bottom_low = (w_bottom - a * self.dw).reshape(-1, 1) 
        w_bottom_high = (w_bottom + a * self.dw).reshape(-1, 1)
        w_top_low = (w_top - a * self.dw).reshape(-1, 1)
        w_top_high = (w_top + a * self.dw).reshape(-1, 1)

        region1 = ( W < w_bottom_low ) 
        region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
        region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
        region4 = ( w_top_low < W ) & ( W <= w_top_high )
        region5 = ( w_top_high < W )

        # region1 and region5 stay at 0
        mask[region2] = (W - w_bottom_low)[region2] / (2 * a * self.dw)
        mask[region3] = 1
        mask[region4] = (w_top_high - W)[region4] / (2 * a * self.dw)

        maskTD = np.broadcast_to(mask, (self.N_s, self.N_alpha, self.N_w))
        return g6 * maskTD
    
    def td_weighting_nonconst_pitch(self, g6, print_tqdm):
        # Apply Tam_Danielson window to the data. Compute the 
        # indicator function maskTD and apply it to the data.
        # For non-constant pitch we have to recompute it for every angle k.
        a=float(0.025)
        if a <= 0:
            raise ValueError("`smoothing_width` must be positive")
        # 2-D grids that are the same for every source angle k
        W, A = np.meshgrid(self.w_vals, self.alpha_vals, indexing="xy")      # (N_alpha, N_w)
        dw   = self.dw
        maskTD = np.zeros_like(g6, dtype=np.float32)

        fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
        r = fov_dia / 2
        delta_s_edge = 2 * np.arccos(r / self.Rs) # self.alpha_m 
        ds_plus = np.linspace(delta_s_edge, 2*np.pi - delta_s_edge, 1000)
        ds_min  = np.linspace(delta_s_edge - 2*np.pi, - delta_s_edge, 1000)

        def nc_window_slice(k):
            s0 = self.geometry.angles[k]  
            s_bot  = s0 + ds_min
            s_top  = s0 + ds_plus
            p0     = self.geometry.src_position(s0)[2]  
            a_bot = (np.sin(ds_min) / (1 - np.cos(ds_min))) 
            a_top = (np.sin(ds_plus) / (1 - np.cos(ds_plus))) 
            try:           
                p_bot  = self.geometry.src_position(s_bot)[:, 2]           # (N_sampling,)
                w_bot  = (p_bot - p0) / (1 - np.cos(ds_min)) 
                exc_bot = False
            except Exception as e:
                # print(f"Error in computing source positions: bottom [{s_bot[0]}, {s_bot[-1]}] not in range of [{self.geometry.angles[0]}, {self.geometry.angles[-1]}]") 
                w_bot = self.w_vals[0] * np.ones_like(ds_min)  # Fallback to min w value         
                exc_bot = True
            try:
                p_top  = self.geometry.src_position(s_top)[:, 2]
                w_top  = (p_top - p0) / (1 - np.cos(ds_plus))
                exc_top = False
            except Exception as e:
                # print(f"Error in computing source positions: top [{s_top[0]}, {s_top[-1]}] not in range of [{geometry.angles[0]}, {geometry.angles[-1]}]") 
                w_top = self.w_vals[-1] * np.ones_like(ds_plus)
                exc_top = True
            if exc_bot:
                w_bot_interp = - self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 + self.alpha_vals) / np.cos(self.alpha_vals)
            else:
                interp_bot = interp1d(a_bot, w_bot, bounds_error=False, fill_value=(w_bot[0], w_bot[-1]))
                w_bot_interp = interp_bot(self.alpha_vals)
            if exc_top:
                w_top_interp =   self.P * self.D / (2 * np.pi * self.Rs) * (np.pi/2 - self.alpha_vals) / np.cos(self.alpha_vals)
            else:
                interp_top = interp1d(a_top, w_top, bounds_error=False, fill_value=(w_top[0], w_top[-1]))
                w_top_interp = interp_top(self.alpha_vals)
            return w_bot_interp, w_top_interp
        
        mayo_data = True

        if not mayo_data:
            range_obj = tqdm(range(self.N_s), desc="CF7: nonconst TD weighting") if print_tqdm else range(self.N_s)
            for k in range_obj:
                w_bottom, w_top = nc_window_slice(k)
                # Expand to 2D so we can compare with W
                w_bottom = w_bottom[:, None]    
                w_top    = w_top[:, None]

                w_bottom_low  = w_bottom - a * dw
                w_bottom_high = w_bottom + a * dw
                w_top_low     = w_top    - a * dw
                w_top_high    = w_top    + a * dw
                # Region tests (same five regions as the constant-pitch version)
                region2 = (w_bottom_low  <= W) & (W <  w_bottom_high)
                region3 = (w_bottom_high <= W) & (W <= w_top_low)
                region4 = (w_top_low     <  W) & (W <= w_top_high)

                m = np.zeros_like(W, dtype=np.float32)
                m[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
                m[region3] = 1.0
                m[region4] = (w_top_high - W)[region4] / (2 * a * dw)
                maskTD[k] = m.astype(np.float32)
            return g6 * maskTD
        else: # mayo data
            print("Using TD for Mayo data")
            w_bottom, _ = nc_window_slice(self.N_s-11)
            w_top, _ = nc_window_slice(10)
            w_top = w_top[:, np.newaxis]      
            w_bottom = w_bottom[:, np.newaxis]

            W, A = np.meshgrid(self.w_vals, self.alpha_vals, indexing='xy')  
            mask = np.zeros(shape=(self.N_alpha, self.N_w), dtype=np.float32)

            w_bottom_low = (w_bottom - a * self.dw).reshape(-1, 1) 
            w_bottom_high = (w_bottom + a * self.dw).reshape(-1, 1)
            w_top_low = (w_top - a * self.dw).reshape(-1, 1)
            w_top_high = (w_top + a * self.dw).reshape(-1, 1)

            region1 = ( W < w_bottom_low ) 
            region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
            region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
            region4 = ( w_top_low < W ) & ( W <= w_top_high )
            region5 = ( w_top_high < W )

            # region1 and region5 stay at 0
            mask[region2] = (W - w_bottom_low)[region2] / (2 * a * self.dw)
            mask[region3] = 1
            mask[region4] = (w_top_high - W)[region4] / (2 * a * self.dw)

            maskTD = np.broadcast_to(mask, (self.N_s, self.N_alpha, self.N_w))
            return g6 * maskTD      


class KatsevichFilterFlat(Operator):
    '''
    Katsevich filter for flat detector geometry.
    This class has been implemented  for the simple case with constant pitch.
    For non-constant pitch, see the curved detector filter class.
    '''

    def __init__(self, ray_trafo):
        self.ray_trafo = ray_trafo
        self.geometry = ray_trafo.geometry

        # extract geometry parameters
        self.D = self.geometry.src_radius + self.geometry.det_radius
        self.P = self.geometry.pitch
        self.Rs = self.geometry.src_radius

        self.u_vals = self.geometry.det_partition.coord_vectors[1]
        self.w_vals = self.geometry.det_partition.coord_vectors[0]
        self.du = self.u_vals[1] - self.u_vals[0]
        self.dw = self.w_vals[1] - self.w_vals[0]
        self.N_u = self.geometry.det_partition.shape[1]
        self.N_w = self.geometry.det_partition.shape[0]
        # self.ds = self.geometry.motion_partition.cell_sides[0]
        self.ds = self.geometry.angles[1:] - self.geometry.angles[:-1]
        self.N_s = self.geometry.motion_partition.shape[0]

        # values for rebinning
        fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
        fov_radius = fov_dia / 2
        if fov_radius > self.Rs:
            raise ValueError('Field of view radius {} is larger than source radius {}.'.format(fov_radius, self.D))
        # alpha_m is the half fan angle
        alpha_m = np.arcsin(fov_radius / self.Rs) # Rs > fov!!!

        # formula (48) Noo et al. 2003
        M = int((np.pi / 2 + alpha_m) * self.D * self.P / (2 * self.dw * self.Rs) * (1 + fov_radius * np.tan(alpha_m) / self.D - (alpha_m + np.pi / 2) * fov_radius / (self.D * np.cos(alpha_m))**2))
        self.detector_rebin_rows = 128 # 2*M + 1 # use 128 to compare to Pykatsevich
        self.psi_vals = np.linspace(-np.pi/2 - alpha_m, np.pi/2 + alpha_m, self.detector_rebin_rows) 
        
        # forward rebinning
        self.fwd_rebin_w = np.zeros((self.detector_rebin_rows, self.N_u))
        eps = 1e-6
        tan_psi = np.tan(self.psi_vals)
        tan_psi = np.where(np.abs(tan_psi) < eps, eps * np.sign(tan_psi), tan_psi)
        # Compute psi / tan(psi) safely
        psi_over_tan = np.where(np.abs(self.psi_vals) < eps, 1.0, self.psi_vals / tan_psi)
        
        # precompute rebinning rows
        # formula (26) Noo et al. 2003
        for i in range(self.N_u):
            self.fwd_rebin_w[:, i] = (self.D * self.P / (2 * np.pi * self.Rs)) * (
                self.psi_vals + psi_over_tan * (self.u_vals[i] / self.D))

        # reverse rebinning 
        # precompute which detector row each point on the rebinning rows belongs to - map rebinning rows to detector rows
        # see formula (55) Noo et al. 2003
        self.rebin_row = np.zeros((self.N_w, self.N_u), dtype=np.int32)
        # self.rebin_fracs_0 = c(u, w, l)
        self.rebin_fracs_0 = np.zeros_like(self.rebin_row, dtype=np.float32)
        # self.rebin_fracs_1 = 1 - c(u, w, l)
        self.rebin_fracs_1 = np.ones_like(self.rebin_row, dtype=np.float32)
        
        for row in range(self.N_w):

            for col in range(self.N_u // 2, self.N_u):
                self.rebin_row[row, col] = 0
                for rebin in range(len(self.psi_vals) - 1):
                    if (self.w_vals[row] >= self.fwd_rebin_w[rebin, col]) \
                        and (self.w_vals[row] <= self.fwd_rebin_w[rebin + 1, col]):
                        self.rebin_row[row, col] = rebin
                        break
                self.rebin_fracs_0[row, col] = (self.w_vals[row] - self.fwd_rebin_w[self.rebin_row[row, col], col]) \
                    / (self.fwd_rebin_w[self.rebin_row[row, col] + 1, col] - self.fwd_rebin_w[self.rebin_row[row, col], col])

            for col in range(self.N_u // 2):
                self.rebin_row[row, col] = 1
                for rebin in range(len(self.psi_vals) - 1, 0, -1):
                    if (self.w_vals[row] >= self.fwd_rebin_w[rebin - 1, col]) \
                        and (self.w_vals[row] <= self.fwd_rebin_w[rebin, col]):
                            self.rebin_row[row, col] = rebin
                            break
                self.rebin_fracs_0[row, col] = (self.w_vals[row] - self.fwd_rebin_w[self.rebin_row[row, col] - 1, col]) \
                    / (self.fwd_rebin_w[self.rebin_row[row, col], col] - self.fwd_rebin_w[self.rebin_row[row, col] - 1, col])

        self.rebin_fracs_1 -= self.rebin_fracs_0

        # initialize as ODL operator
        domain = self.ray_trafo.range
        op_range = self.ray_trafo.range
        print("KatsevichFilterFlat initialized.")
        super().__init__(domain, op_range, linear=False)

    def _call(self, g, out=None, **kwargs):
        diff = kwargs.get('diff', None)
        g = np.asarray(g)
        # use diff=1 to compare with pykatsevich results, if not use diff=2 (more accurate)
        if diff == 1:
            self._g1 = self.ff1_derivative(g)
            print("Using ff1_derivative v1")
        elif diff == 2:
            g1 = self.ff1_derivative_v2(g)
            self._g1 = self.ff2_derivative(g1)
            print("Using ff1_derivative v2")
        else:
            g1 = self.ff1_derivative_v2(g)
            self._g1 = self.ff2_derivative(g1)
        self._g3 = self.ff3_forward_rebin(self._g1)
        self._g4 = self.ff4_hilbert_transform(self._g3)
        self._g5 = self.ff5_backward_rebin(self._g4)
        self._computed = True
        result = self.ray_trafo.range.element(self._g5)

        if out is not None:
            out[:] = result
            return out
        return result

    @property
    def g1(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g1.")
        return self._g1
    @property
    def g3(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g3.")
        return self._g3
    @property
    def g4(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g4.")
        return self._g4
    @property
    def g5(self):
        if not self._computed:
            raise RuntimeError("Run kats(g) before accessing g5.")
        return self._g5
    
    def ff1_derivative(self, g):
        # derivative (forward difference) and length correction steps.
        # see formula (87) Noo et al. 2003 for the derivative and
        # formula (67) Noo et al. 2003 for the length correction.
        dia = self.D
        dia_sqr = dia * dia
        u_coords = self.u_vals[:-1]
        w_coords = self.w_vals[:-1]

        row_col_prod = np.outer(w_coords, u_coords)
        col_sqr = np.broadcast_to(u_coords[None, :] ** 2, row_col_prod.shape)
        row_sqr = np.broadcast_to(w_coords[:, None] ** 2, row_col_prod.shape)

        g1 = np.zeros((self.N_s, self.N_w, self.N_u), dtype=g.dtype)

        for k in tqdm(range(self.N_s - 1), desc='FF1: Derivative'):
            d_proj = (g[k+1, :-1, :-1] - g[k, :-1, :-1] +
                      g[k+1, 1:, :-1] - g[k, 1:, :-1] +
                      g[k+1, :-1, 1:] - g[k, :-1, 1:] +
                      g[k+1, 1:, 1:] - g[k, 1:, 1:]) / (4 * self.geometry.motion_partition.cell_sides[0])
            d_row = (g[k, 1:, :-1] - g[k, :-1, :-1] +
                 g[k, 1:, 1:] - g[k, :-1, 1:] +
                 g[k + 1, 1:, :-1] - g[k + 1, :-1, :-1] +
                 g[k + 1, 1:, 1:] - g[k + 1, :-1, 1:]) / (4 * self.dw)
            d_col = (g[k, :-1, 1:] - g[k, :-1, :-1] +
                     g[k, 1:, 1:] - g[k, 1:, :-1] +
                     g[k + 1, :-1, 1:] - g[k + 1, :-1, :-1] +
                     g[k + 1, 1:, 1:] - g[k + 1, 1:, :-1]) / (4 * self.du)
            g1[k, :-1, :-1] = d_proj + d_col * (col_sqr + dia_sqr) / dia + d_row * row_col_prod / dia
            g1[k, :-1, :-1] *= dia / np.sqrt(col_sqr + dia_sqr + row_sqr)
        return g1
    
    def ff1_derivative_v2(self, g):
        u_coords = self.u_vals[:-1] + 0.5 * self.du
        w_coords = self.w_vals[:-1] + 0.5 * self.dw
        g1us = np.zeros((self.N_s - 2, self.N_w, self.N_u - 1), dtype=g.dtype)
        
        for k in tqdm(range(1, self.N_s - 1), desc='FF1: Derivative'):
            for i in range(self.N_u - 1):
                g1us[k-1, :, i] = (g[k+1, :, i+1] - g[k-1, :, i]) \
                + (1/self.ds[k] - 1) * (g[k+1, :, i] - g[k - 1, :, i+1]) \
                + (1/self.ds[k] + 2 * (u_coords[i]**2 + self.D**2) / (self.D * self.du) - 2) \
                             * (g[k, :, i+1] - g[k, :, i])
        
        g1us *=0.5
        g1us_interp = 0.5 * (g1us[:, :-1, :] + g1us[:, 1:, :])  # shape: [N_s-2, N_w-1, N_u-1]

        # g1w = np.zeros((self.N_s, self.N_w - 1, self.N_u - 1), dtype=g.dtype) 
        # for i in range(self.N_u - 1):
        #     for j in range(self.N_w - 1):
        #         g1w[:, j, i] = u_coords[i]*w_coords[j]/self.D * (g[:, j+1, i] + g[:, j+1, i+1] - g[:, j, i] - g[:, j, i+1])
        # g1w *= 1 / (2 * self.dw)  

        # vectorized version
        scale = (u_coords[None, :] * w_coords[:, None]) / self.D  # shape: (N_w - 1, N_u - 1)
        diff = (g[:, 1:, :-1] + g[:, 1:, 1:] - g[:, :-1, :-1] - g[:, :-1, 1:])
        g1w = scale[None, :, :] * diff / (2 * self.dw)  

        g1 = np.zeros((self.N_s, self.N_w, self.N_u), dtype=g.dtype)
        g1[1:self.N_s-1, :-1, :-1] = g1us_interp
        g1[:, :-1, :-1] += g1w
        # for i in range(self.N_u - 1):
        #     for j in range(self.N_w - 1):
        #         g1[:, j, i] *= self.D / np.sqrt(u_coords[i] **2 + self.D **2 + w_coords[j]** 2 )
        return g1

    def ff2_derivative(self, g1): # vectorized version
        u_coords = self.u_vals
        w_coords = self.w_vals
        denom = np.sqrt(u_coords[None, :]**2 + self.D**2 + w_coords[:, None]**2)  # shape: (N_w - 1, N_u - 1)
        g1 *= self.D / denom[None, :, :] 
        return g1
    
    def ff3_forward_rebin(self, g2):
        # use the precomputed rebinning rows and interpolate the values
        # on the detector rows to find the values on the rebinning rows.
        # see formula (69) Noo et al. 2003.
        g3 = np.zeros((self.N_s, self.detector_rebin_rows, self.N_u), dtype=np.float64)
        # convert from mm to pixel index
        row_scaled = self.fwd_rebin_w / self.dw + 0.5 * self.N_w
        np.clip(row_scaled, 0, self.N_w - 2, out=row_scaled)
        row_idx = np.floor(row_scaled).astype(np.int32)
        row_frac = row_scaled - row_idx
        for k in tqdm(range(self.N_s), desc="FF3: Forward rebin"):
            for i in range(self.N_u):
                g3[k, :, i] = (1 - row_frac[:, i]) * g2[k, row_idx[:, i], i] + row_frac[:, i] * g2[k, row_idx[:, i] + 1, i]
        return g3

    def ff4_hilbert_transform(self, g3):
        # compute hilbert kernal and convolve with data on ribinning rows
        # see formula (20) in Noo et al. 2003 for the kernel, 
        # and formula (70) in Noo et al. 2003 for the convolution.
        def hilbert_kernel():
            kernel_radius = self.N_u - 1
            kernel_width = 2 * kernel_radius + 1
            proj_filter_array = np.zeros(kernel_width, dtype=np.float32)
            for i in range(kernel_width):
                proj_filter_array[i] = (1.0 - np.cos(np.pi * (i - kernel_radius - 0.5))) \
                                    / (np.pi * (i - kernel_radius - 0.5))
            return proj_filter_array

        kernel = hilbert_kernel()
        g4 = np.empty_like(g3)
        for k in tqdm(range(self.N_s), desc='FF4: Hilbert transform'):
            for rebin_row in range(self.detector_rebin_rows):
                conv_result = np.convolve( kernel, g3[k, rebin_row, :])
                tmp = conv_result[self.N_u - 1:2 * self.N_u - 1]
                g4[k, rebin_row, :] = tmp
        return g4

    def ff5_backward_rebin(self, g4):
        # reverse the rebinning process. Use the precomputation 
        # of the weights c, 1-c and of the map rebinning rows -> detector rows.
        # see formula (71) Noo et al. 2003.
        g5 = np.zeros((self.N_s, self.N_w, self.N_u), dtype=np.float64)
        for k in tqdm(range(0, self.N_s), desc="FF5: Reverse rebin"):

            for col in range(self.N_u//2, self.N_u):
                row0 = self.rebin_row[:, col]
                row1 = row0 + 1
                g5[k, :, col] = (self.rebin_fracs_1[:, col] * g4[k, row0, col] +
                                 self.rebin_fracs_0[:, col] * g4[k, row1, col])
            for col in range(self.N_u//2):
                row0 = self.rebin_row[:, col] 
                rowmin1 = row0 - 1
                g5[k, :, col] = (self.rebin_fracs_1[:, col] * g4[k, rowmin1, col] +
                                 self.rebin_fracs_0[:, col] * g4[k, row0, col])
        return g5

    def ff7_td_weighting(self, g5):
        # Apply Tam_Danielson window to the data. Compute the 
        # indicator function maskTD and apply it to the data.
        # see formula (57) Noo et al. 2003.

        a=float(0.025)
        dw = self.dw

        # formula (78) Noo et al. 2003
        w_bottom = - self.P / (2 * np.pi * self.Rs * self.D) * (self.u_vals**2 + self.D**2) * (np.pi/2 + np.arctan(self.u_vals / self.D))
        w_top    =   self.P / (2 * np.pi * self.Rs * self.D) * (self.u_vals**2 + self.D**2) * (np.pi/2 - np.arctan(self.u_vals / self.D))
        
        w_bottom = np.reshape(w_bottom, (1, -1))
        w_top    = np.reshape(w_top, (1, -1))
        if self.P == 0:
            raise ValueError('Tam-Danielson window is only defined with ' '`pitch != 0`')
        if a < 0:
            raise ValueError('`smoothing_width` should be a positive float')

        W, U = np.meshgrid(self.w_vals, self.u_vals, indexing='ij')  # (N_w, N_u)
        mask = np.zeros(shape=(self.N_w, self.N_u), dtype=np.float32)

        w_bottom_low  = np.repeat((w_bottom - a * dw), W.shape[0], axis=0)
        w_bottom_high = np.repeat((w_bottom + a * dw), W.shape[0], axis=0)
        w_top_low  = np.repeat((w_top - a * dw), W.shape[0], axis=0)
        w_top_high = np.repeat((w_top + a * dw), W.shape[0], axis=0)

        region1 = ( W < w_bottom_low ) 
        region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
        region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
        region4 = ( w_top_low < W ) & ( W <= w_top_high )
        region5 = ( w_top_high < W )

        # region1 and region5 stay at 0
        mask[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
        mask[region3] = 1
        mask[region4] = (w_top_high - W)[region4]/ (2 * a * dw)
        
        maskTD = np.broadcast_to(mask, (self.N_s, self.N_w, self.N_u))
        return g5 * maskTD

    def ff7_td_weighting_v2(self, g5):
        window = tam_danielson_window(self.ray_trafo)
        return g5 * window


def katsevich_filter_op(ray_trafo):
    if ray_trafo.geometry.det_curvature_radius is None: 
        # print('redirect to flat filter')  
        return KatsevichFilterFlat(ray_trafo)
    else:
        # print('redirect to curved filter')
        return KatsevichFilterCurved(ray_trafo)


def fbp_filter_op(ray_trafo, padding=True, filter_type='Ram-Lak',
                  frequency_scaling=1.0):
    """Create a filter operator for FBP from a `RayTransform`.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2dGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanBeamGeometry` : Approximate reconstruction, correct in limit of
        fan angle = 0.
        Only flat detectors are supported (det_curvature_radius is None).

        `ConeBeamGeometry`, pitch = 0 (circular) : Approximate reconstruction,
        correct in the limit of fan angle = 0 and cone angle = 0.

        `ConeBeamGeometry`, pitch > 0 (helical) : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : optional
        The type of filter to be used.
        The predefined options are, in approximate order from most noise
        senstive to least noise sensitive:
        ``'Ram-Lak'``, ``'Shepp-Logan'``, ``'Cosine'``, ``'Hamming'`` and
        ``'Hann'``.
        A callable can also be provided. It must take an array of values in
        [0, 1] and return the filter for these frequencies.
    frequency_scaling : float, optional
        Relative cutoff frequency for the filter.
        The normalized frequencies are rescaled so that they fit into the range
        [0, frequency_scaling]. Any frequency above ``frequency_scaling`` is
        set to zero.

    Returns
    -------
    filter_op : `Operator`
        Filtering operator for FBP based on ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data
    """
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'
    alen = ray_trafo.geometry.motion_params.length

    if ray_trafo.domain.ndim == 2:
        # Define ramp filter
        def fourier_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = 1 / (2 * alen)
            return filt * np.max(abs_freq) * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            ran_shp = (ray_trafo.range.shape[0],
                       ray_trafo.range.shape[1] * 2 - 1)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=1, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=1, impl=impl)

    elif ray_trafo.domain.ndim == 3:
        # Find the direction that the filter should be taken in
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

        # Find what axes should be used in the fourier transform
        used_axes = (rot_dir != 0)
        if used_axes[0] and not used_axes[1]:
            axes = [1]
        elif not used_axes[0] and used_axes[1]:
            axes = [2]
        else:
            axes = [1, 2]

        # Add scaling for cone-beam case
        if hasattr(ray_trafo.geometry, 'src_radius'):
            scale = (ray_trafo.geometry.src_radius
                     / (ray_trafo.geometry.src_radius
                        + ray_trafo.geometry.det_radius))

            if ray_trafo.geometry.pitch != 0:
                # In helical geometry the whole volume is not in each
                # projection and we need to use another weighting.
                # Ideally each point in the volume effects only
                # the projections in a half rotation, so we assume that that
                # is the case.
                scale *= alen / (np.pi)
        else:
            scale = 1.0

        # Define ramp filter
        def fourier_filter(x):
            # If axis is aligned to a coordinate axis, save some memory and
            # time by using broadcasting
            if not used_axes[0]:
                abs_freq = np.abs(rot_dir[1] * x[2])
            elif not used_axes[1]:
                abs_freq = np.abs(rot_dir[0] * x[1])
            else:
                abs_freq = np.abs(rot_dir[0] * x[1] + rot_dir[1] * x[2])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = scale * np.max(abs_freq) / (2 * alen)
            return filt * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            if used_axes[0]:
                padded_shape_u = ray_trafo.range.shape[1] * 2 - 1
            else:
                padded_shape_u = ray_trafo.range.shape[1]

            if used_axes[1]:
                padded_shape_v = ray_trafo.range.shape[2] * 2 - 1
            else:
                padded_shape_v = ray_trafo.range.shape[2]

            ran_shp = (ray_trafo.range.shape[0],
                       padded_shape_u,
                       padded_shape_v)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=axes, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=axes, impl=impl)
    else:
        raise NotImplementedError('FBP only implemented in 2d and 3d')

    # Create ramp in the detector direction
    ramp_function = fourier.range.element(fourier_filter)

    weight = 1
    if not ray_trafo.range.is_weighted:
        # Compensate for potentially unweighted range of the ray transform
        weight *= ray_trafo.range.cell_volume

    if not ray_trafo.domain.is_weighted:
        # Compensate for potentially unweighted domain of the ray transform
        weight /= ray_trafo.domain.cell_volume

    ramp_function *= weight

    # Create ramp filter via the convolution formula with fourier transforms
    return fourier.inverse * ramp_function * fourier


def fbp_op(ray_trafo, padding=True, filter_type='Ram-Lak',
           frequency_scaling=1.0):
    """Create filtered back-projection operator from a `RayTransform`.

    The filtered back-projection is an approximate inverse to the ray
    transform.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2dGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanBeamGeometry` : Approximate reconstruction, correct in limit of fan
        angle = 0.
        Only flat detectors are supported (det_curvature_radius is None).

        `ConeBeamGeometry`, pitch = 0 (circular) : Approximate reconstruction,
        correct in the limit of fan angle = 0 and cone angle = 0.

        `ConeBeamGeometry`, pitch > 0 (helical) : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : optional
        The type of filter to be used.
        The predefined options are, in approximate order from most noise
        senstive to least noise sensitive:
        ``'Ram-Lak'``, ``'Shepp-Logan'``, ``'Cosine'``, ``'Hamming'`` and
        ``'Hann'``.
        A callable can also be provided. It must take an array of values in
        [0, 1] and return the filter for these frequencies.
    frequency_scaling : float, optional
        Relative cutoff frequency for the filter.
        The normalized frequencies are rescaled so that they fit into the range
        [0, frequency_scaling]. Any frequency above ``frequency_scaling`` is
        set to zero.

    Returns
    -------
    fbp_op : `Operator`
        Approximate inverse operator of ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data.
    parker_weighting : Windowing for overcomplete fan-beam data.
    """
    if filter_type == 'Katsevich':
        return ray_trafo.adjoint_kats * katsevich_filter_op(ray_trafo)
    else:
        return ray_trafo.adjoint * fbp_filter_op(ray_trafo, padding, filter_type,
                                                 frequency_scaling)


if __name__ == '__main__':
    import odl
    import matplotlib.pyplot as plt
    from odl.util.testutils import run_doctests

    # Display the various filters
    x = np.linspace(0, 1, 100)
    cutoff = 0.7

    plt.figure('fbp filter')
    for filter_name in ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann',
                        np.sqrt]:
        plt.plot(x, _fbp_filter(x, filter_name, cutoff), label=filter_name)

    plt.title('Filters with frequency scaling = {}'.format(cutoff))
    plt.legend(loc=2)

    # Show the Tam-Danielson window

    # Create Ray Transform in helical geometry
    reco_space = odl.uniform_discr(
        min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300])
    angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 2000)
    detector_partition = odl.uniform_partition([-40, -4], [40, 4], [500, 500])
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition, detector_partition, src_radius=100, det_radius=100,
        pitch=5.0)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # Crete and show TD window
    td_window = tam_danielson_window(ray_trafo, smoothing_width=0)
    td_window.show('Tam-Danielson window', coords=[0, None, None])

    # Show the Parker weighting

    # Create Ray Transform in fan beam geometry
    geometry = odl.tomo.cone_beam_geometry(reco_space,
                                           src_radius=40, det_radius=80)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    # Crete and show parker weighting
    parker_weighting = parker_weighting(ray_trafo)
    parker_weighting.show('Parker weighting')

    # Also run the doctests
    run_doctests()
