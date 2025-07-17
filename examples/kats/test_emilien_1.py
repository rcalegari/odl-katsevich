
'''import parameters in /home/rosaca/code/pine_16_1/metadata.json, create 
   geometry, space.
   import data from /home/rosaca/code/pine_16_1/sinogram.npy and run katsevich reco
   on that.'''

from typing import Dict
from pathlib import Path
from functools import partial

import numpy as np
import odl
import json

import matplotlib.pyplot as plt

from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows, td_window, td_window_nonconst_pitch, td_window_nonconst_pitch_video

use_slice = True
use_interval = False
use_full = False

# astra_backend true means to use the 
# backprojection operator ray by ray
# false means to use the new method adjoint_kats

# the first is still wrong

astra_backend = True

'''try fetching the filtered sinogram if available. if yes, use it and run backprojection on it,
otherwise, filter, save it and run backprojection on the filtered sinogram.
'''
sample_path = Path('/home/rosaca/code/pine_16_1')
filtdata_fbp_path = sample_path.joinpath('filtered_fbp.npy')
if use_slice:
    slice_k = 1/2
    print(f"Creating ray transform operator for slice {slice_k}")
    filtdata_kats_path = sample_path.joinpath(f'filtered_kats_slice_{slice_k}.npy')
elif use_interval:
    slice_bounds = [150., 200.]
    filtdata_kats_path = sample_path.joinpath(f'filtered_kats_interval_{slice_bounds[0]}_{slice_bounds[1]}.npy')
elif use_full:
    filtdata_kats_path = sample_path.joinpath('filtered_kats_interval_full.npy')


def make_operators(
        angles:np.ndarray,
        shifts:np.ndarray,
        metadata:Dict,
        local_metadata = None,
        volMinZ = None,
        volMaxZ = None,
        slice_k = None,
        rec_min_z = None,
        rec_max_z = None,
        weight:float = None,
        print_info:bool = True):
    '''
    Create ray transform operator from angles, shifts and metadata.
    
    Parameters
    ----------
    slice_k : float
        If not None, reconstruct only the slice with index `slice_k` * `REC_NPX_Z` in the reco space.
    weight: float
        Use for single angle reconstruction, the weight is the distance between two angles in radians.
    '''
    
    if local_metadata:
        for key, value in local_metadata.items():
            metadata[key] = value
    DET_X_MIN = metadata['DET_X_MIN']
    DET_X_MAX = metadata['DET_X_MAX']
    DET_NPX_X = metadata['DET_NPX_X']
    DET_Z_MIN = metadata['DET_Z_MIN']
    DET_Z_MAX = metadata['DET_Z_MAX']
    DET_NPX_Z = metadata['DET_NPX_Z']
    REC_PIC_SIZE = metadata['REC_PIC_SIZE']

    detector_partition = odl.uniform_partition(
            [DET_X_MIN, DET_Z_MIN],
            [DET_X_MAX, DET_Z_MAX],
            (DET_NPX_X, DET_NPX_Z))
    
    REC_MIN_X, REC_MIN_Y, REC_MIN_Z = metadata['REC_MIN_X'], metadata['REC_MIN_Y'], metadata['REC_MIN_Z']
    REC_MAX_X, REC_MAX_Y, REC_MAX_Z = metadata['REC_MAX_X'], metadata['REC_MAX_Y'], metadata['REC_MAX_Z']
    REC_NPX_X, REC_NPX_Y, REC_NPX_Z = metadata['REC_NPX_X'], metadata['REC_NPX_Y'], metadata['REC_NPX_Z']
    
    if volMinZ is not None and volMaxZ is not None:
        REC_MIN_Z = volMinZ
        REC_MAX_Z = volMaxZ
    
    REC_PIC_SIZE_Z = (REC_MAX_Z - REC_MIN_Z) / REC_NPX_Z
    if print_info:
        print(f"full z reco space: {REC_MIN_Z} to {REC_MAX_Z}, npx: {REC_NPX_Z}")
    if slice_k:
        slice_z = int(slice_k * REC_NPX_Z)  # slice_k is a fraction of the total number of slices
        REC_MIN_Z = slice_z * REC_PIC_SIZE_Z + REC_MIN_Z
        REC_MAX_Z = REC_MIN_Z + REC_PIC_SIZE_Z 
        REC_NPX_Z = 1
        if print_info:
            print(f"slice {slice_z} reco space: {REC_MIN_Z} to {REC_MAX_Z}, npx: {REC_NPX_Z}")

    if rec_min_z is not None and rec_max_z is not None:
        REC_MIN_Z = rec_min_z
        REC_MAX_Z = rec_max_z
        REC_NPX_Z = int((REC_MAX_Z - REC_MIN_Z) / REC_PIC_SIZE_Z)
        if print_info:
            print(f"reco space: {REC_MIN_Z} to {REC_MAX_Z}, npx: {REC_NPX_Z}")

    reco_space = odl.uniform_discr(
        min_pt=[REC_MIN_X, REC_MIN_Y, REC_MIN_Z],
        max_pt=[REC_MAX_X, REC_MAX_Y, REC_MAX_Z],
        shape= [REC_NPX_X, REC_NPX_Y, REC_NPX_Z],
        dtype='float32')
    
    SRC_RADIUS = metadata['SRC_RADIUS']
    DET_RADIUS = metadata['DET_RADIUS'] 
    DET_CURVATURE_RADIUS = metadata['DET_CURVATURE_RADIUS']
    PITCH = metadata['PITCH']
    
    angle_partition = odl.nonuniform_partition(angles)
    # Source Shift
    src_shift_func = partial(odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts)
    # Detector shift 
    det_shift_func = partial(odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts)

    # NB: det_radius is not equal det_curvature_radius!
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SRC_RADIUS,
        det_radius=DET_RADIUS,
        det_curvature_radius=(DET_CURVATURE_RADIUS,None),  # uncomment for curved detector
        pitch=PITCH,
        src_shift_func=src_shift_func,
        det_shift_func=det_shift_func)
    
    geometry_full = geometry
    k_min=None
    k_max=None

    if slice_k:
        if len(angles)>1:
            for k, angle in enumerate(angles):
                if geometry.det_refpoint(angle)[2] + DET_Z_MAX >= REC_MIN_Z:
                    k_min = k - 1 if k > 0 else 0
                    break
            for k, angle in enumerate(angles[::-1]):
                if geometry.det_refpoint(angle)[2] + DET_Z_MIN <= REC_MAX_Z:
                    k_max = len(angles) - k + 1 if k > 0 else len(angles)
                    break
            dk = int((k_max-k_min)/4)
            k_min = k_min - dk
            k_max = k_max + dk
            if print_info:
                print(f"Using projections {k_min} to {k_max} for reconstruction of selected portion of volume. Total {k_max - k_min + 1} projections.")
        else:
            if print_info:
                print("Only one angle in the partition, using that for reconstruction.")
            k_min = 0
            k_max = 1
            
        angle_partition = odl.nonuniform_partition(
            angles[k_min:k_max+1])
        # Source Shift
        src_shift_func = partial(odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts[k_min:k_max+1])
        # Detector shift 
        det_shift_func = partial(odl.tomo.flying_focal_spot, apart=angle_partition, shifts=shifts[k_min:k_max+1])

        geometry = odl.tomo.ConeBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=SRC_RADIUS,
            det_radius=DET_RADIUS,
            det_curvature_radius=(DET_CURVATURE_RADIUS,None),  # uncomment for curved detector
            pitch=PITCH,
            src_shift_func=src_shift_func,
            det_shift_func=det_shift_func)
        
    # to take care of one angle partition!
    if len(angles) == 1:
        if weight is None:
            '''see extent in odl/tomo/operators/ray_trafo.py'''
            raise ValueError("Weighting must be provided for single angle reconstruction.")
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda', angle_weighting=weight)
    else:
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
    
    if PITCH == 0:
        ray_trafo_adjoint = ray_trafo.adjoint
    else:
        ray_trafo_adjoint = ray_trafo.adjoint_kats
    return ray_trafo, ray_trafo_adjoint, k_min, k_max, geometry_full, reco_space

print("Loading sinogram ...")
sinogram = np.load(sample_path.joinpath('sinogram.npy'), mmap_mode='r')
print(f"Sinogram shape: {sinogram.shape}, dtype: {sinogram.dtype}")

print("Loading angles, shifts and metadata ...")
angles   = np.load(sample_path.joinpath('angles.npy'))
angles_old = angles.copy()
shifts   = np.load(sample_path.joinpath('shifts.npy'))
metadata = dict(json.load(open(sample_path.joinpath('metadata.json'))))

volMinZ = 0.0
volMaxZ = 369.36050892522326
npxZ = 738

detMinZ = -34.5
detMaxZ = 34.5
detNpxZ = 230

pixel_size = (detMaxZ - detMinZ) / detNpxZ

pitch = 101.0

# # print("Angles:", angles[0], angles[-1])
angles_mm = angles * pitch / (2 * np.pi)
# OPTION 1: shift angles
angles_shifted_mm = angles_mm - (angles_mm[0] - volMinZ)
# convert back  to radians!!!
angles_shifted = angles_shifted_mm * 2 * np.pi / pitch
# OPTION 2: shift vol to match angles
volMaxZ_shift = volMaxZ + angles_mm[0] - volMinZ
volMinZ_shift = angles_mm[0]

if use_slice:
    forward_operator, backward_operator, k_min, k_max, geom_full, reco_space = make_operators(angles, shifts, metadata, volMinZ=volMinZ_shift, volMaxZ=volMaxZ_shift, slice_k=slice_k)
elif use_full:
    forward_operator, backward_operator, k_min, k_max, geom_full, reco_space = make_operators(angles, shifts, metadata, volMinZ=volMinZ_shift, volMaxZ=volMaxZ_shift)

sinogram = sinogram[k_min:k_max+1, :, :]
angles = angles[k_min:k_max+1]
shifts = shifts[k_min:k_max+1, :]

# if filtdata_fbp_path.exists():
#     print("Loading filtered sinogram for FBP...")
#     filt_sino_fbp = np.load(filtdata_fbp_path, mmap_mode='r')
# else:
#     print("Filtering sinogram for FBP...")
#     filt_op_fbp = odl.tomo.fbp_filter_op(forward_operator, filter_type='Ram-Lak', frequency_scaling=0.8) * odl.tomo.tam_danielson_window(forward_operator)
#     filt_sino_fbp = filt_op_fbp(sinogram)
#     np.save(filtdata_fbp_path, filt_sino_fbp)
#     print("Filtered FBP sinogram saved.")
if filtdata_kats_path.exists():
    print("Loading filtered sinogram for Katsevich...")
    filt_sino_kats = np.load(filtdata_kats_path, mmap_mode='r')
else:
    print("Filtering sinogram for Katsevich...")
    filt_op_kats = odl.tomo.katsevich_filter_op(forward_operator)
    filt_sino_kats = filt_op_kats(sinogram, print=False)
    np.save(filtdata_kats_path, filt_sino_kats)
    print("Filtered KATS sinogram saved.")

print("Reconstructing data with KATS...")

if astra_backend:
    from tqdm import tqdm
    for k in tqdm(range(len(angles)), desc="KATS Reconstruction angle by angle"):
        angle = angles[k]
        forward_op_k, _, _, _, _, reco_space = make_operators(angles[k:k+1], 
                                                shifts[k:k+1, :], 
                                                metadata, 
                                                rec_min_z = reco_space.min_pt[2],
                                                rec_max_z = reco_space.max_pt[2],
                                                weight = angles_old[k+1] - angles_old[k], 
                                                print_info=True if k == 0 else False)
        # compute the voxel weights for this angle
        nx = metadata['REC_NPX_X']
        ny = metadata['REC_NPX_Y']
        nz = 1
        xsize = (metadata['REC_MAX_X'] - metadata['REC_MIN_X']) / nx
        ysize = (metadata['REC_MAX_Y'] - metadata['REC_MIN_Y']) / ny

        ang_cos = np.cos(angle)
        ang_sin = np.sin(angle)
        # ev_x = ang_cos * ev0[0] - ang_sin * ev0[1]
        # ev_y = ang_sin * ev0[0] + ang_cos * ev0[1]
        ev = - geom_full.det_to_src(angle, [0, 0])
        ev_x = ev[0]
        ev_y = ev[1]
        i = np.arange(nx).reshape(-1, 1)
        j = np.arange(ny).reshape(1, -1)
        X = xsize * i + metadata['REC_MIN_X'] + 0.5 * xsize
        Y = ysize * j + metadata['REC_MIN_Y'] + 0.5 * ysize

        scale_coeff = 1. / (metadata['SRC_RADIUS'] + X * ev_x + Y * ev_y)
        voxel_weights = np.broadcast_to(scale_coeff, (nx, ny))
        voxel_weights = voxel_weights[:, :, np.newaxis]

        # print(f"Backprojecting angle {k + k_min} of {k_max + 1}")
        backproj = forward_op_k.adjoint(filt_sino_kats[k, :, :], angle_weighting=angles_old[k+1] - angles_old[k])
        # backproj *= voxel_weights
        if k == 0:
            kats_reco = backproj
        else:
            kats_reco += backproj

else:
    kats_reco = backward_operator(odl.tomo.td_window_curved(forward_operator, filt_sino_kats, force_const=True))

print("KATS Reconstruction done.")

kats_reco_arr = np.transpose(kats_reco.asarray(), (2, 0, 1))

save_xy_video(kats_reco_arr, sample_path, 'kats_reco', step=1, interval=1)

kats_reco.show(saveto=sample_path.joinpath(f'kats_reco.svg'),
            title=f'Katsevich Reconstruction of slice at {slice_k} of volume', 
            cmap='gray', 
            colorbar=True)

# diff = fbp_reco - kats_reco
# diff.show(saveto=sample_path.joinpath('diff_reco.svg'),
#         title='FBP - Katsevich Reconstruction', 
#         cmap='gray', 
#         colorbar=True)
