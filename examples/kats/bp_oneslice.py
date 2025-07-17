from typing import Dict
from pathlib import Path
from functools import partial

import numpy as np
import odl
import json
import os

import matplotlib.pyplot as plt

from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows
from examples.kats.pykats_config import create_pykats_config, create_space_geom
from examples.kats.test_flat import katsevich_filter_flat
from examples.kats.test_curved import katsevich_filter_curved
from examples.kats.utils import save_intensity_row, save_intensity_error_rows

def make_operators(angles:np.ndarray, metadata:Dict):
    
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

    REC_PIC_SIZE_Z = (REC_MAX_Z - REC_MIN_Z) / REC_NPX_Z
    print(f"pixel size in z: {REC_PIC_SIZE_Z} mm")

    print(f"full z reco space: {REC_MIN_Z} to {REC_MAX_Z}, npx: {REC_NPX_Z}")
    slice_z = REC_NPX_Z // 2  # middle slice
    REC_MIN_Z = slice_z * REC_PIC_SIZE_Z + REC_MIN_Z
    REC_MAX_Z = REC_MIN_Z + REC_PIC_SIZE_Z 
    REC_NPX_Z = 1
    print(f"slice {slice_z} reco space: {REC_MIN_Z} to {REC_MAX_Z}, npx: {REC_NPX_Z}")

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

    # NB: det_radius is not equal det_curvature_radius!
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SRC_RADIUS,
        det_radius=DET_RADIUS,
        det_curvature_radius=(DET_CURVATURE_RADIUS,None),  # uncomment for curved detector
        pitch=PITCH)
    
    geometry_old = geometry
    
    for k, angle in enumerate(angles):
        if geometry.det_refpoint(angle)[2] + DET_Z_MAX >= REC_MIN_Z:
            k_min = k - 1 if k > 0 else 0
            break
    for k, angle in enumerate(angles[::-1]):
        if geometry.det_refpoint(angle)[2] + DET_Z_MIN <= REC_MAX_Z:
            k_max = len(angles) - k + 1 if k > 0 else len(angles)
            break
    print(f"Using projections {k_min} to {k_max} for reconstruction of slice {slice_z}. Total {k_max - k_min + 1} projections.")

    angle_partition = odl.nonuniform_partition(
        angles[k_min:k_max+1])

    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SRC_RADIUS,
        det_radius=DET_RADIUS,
        det_curvature_radius=(DET_CURVATURE_RADIUS,None),  
        pitch=PITCH)

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
    ray_trafo_adjoint = ray_trafo.adjoint

    return ray_trafo, ray_trafo_adjoint, k_min, k_max, geometry_old

path = f'/home/rosaca/code/odl/examples/kats/bp_oneslice/'
if not os.path.exists(path):
    os.makedirs(path)

params, space, geometry = create_space_geom(curved=True)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

ellipsoids = [[1.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.4, 0.2, 0.7, 0.3, -0.2, 0.2, np.pi/4, np.pi/6, np.pi/3],
        [0.6, 0.5, 0.5, 0.5, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0],
        [-0.6, 0.3, 0.3, 0.3, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0]]
phantom= odl.phantom.ellipsoid_phantom(space, ellipsoids)

proj_data = ray_trafo(phantom)  
# save the angles and the sinogram in npy files
angles = geometry.angles
sinogram = proj_data.asarray()
np.save(os.path.join(path, 'angles.npy'), angles)
np.save(os.path.join(path, 'sinogram.npy'), sinogram)

# create metadata dictionary
metadata = {
    'DET_X_MIN': geometry.det_partition.min_pt[0],
    'DET_X_MAX': geometry.det_partition.max_pt[0],
    'DET_NPX_X': geometry.det_partition.shape[0],
    'DET_Z_MIN': geometry.det_partition.min_pt[1],
    'DET_Z_MAX': geometry.det_partition.max_pt[1],
    'DET_NPX_Z': geometry.det_partition.shape[1],
    'REC_MIN_X': space.min_pt[0],
    'REC_MIN_Y': space.min_pt[1],
    'REC_MIN_Z': space.min_pt[2],
    'REC_MAX_X': space.max_pt[0],
    'REC_MAX_Y': space.max_pt[1],
    'REC_MAX_Z': space.max_pt[2],
    'REC_NPX_X': space.shape[0],
    'REC_NPX_Y': space.shape[1],
    'REC_NPX_Z': space.shape[2],
    'SRC_RADIUS': geometry.src_radius,
    'DET_RADIUS': geometry.det_radius,
    'DET_CURVATURE_RADIUS': geometry.det_curvature_radius,
    'PITCH': geometry.pitch,
    'REC_PIC_SIZE': space.cell_sides[0],  # assuming isotropic voxel size
    }

forward_operator, backward_operator, k_min, k_max, geometry_old = make_operators(angles, metadata)
sinogram = sinogram[k_min:k_max+1, :, :]
angles = angles[k_min:k_max+1]
print('Angles and sinogram windowed to the reconstruction range.')

# filt_op_fbp = odl.tomo.fbp_filter_op(forward_operator, filter_type='Ram-Lak', frequency_scaling=0.8) * odl.tomo.tam_danielson_window(forward_operator)
# filt_sino_fbp = filt_op_fbp(sinogram)
# np.save(path + 'fbp_filt_data', filt_sino_fbp)
# print("Filtered FBP sinogram saved.")

filt_op_kats = odl.tomo.katsevich_filter_op(forward_operator)
filt_sino_kats = filt_op_kats(sinogram)
np.save(path + 'kats_filt_data', filt_sino_kats)
print("Filtered KATS sinogram saved.")

# KATS Backprojection
# precompute the voxel weights, it is a rec_npx_x * rec_npx_y * 1 array and depends on the projection angle
# voxel_weights
# float scale_coeff = scale_const*(scan_radius + X*ev_x + Y*ev_y);


# now do the backprojection for each angle
# reco empty 3d array

for k, angle in enumerate(angles):
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
    ev_x = geometry_old.det_refpoint(angle)[0]
    ev_y = geometry_old.det_refpoint(angle)[1]
    i = np.arange(nx).reshape(-1, 1)
    j = np.arange(ny).reshape(1, -1)
    X = xsize * i + metadata['REC_MIN_X'] + 0.5 * xsize
    Y = ysize * j + metadata['REC_MIN_Y'] + 0.5 * ysize
    scale_coeff = 1. / (geometry.src_radius + X * ev_x + Y * ev_y)
    voxel_weights = np.broadcast_to(scale_coeff, (nx, ny, nz))
    if k == 0:
        print(f"Voxel weights shape: {voxel_weights.shape}, min: {voxel_weights.min()}, max: {voxel_weights.max()}")
        print(voxel_weights[:10, :10, :])

    print(f"Backprojecting angle {k + k_min} of {k_max - k_min + 1}")
    backproj = backward_operator(filt_sino_kats[k, :, :])
    backproj *= voxel_weights
    if k == 0:
        reco = backproj
    else:
        reco += backproj

# save the reconstruction
reco.show(path=path + 'reco_kats.png', title='KATS Backprojection Reconstruction')