"""Example using FBP in helical 3D geometry using `fbp_op`.

This FBP is theretically exact as it uses the Katsevich algorithm
"""

import numpy as np
import odl
import os
from functools import partial

def create_space_geom(curved=True, shift_func=None):

    # taken from pykats test03.yaml
    volCols = 560
    volRows = 540
    volSlices = 580
    voxel_size = 0.032

    angles_count = 512 # total number of angles in the helical trajectory
    angles_range = 7.084794 # in radians
    Rs = 50
    D = 80
    pitch_mm_rad = 5.25

    # detector
    detCols = 1056 
    detRows = 928 
    pixel_size = 0.078125 # detector pixel size in mm

    minX = -volCols * voxel_size / 2
    maxX = volCols * voxel_size / 2
    minY = -volRows * voxel_size / 2
    maxY = volRows * voxel_size / 2
    minZ = -volSlices * voxel_size / 2
    maxZ = volSlices * voxel_size / 2

    # print(f"({minZ}, {maxZ}) should be inside ({-0.5 * pitch_mm_rad * angles_range}, {0.5 * pitch_mm_rad * angles_range})")
    # assert that -0.5 * pitch_mm_rad * angles_range <= minZ <= maxZ <= 0.5 * pitch_mm_rad * angles_range, \
    
    assert -0.5 * pitch_mm_rad * angles_range <= minZ <= maxZ <= 0.5 * pitch_mm_rad * angles_range

    space = odl.uniform_discr(min_pt=[minZ, minY, minX],
                              max_pt=[maxZ, maxY, maxX],
                              shape=[volSlices, volRows, volCols], 
                              dtype='float32')
    
    angles_per_turn = angles_count * 2 * np.pi / angles_range

    params = {"SRC_RADIUS": Rs,
        "DET_RADIUS": D - Rs,
        "PITCH": pitch_mm_rad * 2 * np.pi,
        "DET_X_MIN": - detCols / 2 * pixel_size,
        "DET_X_MAX":   detCols / 2 * pixel_size,
        "DET_Z_MIN": - detRows / 2 * pixel_size, 
        "DET_Z_MAX":   detRows / 2 * pixel_size,
        "DET_NPX_X":   detCols, 
        "DET_NPX_Z":   detRows,
        "REC_MIN_X":   minX,
        "REC_MAX_X":   maxX,
        "REC_MIN_Y":   minY,
        "REC_MAX_Y":   maxY,
        "REC_MIN_Z":   -0.5 * pitch_mm_rad * angles_range,
        "REC_MAX_Z":    0.5 * pitch_mm_rad * angles_range,
        "REC_NPX_X": volCols,
        "REC_NPX_Y": volRows,
        "REC_NPX_Z": volSlices,
        "ANGLES_PER_TURN": angles_per_turn,
        "DET_CURVATURE_RADIUS": None if not curved else D,
        "DET_PIXEL_SIZE": pixel_size
        }
    
    if params['DET_CURVATURE_RADIUS'] is not None:
        params['DET_X_MIN'] = np.arctan(params['DET_X_MIN'] / params['DET_CURVATURE_RADIUS'])
        params['DET_X_MAX'] = np.arctan(params['DET_X_MAX'] / params['DET_CURVATURE_RADIUS'])
        curv = (params['DET_CURVATURE_RADIUS'], None)
        min_pt = (params["DET_X_MIN"], params["DET_Z_MIN"]) # see astra_setup.py
        max_pt = (params["DET_X_MAX"], params["DET_Z_MAX"])
        shape_dpart = (params["DET_NPX_X"], params["DET_NPX_Z"])
        det_axes = [(0, 1, 0), (0, 0, 1)]
    else:
        curv = None
        min_pt = (params["DET_Z_MIN"], params["DET_X_MIN"]) # see astra_setup.py
        max_pt = (params["DET_Z_MAX"], params["DET_X_MAX"])
        shape_dpart = (params["DET_NPX_Z"], params["DET_NPX_X"])
        det_axes = [(0, 0, 1), (0, 1, 0)]

    s_len = angles_range
    s_min = -s_len * 0.5
    delta_s = 2 * np.pi / angles_per_turn # Turn in radians per projection 
    angles = s_min + delta_s * (np.arange(angles_count, dtype=np.float32) + 0.5 )  # only with nodes_on_bdry=True
    # N_s = angles_count
    # apart = odl.uniform_partition(angles[0], angles[-1], N_s, nodes_on_bdry=True)
    
    apart = odl.nonuniform_partition(angles)
    dpart = odl.uniform_partition(min_pt, max_pt, shape_dpart)

    if shift_func is not None:
        shift_func = partial(odl.tomo.flying_focal_spot, apart=apart, shifts=shift_func)
    
    geometry = odl.tomo.ConeBeamGeometry(apart=apart,
                                         dpart=dpart,
                                         src_radius=params["SRC_RADIUS"],
                                         det_radius=params["DET_RADIUS"],
                                         det_curvature_radius=curv,
                                         pitch=params["PITCH"],
                                         src_shift_func=shift_func,
                                         det_shift_func=shift_func,
                                         src_to_det_init=(-1, 0, 0),
                                         det_axes_init=det_axes)
    
    return params, space, geometry

path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(path + '/pictures'):
    os.makedirs(path + '/pictures')

curved = False
det = 'c' if curved else 'f'
params, space, geometry = create_space_geom(curved=curved)

# --- Create Katsevich Filtered Back-projection operator --- #

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(space, geometry)

# Create phantom and sinogram data.
print('Create phantom and sinogram data...')
phantom = odl.phantom.shepp_logan(space, modified=True)
proj_data = ray_trafo(phantom)

phantom.show(saveto = path + '/pictures/phantom.png')
proj_data.show(saveto = path + f'/pictures/{det}_sino.png')

# fbp
print('Create FBP operator...')
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hamming', frequency_scaling=0.8)
windowed_fbp = fbp * odl.tomo.tam_danielson_window(ray_trafo)
fbp_reco = windowed_fbp(proj_data)
fbp_reco.show(saveto = path + f'/pictures/{det}_fbp_reco.png')
(phantom - fbp_reco).show(saveto = path + f'/pictures/{det}_fbp_error.png')

# kats
print('Create Katsevich operator...')
kats = odl.tomo.fbp_op(ray_trafo, filter_type='Katsevich')
kats_reconstruction = kats(proj_data)
kats_reconstruction.show(saveto = path + f'/pictures/{det}_kats_reco.png')
(phantom - kats_reconstruction).show(saveto = path + f'/pictures/{det}_kats_error.png')

(kats_reconstruction - fbp_reco).show(saveto = path + f'/pictures/{det}_diff.png')