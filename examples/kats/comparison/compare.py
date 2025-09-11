'''compare pykatsevich and odl:
   - space
   - geometry
   - phantom
   - sinogram

   You need the pykatsevich files.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import odl
import sys
import yaml
import matplotlib.gridspec as gridspec

from  examples.kats.utils import save_uv_video

t = 3
curved = False

# Paths
output_path = '/home/rosaca/code/odl/examples/kats/comparison/'
os.makedirs(output_path, exist_ok=True)
pykatsevich_path = '/home/rosaca/code/helical-kats/pykatsevich'
test_path = '/home/rosaca/code/helical-kats/tests'

# Load initialize.py
spec_init = importlib.util.spec_from_file_location("initialize", os.path.join(pykatsevich_path, 'initialize.py'))
initialize = importlib.util.module_from_spec(spec_init)
spec_init.loader.exec_module(initialize)
sys.path.append('/home/rosaca/code/helical-kats')

# Load common.py
spec_phantom = importlib.util.spec_from_file_location("common", os.path.join(test_path, 'common.py'))
common = importlib.util.module_from_spec(spec_phantom)
spec_phantom.loader.exec_module(common)

# load data from yaml file
with open(os.path.join(test_path, f'test0{t}.yaml'), 'r') as f:
    yaml_settings = yaml.safe_load(f)

phantom_settings = yaml_settings['phantom']
voxel_size = phantom_settings['voxel_size']
geom = yaml_settings['geometry']
shape = (phantom_settings['columns'], phantom_settings['rows'], phantom_settings['slices'])
angles_per_turn = geom['helix']['angles_count'] * 2 * np.pi / geom['helix']['angles_range'] 
kat_conf = initialize.create_configuration(geom, {
    'GridColCount': shape[0],
    'GridRowCount': shape[1],
    'GridSliceCount': shape[2],
    'option': {
        'WindowMinX': -shape[0]*voxel_size/2,
        'WindowMaxX':  shape[0]*voxel_size/2,
        'WindowMinY': -shape[1]*voxel_size/2,
        'WindowMaxY':  shape[1]*voxel_size/2,
        'WindowMinZ': -0.5 * geom['helix']['pitch_mm_rad'] * geom['helix']['angles_range'],
        'WindowMaxZ':  0.5 * geom['helix']['pitch_mm_rad'] * geom['helix']['angles_range'],
    }
})

# --- CREATE PHANTOM AND PROJECTION --- #

# phantom pykats
phantom_pykats = common.phantom_objects_3d(
    phantom_settings['rows'],
    phantom_settings['columns'],
    phantom_settings['slices'],
    voxel_size=voxel_size,
    objects_list=phantom_settings['objects'])


if __name__ == '__main__':

    sinogram_pykats, vol_geom, proj_geom, vertical_shift = common.project(phantom_pykats, voxel_size, geom)
    # vol_geom = astra.create_vol_geom(phantom_pykats.shape[0], phantom_pykats.shape[1], phantom_pykats.shape[2], -phantom_pykats.shape[1]*voxel_size*0.5, phantom_pykats.shape[1]*voxel_size*0.5,    # along X
    #     -phantom_pykats.shape[0]*voxel_size*0.5, phantom_pykats.shape[0]*voxel_size*0.5,    # along Y
    #     -phantom_pykats.shape[2]*voxel_size*0.5, phantom_pykats.shape[2]*voxel_size*0.5)
    space = odl.uniform_discr(min_pt=[
                                    vol_geom["option"]["WindowMinZ"],
                                    vol_geom["option"]["WindowMinY"],
                                    vol_geom["option"]["WindowMinX"]],
                                max_pt=[
                                    vol_geom["option"]["WindowMaxZ"],
                                    vol_geom["option"]["WindowMaxY"],
                                    vol_geom["option"]["WindowMaxX"]],
                                shape=[
                                    vol_geom["GridSliceCount"], # slice
                                    vol_geom["GridRowCount"], # row
                                    vol_geom["GridColCount"]], # cols
                                dtype='float32')
    
    vol_geom_from_odl = odl.tomo.backends.astra_setup.astra_volume_geometry(space)
    print('Are pykats and ODL vol_geoms equal?')
    print(vol_geom)
    print(vol_geom_from_odl)

    params = {"SRC_RADIUS": geom['SOD'],
        "DET_RADIUS": geom['SDD'] - geom['SOD'],
        "PITCH": geom['helix']['pitch_mm_rad'] * 2 * np.pi,
        "DET_X_MIN": - geom['detector']['detector cols'] / 2 * geom['detector']['detector psize'],
        "DET_X_MAX":   geom['detector']['detector cols'] / 2 * geom['detector']['detector psize'],
        "DET_Z_MIN": - geom['detector']['detector rows'] / 2 * geom['detector']['detector psize'], 
        "DET_Z_MAX": + geom['detector']['detector rows'] / 2 * geom['detector']['detector psize'],
        "DET_NPX_X": geom['detector']['detector cols'], 
        "DET_NPX_Z": geom['detector']['detector rows'],
        "REC_MIN_X": -shape[0] * voxel_size / 2,
        "REC_MAX_X": shape[0] * voxel_size / 2,
        "REC_MIN_Y": -shape[1] * voxel_size / 2,
        "REC_MAX_Y": shape[1] * voxel_size / 2,
        "REC_MIN_Z": kat_conf['z_min'],
        "REC_MAX_Z": kat_conf['z_max'],
        "REC_NPX_X": shape[0],
        "REC_NPX_Y": shape[1],
        "REC_NPX_Z": shape[2],
        "ANGLES_PER_TURN": angles_per_turn,
        "DET_CURVATURE_RADIUS": None if not curved else geom['SDD'],
        "DET_PIXEL_SIZE": geom['detector']['detector psize']
        }
    
    if curved:
        params['DET_X_MIN'] = np.arctan(params['DET_X_MIN'] / params['DET_CURVATURE_RADIUS'])
        params['DET_X_MAX'] = np.arctan(params['DET_X_MAX'] / params['DET_CURVATURE_RADIUS'])
        curv = (params['DET_CURVATURE_RADIUS'], None)
        det_type = 'curved'
        x_axis = r'$\alpha$'
        min_pt = (params["DET_X_MIN"], params["DET_Z_MIN"]) # see astra_setup.py
        max_pt = (params["DET_X_MAX"], params["DET_Z_MAX"])
        shape_dpart = (params["DET_NPX_X"], params["DET_NPX_Z"])
        det_axes = [(0, 1, 0), (0, 0, 1)]
    else:
        curv = None
        det_type = 'flat'
        x_axis = 'u'
        min_pt = (params["DET_Z_MIN"], params["DET_X_MIN"]) # see astra_setup.py
        max_pt = (params["DET_Z_MAX"], params["DET_X_MAX"])
        shape_dpart = (params["DET_NPX_Z"], params["DET_NPX_X"])
        det_axes = [(0, 0, 1), (0, 1, 0)]
        # for how proj_geom is created,
        # rows are on first axis and cols on second (for flat)


    s_len = geom['helix']['angles_range']
    s_min = -s_len * 0.5
    s_max =  s_len * 0.5
    delta_s = 2 * np.pi / angles_per_turn # Turn in radians per projection 
    angles = s_min + delta_s * (np.arange(geom['helix']["angles_count"], dtype=np.float32) + 0.5 )  # only with nodes_on_bdry=True

    # apart = odl.uniform_partition(angles[0], angles[-1], N_s, nodes_on_bdry=True)
    apart = odl.nonuniform_partition(angles)
    dpart = odl.uniform_partition(min_pt, max_pt, shape_dpart)

    geometry = odl.tomo.ConeBeamGeometry(apart=apart,
                                         dpart=dpart,
                                         src_radius=params["SRC_RADIUS"],
                                         det_radius=params["DET_RADIUS"],
                                         det_curvature_radius=curv,
                                         pitch=params["PITCH"],
                                         src_to_det_init=(-1, 0, 0),
                                         det_axes_init=det_axes,
                                         vertical_shift=vertical_shift)
    # create transform operator
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
 
    # update path
    output_path = output_path + f'/{det_type}/t0{t}/'
    os.makedirs(output_path, exist_ok=True)

    # check that projection geometry is correct (to check if geometry is correct)
    proj_geom_from_odl = odl.tomo.backends.astra_setup.astra_projection_geometry(geometry)
    plot_vectors = True
    if plot_vectors:
        vecs1 = proj_geom['Vectors']
        vecs2 = proj_geom_from_odl['Vectors']
        fig, ax = plt.subplots(3, 1, figsize=(10, 6))
        ax[0].plot(vecs1[:, 0], label='vecs1')
        ax[0].plot(vecs2[:, 0], label='vecs2')
        ax[0].legend()
        ax[0].set_title('vecs[0] = z position')
        ax[1].plot(vecs1[:, 1])
        ax[1].plot(vecs2[:, 1])
        ax[1].set_title('vecs[1] = y position')
        ax[2].plot(vecs1[:, 2])
        ax[2].plot(vecs2[:, 2])
        ax[2].set_title('vecs[2] = x position')
        plt.tight_layout()
        plt.savefig(output_path + f'vecs1_vecs2_{det_type}.png')
        plt.close()
        diff_vecs = np.abs(vecs1[:, 0:3] - vecs2[:, 0:3])
        fig, ax = plt.subplots(3, 1, figsize=(10, 6))
        ax[0].plot(diff_vecs[:, 0])
        ax[0].set_title('vecs[0] = z position')
        ax[1].plot(diff_vecs[:, 1])
        ax[1].set_title('vecs[1] = y position')
        ax[2].plot(diff_vecs[:, 2])
        ax[2].set_title('vecs[2] = x position')
        plt.tight_layout()
        plt.savefig(output_path + f'vec_error_{det_type}.png')
        plt.close()

    print('\nAre pykats and ODL proj_geoms equal?')
    print('pykats-->', 'type:', proj_geom['type'], 'det row count:', proj_geom['DetectorRowCount'], 'det col count:', proj_geom['DetectorColCount'])
    print('ODL-->', 'type:', proj_geom_from_odl['type'], 'det row count:', proj_geom_from_odl['DetectorRowCount'], 'det col count:', proj_geom_from_odl['DetectorColCount'])
    print('diff vectors:', np.abs(vecs1[:, 0] - vecs2[:, 0]).max(), np.abs(vecs1[:, 1] - vecs2[:, 1]).max(), np.abs(vecs1[:, 2] - vecs2[:, 2]).max())
    # proj_geoms equal

    # (rows, columns, slices) -> needs to be transposed
    phantom_pykats = common.phantom_objects_3d(phantom_settings['rows'],
                                                phantom_settings['columns'],
                                                phantom_settings['slices'],
                                                voxel_size=voxel_size,
                                                objects_list=phantom_settings['objects'])
    # (slices, rows, columns)
    phantom_odl = space.element(np.transpose(phantom_pykats, (2, 0, 1))) 
        
    sinogram_odl = ray_trafo(phantom_odl).asarray() # (slices, rows, columns)

    # sinogram_odl.shape    = (slices, rows, columns) 
    # sinogram_pykats.shape = (rows, slices, columns) 
    # transpose odl to match pykats
    sinogram_pykats_T = np.transpose(sinogram_pykats, (1, 0, 2) if curv is None else (1, 2, 0)) # 1, 0, 2

    diff = sinogram_odl - sinogram_pykats_T

    if np.shape(diff[0])[0] == geom['detector']['detector rows']:
        axes = (x_axis, 'w')
    else:
        axes = ('w', x_axis)
    save_uv_video(np.transpose(diff, (0, 2, 1)), output_path, f'error_sino_{det_type}', step=10, cmap='grey', axes=axes, save_slice=True)







