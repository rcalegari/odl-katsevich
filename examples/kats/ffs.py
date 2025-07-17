'''Test the Katsevich algorithm with ffs.'''

import numpy as np
import odl
from tqdm import tqdm
from functools import partial
import os
from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows
from examples.kats.pykats_config import create_pykats_config, create_space_geom
from examples.kats.test_curved import katsevich_filter_curved
from examples.kats.test_flat import katsevich_filter_flat

curved = True

def plot_src(geometry, path):

    pos = geometry.src_position(geometry.angles)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(3, 1)
    ax[0].plot(geometry.angles[30:50], pos[30:50, 0])
    ax[0].set_title('X')
    ax[2].set_xlabel('Angle (rad)')
    ax[1].set_ylabel('Position (mm)')
    ax[1].plot(geometry.angles[30:50], pos[30:50, 1])
    ax[1].set_title('Y')
    ax[2].plot(geometry.angles[30:50], pos[30:50, 2])
    ax[2].set_title('Z')
    plt.tight_layout()
    plt.savefig(path+'src_coords.svg')

def plot_det(geometry, path):

    pos = geometry.det_refpoint(geometry.angles)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(3, 1)
    ax[0].plot(geometry.angles[30:50], pos[30:50, 0])
    ax[0].set_title('X')
    ax[2].set_xlabel('Angle (rad)')
    ax[1].set_ylabel('Position (mm)')
    ax[1].plot(geometry.angles[30:50], pos[30:50, 1])
    ax[1].set_title('Y')
    ax[2].plot(geometry.angles[30:50], pos[30:50, 2])
    ax[2].set_title('Z')
    plt.tight_layout()
    plt.savefig(path+'det_coords.svg')

# Create the geometry and space
print('computing new config')
params, space, geometry = create_space_geom(curved=curved, ffs=True, config='simple')

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
ellipsoids = [[1., 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, np.pi/8, 0.0]]
phantom = odl.phantom.ellipsoid_phantom(space, ellipsoids)
sino = ray_trafo(phantom)

path = '/home/rosaca/code/odl/examples/kats/ffs/'
# Ensure the path exists and if not create it
if not os.path.exists(path):
    os.makedirs(path)
#plot coords src and det
plot_src(geometry, path)
plot_det(geometry, path)
# save sino
save_uv_video(sino, path, 'sinogram', step=5, interval=1)

curved=False
if curved:
    filtered_data = katsevich_filter_curved(geometry, space, ray_trafo=ray_trafo, proj_data=sino, phantom=phantom, video=False, path=path, params=params, step=5, interval=1)
    reco_data = ray_trafo.adjoint_kats(filtered_data)
    reco_data = reco_data.asarray()
    error = phantom - reco_data
    save_xy_video(reco_data, path, 'kats_reco_curv', step=10, interval=1, save_slice=True, clip=True)
    save_xy_video(error, path, 'kats_error_curv', step=10, interval=1, save_slice=True, clip=True)
    save_xy_video(phantom, path, 'phantom', step=10, interval=1, save_slice=True)
    # save_intensity_error_rows(phantom, reco_data, 3, path)
    save_intensity_row(phantom, reco_data, 3, path)

