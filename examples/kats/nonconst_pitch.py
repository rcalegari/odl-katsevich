'''Test the Katsevich algorithm with non-constant pitch.
   Not completely correct yet but results not too bad.'''


import numpy as np
import odl
from tqdm import tqdm
from functools import partial


from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows
from examples.kats.pykats_config import create_pykats_config, create_space_geom
from examples.kats.test_curved import katsevich_filter_curved
from examples.kats.test_flat import katsevich_filter_flat
from examples.kats.utils import td_window, td_window_nonconst_pitch

curved = True
not_const = True

# create an array that has len(angles) 3d vector, the first 2 dimensions always zero,
# while the third one oscillates smoothly bewteen -0.1 and 1.

shifts = 1e-10 * np.ones((256, 3))
if not_const:
    shifts[:, 2] = 0.2 * np.sin(np.linspace(0, 10 * np.pi, 256))
# make shifts a random smooth function that ranges between -1 and 1
# shifts[:, 2] = np.random.uniform(-0.1, 0.1, size=256)

# Create the geometry and space
print('computing new config')
params, space, geometry = create_space_geom(curved=curved, shift_func=shifts, config='simple')
'''
# params, geometry, space, phantom, ray_trafo, sino, kat_conf = create_pykats_config(curved=curved, test=3, shifts=shifts)
# def plot_safe_src():
#     from scipy.interpolate import interp1d
#     from scipy.interpolate import UnivariateSpline
#     angles = geometry.angles
#     positions = geometry.src_position(angles)

#     # plot in three subplots the x, y, z coordinates of the source positions
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.plot(angles, positions[:, 0], label='x')
#     plt.xlabel('Angle (rad)')
#     plt.ylabel('x position (m)')
#     plt.title('Source X Position')
#     plt.grid()
#     plt.subplot(1, 3, 2)
#     plt.plot(angles, positions[:, 1], label='y', color='orange')
#     plt.xlabel('Angle (rad)')
#     plt.ylabel('y position (m)')
#     plt.title('Source Y Position')
#     plt.grid()
#     plt.subplot(1, 3, 3)
#     plt.plot(angles, positions[:, 2], label='z', color='green')
#     plt.xlabel('Angle (rad)')
#     plt.ylabel('z position (m)')
#     plt.title('Source Z Position')
#     plt.grid()
#     plt.tight_layout()
#     path = '/home/rosaca/code/odl/examples/kats/pitch/'
#     plt.savefig(path+'source_positions.png')

# plot_safe_src()
'''
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')



# phantom = odl.phantom.shepp_logan(space, modified=True)
ellipsoids = [
            # Outer ellipsoid of hollow shell
            [1., 0.5, 0.5, 0.5, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0],
            # Inner ellipsoid of hollow shell (subtracted)
            [-1., 0.3, 0.3, 0.3, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0]
        ]
phantom = odl.phantom.ellipsoid_phantom(space, ellipsoids)
# phantom = odl.phantom.shepp_logan(space, modified=True)
sino = ray_trafo(phantom)

path = '/home/rosaca/code/odl/examples/kats/pitch/nonconst/'
import os
if not os.path.exists(path):
    os.makedirs(path)

td_window_nonconst_pitch(ray_trafo, path=path)

if curved:

    filtered_data = katsevich_filter_curved(geometry, space, ray_trafo=ray_trafo, proj_data=sino, phantom=phantom, video=True, path=path, params=params, step=5, interval=1)
    reco_data = ray_trafo.adjoint_kats(filtered_data)
    reco_data = reco_data.asarray()
    error = phantom - reco_data
    save_xy_video(reco_data, path, 'kats_reco_curv', step=10, interval=1, save_slice=True, clip=True)
    save_xy_video(error, path, 'kats_error_curv', step=10, interval=1, save_slice=True, clip=True)
    save_xy_video(phantom, path, 'phantom', step=10, interval=1, save_slice=True)
    save_intensity_error_rows(phantom, reco_data, 3, path)
    save_intensity_row(phantom, reco_data, 3, path)


