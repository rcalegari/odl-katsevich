'''
Script to test the Katsevich algorithm implementation in ODL
on artificial data
'''
from examples.kats.pykats_config import create_pykats_config
from examples.kats.test_flat import katsevich_filter_flat
from examples.kats.test_curved import katsevich_filter_curved
from examples.kats.utils import save_intensity_row, save_intensity_error_rows, save_intensity_error_derivative_comparison
from examples.kats.utils import plot_psnr, plot_ssim, plot_rmse

import os
import numpy as np
import odl
from examples.kats.utils import save_xy_video
import importlib.util

pykatsevich_path = '/home/rosaca/code/helical-kats/pykatsevich'
test_path = '/home/rosaca/code/helical-kats/tests/'

spec_init = importlib.util.spec_from_file_location("initialize", os.path.join(pykatsevich_path, 'initialize.py'))
initialize = importlib.util.module_from_spec(spec_init)
spec_init.loader.exec_module(initialize)

spec_phantom = importlib.util.spec_from_file_location("common", os.path.join(test_path, 'common.py'))
common = importlib.util.module_from_spec(spec_phantom)
spec_phantom.loader.exec_module(common)

spec_filter = importlib.util.spec_from_file_location("filter", os.path.join(pykatsevich_path, 'filter.py'))
filter = importlib.util.module_from_spec(spec_filter)
spec_filter.loader.exec_module(filter)


phantom_type = 'shepp' # 'shepp', 'ellipsoid', 'simple'
step = 10
interval = 1
curved = True
t = 2
video = False
new = False
new1 = False
num_angles = 300

det = 'curved' if curved else 'flat'

if phantom_type == 'simple':
    path = f'/home/rosaca/code/odl/examples/kats/phantom_simple/{det}/'
    # needs pykats files
    params, geometry, space, phantom, ray_trafo, sinogram, conf = create_pykats_config(curved=curved, test=t)

elif phantom_type == 'shepp':
    t = 3
    path = f'/home/rosaca/code/odl/examples/kats/phantom_complex/{phantom_type}/{det}/'
    if not os.path.exists(path):
        os.makedirs(path)
    params, space, geometry = create_space_geom(curved=True)
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    phantom = odl.phantom.shepp_logan(space, modified=True)

elif phantom_type == 'ellipsoid':
    t = 3
    path = f'/home/rosaca/code/odl/examples/kats/phantom_complex/{phantom_type}/{det}/'
    if not os.path.exists(path):
        os.makedirs(path)
    params, space, geometry = create_space_geom(curved=True)
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    
    ellipsoids = [[1.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.4, 0.2, 0.7, 0.3, -0.2, 0.2, np.pi/4, np.pi/6, np.pi/3],
            [0.6, 0.5, 0.5, 0.5, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0],
            [-0.6, 0.3, 0.3, 0.3, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0]]
    phantom= odl.phantom.ellipsoid_phantom(space, ellipsoids)

else:
    raise ValueError(f'Unknown phantom type: {phantom_type}')

print(f'Using {phantom_type} phantom with {det} detector. Using data from test=0{t}.')

phantom_arr = phantom.asarray()
proj_data = ray_trafo(phantom)
save_xy_video(phantom, path, 'phantom', step=step, interval=interval)

if not curved:
    filt_data = katsevich_filter_flat(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval, params=params)
    kats_reco = ray_trafo.adjoint_kats(filt_data)
    save_xy_video(kats_reco, path, 'kats_reco', step=step, interval=interval)
    save_xy_video(kats_reco - phantom_arr, path, 'kats_error', step=step, interval=interval)
    save_intensity_error_rows(phantom_arr, kats_reco, test=t, path=path)
    save_intensity_row(phantom, kats_reco, test=t, path=path)
else:
    filt_data = katsevich_filter_curved(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval)
    kats_reco = ray_trafo.adjoint_kats(filt_data)
    kats_reco = kats_reco.asarray()
    error = kats_reco - phantom_arr
    save_xy_video(kats_reco, path, 'kats_reco', step=step, interval=interval)
    save_xy_video(error, path, 'kats_error', step=step, interval=interval)
    save_intensity_row(phantom, kats_reco, test=t, path=path)
    save_intensity_error_rows(phantom_arr, kats_reco, test=t, path=path)
    
    