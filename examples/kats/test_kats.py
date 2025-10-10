'''
Script to test the Katsevich algorithm implementation in ODL
on artificial data

modify path, pykatsevich_path and test_path to your local repos!!
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
# test number from pykats files, to test simple phantom
# for complex phantoms, fixed test case 102
# check create_pykats_config in pykats_config.py for details
video = False
new = False
# new to recompute kats and fbp reco even if npy files already exist
num_angles = 300
# number of views on source trajectory
save_videos = False

det = 'curved' if curved else 'flat'

# build space/geometry/phantom based on phantom type
if phantom_type == 'simple':
    path = f'/home/rosaca/odl_remote_repo/odl-katsevich/examples/kats/phantom_simple/{det}/'
    params, geometry, space, phantom, ray_trafo, sinogram, conf = create_pykats_config(curved=curved, test=t)

elif phantom_type == 'shepp':
    path = f'/home/rosaca/odl_remote_repo/odl-katsevich/examples/kats/phantom_complex/{phantom_type}/{det}/{num_angles}/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    testnr = 102
    params, geometry, space, phantom, ray_trafo, sinogram, conf = create_pykats_config(curved=curved, test=testnr, complex_phantom='shepp')
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    phantom = odl.phantom.shepp_logan(space, modified=True)

elif phantom_type == 'ellipsoid':
    path = f'/home/rosaca/code/odl/examples/kats/phantom_complex/{phantom_type}/{det}/{num_angles}'
    if not os.path.exists(path):
        os.makedirs(path)
    params, geometry, space, phantom, ray_trafo, sinogram, conf = create_pykats_config(curved=curved, test=102, complex_phantom='ellipsoid')
    # ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    
    # ellipsoids = [[1.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.8, 0.4, 0.2, 0.7, 0.3, -0.2, 0.2, np.pi/4, np.pi/6, np.pi/3],
    #         [0.6, 0.5, 0.5, 0.5, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0],
    #         [-0.6, 0.3, 0.3, 0.3, -0.4, 0.4, -0.3, 0.0, np.pi/8, 0.0]]
    # phantom = odl.phantom.defrise(space, nellipses=4)
    # phantom= odl.phantom.ellipsoid_phantom(space, ellipsoids)

else:
    raise ValueError(f'Unknown phantom type: {phantom_type}')

# put in a text file in the path given the logs of the settings used and geometry used
with open(os.path.join(path, f'settings.txt'), 'w') as f:
    # f.write(f'test case: {t}\n')
    f.write(f'geometry: {geometry}\n')
    f.write(f'space: {space}\n')


print(f'Using {phantom_type} phantom with {det} detector. Using data from test=0{t}.')
phantom.show(saveto=os.path.join(path, f'phantom.svg'), coords=[None, None, 0])
phantom_arr = phantom.asarray()
proj_data = ray_trafo(phantom)
save_xy_video(phantom, path, 'phantom', step=step, interval=interval, save_slice=True)

# if npy fbp file exists, load it
if os.path.exists(os.path.join(path, f'fbp_reco.npy')) and not new:
    print(f'Loading FBP reco')
    fbp_reco = np.load(os.path.join(path, f'fbp_reco.npy'))
else:
    fbp_filt_op = odl.tomo.fbp_filter_op(ray_trafo, filter_type='Shepp-Logan')
    fbp_filt = fbp_filt_op(proj_data)
    if curved:
        w_fbp = odl.tomo.td_window_curved(ray_trafo, fbp_filt)
    else:
        w_fbp = odl.tomo.ff7_td_weighting(ray_trafo, fbp_filt)
    fbp_reco = ray_trafo.adjoint(w_fbp)
    fbp_reco.show(saveto=os.path.join(path, f'fbp_reco.svg'), coords=[None, None, 0])
    save_xy_video(fbp_reco, path, f'fbp_reco', step=step, interval=interval, save_slice=True)
    save_xy_video(fbp_reco - phantom_arr, path, f'fbp_error', step=step, interval=interval, save_slice=True)
    np.save(os.path.join(path, f'fbp_reco.npy'), fbp_reco.asarray())

'''
-------------------------------------
------------- FLAT CASE -------------
-------------------------------------
'''

if not curved:
    # if kats reco npy file exists, load it
    if os.path.exists(os.path.join(path, f'kats_reco.npy')) and not new:
        kats_reco = np.load(os.path.join(path, f'kats_reco.npy'))
        print('Load kats reco')
    else:
        filt_data = katsevich_filter_flat(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval, params=params)
        kats_reco = ray_trafo.adjoint_kats(filt_data)
        kats_reco.show(saveto=os.path.join(path, f'kats_reco.svg'), coords=[None, None, 0])
        kats_reco = kats_reco.asarray()
        np.save(os.path.join(path, f'kats_reco.npy'), kats_reco)
    
    if os.path.exists(os.path.join(path, f'kats_reco_v1.npy')) and not new:
        kats_reco_v1 = np.load(os.path.join(path, f'kats_reco_v1.npy'))
        print('Load kats reco v1')
    else:
        filt_data_v1 = katsevich_filter_flat(geometry, space, ray_trafo, proj_data, phantom, video=False, path=path, step=step, interval=interval, params=params, diff=1)
        kats_reco_v1 = ray_trafo.adjoint_kats(filt_data_v1)
        kats_reco_v1.show(saveto=os.path.join(path, f'kats_reco_v1.svg'), coords=[None, None, 0])
        kats_reco_v1 = kats_reco_v1.asarray()
        np.save(os.path.join(path, f'kats_reco_v1.npy'), kats_reco_v1)
'''
---------------------------------------
------------- CURVED CASE -------------
---------------------------------------
'''
else:
    if os.path.exists(os.path.join(path, f'kats_reco.npy')) and not new:
        kats_reco = np.load(os.path.join(path, f'kats_reco.npy'))
        print('Load kats reco')
    else:
        filt_data = katsevich_filter_curved(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval)
        kats_reco = ray_trafo.adjoint_kats(filt_data)
        kats_reco.show(saveto=os.path.join(path, f'kats_reco.svg'), coords=[None, None, 0])
        kats_reco = kats_reco.asarray()
        np.save(os.path.join(path, f'kats_reco.npy'), kats_reco)
    if os.path.exists(os.path.join(path, f'kats_reco_v1.npy')) and not new:
        kats_reco_v1 = np.load(os.path.join(path, f'kats_reco_v1.npy'))
        print('Load kats reco v1')
    else:
        filt_data_v1 = katsevich_filter_curved(geometry, space, ray_trafo, proj_data, phantom, video=False, path=path, step=step, interval=interval, diff=1)
        kats_reco_v1 = ray_trafo.adjoint_kats(filt_data_v1)
        kats_reco_v1.show(saveto=os.path.join(path, f'kats_reco_v1.svg'), coords=[None, None, 0])
        kats_reco_v1 = kats_reco_v1.asarray()
        error = kats_reco - phantom_arr
        np.save(os.path.join(path, f'kats_reco_v1.npy'), kats_reco_v1)

# save_xy_video(kats_reco, path, f'kats_reco', step=step, interval=interval, save_slice=True)
save_intensity_row(phantom, kats_reco, test=t, path=path, reco2=fbp_reco)
save_intensity_row(phantom, kats_reco_v1, test=t, path=path, reco2=fbp_reco, v1=True)
# save_intensity_error_derivative_comparison(phantom, kats_reco_v1, kats_reco, test=t, path=path)

# load npy files of kats and fbp reco and of phantom and compute PSNR and store it in settings.txt

# if fbp is not array convert it
if not isinstance(fbp_reco, np.ndarray):
    fbp_reco = np.array(fbp_reco)
with open(os.path.join(path, f'settings.txt'), 'a') as f:
    print(f'PSNR FBP: {compute_psnr(phantom_arr, fbp_reco)}')
    print(f'PSNR Kats K: {compute_psnr(phantom_arr, kats_reco)}')
    print(f'PSNR Kats NPH: {compute_psnr(phantom_arr, kats_reco_v1)}')
    f.write(f'PSNR FBP: {compute_psnr(phantom_arr, fbp_reco)}\n')
    f.write(f'PSNR Kats K: {compute_psnr(phantom_arr, kats_reco)}\n')
    f.write(f'PSNR Kats NPH: {compute_psnr(phantom_arr, kats_reco_v1)}\n')
    print(f'SSIM FBP: {compute_ssim(phantom_arr, fbp_reco)}')
    print(f'SSIM Kats K: {compute_ssim(phantom_arr, kats_reco)}')
    print(f'SSIM Kats NPH: {compute_ssim(phantom_arr, kats_reco_v1)}')
    f.write(f'SSIM FBP: {compute_ssim(phantom_arr, fbp_reco)}\n')
    f.write(f'SSIM Kats K: {compute_ssim(phantom_arr, kats_reco)}\n')
    f.write(f'SSIM Kats NPH: {compute_ssim(phantom_arr, kats_reco_v1)}\n')

# plot_psnr(phantom_arr, path, t=t, kats_reco=kats_reco, fbp_reco=fbp_reco, kats_reco_v1=kats_reco_v1)
# plot_ssim(phantom_arr, path, t=t, kats_reco=kats_reco, fbp_reco=fbp_reco, kats_reco_v1=kats_reco_v1)
# plot_rmse(phantom_arr, path, t=t, kats_reco=kats_reco, fbp_reco=fbp_reco, kats_reco_v1=kats_reco_v1)

if save_videos:
    if not curved:
        filt_data = katsevich_filter_flat(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval, params=params)
        kats_reco = ray_trafo.adjoint_kats(filt_data)
        save_xy_video(kats_reco, path, 'kats_reco', step=step, interval=interval)
        save_intensity_error_rows(phantom_arr, kats_reco, test=t, path=path)
        save_intensity_row(phantom, kats_reco, test=t, path=path)
    else:
        filt_data = katsevich_filter_curved(geometry, space, ray_trafo, proj_data, phantom, video=video, path=path, step=step, interval=interval)
        kats_reco = ray_trafo.adjoint_kats(filt_data)
        kats_reco = kats_reco.asarray()
        save_xy_video(kats_reco, path, 'kats_reco', step=step, interval=interval)
        save_intensity_row(phantom, kats_reco, test=t, path=path)
        save_intensity_error_rows(phantom_arr, kats_reco, test=t, path=path)
        
    