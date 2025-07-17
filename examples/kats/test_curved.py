import os
import numpy as np
import odl
from examples.kats.utils import save_uv_video, save_xy_video
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
 
def fbp_filter(geometry, space, phantom = None, params=None, video=False, path=None, step=10, interval=1):
    '''
    Returns filtered data using the FBP filter.
    If video is True, saves videos.
    '''
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__)) + '/fbp/'
        print('no path provided, using default: ', path)
    os.makedirs(path, exist_ok=True)
    if phantom is None:
        phantom = odl.phantom.shepp_logan(space, modified=True)

    # create ray trafo and project data
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl = 'astra_cuda')
    proj_data = ray_trafo(phantom)

    # filter data
    filter_fbp = odl.tomo.fbp_filter_op(ray_trafo, filter_type='Hamming', frequency_scaling=0.8)
    filtered_data = filter_fbp(proj_data)

    if geometry.det_curvature_radius is None:
        det = 'flat'
    else:
        det = 'curved'

    if video and params:
        save_uv_video(filtered_data, path, f'fbp_{det}_filt_data', params, step=step, interval=interval, save_slice=True)
    
    return filtered_data

def katsevich_filter_curved(geometry, space, ray_trafo=None, proj_data=None, phantom=None,  params=None, video=False, path=None, step=10, interval=1):
    '''
    Returns kats filtered data. 
    If video is True, saves videos.
    '''
    if phantom is None:
        phantom = odl.phantom.shepp_logan(space, modified=True)
    if proj_data is None:
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl = 'astra_cuda')
        proj_data = ray_trafo(phantom)    
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))+ '/comparison/curved/'
        print('no path provided, using default: ', path)
    os.makedirs(path, exist_ok=True)

    # Create Katsevich operator 
    kats_filter = odl.tomo.katsevich_filter_op(ray_trafo)
    kats = kats_filter(proj_data) # shape (N_s, N_alpha, N_w)
    g1 = kats_filter.g1
    g2 = kats_filter.g2
    g3 = kats_filter.g3
    g4 = kats_filter.g4
    g5 = kats_filter.g5
    g6 = kats_filter.g6
    gF = kats_filter.gF

    if video:
        save_uv_video(proj_data, path, 'curved_sino', step=step, interval=interval, axes=(r'$\alpha$', 'w'))
        
        if params and g1.shape[1] == params['DET_NPX_Z']: # put detector cols in x axis
            g1 = g1.transpose(0, 2, 1) 
            g2 = g2.transpose(0, 2, 1)
            g3 = g3.transpose(0, 2, 1)
            g4 = g4.transpose(0, 2, 1)
            g5 = g5.transpose(0, 2, 1)
            gF = gF.transpose(0, 2, 1)

        save_uv_video(g1, path,'g1_curv', step=step, interval=interval)
        save_uv_video(g2, path,'g2_curv', step=step, interval=interval)
        save_uv_video(g3, path,'g3_curv', step=step, interval=interval, axes=(r'$\alpha$', r'$\psi$'))
        save_uv_video(g4, path,'g4_curv', step=step, interval=interval, axes=(r'$\alpha$', r'$\psi$'))
        save_uv_video(g5, path,'g5_curv', step=step, interval=interval)
        save_uv_video(g6, path,'g6_curv', step=step, interval=interval)
        save_uv_video(gF, path, 'gF_curv', step=step, interval=interval)

    return kats
