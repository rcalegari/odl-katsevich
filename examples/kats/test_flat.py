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


def katsevich_filter_flat(geometry, space, ray_trafo, proj_data=None, phantom=None, video=False, path=None, conf=None, step=10, interval=1, params=None, diff=None):
    '''
    Returns kats filtered data for flat geometry.
    If video is True, saves the intermediate results as videos.
    If conf is given, compares to the pykats results.
    '''
    if phantom is None:
        phantom = odl.phantom.shepp_logan(space, modified=True) # (z, y, x) shape
    if proj_data is None:
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl = 'astra_cuda')
        proj_data = ray_trafo(phantom) # (N_s, N_w, N_u)
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))+ '/comparison/flat/'

    # Create Katsevich operator 
    kats_filter = odl.tomo.katsevich_filter_op(ray_trafo)     
    kats = kats_filter(proj_data, diff=diff) # shape (N_s, N_w, N_u)

    if video:
        g1 = kats_filter.g1
        g3 = kats_filter.g3
        g4 = kats_filter.g4
        g5 = kats_filter.g5
        gF = kats_filter.gF
        gF = np.asarray(gF)

        proj_data_arr = np.asarray(proj_data, order='C')
        if params and proj_data_arr.shape[1] == params['DET_NPX_Z']: # rows 
            proj_data_arr = np.transpose(proj_data_arr, (0, 2, 1)) # we want shape (N_s, N_u, N_w)
        save_uv_video(proj_data_arr, path, 'flat_sino', step=step, interval=interval, axes=('u', 'w'))
        
        if conf is not None:
            sino_diff = filter.differentiate(proj_data, conf)
            sino_rebin = filter.fw_height_rebinning(sino_diff, conf)
            hilbert_array = filter.compute_hilbert_kernel(conf)
            sino_heilbert_trans = filter.hilbert_conv(sino_rebin, hilbert_array, conf)
            sino_reverse_rebin = filter.rev_rebin_vec(sino_heilbert_trans, conf)
            sino_td = filter.sino_weight_td(sino_reverse_rebin, conf, False)

            g1_diff = g1 - sino_diff
            g3_diff = g3 - sino_rebin
            g4_diff = g4 - sino_heilbert_trans
            g5_diff = g5 - sino_reverse_rebin
            gF_diff = gF - sino_td

        if params and g1.shape[1] == params['DET_NPX_Z']: # put detector cols on x axis
            g1 = np.transpose(g1, (0, 2, 1))
            g3 = np.transpose(g3, (0, 2, 1))
            g4 = np.transpose(g4, (0, 2, 1))
            g5 = np.transpose(g5, (0, 2, 1))
            gF = np.transpose(gF, (0, 2, 1))
            if conf is not None:
                g1_diff = np.transpose(g1_diff, (0, 2, 1))
                g3_diff = np.transpose(g3_diff, (0, 2, 1))
                g4_diff = np.transpose(g4_diff, (0, 2, 1))
                g5_diff = np.transpose(g5_diff, (0, 2, 1))
                gF_diff = np.transpose(gF_diff, (0, 2, 1))
        save_uv_video(g1, path,'g1_flat', step=step, interval=interval, axes=('u', 'w'))
        save_uv_video(g3, path,'g3_flat', step=step, interval=interval, axes=('u', 'psi'))
        save_uv_video(g4, path,'g4_flat', step=step, interval=interval, axes=('u', 'psi'))
        save_uv_video(g5, path,'g5_flat', step=step, interval=interval, axes=('u', 'w'))
        save_uv_video(gF, path, 'gF_flat', step=step, interval=interval, axes=('u', 'w'))
        if conf is not None:
            save_uv_video(g1_diff, path,'g1_diff_flat', step=step, interval=interval, axes=('u', 'w'))
            save_uv_video(g3_diff, path,'g3_diff_flat', step=step, interval=interval, axes=('u', 'psi'))
            save_uv_video(g4_diff, path,'g4_diff_flat', step=step, interval=interval, axes=('u', 'psi'))
            save_uv_video(g5_diff, path,'g5_diff_flat', step=step, interval=interval, axes=('u', 'w'))
            save_uv_video(gF_diff, path, 'gF_diff_flat', step=step, interval=interval, axes=('u', 'w'))
    
    return kats

def compare_reco_flat(geometry, space, ray_trafo, filt_data, phantom, video=False, path=None, conf=None, step=10, interval=1, clip=False):
    ''' 
    Returns the Kats reconstruction for flat geometry.
    If video is True, saves the result as videos.
    If conf is given, compares to the pykats results.

    filt_data and phantom should be odl objects.
    '''
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))+ '/comparison/'
    kats_reco = ray_trafo.adjoint_kats(filt_data)
    kats_reco = kats_reco.asarray()
    # backprojection pykats
    vol_geom = odl.tomo.backends.astra_setup.astra_volume_geometry(space)
    proj_geom = odl.tomo.backends.astra_setup.astra_projection_geometry(geometry)
    filt_data = filt_data.asarray() # pykats expects (N_s, N_w, N_u))

    if conf is not None:
        proj_data = ray_trafo(phantom) 
        sino_diff = filter.differentiate(proj_data, conf)
        sino_rebin = filter.fw_height_rebinning(sino_diff, conf)
        hilbert_array = filter.compute_hilbert_kernel(conf)
        sino_heilbert_trans = filter.hilbert_conv(sino_rebin, hilbert_array, conf)
        sino_reverse_rebin = filter.rev_rebin_vec(sino_heilbert_trans, conf)
        sino_td = filter.sino_weight_td(sino_reverse_rebin, conf, False)

        pykats_reco = filter.backproject_a(sino_td, conf, vol_geom, proj_geom)
        pykats_reco = np.transpose(pykats_reco, (2, 0, 1))
        error = kats_reco - pykats_reco
    

    if video:
        save_xy_video(phantom, path, 'phantom', step=step, interval=interval, axes=('x', 'y'), clip=clip, save_slice=True)
        save_xy_video(kats_reco, path, 'kats_reco_flat', step=step, interval=interval, clip=clip, save_slice=True)
        if conf is not None:
            save_xy_video(error, path, 'reco_diff_flat', step=step, interval=interval, clip=clip)
        save_xy_video(kats_reco - phantom.asarray(), path, 'kats_error_flat', step=step, interval=interval, clip=clip, save_slice=True)
    return kats_reco

