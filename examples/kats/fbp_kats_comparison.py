from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows
from examples.kats.pykats_config import create_pykats_config, create_space_geom

import os
import numpy as np
import odl
from examples.kats.utils import save_xy_video
import importlib.util

curved = False
det = 'curved' if curved else 'flat'

params, space, geometry = create_space_geom(curved=curved) # using t03

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
phantom = odl.phantom.shepp_logan(space, modified=True)
sino = ray_trafo(phantom)

path = f'/home/rosaca/code/odl/examples/kats/fbp/{det}/'
if not os.path.exists(path):
    os.makedirs(path)

# phantom 
save_xy_video(phantom, path, 'phantom', step=10, interval=1, save_slice=True)

# fbp reconstruction
print("FBP reconstruction...")
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Shepp-Logan')
reco_fbp = fbp(sino)
save_xy_video(reco_fbp, path, 'fbp_reco', step=10, interval=1, save_slice=True)
error_fbp = reco_fbp - phantom
save_xy_video(error_fbp, path, 'fbp_error', step=10, interval=1)

# katsevich reconstruction
print("Katsevich reconstruction...")
kats = odl.tomo.fbp_op(ray_trafo, filter_type='Katsevich')
reco_kats = kats(sino)
save_xy_video(reco_kats, path, 'kats_reco', step=10, interval=1, save_slice=True)
error_kats = reco_kats - phantom
save_xy_video(error_kats, path, 'kats_error', step=10, interval=1)

diff = reco_fbp - reco_kats
save_xy_video(diff, path, 'diff', step=10, interval=1)

