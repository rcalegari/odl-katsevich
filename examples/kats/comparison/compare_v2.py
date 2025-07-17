'''
Compare the filter and backprojection results 
of ODL and pykats for flat geometry.

You need the pyykatsevich files.
'''

import odl

from examples.kats.pykats_config import create_pykats_config
from examples.kats.test_flat import katsevich_filter_flat, compare_reco_flat

step = 10
interval = 1
test = 1


params, geometry, space, phantom, ray_trafo, sinogram_odl, conf = create_pykats_config(curved=False, test=test)

path = f'/home/rosaca/code/odl/examples/kats/comparison/flat/t0{test}/'
# my filter (flat)
filt_data = katsevich_filter_flat(geometry, space, ray_trafo, sinogram_odl, phantom, video=True, path=path, conf=conf, step=step, interval=interval, params=params, 
                                    diff=1) # use v1 of the differentiation step
# backprojection
reco = compare_reco_flat(geometry, space, ray_trafo, filt_data, phantom, video=True, path=path, conf=conf, step=step, interval=interval, clip=False)


           