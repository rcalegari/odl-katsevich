"""Reconstruct Mayo dataset using FBP and compare to reference recon.

Note that this example requires that projection and reconstruction data
of a CT scan from the Mayo dataset have been previously downloaded (see
[the webpage](https://ctcicblog.mayo.edu/hubcap/patient-ct-projection-data-library/))
and are stored in the locations indicated by "proj_dir" and "rec_dir".


In this example we only use a subset of the data for performance reasons.
The number of projections per patient varies in the full dataset. To get
a reconstruction of the central part of the volume, modify the indices in
the argument of the `mayo.load_projections` function accordingly.
"""
import numpy as np
import os
import odl
from odl.contrib.datasets.ct import mayo
from time import perf_counter
from examples.kats.utils import save_uv_video, save_xy_video, save_intensity_row, save_intensity_error_rows, td_window, td_window_nonconst_pitch, td_window_nonconst_pitch_video

reco_slice = True
ffs = False
print("Using FFS: {}".format(ffs))

# replace with your local directory
mayo_dir = "/mnt/data0/manifest-1648648375084/LDCT-and-Projection-data"
# define projection and reconstruction data directories
# e.g. for patient L004 full dose CT scan:
# proj_dir = os.path.join(
#     mayo_dir, 'L004/08-21-2018-NA-NA-10971/1.000000-Full dose projections-24362/')
proj_dir = os.path.join(
    mayo_dir, 'L014/08-22-2018-NA-NA-20022/1.000000-Full dose projections-35722/')

# rec_dir = os.path.join(
#     mayo_dir, 'L004/12-23-2021-NA-NA-40641/1.000000-Full Dose Images-63186/')
rec_dir = os.path.join(
    mayo_dir, 'L014/12-23-2021-NA-NA-83637/1.000000-Full Dose Images-28591/')


if reco_slice:
    slice_i = 16000
    slice_f = 19000    
    geometry, proj_data, photon_stat = mayo.load_projections(proj_dir,
                                           indices=slice(slice_i, slice_f), use_ffs=ffs)
    print("Using slices {:d} to {:d} for reconstruction".format(slice_i, slice_f))             
else:
    # use all projs
    geometry, proj_data, photon_stat = mayo.load_projections(proj_dir, use_ffs=ffs)
# Load projection data restricting to a central slice
print("Loading projection data from {:s}".format(proj_dir))

# Load reconstruction data
print("Loading reference data from {:s}".format(rec_dir))
recon_space, volume = mayo.load_reconstruction(rec_dir)

# ray transform
ray_trafo = odl.tomo.RayTransform(recon_space, geometry)

# Interpolate projection data for a flat grid
radial_dist = geometry.src_radius + geometry.det_radius
# I guess we can get rid of this since we have the backend for curved now?
# flat_proj_data = mayo.interpolate_flat_grid(proj_data,
#                                             ray_trafo.range.grid,
#                                             radial_dist)

path = "/home/rosaca/code/odl/odl/contrib/datasets/ct/examples/pictures/"

def fbp_reco():
    # --------------------
    # FBP reconstruction
    # --------------------

    # Define FBP operator
    start = perf_counter()
    fbp = odl.tomo.fbp_op(ray_trafo, padding=True)
    # Tam-Danielsson window to handle redundant data
    td_window = odl.tomo.tam_danielson_window(ray_trafo, n_pi=3)
    # Calculate FBP reconstruction
    fbp_result = fbp(td_window * proj_data) # flat_proj_data
    stop = perf_counter()
    print('FBP done after {:.3f} seconds'.format(stop-start))

    # Compare the computed recon to reference reconstruction (coronal slice)
    ref = recon_space.element(volume)
    # diff = recon_space.element(np.linalg.norm(volume - fbp_result_HU.asarray())/np.linalg.norm(volume))

    A = volume                               # reference (numpy ndarray)
    B = fbp_result.asarray()              # reconstruction (numpy ndarray)

    s = (A * B).sum() / (B * B).sum()        # best gain  (Noo 2003, Eq. 57 idea)

    # -------------------------------------------------------------------------
    # 2.  Signed, per-voxel relative error after rescaling
    eps = np.finfo(A.dtype).eps              # avoid divide-by-zero at A == 0
    rel_err = (s * B - A) / (np.abs(A) + eps)
    rel_err = np.clip(rel_err, -0.5, 0.5)  # clip to [-1, 1] for better visualization
    # -------------------------------------------------------------------------
    # Wrap into an ODL element so .show() works
    diff = recon_space.element(rel_err)
    # Compare the computed recon to reference reconstruction (coronal slice)
    fbp_result.show(saveto=path+'FBP Recon (axial).svg')
    ref.show(saveto=path+'Reference (axial).svg')
    diff.show(saveto=path+'FBP Diff Normalized (axial).svg')

    coords = [0, None, None]
    fbp_result.show(saveto=path+'FBP Recon (sagittal).svg', coords=coords)
    ref.show(saveto=path+'Reference (sagittal).svg', coords=coords)

def kats_reco():
    # # --------------------
    # # KATS reconstruction
    # # --------------------

    # # Define KATS operator
    # start = perf_counter()
    ffs_fn = '' if ffs else '_no_ffs'
    if reco_slice:
        filt_data_path = path + f'filtered_kats_{slice_f - slice_i}_projs{ffs_fn}.npy'
        print(filt_data_path)
    else:
        filt_data_path = path + f'filtered_kats_full_projs{ffs_fn}.npy'
    if not os.path.exists(filt_data_path):
        kats_filter = odl.tomo.katsevich_filter_op(ray_trafo)
        filt_data = kats_filter(proj_data)
        print('Saving filtered data to {}'.format(filt_data_path))
        np.save(filt_data_path, filt_data.asarray())
    else:
        print('Loading filtered data from {}'.format(filt_data_path))
        filt_data = np.load(filt_data_path)
        
    windowed_filt_data = odl.tomo.td_window_curved(ray_trafo, filt_data, print_tqdm=True, force_const=True)
    kats_result = ray_trafo.adjoint(windowed_filt_data)
    # stop = perf_counter()
    # print('KATS done after {:.3f} seconds'.format(stop-start))

    # diff_kats = recon_space.element(np.linalg.norm(volume - kats_result_HU.asarray())/np.linalg.norm(volume))
    A = volume                               # reference (numpy ndarray)
    B = kats_result.asarray()              # reconstruction (numpy ndarray)

    s = (A * B).sum() / (B * B).sum()        # best gain  (Noo 2003, Eq. 57 idea)

    # -------------------------------------------------------------------------
    # 2.  Signed, per-voxel relative error after rescaling
    eps = np.finfo(A.dtype).eps              # avoid divide-by-zero at A == 0
    rel_err = (s * B - A) / (np.abs(A) + eps)
    rel_err = np.clip(rel_err, -0.5, 0.5)  # clip to [-1, 1] for better visualization
    # -------------------------------------------------------------------------
    # Wrap into an ODL element so .show() works
    diff_kats = recon_space.element(rel_err)
    file_name = 'KATS Recon (a' \
    'xial).svg' if ffs else 'KATS Recon (axial) No FFS.svg'
    kats_result.show(saveto=path+file_name)
    diff_kats.show(saveto=path+'KATS Diff Normalized (axial).svg')

    coords = [0, None, None]
    file_name = 'KATS Recon (sagittal).svg' if ffs else 'KATS Recon (sagittal) No FFS.svg'
    kats_result.show(saveto=path+file_name, coords=coords)
    diff_kats.show(saveto=path+'KATS Diff Normalized (sagittal).svg', coords=coords)

    coords = [None, 0, None]
    file_name = 'KATS Recon (coronal).svg' if ffs else 'KATS Recon (coronal) No FFS.svg'
    kats_result.show(saveto=path+file_name, coords=coords)
    diff_kats.show(saveto=path+'KATS Diff Normalized (coronal).svg', coords=coords)
def plot_src():
    # plot the src_position of the geometry in a 3d space
    color_even = '#08306B'   # dark blue
    color_odd  = "#6DB1D6"   # light blue
    pos = geometry.src_position(geometry.angles)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    indices = np.arange(30, 300)
    even_idx = indices[indices % 2 == 0]
    odd_idx = indices[indices % 2 == 1]
    ax.plot(pos[even_idx, 0], pos[even_idx, 1], pos[even_idx, 2],
            'o', color=color_even, markersize=2, linewidth=0.1, label='smaller, lower')
    ax.plot(pos[odd_idx, 0], pos[odd_idx, 1], pos[odd_idx, 2],
            'o', color=color_odd, markersize=2, linewidth=0.1, label='bigger, higher')
    ax.plot(pos[indices, 0], pos[indices, 1], pos[indices, 2], linewidth=0.1)
    plt.title('Close-up positions')
    plt.tight_layout()
    plt.savefig(path+'src_closeup.svg')

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

    # Indices and colors
    indices = np.arange(len(geometry.angles))
    even_idx = indices[indices % 2 == 0]
    odd_idx = indices[indices % 2 == 1]

    # Colors
    color_even = '#08306B'   # dark blue
    color_odd  = "#6DB1D6"   # light blue

    fig = plt.figure(figsize=(16, 8))

    # ── Top view (XY) ──
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_proj_type('ortho')
    ax1.plot(pos[even_idx, 0], pos[even_idx, 1], pos[even_idx, 2],
            'o', color=color_even, markersize=2, linewidth=0.1, label='smaller, lower')
    ax1.plot(pos[odd_idx, 0], pos[odd_idx, 1], pos[odd_idx, 2],
            'o', color=color_odd, markersize=2, linewidth=0.1, label='bigger, higher')
    ax1.plot(pos[indices, 0], pos[indices, 1], pos[indices, 2], linewidth=0.1, color='gray')
    ax1.view_init(elev=90, azim=-90)
    ax1.set_title('Top view (XY)', pad=20)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
    ax1.zaxis.set_tick_params(labelsize=8)
    ax1.tick_params(axis='x', pad=5)
    ax1.tick_params(axis='y', pad=5)
    ax1.tick_params(axis='z', pad=5)

    # ── Side view (XZ) ──
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_proj_type('ortho')
    ax2.plot(pos[even_idx, 0], pos[even_idx, 1], pos[even_idx, 2],
            'o', color=color_even, markersize=2, linewidth=0.1, label='smaller, lower')
    ax2.plot(pos[odd_idx, 0], pos[odd_idx, 1], pos[odd_idx, 2],
            'o', color=color_odd, markersize=2, linewidth=0.1, label='bigger, higher')
    ax2.plot(pos[indices, 0], pos[indices, 1], pos[indices, 2], linewidth=0.1, color='gray')
    ax2.view_init(elev=0, azim=-90)
    ax2.set_title('Side view (XZ)', pad=20)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.zaxis.set_tick_params(labelsize=8)
    ax2.tick_params(axis='x', pad=5)
    ax2.tick_params(axis='y', pad=5)
    ax2.tick_params(axis='z', pad=5)

    plt.tight_layout(pad=2.0)
    plt.savefig(path + 'src_2views.svg', bbox_inches='tight')

def plot_det():
    # plot the det_position of the geometry in a 3d space
    color_even = '#08306B'   # dark blue
    color_odd  = "#6DB1D6"   # light blue
    pos = geometry.det_refpoint(geometry.angles)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    indices = np.arange(30, 300)
    even_idx = indices[indices % 2 == 0]
    odd_idx = indices[indices % 2 == 1]
    ax.plot(pos[even_idx, 0], pos[even_idx, 1], pos[even_idx, 2],
            'o', color=color_even, markersize=2, linewidth=0.1, label='smaller, lower')
    ax.plot(pos[odd_idx, 0], pos[odd_idx, 1], pos[odd_idx, 2],
            'o', color=color_odd, markersize=2, linewidth=0.1, label='bigger, higher')
    ax.plot(pos[indices, 0], pos[indices, 1], pos[indices, 2], linewidth=0.1)
    plt.title('Close-up positions')
    plt.tight_layout()
    plt.savefig(path+'det_closeup.svg')

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

# fbp_reco()
kats_reco()
# plot_src()
# plot_det()

# td_window(ray_trafo, path=path)
# td_window_nonconst_pitch(ray_trafo, path=path, N_sampling=1000, use_ffs=ffs)
# td_window_nonconst_pitch_video(ray_trafo, path=path, N_sampling=1000, step=100)


