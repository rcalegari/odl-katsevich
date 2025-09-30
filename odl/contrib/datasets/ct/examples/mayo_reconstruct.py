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
from examples.kats.utils import plot_psnr, plot_rmse, plot_ssim

reco_slice = False
ffs = False
full = True  
body_part = 'chest'  # 'head', 'abdomen', 'chest'

# import argparse

# # Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--reco_slice', type=str, default='True')
# parser.add_argument('--body_part', type=str, default='head', choices=['head', 'abdomen', 'abdomen-L019', 'abdomen-L033', 'abdomen-L049', 'abdomen-L056', 'chest', 'chest-C004', 'chest-C012', 'chest-C016', 'chest-C027'])
# parser.add_argument('--full', type=str, default='True')
# parser.add_argument('--ffs', type=str, default='False')
# args = parser.parse_args()

# # Convert to booleans (argparse only returns strings)
# reco_slice = args.reco_slice.lower() == 'true'
# body_part = args.body_part
# full = args.full.lower() == 'true'
# ffs = args.ffs.lower() == 'true'

print("Using FFS: {}".format(ffs))


# replace with your local directory
mayo_dir = "/mnt/data0/manifest-1648648375084/LDCT-and-Projection-data"
# define projection and reconstruction data directories
# e.g. for patient L004 full dose CT scan:
# proj_dir = os.path.join(
#     mayo_dir, 'L004/08-21-2018-NA-NA-10971/1.000000-Full dose projections-24362/')
if body_part == 'abdomen':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'L014/08-22-2018-NA-NA-20022/1.000000-Full dose projections-35722/')
        rec_dir = os.path.join(
            mayo_dir, 'L014/12-23-2021-NA-NA-83637/1.000000-Full Dose Images-28591/')
    else:
        raise ValueError("Low dose projections for abdomen not available in this example.")
elif body_part == 'abdomen-L019':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'L019/08-23-2018-NA-NA-11492/1.000000-Full dose projections-77764/')
        rec_dir = os.path.join(
            mayo_dir, 'L019/12-23-2021-NA-NA-03411/1.000000-Full Dose Images-23860/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'L019/08-23-2018-NA-NA-11492/1.000000-Low dose projections-22295/')
        rec_dir = os.path.join(
            mayo_dir, 'L019/12-23-2021-NA-NA-03411/1.000000-Low Dose Images-71448/')
elif body_part == 'abdomen-L033':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'L033/08-28-2018-NA-NA-93540/1.000000-Full dose projections-02630/')
        rec_dir = os.path.join(
            mayo_dir, 'L033/12-23-2021-NA-NA-20514/1.000000-Full Dose Images-12321/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'L033/08-28-2018-NA-NA-93540/1.000000-Low dose projections-48288/')
        rec_dir = os.path.join(
            mayo_dir, 'L033/12-23-2021-NA-NA-20514/1.000000-Low Dose Images-30854/')
elif body_part == 'abdomen-L049':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'L049/08-26-2018-NA-NA-28158/1.000000-Full dose projections-15765/')
        rec_dir = os.path.join(
            mayo_dir, 'L049/12-23-2021-NA-NA-40818/1.000000-Full Dose Images-62638/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'L049/08-26-2018-NA-NA-28158/1.000000-Low dose projections-84771/')
        rec_dir = os.path.join(
            mayo_dir, 'L049/12-23-2021-NA-NA-40818/1.000000-Low Dose Images-38720/')
elif body_part == 'abdomen-L056':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'L056/08-24-2018-NA-NA-07351/1.000000-Full dose projections-52723/')
        rec_dir = os.path.join(
            mayo_dir, 'L056/12-23-2021-NA-NA-43434/1.000000-Full Dose Images-72444/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'L056/08-24-2018-NA-NA-07351/1.000000-Low dose projections-66654/')
        rec_dir = os.path.join(
            mayo_dir, 'L056/12-23-2021-NA-NA-43434/1.000000-Low Dose Images-61966/')
elif body_part == 'chest': # C016
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'C016/08-29-2018-NA-NA-99478/1.000000-Full dose projections-68344/')
        rec_dir = os.path.join(
            mayo_dir, 'C016/12-23-2021-NA-NA-72558/1.000000-Full Dose Images-51981/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'C016/08-29-2018-NA-NA-99478/1.000000-Low dose projections-60159/')
        rec_dir = os.path.join(
            mayo_dir, 'C016/12-23-2021-NA-NA-72558/1.000000-Low Dose Images-82312/')
elif body_part == 'chest-C004':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'C004/08-30-2018-NA-NA-33450/1.000000-Full dose projections-47507/')
        rec_dir = os.path.join(
            mayo_dir, 'C004/12-23-2021-NA-NA-40816/1.000000-Full Dose Images-73627/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'C004/08-30-2018-NA-NA-33450/1.000000-Low dose projections-04813/')
        rec_dir = os.path.join(
            mayo_dir, 'C004/12-23-2021-NA-NA-40816/1.000000-Low Dose Images-30622/')
elif body_part == 'chest-C012':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'C012/08-30-2018-NA-NA-87735/1.000000-Full dose projections-26638/')
        rec_dir = os.path.join(
            mayo_dir, 'C012/12-23-2021-NA-NA-93425/1.000000-Full Dose Images-93580/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'C012/08-30-2018-NA-NA-87735/1.000000-Low dose projections-78672/')
        rec_dir = os.path.join(
            mayo_dir, 'C012/12-23-2021-NA-NA-93425/1.000000-Low Dose Images-12506/')
elif body_part == 'chest-C027':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'C027/08-29-2018-NA-NA-03765/1.000000-Full dose projections-25119/')
        rec_dir = os.path.join(
            mayo_dir, 'C027/01-13-2022-NA-NA-40777/1.000000-Full Dose Images-03450/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'C027/08-29-2018-NA-NA-03765/1.000000-Low dose projections-90270/')
        rec_dir = os.path.join(
            mayo_dir, 'C027/01-19-2022-NA-NA-40777/1.000000-Low Dose Images-53324/')


# head scans don't work, M in filter becomes negative
elif body_part == 'head':
    if full:
        proj_dir = os.path.join(
            mayo_dir, 'N012/09-03-2018-NA-NA-06804/1.000000-Full dose projections-88808/')
        rec_dir = os.path.join(
            mayo_dir, 'N012/12-23-2021-NA-NA-36085/1.000000-Full Dose Images-20199/')
    else:
        proj_dir = os.path.join(
            mayo_dir, 'N012/09-03-2018-NA-NA-06804/1.000000-Low dose projections-97331/')
        rec_dir = os.path.join(
            mayo_dir, 'N012/12-23-2021-NA-NA-36085/1.000000-Low Dose Images-40756/')

else:
    raise ValueError("Unknown body part: {}".format(body_part))
        
print("Projection data directory: {:s}".format(proj_dir))

if reco_slice:
    slice_i = 16000
    slice_f = 19000    
    geometry, proj_data, photon_stat = mayo.load_projections(proj_dir,
                                           indices=slice(slice_i, slice_f), use_ffs=ffs)
    print("Using slices {:d} to {:d} for reconstruction".format(slice_i, slice_f))             
else:
    # use all projs
    print("Reconstructing full object")
    geometry, proj_data, photon_stat = mayo.load_projections(proj_dir, use_ffs=ffs)
# Load projection data restricting to a central slice
print("Loading projection data from {:s}".format(proj_dir))

if len(geometry.angles) > 40000:
    print("Warning: More than 40000 angles in the geometry. It will probably get killed.\n Skip to the next.")

else:
    print("Number of angles in the geometry: {:d}".format(len(geometry.angles)))


    # Load reconstruction data
    print("Loading reference data from {:s}".format(rec_dir))
    recon_space, volume = mayo.load_reconstruction(rec_dir)

    # ray transform
    ray_trafo = odl.tomo.RayTransform(recon_space, geometry)

    # Interpolate projection data for a flat grid
    radial_dist = geometry.src_radius + geometry.det_radius
    # I guess we can get rid of this since we have the backend for curved now?
    # flat_proj_data = mayo.interpolate_flat_grid(proj_data,
    #                             ray_trafo.range.grid,
    #                                             radial_dist)

    dose = 'full' if full else 'low'
    if reco_slice:
        path = f"/home/rosaca/code/odl/odl/contrib/datasets/ct/examples/pictures/slices/{body_part}/{dose}/"
    else:
        path = f"/home/rosaca/code/odl/odl/contrib/datasets/ct/examples/pictures/{body_part}/{dose}/"

    print("Saving results to {:s}".format(path))

    if not os.path.exists(path+'settings.txt'):
        with open(path + 'settings.txt', 'w') as f:
            f.write('Geometry:\n')
            f.write(str(geometry) + '\n\n')
            f.write('Number of views: {}\n'.format(len(geometry.angles)))
            f.write('Reconstruction space:\n')
            f.write(str(recon_space) + '\n\n')
            f.write('Using FFS: {}\n'.format(ffs))
    else:
        with open(path + 'settings.txt', 'a') as f:
            f.write('Using FFS: {}\n'.format(ffs))

    def reco(fbp=True, kats=True, video=True, save=False):
        if fbp:
            print('------------------------')
            print('-- FBP reconstruction --')
            print('------------------------')

            if os.path.exists(path + 'fbp_reco.npy'):
                print('Loading FBP data from {}'.format(path + 'fbp_reco.npy'))
                fbp_result = np.load(path + 'fbp_reco.npy')
                fbp_result = recon_space.element(fbp_result)

            else:
                print('Filtering data with FBP filter...')
                start = perf_counter()
                td_window = odl.tomo.tam_danielson_window(ray_trafo, n_pi=3)
                fbp = odl.tomo.fbp_op(ray_trafo, padding=True)
                fbp_result = fbp(td_window * proj_data) # flat_proj_data
                end = perf_counter()
                np.save(path + 'fbp_reco.npy', fbp_result.asarray())
                fbp_time = end - start
                print('FBP reconstruction done in {:.3f} seconds'.format(fbp_time))
                with open(path + 'settings.txt', 'a') as f:
                    f.write('Time for FBP: {:.3f} seconds\n'.format(fbp_time))

            # Compare the computed recon to reference reconstruction (coronal slice)
            ref = recon_space.element(volume)
            # diff = recon_space.element(np.linalg.norm(volume - fbp_result_HU.asarray())/np.linalg.norm(volume))

            A = volume                               # reference (numpy ndarray)
            B = fbp_result.asarray()                 # reconstruction (numpy ndarray)

            # compute PSNR, RMSE, SSIM
            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
            psnr_value = psnr(A, B, data_range=A.max() - A.min())
            rmse_value = np.sqrt(mse(A, B))
            ssim_value = ssim(A, B, data_range=A.max() - A.min())
            with open(path + 'settings.txt', 'a') as f:
                f.write('FBP PSNR: {:.3f} dB\n'.format(psnr_value))
                f.write('FBP RMSE: {:.3f}\n'.format(rmse_value))
                f.write('FBP SSIM: {:.3f}\n'.format(ssim_value))
            print('FBP PSNR: {:.3f} dB'.format(psnr_value))
            print('FBP RMSE: {:.3f}'.format(rmse_value))
            print('FBP SSIM: {:.3f}'.format(ssim_value))

            print('Saving FBP reconstructions ...')

            fbp_result.show(saveto=path+'FBP Recon (axial).svg')
            ref.show(saveto=path+'Reference (axial).svg')

            coords = [0, None, None]
            fbp_result.show(saveto=path+'FBP Recon (sagittal).svg', coords=coords)
            ref.show(saveto=path+'Reference (sagittal).svg', coords=coords)

            coords = [None, 0, None]
            fbp_result.show(saveto=path+'FBP Recon (coronal).svg', coords=coords)
            ref.show(saveto=path+'Reference (coronal).svg', coords=coords)

            if video:
                save_xy_video(B, path=path, filename_prefix=f'FBP axial {body_part}', step=1)
                save_xy_video(np.transpose(B, (0, 2, 1)), path=path, filename_prefix=f'FBP coronal {body_part}', step=1)
                save_xy_video(np.transpose(B, (2, 1, 0)), path=path, filename_prefix=f'FBP sagittal {body_part}', step=1)


        if kats:
            if not fbp:
                ref = recon_space.element(volume)
                A = volume  
                ref.show(saveto=path+'Reference (axial).svg')
                coords = [0, None, None]
                ref.show(saveto=path+'Reference (sagittal).svg', coords=coords)
                coords = [None, 0, None]
                ref.show(saveto=path+'Reference (coronal).svg', coords=coords)                             
            # # --------------------
            # # KATS reconstruction
            # # --------------------
            print('-------------------------')
            print('-- KATS reconstruction --')
            print('-------------------------')
            # # Define KATS operator
            # start = perf_counter()
            '''create settings file that contains the geometry and space parameters
            and the time  it takes for each step of the reconstruction '''

            ffs_fn = '' if ffs else '_no_ffs'
            init_time = np.NaN
            filter_time = np.NaN

            if reco_slice:
                filt_data_path = path + f'filtered_kats_{slice_f - slice_i}_projs{ffs_fn}.npy'
                print(filt_data_path)
            else:
                filt_data_path = path + f'filtered_kats_full_projs{ffs_fn}.npy'

            if not os.path.exists(filt_data_path):
                print('Filtering data with KATS filter...')
                start = perf_counter()
                kats_filter = odl.tomo.katsevich_filter_op(ray_trafo)
                end = perf_counter()
                init_time = end - start
                start = perf_counter()
                filt_data = kats_filter(proj_data)
                end = perf_counter()
                filter_time = end - start
                # print('Saving kats filtered data to {}'.format(filt_data_path))
                # np.save(filt_data_path, filt_data.asarray())
                with open(path + 'settings.txt', 'a') as f:
                    f.write('Time for KATS initialization: {:.3f} seconds\n'.format(init_time))
                    f.write('Time for KATS filter: {:.3f} seconds\n'.format(filter_time))
                print('Kats filter initialization done in {:.3f} seconds'.format(init_time))
                print('Kats filtering done in {:.3f} seconds'.format(filter_time))

            else:
                print('\nLoading filtered data from {}'.format(filt_data_path))
                filt_data = np.load(filt_data_path)

            start = perf_counter() 
            windowed_filt_data = odl.tomo.td_window_curved(ray_trafo, filt_data, print_tqdm=True, force_const=True)
            kats_result = ray_trafo.adjoint(windowed_filt_data)
            end = perf_counter()
            bp_time = end - start
            
            with open(path + 'settings.txt', 'a') as f:
                f.write('Time for KATS backprojection: {:.3f} seconds\n'.format(bp_time))
                f.write('Total time for KATS reconstruction: {:.3f} seconds\n'.format(init_time + filter_time + bp_time))
            print('KATS backprojection done in {:.3f} seconds'.format(bp_time))
            C = kats_result.asarray()              # reconstruction (numpy ndarray)
            C_rescaled = A.max() * (C - C.min()) / (C.max() - C.min())

            # if save:
                # np.save(path + 'kats_reco.npy', C)
                # np.save(path + 'reference.npy', A)
            kats_rescaled = recon_space.element(C_rescaled)

            # compute PSNR, RMSE, SSIM
            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
            psnr_value = psnr(A, C_rescaled, data_range=A.max() - A.min())
            rmse_value = np.sqrt(mse(A, C_rescaled))
            ssim_value = ssim(A, C_rescaled, data_range=A.max() - A.min())
            with open(path + 'settings.txt', 'a') as f:
                f.write('Kats PSNR: {:.3f} dB\n'.format(psnr_value))
                f.write('Kats RMSE: {:.3f}\n'.format(rmse_value))
                f.write('Kats SSIM: {:.3f}\n'.format(ssim_value))
            print('Kats PSNR: {:.3f} dB'.format(psnr_value))
            print('Kats RMSE: {:.3f}'.format(rmse_value))
            print('Kats SSIM: {:.3f}'.format(ssim_value))

            ffs_str = '' if ffs else ' No FFS'
            file_name = 'KATS Recon (axial).svg' if ffs else 'KATS Recon (axial) No FFS.svg'
            kats_result.show(saveto=path+file_name)
            kats_rescaled.show(saveto=path+f'KATS Rescaled (axial){ffs_str}.svg')

            coords = [0, None, None]
            file_name = 'KATS Recon (sagittal).svg' if ffs else 'KATS Recon (sagittal) No FFS.svg'
            kats_result.show(saveto=path+file_name, coords=coords)
            kats_rescaled.show(saveto=path+f'KATS Rescaled (sagittal){ffs_str}.svg', coords=coords)

            coords = [None, 0, None]
            file_name = 'KATS Recon (coronal).svg' if ffs else 'KATS Recon (coronal) No FFS.svg'
            kats_result.show(saveto=path+file_name, coords=coords)
            kats_rescaled.show(saveto=path+f'KATS Rescaled (coronal){ffs_str}.svg', coords=coords)


            if kats and fbp:
                plot_psnr(A, path=path, kats_reco=C_rescaled, fbp_reco=B, ffs=ffs)
                plot_rmse(A, path=path, kats_reco=C_rescaled, fbp_reco=B, ffs=ffs)
                plot_ssim(A, path=path, kats_reco=C_rescaled, fbp_reco=B, ffs=ffs)

            if video:
                save_xy_video(C_rescaled, path=path, filename_prefix=f'Rescaled KATS axial {body_part}', step=1)
                save_xy_video(np.transpose(C_rescaled, (0, 2, 1)), path=path, filename_prefix=f'Rescaled KATS coronal {body_part}', step=1)
                save_xy_video(np.transpose(C_rescaled, (2, 1, 0)), path=path, filename_prefix=f'Rescaled KATS sagittal {body_part}', step=1)

                save_xy_video(A, path=path, filename_prefix=f'Reference axial {body_part}', step=1)
                save_xy_video(np.transpose(A, (0, 2, 1)), path=path, filename_prefix=f'Reference coronal {body_part}', step=1)
                save_xy_video(np.transpose(A, (2, 1, 0)), path=path, filename_prefix=f'Reference sagittal {body_part}', step=1) 


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




    reco(video=False)

    # plot_src()
    # plot_det()

    # td_window(ray_trafo, path=path)
    # td_window_nonconst_pitch(ray_trafo, path=path, N_sampling=1000, use_ffs=ffs)
    # td_window_nonconst_pitch_video(ray_trafo, path=path, N_sampling=1000, step=100)


