import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.animation as animation
from scipy.interpolate import interp1d

def save_uv_video(data_arr, path, filename_prefix, step=20, interval=1, cmap='gray', axes=(r'$\alpha$', 'w'), save_slice=False):
    """Save a (alpha, w) slice animation sampled every `step` s.
       Expected data shape: (slices, rows, columns)"""

    if not isinstance(data_arr, np.ndarray):
        data_arr = np.asarray(data_arr)

    projIdx = list(range(0, data_arr.shape[0], step))
    fig, ax = plt.subplots()
    vmax = np.max(np.transpose(data_arr, (0, 2, 1)))
    vmin = np.min(np.transpose(data_arr, (0, 2, 1)))
    # shows axis 0 (w) on y-axis and axis 1 (u) on x-axis
    im = ax.imshow(data_arr[0].T, cmap=cmap, aspect='equal', # data_arr[0].T
                   extent=None, vmin=vmin, vmax=vmax, interpolation='none') 
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    def update(i):
        idx = projIdx[i]
        frame = np.squeeze(data_arr[idx].T)
        im.set_data(frame) 
        ax.set_title(filename_prefix + " in detector space {} vs {}".format(axes[0], axes[1]))
        return [im]
    ani = FuncAnimation(fig, update, frames=len(projIdx), interval=interval, blit=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_path = os.path.join(path, filename_prefix + "_uv_video.mp4")
    writer = FFMpegWriter(fps=30, bitrate=1800)
    ani.save(save_path, writer=writer, dpi=150)
    plt.close()
    if save_slice:
        plt.figure()
        plt.imshow(data_arr[data_arr.shape[0] // 2].T, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{filename_prefix} slice at index {data_arr.shape[0] // 2}")
        plt.colorbar()
        plt.savefig(os.path.join(path, f"{filename_prefix}_slice.png"), dpi=150)
        plt.close()
    return save_path

def save_xy_video(data_arr, path, filename_prefix, step=20, interval=1, axes=('x', 'y'), clip=False, save_slice=False):
    """Save a (x, y) slice animation sampled every `step` s."""

    if not isinstance(data_arr, np.ndarray):
        data_arr = np.asarray(data_arr)

    if clip:
        data_arr = np.clip(data_arr, -1, 2)

    phi_indices = list(range(0, data_arr.shape[0], step))

    fig, ax = plt.subplots(1, 1)
    extent = None
    vmax = np.max(data_arr)
    vmin = np.min(data_arr)

    im = ax.imshow(data_arr[0], cmap='grey', aspect='equal',
                   extent=extent, vmin=vmin, vmax=vmax, interpolation='none')

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(i):
        idx = phi_indices[i]
        frame = np.squeeze(data_arr[idx])
        im.set_data(frame) # data_arr[idx]
        ax.set_title(filename_prefix + " in reconstruction space {} vs {}".format(axes[0], axes[1]))
        return [im]

    ani = FuncAnimation(fig, update, frames=len(phi_indices), interval = interval, blit=True)

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename_prefix + "_xy_video.mp4")
    writer = FFMpegWriter(fps=30, bitrate=1800)
    ani.save(save_path, writer=writer, dpi=150)
    plt.close()

    if save_slice:
        plt.figure()
        plt.imshow(data_arr[data_arr.shape[0] // 2], cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{filename_prefix} slice at index {data_arr.shape[0] // 2}")
        plt.colorbar()
        plt.savefig(os.path.join(path, f"{filename_prefix}_slice.png"), dpi=150)
        plt.close()
        
    return save_path

def save_side_by_side_video(
    volume1,
    volume2,
    path,
    title1="Left",
    title2="Right",
    filename="side_by_side.mp4",
    axis=2,
    fps=30,
    dpi=150,
    axis1 = None,
    axis2 = None
):
    """
    Save a side-by-side animation of two 3D volumes along a slicing axis.
    """
    if axis1 is None:
        axis1 = axis
    if axis2 is None:
        axis2 = axis
    volume1 = np.asarray(volume1)
    volume2 = np.asarray(volume2)
    assert volume1.shape == volume2.shape, "Volumes must have same shape."

    frames = volume1.shape[axis]
    vmin = min(volume1.min(), volume2.min())
    vmax = max(volume1.max(), volume2.max())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axs[0].imshow(np.take(volume1, 0, axis=axis1), cmap='gray', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(np.take(volume2, 0, axis=axis2), cmap='gray', vmin=vmin, vmax=vmax)

    axs[0].set_title(title1)
    axs[1].set_title(title2)

    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.axis('off')

    pbar = tqdm(total=frames, desc="Side-by-side video frames")

    def update(i):
        im1.set_data(np.take(volume1, i, axis=axis1))
        im2.set_data(np.take(volume2, i, axis=axis2))
        pbar.update(1)
        return [im1, im2]

    ani = FuncAnimation(fig, update, frames=frames, blit=True)

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(save_path, writer=writer, dpi=dpi)
    pbar.close()
    plt.close()
    print(f"Side-by-side video saved to {save_path}")

def plot_source_and_detector_centers(src_pykats, src_odl, geometry, kat_conf, save_path=None):
    """Compare source and detector center trajectories in 3D."""

    # Get ODL detector center (position at each angle s)
    det_odl = np.array([geometry.det_refpoint(s) for s in geometry.angles])

    # Get Pykatsevich detector centers (assumes same s_vals as src_pykats)
    # direction from source to center of object (origin), normalized
    rays_dir = - src_pykats.copy()
    rays_dir[:, 2] = 0  # Zero out the z component
    rays_dir /= np.linalg.norm(rays_dir, axis=1)[:, None]
    rot = np.array([
        [0, 1],
        [-1, 0]
    ])
    xy_rotated = src_pykats[:, :2] @ rot.T
    src_pykats_rot = np.concatenate([xy_rotated, src_pykats[:, 2:]], axis=1)

    # rays_dir = -src_pykats / np.linalg.norm(src_pykats, axis=1)[:, None]
    det_pykats = src_pykats + kat_conf['scan_diameter'] * rays_dir
    det_pykats_rot = np.concatenate([det_pykats[:, :2] @ rot.T, det_pykats[:, 2:]], axis=1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # check that z position of source and detector center is the same
    if not np.allclose(src_pykats_rot[:, 2], det_pykats_rot[:, 2]):
        print("Warning: Pykats source and detector center z-positions do not match.")
        print(f"max distance: {np.max(np.abs(src_pykats_rot[:, 2] - det_pykats_rot[:, 2])):.3f} mm")
    if not np.allclose(src_odl[:, 2], det_odl[:, 2]):
        print("Warning: ODL source and detector center z-positions do not match.")
        print(f"max distance: {np.max(np.abs(src_odl[:, 2] - det_odl[:, 2])):.3f} mm")

    # # Plot source trajectories
    ax.plot(src_pykats_rot[:, 0], src_pykats_rot[:, 1], src_pykats_rot[:, 2], label="Source (Pykatsevich)", color='C0')
    ax.plot(src_odl[:, 0],    src_odl[:, 1],    src_odl[:, 2],    label="Source (ODL)",         color='C1', linestyle='--')

    # Plot detector center trajectories
    ax.plot(det_pykats_rot[:, 0], det_pykats_rot[:, 1], det_pykats_rot[:, 2], label="Detector center (Pykatsevich)", color='C0')
    ax.plot(det_odl[:, 0],    det_odl[:, 1],    det_odl[:, 2],    label="Detector center (ODL)",         color='C1', linestyle='--')

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Detector Trajectories")
    ax.legend()
    ax.view_init(elev=30, azim=45)
    # ax.view_init(elev=90, azim=-90)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    else:
        print("No save path provided. Plot not saved.")
    plt.close()

def c_window(geometry):
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    R = geometry.src_radius
    alpha_vals = geometry.det_partition.coord_vectors[0]
    w_vals = geometry.det_partition.coord_vectors[1]
    N_alpha = len(alpha_vals)
    N_w = len(w_vals)
    N_s = geometry.motion_partition.shape[0]
    w_bottom = - P * D / (2 * np.pi * R) * (np.pi/2 + alpha_vals) / np.cos(alpha_vals)
    w_top    =   P * D / (2 * np.pi * R) * (np.pi/2 - alpha_vals) / np.cos(alpha_vals)
    return w_bottom, w_top

def nc_window_slice(geometry, ds_plus, ds_min, k, N_sampling=1000):
        alpha_vals = geometry.det_partition.coord_vectors[0]
        w_vals = geometry.det_partition.coord_vectors[1]
        P = geometry.pitch
        D = geometry.src_radius + geometry.det_radius
        R = geometry.src_radius
        
        s0 = geometry.angles[k]  
        s_bot  = s0 + ds_min
        s_top  = s0 + ds_plus
        p0     = geometry.src_position(s0)[2]  
        a_bot =  (np.sin(ds_min) / (1 - np.cos(ds_min))) 
        a_top =  (np.sin(ds_plus) / (1 - np.cos(ds_plus))) 
        try:           
            p_bot  = geometry.src_position(s_bot)[:, 2]           # (N_sampling,)
            w_bot  = D / R *(p_bot - p0) / (1 - np.cos(ds_min)) 
            exc_bot = False
        except Exception as e:
            # print(f"Error in computing source positions: bottom [{s_bot[0]}, {s_bot[-1]}] not in range of [{self.geometry.angles[0]}, {self.geometry.angles[-1]}]") 
            w_bot = w_vals[0] * np.ones_like(ds_min)  # Fallback to min w value         
            exc_bot = True
        
        try:
            p_top  = geometry.src_position(s_top)[:, 2]
            w_top  = D / R *(p_top - p0) / (1 - np.cos(ds_plus))
            exc_top = False
        except Exception as e:
            # print(f"Error in computing source positions: top [{s_top[0]}, {s_top[-1]}] not in range of [{geometry.angles[0]}, {geometry.angles[-1]}]") 
            w_top = w_vals[-1] * np.ones_like(ds_plus)
            exc_top = True
        
        if exc_bot:
            w_bot_interp = c_window(geometry)[0]  # use constant window
        else:
            interp_bot = interp1d(a_bot, w_bot, bounds_error=False, fill_value=(w_bot[0], w_bot[-1]))
            w_bot_interp = interp_bot(alpha_vals)
        
        if exc_top:
            w_top_interp = c_window(geometry)[1] # use constant window
        else:
            interp_top = interp1d(a_top, w_top, bounds_error=False, fill_value=(w_top[0], w_top[-1]))
            w_top_interp = interp_top(alpha_vals)
        
        return w_bot_interp, w_top_interp

def td_window(ray_trafo, path: str = '/home/rosaca/code/pine_16_1/'):
    geometry = ray_trafo.geometry
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    Rs = geometry.src_radius
    a=float(0.025) # smoothing width
    if geometry.det_curvature_radius is None:
        det = 'flat'
        u_vals = geometry.det_partition.coord_vectors[1]
        w_vals = geometry.det_partition.coord_vectors[0]
        N_u = len(u_vals)
        N_w = len(w_vals)
        w_bottom = - P / (2 * np.pi * Rs * D) * (u_vals**2 + D**2) * (np.pi/2 + np.arctan(u_vals / D))
        w_top    =   P / (2 * np.pi * Rs * D) * (u_vals**2 + D**2) * (np.pi/2 - np.arctan(u_vals / D))
        w_bottom = np.reshape(w_bottom, (1, -1))
        w_top    = np.reshape(w_top, (1, -1))
    else:
        print('plotting td window curv with constant pitch')
        det = 'curv'
        alpha_vals = geometry.det_partition.coord_vectors[0]
        w_vals = geometry.det_partition.coord_vectors[1]
        N_alpha = len(alpha_vals)
        N_w = len(w_vals)
        w_bottom, w_top = c_window(geometry)
        w_top = w_top[:, np.newaxis]      # shape (1, N_alpha)
        w_bottom = w_bottom[:, np.newaxis]

    W, A = np.meshgrid(w_vals, alpha_vals, indexing='xy')  # W = (N_alpha, N_w)
    mask = np.zeros(shape=(N_alpha, N_w), dtype=np.float32)

    N_s = geometry.motion_partition.shape[0]
    dw = (w_vals[1] - w_vals[0]) 

    w_bottom_low = (w_bottom - a * dw).reshape(-1, 1)  # shape (275, 1)
    w_bottom_high = (w_bottom + a * dw).reshape(-1, 1)
    w_top_low = (w_top - a * dw).reshape(-1, 1)
    w_top_high = (w_top + a * dw).reshape(-1, 1)

    region1 = ( W < w_bottom_low ) 
    region2 = ( w_bottom_low <= W ) & ( W < w_bottom_high )
    region3 = ( w_bottom_high <= W ) & ( W <= w_top_low )
    region4 = ( w_top_low < W ) & ( W <= w_top_high )
    region5 = ( w_top_high < W )

    # region1 and region5 stay at 0
    mask[region2] = (W - w_bottom_low)[region2] / (2 * a * dw)
    mask[region3] = 1
    mask[region4] = (w_top_high - W)[region4] / (2 * a * dw)

    if geometry.det_curvature_radius is None:
        maskTD = np.broadcast_to(mask, (N_s, N_w, N_u))
        col_vals = u_vals
    else:
        maskTD = np.broadcast_to(mask, (N_s, N_alpha, N_w))
        col_vals = alpha_vals

    plt.figure()
    plt.imshow(maskTD[N_s//2].T if geometry.det_curvature_radius is not None else maskTD[N_s//2], cmap='gray')
    plt.title("ODL TD mask")
    plt.xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    plt.ylabel("w")
    plt.colorbar()
    plt.savefig(path+f"td_mask_odl_{det}.svg")
    plt.close()
    print('td window saved in pine 16 1')

    plt.plot(col_vals, w_bottom.flatten(), color='blue')
    plt.plot(col_vals, w_top.flatten(), label='window', color='blue')
    plt.plot(col_vals, [w_vals[0]]*len(alpha_vals), '--', color='red', label='detector bounds')
    plt.plot(col_vals, [w_vals[-1]]*len(alpha_vals), '--', color='red')
    plt.xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    plt.ylabel("w")
    plt.legend()
    plt.title("TD window vs detector range")
    plt.grid()
    plt.savefig(path+f"td_vs_range_{det}.svg")
    plt.close()

def td_window_nonconst_pitch(ray_trafo,
                                path: str = '/home/rosaca/code/pine_16_1/',
                                a_ramp: float = 0.025,
                                N_sampling: int = 128,
                                slice_k: int = None,
                                use_ffs : bool = False):
    geometry = ray_trafo.geometry
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    R = geometry.src_radius
    alpha_vals = geometry.det_partition.coord_vectors[0]
    w_vals = geometry.det_partition.coord_vectors[1]
    N_s = geometry.motion_partition.shape[0]
    fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
    r = fov_dia / 2
    if a_ramp <= 0:
        raise ValueError("`a_ramp` must be positive")

    delta_s_edge = 2*np.arccos(r / R) 
    # Sampling grid in delta-s for one branch
    ds_plus = np.linspace(delta_s_edge, 2*np.pi - delta_s_edge, N_sampling)
    ds_min  = np.linspace(delta_s_edge - 2*np.pi, - delta_s_edge, N_sampling)

    if slice_k is None:
        k = N_s // 2  # middle slice
    else:
        k = slice_k
        if k < 0 or k >= N_s:
            raise ValueError(f"Slice index {k} out of bounds for N_s={N_s}")

    w_bot_interp, w_top_interp = nc_window_slice(geometry, ds_plus, ds_min, k, N_sampling=N_sampling)

    plt.figure()
    plt.plot(alpha_vals, w_bot_interp, color='blue')
    plt.plot(alpha_vals, w_top_interp, color='blue', label='window')
    plt.plot(alpha_vals, [w_vals[0]]*len(alpha_vals), '--', color='red', label='detector bounds')
    plt.plot(alpha_vals, [w_vals[-1]]*len(alpha_vals), '--', color='red')
    plt.xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    plt.ylabel("w")
    plt.legend()
    plt.title(f"TD window vs detector range at slice {k} ")
    plt.grid()
    if use_ffs:
        plt.savefig(path+"td_nonconst_ffs.svg")
    else:
        plt.savefig(path+"td_nonconst.svg")
    plt.close() 
    print('done')

def td_window_nonconst_pitch_video(ray_trafo,
                                path: str = '/home/rosaca/code/pine_16_1/',
                                a_ramp: float = 0.025,
                                N_sampling: int = 128,
                                step: int = 100):
    geometry = ray_trafo.geometry
    P = geometry.pitch
    D = geometry.src_radius + geometry.det_radius
    R = geometry.src_radius

    alpha_vals = geometry.det_partition.coord_vectors[0]
    w_vals = geometry.det_partition.coord_vectors[1]

    N_s = geometry.motion_partition.shape[0]

    # fov_dia = max(ray_trafo.domain.max_pt[2] - ray_trafo.domain.min_pt[2], ray_trafo.domain.max_pt[1] - ray_trafo.domain.min_pt[1])
    fov_dia = np.linalg.norm(ray_trafo.domain.max_pt - ray_trafo.domain.min_pt)
    r = fov_dia / 2
    delta_s_edge = 2*np.arccos(r / R) 

    # Sampling grid in delta-s for one branch
    ds_plus = np.linspace(delta_s_edge, 2*np.pi - delta_s_edge, N_sampling)
    ds_min  = np.linspace(delta_s_edge - 2*np.pi, - delta_s_edge, N_sampling)
    
    if a_ramp <= 0:
        raise ValueError("`a_ramp` must be positive")
    
    fig, ax = plt.subplots()
    line_bot, = ax.plot([], [], color='blue')
    line_top, = ax.plot([], [], color='blue')
    const_line_bot, = ax.plot(alpha_vals, c_window(geometry)[0], color='orange')
    const_line_top, = ax.plot(alpha_vals, c_window(geometry)[1], color='orange')
    line_det_min, = ax.plot(alpha_vals, [w_vals[0]] * len(alpha_vals), '--', color='red')
    line_det_max, = ax.plot(alpha_vals, [w_vals[-1]] * len(alpha_vals), '--', color='red')
    ax.set_xlim(alpha_vals[0]-0.1, alpha_vals[-1]+0.1)
    ax.set_ylim(w_vals[0]-5, w_vals[-1]+5)
    ax.set_xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    ax.set_ylabel("w")
    ax.set_title("TD window vs detector range")
    ax.grid()

    Idx = list(range(0, N_s, step))

    def update(i):
        i = Idx[i]
        w_bot_interp, w_top_interp = nc_window_slice(geometry, ds_plus, ds_min, i)
        line_bot.set_data(alpha_vals, w_bot_interp)
        line_top.set_data(alpha_vals, w_top_interp)
        ax.set_title(f"slice {i} / {N_s - 1}")
        return line_bot, line_top, line_det_min, line_det_max, const_line_bot, const_line_top
    ani = FuncAnimation(fig, update, frames=len(Idx), blit=True, interval=1)
    ani.save(path + "td_nonconst_anim.mp4", writer="ffmpeg", fps=15)

def save_intensity_row(phantom, reco, test, path):

    # Set up the figure
    if not isinstance(phantom, np.ndarray):
        phantom = phantom.asarray()
    Nz, Ny, Nx = phantom.shape
    slicesIdx = [Nz//3, Nz//2]
    yIdx = Ny // 2  # Fixed y index for the row

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], 'b-', label='Phantom')
    line2, = ax.plot([], [], 'b--', label='Reconstruction')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

    ax.set_xlim(0, Nx)
    ax.set_ylim(max(np.min([phantom, reco]), -2.), min(np.max([phantom, reco]), 2.))
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity')
    ax.set_title(f't0{test} Phantom vs. Reconstruction Profiles at y={yIdx}')
    ax.grid(True)
    ax.legend()

    # Update function for animation
    def update(z):
        phantom_profile = phantom[z, yIdx, :]
        reco_profile = reco[z, yIdx, :]

        line1.set_ydata(phantom_profile)
        line2.set_ydata(reco_profile)
        line1.set_xdata(np.arange(Nx))
        line2.set_xdata(np.arange(Nx))
        title.set_text(f'z = {z} / {Nz - 1}')
        return line1, line2, title

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nz, interval=100, blit=True)
    # Save as MP4 (requires ffmpeg) or show inline
    ani.save(path + f'profiles_t0{test}.mp4', fps=10)

def save_intensity_error_derivative_comparison(phantom, reco1, reco2, test, path):

    if not isinstance(phantom, np.ndarray):
        phantom = phantom.asarray()

    Nz, Ny, Nx = phantom.shape
    slicesIdx = [Nz//3, Nz//2]
    yIdx = Ny // 2  # Fixed y index for the row

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], 'r-', label='v1')
    line2, = ax.plot([], [], 'b-', label='v2')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

    ax.set_xlim(0, Nx)
    ax.set_ylim(np.min([phantom - reco1, phantom - reco2]), np.max([phantom - reco1, phantom - reco2]))
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity Error phantom - reco')
    ax.set_title(f't0{test} y={yIdx}')
    ax.grid(True)
    ax.legend()

    # Update function for animation
    def update(z):
        phantom_profile = phantom[z, yIdx, :]
        reco1_profile = reco1[z, yIdx, :]
        reco2_profile = reco2[z, yIdx, :]

        line1.set_ydata(phantom_profile - reco1_profile)
        line2.set_ydata(phantom_profile - reco2_profile)
        line1.set_xdata(np.arange(Nx))
        line2.set_xdata(np.arange(Nx))
        title.set_text(f'z = {z} / {Nz - 1}')
        return line1, line2, title

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nz, interval=100, blit=True)
    # Save as MP4 (requires ffmpeg) or show inline
    ani.save(path + f'profiles_error_derivatives.mp4', fps=10)

def save_intensity_error_rows(phantom, reco, test, path):

    # Set up the figure
    if not isinstance(phantom, np.ndarray):
        phantom = np.asarray(phantom)

    Nz, Ny, Nx = phantom.shape
    slicesIdx = [Ny // 6, Ny // 3, Ny // 2, 2 * Ny // 3, 5 * Ny // 6]  # Fixed y index for the row
    fig, ax = plt.subplots(figsize=(10, 6))

    lines = []
    for yIdx in slicesIdx:
        (line,) = ax.plot([], [], label=f'y={yIdx}')
        lines.append(line)

    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)
    ax.set_xlim(0, Nx)
    ax.set_ylim(np.min(phantom - reco), np.max(phantom - reco))
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity error')
    ax.set_title(f't0{test} Error phantom - recontruction Profiles')
    ax.grid(True)
    ax.legend()

    error = phantom - reco
    # Update function for animation
    def update(z):
        for line, yIdx in zip(lines, slicesIdx):
            profile = error[z, yIdx, :]
            line.set_data(np.arange(Nx), profile)
        title.set_text(f'z = {z} / {Nz - 1}')
        return lines + [title]

    ani = animation.FuncAnimation(fig, update, frames=Nz, interval=100, blit=True)
    ani.save(path + f'profiles_error_t0{test}.mp4', fps=10)