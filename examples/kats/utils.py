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

def save_xy_video(data_arr, path, filename_prefix, step=20, interval=1, axes=('x', 'y'), save_slice=False):
    """Save a (x, y) slice animation sampled every `step` s."""

    if not isinstance(data_arr, np.ndarray):
        data_arr = np.asarray(data_arr)

    phi_indices = list(range(0, data_arr.shape[2], step))

    fig, ax = plt.subplots(1, 1)
    extent = None
    vmax = np.max(data_arr)
    vmin = np.min(data_arr)

    im = ax.imshow(data_arr[:, :, 0].T, cmap='grey', aspect='equal',
                   extent=extent, vmin=vmin, vmax=vmax, interpolation='none')

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(i):
        idx = phi_indices[i]
        frame = np.squeeze(data_arr[:, :, idx].T)
        im.set_data(frame) # data_arr[idx]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
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
        plt.imshow(data_arr[:, :, data_arr.shape[2] // 2].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        plt.title(f"{filename_prefix} slice at index {data_arr.shape[2] // 2}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig(os.path.join(path, f"{filename_prefix}_slice.svg"), dpi=150)
        plt.close()
        
    return save_path

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
            interp_bot = interp1d(a_bot, w_bot, bounds_error=False, fill_value='extrapolate') # (w_bot[0], w_bot[-1])
            w_bot_interp = interp_bot(alpha_vals)
        
        if exc_top:
            w_top_interp = c_window(geometry)[1] # use constant window
        else:
            interp_top = interp1d(a_top, w_top, bounds_error=False, fill_value='extrapolate') # (w_top[0], w_top[-1])
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
        alpha_vals = u_vals
        N_alpha = N_u
        print('using flat detector')
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
        maskTD = np.broadcast_to(mask, (N_s, N_u, N_w))
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
    
    plt.fill_between(alpha_vals, w_vals[0], w_vals[-1], color='grey', alpha=0.5)

    plt.plot([alpha_vals[0]]*len(w_vals), w_vals, '-', color='grey', alpha=0.5, linewidth=3)
    plt.plot([alpha_vals[-1]]*len(w_vals), w_vals, '-', color='grey', alpha=0.5, linewidth=3)

    # Get plot limits
    alpha_min, alpha_max = alpha_vals[0] - 0.05, alpha_vals[-1] + 0.05
    w_min, w_max = np.min(w_bot_interp) - 5, np.max(w_top_interp) + 5
    
    plt.plot(alpha_vals, [w_vals[0]]*len(alpha_vals), '-', color='gray', label='detector bounds', alpha = 0.5, linewidth=3)
    plt.plot(alpha_vals, [w_vals[-1]]*len(alpha_vals), '-', color='gray', alpha = 0.5, linewidth=3)
    plt.plot(alpha_vals, w_bot_interp, color='red', linewidth=4)
    plt.plot(alpha_vals, w_top_interp, color='red', label='TD window', linewidth=4)
    plt.vlines(alpha_vals[0], w_bot_interp[0], w_top_interp[0], color='red', linestyle='-', linewidth=4)
    plt.vlines(alpha_vals[-1], w_bot_interp[-1], w_top_interp[-1], color='red', linestyle='-', linewidth=4)
    
    plt.fill_between(alpha_vals, w_bot_interp, w_top_interp, color='red', alpha=0.5)

    plt.xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    plt.ylabel("w")
    plt.xlim(alpha_min, alpha_max)
    plt.ylim(w_min, w_max)
    plt.legend(loc='upper right')

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
    delta_s_edge = np.arccos(r / R) 

    # Sampling grid in delta-s for one branch
    ds_plus = np.linspace(delta_s_edge, 2*np.pi - delta_s_edge, N_sampling)
    ds_min  = np.linspace(delta_s_edge - 2*np.pi, - delta_s_edge, N_sampling)
    
    if a_ramp <= 0:
        raise ValueError("`a_ramp` must be positive")
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
    ax.set_ylabel("w")
    ax.fill_between(alpha_vals, w_vals[0], w_vals[-1], color='grey', alpha=0.5, label='Detector bounds')
    # Detector bounds
    # line_det_min, = ax.plot(alpha_vals, [w_vals[0]] * len(alpha_vals), '-', color='gray', alpha=0.5, linewidth=3)
    # line_det_max, = ax.plot(alpha_vals, [w_vals[-1]] * len(alpha_vals), '-', color='gray', alpha=0.5, linewidth=3)
    # TD window (costant)
    # const_line_bot, = ax.plot(alpha_vals, c_window(geometry)[0], color='orange', label='TD const', linewidth=4)
    # const_line_top, = ax.plot(alpha_vals, c_window(geometry)[1], color='orange', linewidth=4)
    ax.fill_between(alpha_vals, c_window(geometry)[0], c_window(geometry)[1], color='orange', alpha=0.5, label='Constant window')
    # TD window (variable)
    line_bot, = ax.plot([], [], color='red', linewidth=1, label='Variable window')
    line_top, = ax.plot([], [], color='red', linewidth=1)
    line_left, = ax.plot([], [], color='red', linestyle='-', linewidth=1)
    line_right, = ax.plot([], [], color='red', linestyle='-', linewidth=1)

    ax.set_xlim(alpha_vals[0] - 0.1, alpha_vals[-1] + 0.1)
    ax.set_ylim(w_vals[0] - 10, w_vals[-1] + 10)

    Idx = list(range(0, N_s, step))

    def update(i):
        i = Idx[i]
        w_bot_interp, w_top_interp = nc_window_slice(geometry, ds_plus, ds_min, i)
        line_bot.set_data(alpha_vals, w_bot_interp)
        line_top.set_data(alpha_vals, w_top_interp)
        x_val_0 = alpha_vals[0]
        y1 = w_bot_interp[0]
        y2 = w_top_interp[0]
        x_val_end = alpha_vals[-1]
        y3 = w_bot_interp[-1]
        y4 = w_top_interp[-1]
        line_left.set_data([x_val_0, x_val_0], [y1, y2])
        line_right.set_data([x_val_end, x_val_end], [y3, y4])
        ax.legend(loc='upper right')

        ax.set_title(f"slice {i} / {N_s - 1}")
        return line_bot, line_top #, line_det_min, line_det_max, const_line_bot, const_line_top

    ani = FuncAnimation(fig, update, frames=len(Idx), blit=True, interval=1)
    ani.save(path + "td_nonconst_anim.mp4", writer="ffmpeg", fps=15)

    IdxImages = [3*N_s//7, 4*N_s//9, N_s//2, 5*N_s//9, 4*N_s//7]
    # saave the image of given frames
    for i in IdxImages:
        w_bot_interp, w_top_interp = nc_window_slice(geometry, ds_plus, ds_min, i)
        plt.figure()
        plt.fill_between(alpha_vals, w_vals[0], w_vals[-1], color='grey', alpha=0.5, label='Detector')
        plt.fill_between(alpha_vals, c_window(geometry)[0], c_window(geometry)[1], color='orange', alpha=0.5, label='Constant TD window')
        
        plt.plot(alpha_vals, w_bot_interp, color='red', linewidth=1)
        plt.plot(alpha_vals, w_top_interp, color='red', label='Variable TD window', linewidth=1)
        plt.vlines(alpha_vals[0], w_bot_interp[0], w_top_interp[0], color='red', linestyle='-', linewidth=1)
        plt.vlines(alpha_vals[-1], w_bot_interp[-1], w_top_interp[-1], color='red', linestyle='-', linewidth=1)

        plt.xlabel(r"$\alpha$" if geometry.det_curvature_radius is not None else "u")
        plt.ylabel("w")
        plt.xlim(alpha_vals[0] - 0.1, alpha_vals[-1] + 0.1)
        plt.ylim(w_vals[0] - 10, w_vals[-1] + 10)
        if i == 3279:
            plt.legend(loc='upper right')
        plt.savefig(path+f"td_nonconst_{i}.svg")
        plt.close()
    print('done with video and images')

def save_intensity_row(phantom, reco, test, path, reco2 = None, v1=False):

    if not isinstance(phantom, np.ndarray):
        phantom = phantom.asarray()
    Nz, Ny, Nx = phantom.shape
    slicesIdx = [Nz*2//8, Nz * 3//8, Nz * 4//8, Nz * 5//8, Nz * 6//8]  # Fixed y index for the row

    yIdx = Ny // 2  # Fixed y index for the row
    xIdx = Nx // 2  # Fixed x index for the column

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], '--', color='grey', label='Phantom')
    line2, = ax.plot([], [], '-', color='red' if v1 else 'blue', label='Kats Reconstruction')
    if reco2 is not None:
        line3, = ax.plot([], [], '-', color='orange', label='FBP Reconstruction')
    else:
        line3 = None
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

    ax.set_xlim(0, Ny)
    ax.set_ylim(max(np.min([phantom, reco]), -2.), min(np.max([phantom, reco]), 2.))
    ax.set_xlabel('y')
    ax.set_ylabel('Intensity')
    ax.grid(True)
    ax.legend()

    # Update function for animation
    def update(z):
        phantom_profile = phantom[z, :, xIdx]
        reco_profile = reco[z, :, xIdx]

        line1.set_ydata(phantom_profile)
        line2.set_ydata(reco_profile)
        if reco2 is not None:
            reco2_profile = reco2[z, :, xIdx]
            line3.set_ydata(reco2_profile)
        line1.set_xdata(np.arange(Ny))
        line2.set_xdata(np.arange(Ny))
        title.set_text(f'z = {z} / {Nz - 1}')
        if reco2 is not None:
            line3.set_xdata(np.arange(Ny))
            return line1, line2, line3, title
        else:
            return line1, line2, title

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nz, interval=100, blit=True)
    # Save as MP4 (requires ffmpeg) or show inline
    if v1:
        ani.save(path + f'profiles_v1_t0{test}.mp4', fps=10)
    else:
        ani.save(path + f'profiles_t0{test}.mp4', fps=10)
    # Save static plots for selected z-slices
    # ylim = (max(np.min([phantom, reco]), -2.), min(np.max([phantom, reco]), 2.))
    ylim = (-0.5, 1.1)
    for z in slicesIdx:
        plt.figure(figsize=(6, 4))
        phantom_profile = phantom[z, :, xIdx]
        reco_profile = reco[z, :, xIdx]

        plt.plot(phantom_profile, '--', color='grey', label='Phantom')
        plt.plot(reco_profile, '-', color='red' if v1 else 'blue', label='Kats Reconstruction')
        if reco2 is not None:
            reco2_profile = reco2[z, :, xIdx]
            plt.plot(reco2_profile, '-', color='orange', label='FBP Reconstruction')
        plt.legend()
        # plt.title(f'{z}/{Nz-1}')
        plt.xlabel('y')
        plt.ylabel('Intensity')
        plt.ylim(ylim)
        # plt.grid(True)
        if v1:
            plt.savefig(os.path.join(path, f'profile_v1_t0{test}_z{z}.svg'), format='svg')
        else:
            plt.savefig(os.path.join(path, f'profile_t0{test}_z{z}.svg'), format='svg')
        plt.close()

def save_intensity_row_wood(reco1, reco2, path):

    Nz, Ny, Nx = reco1.shape

    yIdxList = [Ny // 4, 3 * Ny // 8, 5 * Ny // 8, 6 * Ny // 8]  # Fixed y index for the row
    # xIdx = Nx // 2  # Fixed x index for the column
    for yIdx in yIdxList:
        plt.figure(figsize=(10, 8))

        reco1_profile = reco1[0, yIdx, :]
        reco2_profile = reco2[0, yIdx, :]

        plt.plot(reco1_profile, '-', color='orange', label='FBP', linewidth=2)
        plt.plot(reco2_profile, '-', color='red', label='Kats', linewidth=2)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Intensity')

        plt.savefig(os.path.join(path, f'profile_y_{yIdx}.svg'), format='svg')
        plt.close()

def save_intensity_error_derivative_comparison(phantom, reco1, reco2, test, path):

    if not isinstance(phantom, np.ndarray):
        phantom = phantom.asarray()

    Nz, Ny, Nx = phantom.shape
    slicesIdx = [Nz * 2//8, Nz * 3//8, Nz * 4//8, Nz * 5//8, Nz * 6//8]  # Fixed y index for the row
    yIdx = Ny // 2  # Fixed y index for the row

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], 'r-', label='NPH')
    line2, = ax.plot([], [], 'b-', label='K')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

    ax.set_xlim(0, Nx)
    ax.set_ylim(np.min([phantom - reco1, phantom - reco2]), np.max([phantom - reco1, phantom - reco2]))
    ax.set_xlabel('x')
    ax.set_ylabel('Intensity Error')
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
        title.set_text(f'Index z = {z} / {Nz - 1}')
        return line1, line2, title

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nz, interval=100, blit=True)
    # Save as MP4 (requires ffmpeg) or show inline
    ani.save(path + f'profiles_error_der_compare_t0{test}.mp4', fps=10)

    # Save static plots for selected z-slices
    for z in slicesIdx:
        plt.figure(figsize=(6, 4))
        phantom_profile = phantom[z, yIdx, :]
        reco1_profile = reco1[z, yIdx, :]
        reco2_profile = reco2[z, yIdx, :]

        plt.plot(phantom_profile - reco1_profile, 'r-', label='NPH')
        plt.plot(phantom_profile - reco2_profile, 'b-', label='K')
        plt.title(f'Index z={z}/{Nz-1}')
        plt.xlabel('x')
        plt.ylabel('Intensity Error')
        plt.grid(True)
        plt.savefig(os.path.join(path, f'compare_diff_profile_t0{test}_z{z}.svg'), format='svg')
        plt.close()

def save_intensity_error_rows(phantom, reco, test, path):

    # Set up the figure
    if not isinstance(phantom, np.ndarray):
        phantom = np.asarray(phantom)

    Nz, Ny, Nx = phantom.shape
    slicesIdx = [2 * Ny // 8, 3 * Ny // 8, 4 * Ny // 8, 5 * Ny // 8, 6 * Ny // 8]  # Fixed y index for the row
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
    ax.set_title(f'Error on different detector rows')
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

def plot_psnr(phantom_arr, path, t=None, kats_reco=None, fbp_reco=None, kats_reco_v1=None, ffs=False):
    """Compute PSNR between two slices."""
    from skimage.metrics import peak_signal_noise_ratio
    from matplotlib import pyplot as plt
    import numpy as np
    if kats_reco is None and fbp_reco is None and kats_reco_v1 is None:
        raise ValueError("At least one reconstruction must be provided (kats_reco, fbp_reco, kats_reco_v1).")
    psnr_per_slice = []
    psnr_per_slice_fbp = []
    if kats_reco_v1 is not None:
        psnr_per_slice_nph = []
    step = 3
    phantom_max = np.max(phantom_arr)
    phantom_min = np.min(phantom_arr)
    data_range = phantom_max - phantom_min
    # if fbp_reco is not array, convert it to numpy array
    if fbp_reco is not None and not isinstance(fbp_reco, np.ndarray):
        fbp_reco = fbp_reco.asarray()
    for z in range(0, phantom_arr.shape[0], step):
        psnr = peak_signal_noise_ratio(phantom_arr[z], kats_reco[z], data_range=data_range)
        psnr_fbp = peak_signal_noise_ratio(phantom_arr[z], fbp_reco[z], data_range=data_range)
        psnr_per_slice.append(psnr)
        psnr_per_slice_fbp.append(psnr_fbp)
        if kats_reco_v1 is not None:
            psnr_nph = peak_signal_noise_ratio(phantom_arr[z], kats_reco_v1[z], data_range=data_range)
            psnr_per_slice_nph.append(psnr_nph)

    plt.figure(figsize=(10, 4))
    # plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice)), cap_inf(psnr_per_slice), marker='o', markersize=3, color='blue', linewidth=0.4, label='Katsevich K')
    # plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice_nph)), cap_inf(psnr_per_slice_nph), marker='x', markersize=3, color='red', linewidth=0.4, label='Katsevich NPH')
    # plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice_fbp)), cap_inf(psnr_per_slice_fbp), marker='^', markersize=3, color='orange', linewidth=0.4, label='FBP')
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice)), psnr_per_slice, marker='o', markersize=3, color='blue', linewidth=0.4, label='Katsevich K')
    if kats_reco_v1 is not None:
        plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice_nph)), psnr_per_slice_nph, marker='x', markersize=3, color='red', linewidth=0.4, label='Katsevich NPH')
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(psnr_per_slice_fbp)), psnr_per_slice_fbp, marker='^', markersize=3, color='orange', linewidth=0.4, label='FBP')
    plt.xlabel("Slice index (z)")
    plt.ylabel("PSNR (dB)")
    # plt.title("PSNR per slice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if t:
        plt.savefig(os.path.join(path, f'psnr_per_slice_t0{t}.svg'))
    elif ffs:
        plt.savefig(os.path.join(path, 'psnr_per_slice_ffs.svg'))
    else:
        plt.savefig(os.path.join(path, 'psnr_per_slice.svg'))

def plot_ssim(phantom_arr, path, t=None, kats_reco=None, fbp_reco=None, kats_reco_v1=None, ffs=False):

    from skimage.metrics import structural_similarity as ssim
    import numpy as np  
    ssim_list = []
    ssim_list_fbp = []
    if kats_reco_v1 is not None:
        ssim_list_nph = []
    phantom_min = np.min(phantom_arr)
    phantom_max = np.max(phantom_arr)
    data_range = phantom_max - phantom_min
    step = 3
    # if fbp_reco is not array, convert it to numpy array
    if fbp_reco is not None and not isinstance(fbp_reco, np.ndarray):
        fbp_reco = fbp_reco.asarray()
    for z in range(0, phantom_arr.shape[0], step):
        ssim_k = ssim(
            phantom_arr[z], kats_reco[z],
            data_range=data_range)
        ssim_fbp = ssim(
            phantom_arr[z], fbp_reco[z],
            data_range=data_range)
        
        ssim_list.append(ssim_k)
        ssim_list_fbp.append(ssim_fbp)
        if kats_reco_v1 is not None:
            ssim_nph = ssim(
                phantom_arr[z], kats_reco_v1[z],
                data_range=data_range)
            ssim_list_nph.append(ssim_nph)
    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(ssim_list)),
             ssim_list, marker='o', markersize=3, color='blue', linewidth=0.4, label='Katsevich K')
    if kats_reco_v1 is not None:
        plt.plot(np.linspace(0, phantom_arr.shape[0], len(ssim_list_nph)),
             ssim_list_nph, marker='x', markersize=3, color='red', linewidth=0.4, label='Katsevich NPH')
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(ssim_list_fbp)),
             ssim_list_fbp, marker='^', markersize=3, color='orange', linewidth=0.4, label='FBP')
    plt.xlabel("Slice index (z)")
    plt.ylabel("SSIM")
    # plt.title("SSIM per slice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if t:
        plt.savefig(os.path.join(path, f'ssim_per_slice_t0{t}.svg'))
    elif ffs:
        plt.savefig(os.path.join(path, 'ssim_per_slice_ffs.svg'))
    else:
        plt.savefig(os.path.join(path, 'ssim_per_slice.svg'))

def plot_rmse(phantom_arr, path, t=None, kats_reco=None, fbp_reco=None, kats_reco_v1=None, ffs=False):

    from skimage.metrics import mean_squared_error
    import numpy as np  
    rmse_list = []
    rmse_list_fbp = []
    if kats_reco_v1 is not None:
        rmse_list_nph = []
    phantom_min = np.min(phantom_arr)
    phantom_max = np.max(phantom_arr)
    data_range = phantom_max - phantom_min
    step = 3
    # if fbp_reco is not array, convert it to numpy array
    if fbp_reco is not None and not isinstance(fbp_reco, np.ndarray):
        fbp_reco = fbp_reco.asarray()
    for z in range(0, phantom_arr.shape[0], step):
        rmse_k = np.sqrt(mean_squared_error(
            phantom_arr[z], kats_reco[z]))
        rmse_fbp = np.sqrt(mean_squared_error(
            phantom_arr[z], fbp_reco[z]))
        
        rmse_list.append(rmse_k)
        rmse_list_fbp.append(rmse_fbp)
        if kats_reco_v1 is not None:
            rmse_nph = np.sqrt(mean_squared_error(
                phantom_arr[z], kats_reco_v1[z]))
            rmse_list_nph.append(rmse_nph)
    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(rmse_list)),
             rmse_list, marker='o', markersize=3, color='blue', linewidth=0.4, label='Katsevich K')
    if kats_reco_v1 is not None:
        plt.plot(np.linspace(0, phantom_arr.shape[0], len(rmse_list_nph)),
             rmse_list_nph, marker='x', markersize=3, color='red', linewidth=0.4, label='Katsevich NPH')
    plt.plot(np.linspace(0, phantom_arr.shape[0], len(rmse_list_fbp)),
             rmse_list_fbp, marker='^', markersize=3, color='orange', linewidth=0.4, label='FBP')
    plt.xlabel("Slice index (z)")
    plt.ylabel("RMSE")
    # plt.title("RMSE per slice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  
    if t:
        plt.savefig(os.path.join(path, f'rmse_per_slice_t0{t}.svg'))
    elif ffs:
        plt.savefig(os.path.join(path, 'rmse_per_slice_ffs.svg'))
    else:
        plt.savefig(os.path.join(path, 'rmse_per_slice.svg'))
