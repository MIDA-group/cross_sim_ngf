
###
### Multimodal registration with exhaustive search mutual information
### Author: Johan \"{O}fverstedt
###

from numpy.random.mtrand import random
import time
import torch
import torch.nn.functional
import numpy as np
import torch.nn.functional as F
import torch.fft
import torchvision.transforms.functional as TF
import transformations

import util3d

VALUE_TYPE = torch.float32
ALIGN_CORNERS = True

def matrix_string(matrix):
    s = '['
    for i in range(matrix.size):
        if i > 0:
            s += ', '
        s += f'{matrix[i]:.02f}'
    s += ']'
    return s       

# Creates a list of random angles
def grid_angles(center, radius, n = 32):
    angles = []

    n_denom = n
    if radius < 180:
        n_denom -= 1

    for i in range(n):
        i_frac = i/n_denom
        ang = center + (2.0 * i_frac - 1.0) * radius
        angles.append(ang)

    return angles

def random_angles_3d(centers, center_prob, radius, n = 32, include_centers=True):
    angles = []

    if not isinstance(centers, list):
        centers = [centers]
    if center_prob is not None:
        mass = np.sum(center_prob)
        p = center_prob / mass
    else:
        p = None
    center_inds = np.arange(len(centers))
    if include_centers:
        angles = angles + centers
    
    for i in range(n):
        c = np.random.choice(center_inds, p=p, replace=True)
        c = centers[c]
        frac = np.random.random(size=(c.size,))
        ang = c + (2.0 * frac - 1.0) * radius

        angles.append(ang)

    return angles

def quasi_random_angles_3d(center, radius, n = 32):
    angles = []
    
    phi3 = 1.220744084605759475361686349108831
    alpha = np.array([1.0/phi3, 1.0/(phi3**2.0), 1.0/(phi3**3.0)])
    state = np.random.random(size=(3,))#np.zeros((3,))    
    for i in range(n):
        frac = state + i * alpha
        frac -= np.floor(frac)
        
        state[:] = frac[:]
        ang = center + (2.0 * frac - 1.0) * radius

        angles.append(ang)

    return angles


### Helper functions

def sum_pool(A, ds_factor):
    return torch.nn.functional.avg_pool3d(A, ds_factor, divisor_override=1)

def compute_entropy(C, N, eps=1e-7):
    p = C/N
    return p*torch.log2(torch.clamp(p, min=eps, max=None))

def float_compare(A, c):
    return torch.clamp(1-torch.abs(A-c), 0.0)

def fft_of_levelsets(A, Q, packing, ds, setup_fn):
    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = np.minimum(a_start + packing, Q)
        levelsets = []
        for a in range(a_start, a_end):
            levelsets.append(float_compare(A, a))
        A_cat = torch.cat(levelsets, 0)
        del levelsets
        ffts = setup_fn(A_cat, ds)
        del A_cat
        fft_list.append((ffts, a_start, a_end))
    return fft_list

def fft(A):
    spectrum = torch.fft.rfftn(A, dim=(2, 3, 4))
    return spectrum

def ifft(Afft):
    res = torch.fft.irfftn(Afft, dim=(2, 3, 4))
    return res

def fftconv(A, B):
    C = A * B
    return C

def corr_target_setup(A, ds=1):
    if ds > 1:
        A = sum_pool(A, ds)
    B = fft(A)
    return B

def corr_template_setup(B, ds=1):
    if ds > 1:
        B = sum_pool(B, ds)
    B_FFT = torch.conj(fft(B))

    return B_FFT

def corr_apply(A, B, sz, do_rounding = True):
    C = fftconv(A, B)

    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3], :sz[4]]

    if do_rounding:
        C = torch.round(C)

    return C

def tf_rotate(I, angle, fill_value, center=None):
    return TF.rotate(I, -angle, center=center, fill=[fill_value, ])

def make_torch_grad_3d(on_gpu=True):
    kernel = np.zeros((3, 1, 3, 3, 3), dtype='float32')
    kernel[0, 0, 1, 1, 0] = -0.5
    kernel[0, 0, 1, 1, 2] = 0.5
    kernel[1, 0, 1, 0, 1] = -0.5
    kernel[1, 0, 1, 2, 1] = 0.5
    kernel[2, 0, 0, 1, 1] = -0.5
    kernel[2, 0, 2, 1, 1] = 0.5
    tkern = torch.from_numpy(kernel)
    if on_gpu:
        tkern = tkern.cuda()
    def apply_grad(B, epsilon=1e-5):
        G = torch.conv3d(B, tkern, bias=None, stride=1, padding=1)
        G_norm = torch.sqrt(torch.sum(G**2, dim=1) + epsilon**2)
        return G / G_norm
    return apply_grad

def corr_apply_multiple(A, B, sz):
    C = fftconv(A[0], B[0])
    for i in range(1, len(A)):
        C += fftconv(A[i], B[i])

    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3], :sz[4]]

    return C

def create_float_tensor(shape, on_gpu, fill_value=None):
    if on_gpu:
        res = torch.cuda.FloatTensor(shape[0], shape[1], shape[2], shape[3], shape[4])
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full((shape[0], shape[1], shape[2], shape[3], shape[4]), fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3], shape[4]), dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

def to_tensor(A, on_gpu=True):
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)

def tf_apply_scale_euler_3d_from_matrix(input, sz, thetas, interpolation_mode='bilinear', on_gpu=False):
    grid = TF.affine_grid(thetas.reshape(1, 3, 4), sz, align_corners=ALIGN_CORNERS).float()
    if on_gpu:
        grid = grid.cuda()
    return TF.grid_sample(input, grid, mode=interpolation_mode, align_corners=ALIGN_CORNERS)

### End helper functions

def align_rigid_ngf(A, B, M_A, M_B, angles, overlap=0.5, enable_partial_overlap=True, squared_mode=True, on_gpu=True, save_maps=False, display_progress=True):
    eps=1e-7

    results = []
    maps = []

    A_tensor = to_tensor(A, on_gpu=on_gpu)
    B_tensor = to_tensor(B, on_gpu=on_gpu)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = to_tensor(M_A, on_gpu)
        A_tensor = M_A * A_tensor
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = to_tensor(M_B, on_gpu)
        
    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_sz = (round(B.shape[-1]*(1.0-overlap)), round(B.shape[-2]*(1.0-overlap)), round(B.shape[-3]*(1.0-overlap)))
        A_tensor = F.pad(A_tensor, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1], partial_overlap_pad_sz[2], partial_overlap_pad_sz[2]), mode='constant', value=0)
        M_A = F.pad(M_A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1], partial_overlap_pad_sz[2], partial_overlap_pad_sz[2]), mode='constant', value=0)
    else:
        partial_overlap_pad_sz = (0, 0, 0)

    ext_ashape = A_tensor.shape
    ext_bshape = B_tensor.shape
    ext_valid_shape = torch.tensor([1, 1, (A_tensor.shape[2])-(B_tensor.shape[2])+1, (A_tensor.shape[3])-(B_tensor.shape[3])+1, (A_tensor.shape[4])-(B_tensor.shape[4])+1], dtype=torch.long)

    grad_fun = make_torch_grad_3d(on_gpu=on_gpu)
    A_grad = grad_fun(A_tensor)
    A_grad = A_grad * M_A

    # use default center of rotation (which is the center point)
    center_of_rotation = [(B_tensor.shape[4]-1) / 2.0, (B_tensor.shape[3]-1) / 2.0, (B_tensor.shape[2]-1) / 2.0]

    M_A_FFT = corr_target_setup(M_A)

    if squared_mode:
        # square terms
        A1_FFT = corr_target_setup(A_grad[:, 0:1, :, :, :]**2)
        A2_FFT = corr_target_setup(A_grad[:, 1:2, :, :, :]**2)
        A3_FFT = corr_target_setup(A_grad[:, 2:3, :, :, :]**2)
        # cross terms (0 and 1, 0 and 2, 1 and 2)
        A4_FFT = 2.0 * corr_target_setup(A_grad[:, 0:1, :, :, :] * A_grad[:, 1:2, :, :, :])
        A5_FFT = 2.0 * corr_target_setup(A_grad[:, 0:1, :, :, :] * A_grad[:, 2:3, :, :, :])
        A6_FFT = 2.0 * corr_target_setup(A_grad[:, 1:2, :, :, :] * A_grad[:, 2:3, :, :, :])
    else:
        A1_FFT = corr_target_setup(A_grad[:, 0:1, :, :, :])
        A2_FFT = corr_target_setup(A_grad[:, 1:2, :, :, :])
        A3_FFT = corr_target_setup(A_grad[:, 2:3, :, :, :])

    print('#Angles: ', len(angles))
    best_mi = -1.0
    best_ang = np.array([0.0, 0.0, 0.0])
    best_matrix = np.zeros((12,))       
    ang_tensors = [torch.tensor(util3d.make_rigid3d_matrix(2.0 * np.pi * angles[i] / 360.0, np.zeros((3,)), np.zeros((3,)), np.zeros((3,)), xyz_mode=True, hom=False)) for i in range(len(angles))]
    if on_gpu:
        ang_tensors = [ang_tensors[i].cuda() for i in range(len(ang_tensors))]
    for ang_ind, ang in enumerate(angles):
        # preprocess B for angle
        #
        B_tensor_rotated = tf_apply_scale_euler_3d_from_matrix(B_tensor, B_tensor.shape, ang_tensors[ang_ind], interpolation_mode='bilinear', on_gpu=False)
        M_B_rotated = tf_apply_scale_euler_3d_from_matrix(M_B, M_B.shape, ang_tensors[ang_ind], interpolation_mode='nearest', on_gpu=False)

        B_tensor_rotated = B_tensor_rotated * M_B_rotated

        B_tensor_rotated = F.pad(B_tensor_rotated, (0, ext_ashape[-1]-ext_bshape[-1], 0, ext_ashape[-2]-ext_bshape[-2], 0, ext_ashape[-3]-ext_bshape[-3], 0, 0, 0, 0), mode='constant', value=0)
        M_B_rotated = F.pad(M_B_rotated, (0, ext_ashape[-1]-ext_bshape[-1], 0, ext_ashape[-2]-ext_bshape[-2], 0, ext_ashape[-3]-ext_bshape[-3], 0, 0, 0, 0), mode='constant', value=0)
        
        B_grad = grad_fun(B_tensor_rotated)
        del B_tensor_rotated

        B_grad = B_grad * M_B_rotated

        M_B_FFT = corr_template_setup(M_B_rotated)
        del M_B_rotated

        N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape), min=eps, max=None)

        del M_B_FFT

        if squared_mode:
            # square terms
            B1_FFT = corr_template_setup(B_grad[:, 0:1, :, :, :]**2)
            B2_FFT = corr_template_setup(B_grad[:, 1:2, :, :, :]**2)
            B3_FFT = corr_template_setup(B_grad[:, 2:3, :, :, :]**2)
            # cross terms (0 and 1, 0 and 2, 1 and 2)
            B4_FFT = corr_template_setup(B_grad[:, 0:1, :, :, :] * B_grad[:, 1:2, :, :, :])
            B5_FFT = corr_template_setup(B_grad[:, 0:1, :, :, :] * B_grad[:, 2:3, :, :, :])
            B6_FFT = corr_template_setup(B_grad[:, 1:2, :, :, :] * B_grad[:, 2:3, :, :, :])
        else:
            B1_FFT = corr_template_setup(B_grad[:, 0:1, :, :, :])
            B2_FFT = corr_template_setup(B_grad[:, 1:2, :, :, :])
            B3_FFT = corr_template_setup(B_grad[:, 2:3, :, :, :])

        if squared_mode:
            NGF = corr_apply_multiple([A1_FFT, A2_FFT, A3_FFT, A4_FFT, A5_FFT, A6_FFT], [B1_FFT, B2_FFT, B3_FFT, B4_FFT, B5_FFT, B6_FFT], ext_valid_shape)
        else:
            NGF = corr_apply_multiple([A1_FFT, A2_FFT, A3_FFT], [B1_FFT, B2_FFT, B3_FFT], ext_valid_shape)
        NGF = NGF / N

        if save_maps:
            maps.append(NGF.cpu().numpy())

        (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
        N_filt = torch.lt(N, overlap*max_n)
        NGF[N_filt] = -1.0
        del N_filt, N

        NGF_vec = torch.reshape(NGF, (-1,))
        (val, ind) = torch.max(NGF_vec, -1)

        val_cpu = val.item()
        if val_cpu > best_mi:
            best_mi = val
            best_ang = ang
            best_matrix[:] = ang_tensors[ang_ind].cpu().numpy().reshape((12,))
        
        results.append((ang, val, ind, ang_tensors[ang_ind]))

        #NGF.fill_(-1.0)
        unsquared_str = '' if squared_mode else ' unsquared'
        print(f'{100.0*(1+ang_ind)/len(angles):.1f}: NGF{unsquared_str}: {best_mi:.4f}, Angle: {best_ang[0]:.2f}, {best_ang[1]:.2f}, {best_ang[2]:.2f}, {matrix_string(best_matrix)}           \r', end='')


    print('\n-------------------------------------------')
    cpu_results = []
    for i in range(len(results)):
        ang = results[i][0]
        maxval = results[i][1].cpu().numpy()
        maxind = results[i][2].cpu().numpy()
        rotmatrix = results[i][3].cpu().numpy()
        sub = np.unravel_index(maxind, ext_valid_shape.numpy().astype('int'))
        z = sub[-3]
        y = sub[-2]
        x = sub[-1]
        cpu_results.append((maxval, ang, -(z - partial_overlap_pad_sz[2]), -(y - partial_overlap_pad_sz[1]), -(x - partial_overlap_pad_sz[0]), center_of_rotation[2], center_of_rotation[1], center_of_rotation[0], rotmatrix))
    cpu_results = sorted(cpu_results, key=(lambda tup: tup[0]), reverse=True)
    # print top 10
    top_ind = np.minimum(20, len(cpu_results))
    for i in range(top_ind):
        res = cpu_results[i]
        print(float(res[0]), res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8].reshape((res[8].size,)))
    print('-------------------------------------------')
    # Return the maximum found
    if save_maps:
        return cpu_results, maps
    else:
        return cpu_results, None

def to_tup(value, n):
    if isinstance(value, tuple) or isinstance(value, list):
        assert(len(value) == n)
        return value
    else:
        return tuple([value] * n)

def align_rigid_and_refine_ngf(A, B, M_A, M_B, angles_n, angles_max, start_angle, starting_points, rand_methods, overlap=0.5, enable_partial_overlap=True, algo='gpu_squared', save_maps=False, display_progress=True):
    stages = len(angles_n)
    A = to_tup(A, stages)
    B = to_tup(B, stages)
    M_A = to_tup(M_A, stages)
    M_B = to_tup(M_B, stages)
    algo = to_tup(algo, stages)
    starting_points = to_tup(starting_points, stages-1)
    
    # Currently, only cube images are supported for image B.
    for i in range(len(B)):
        assert(B[i].shape[0]==B[i].shape[1] and B[i].shape[0]==B[i].shape[2])
    
    if start_angle is None:
        start_angle = np.array([0.0, 0.0, 0.0])
    maps = []
    for r in range(stages):
        assert(algo[r] == 'gpu_ngf' or algo[r] == 'gpu_ngf_unsquared' or algo[r] == 'cpu_ngf' or algo[r] == 'cpu_ngf_unsquared')
        squared_mode = not ('unsquared' in algo[r])
        on_gpu = 'gpu' in algo[r]

        if rand_methods[r] == 'quasi':
            ang = quasi_random_angles_3d(start_angle, angles_max[r], angles_n[r])
        elif rand_methods[r] == 'rand':
            ang = random_angles_3d(start_angle, None, angles_max[r], angles_n[r])
        t1 = time.time()
        param, maps_r = align_rigid_ngf(A[r], B[r], M_A[r], M_B[r], ang, overlap, enable_partial_overlap, squared_mode=squared_mode, on_gpu=on_gpu, save_maps=save_maps, display_progress=display_progress)
        t2 = time.time()
        if display_progress:
            print(f'Time elapsed for stage {r+1}: {t2-t1:.02f}s.')
        if maps_r is not None:
            maps = maps + maps_r
        last_param = param[0]
        if r + 1 < stages:
            start_angle = [np.array(param[i][1]) for i in range(starting_points[r])]
    return last_param, maps, param
