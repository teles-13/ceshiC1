import h5py
import torch
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心 G 矩阵代码 (保留原始逻辑)
# ==========================================
def even_pisco(int_val):
    return int_val % 2 == 0

def ChC_FFT_convolutions(X, N1, N2, Nc, tau, pad, kernel_shape):
    device = X.device
    grid_1d = torch.arange(-tau, tau+1, device=device)
    in1, in2 = torch.meshgrid(grid_1d, grid_1d, indexing='xy')
    
    if kernel_shape == 1:
        mask = in1**2 + in2**2 <= tau**2
        mask_flat = mask.t().flatten() 
        i = torch.where(mask_flat)[0]
    else:
        i = torch.arange(in1.numel(), device=device)
        
    in1 = in1.t().flatten()[i]
    in2 = in2.t().flatten()[i]
    patchSize = len(in1)

    if pad:
        N1n = 2 ** math.ceil(math.log2(N1 + 2*tau))
        N2n = 2 ** math.ceil(math.log2(N2 + 2*tau))
    else:
        N1n = N1
        N2n = N2

    row_inds = (N1n // 2) - in1.unsqueeze(1) + in1.unsqueeze(0)
    col_inds = (N2n // 2) - in2.unsqueeze(1) + in2.unsqueeze(0)
    row_inds = torch.clamp(row_inds, 0, N1n-1).long()
    col_inds = torch.clamp(col_inds, 0, N2n-1).long()
    
    inds = col_inds * N1n + row_inds
    
    n1_freq = torch.fft.fftshift(torch.fft.fftfreq(N1n, device=device))
    n2_freq = torch.fft.fftshift(torch.fft.fftfreq(N2n, device=device))
    n2, n1 = torch.meshgrid(n2_freq, n1_freq, indexing='xy')
    
    phaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * ((N1n+1)//2 + tau) + n2 * ((N2n+1)//2 + tau)))
    cphaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * ((N1n+1)//2) + n2 * ((N2n+1)//2)))

    x = torch.fft.fft2(X, s=(N1n, N2n), dim=(0,1)) * phaseKernel.unsqueeze(2)

    PhP = torch.zeros((patchSize, patchSize, Nc, Nc), dtype=torch.complex64, device=device)
    for q in range(Nc):
        x_rest = x[:, :, q:]
        x_q = x[:, :, q] 
        prod = torch.conj(x_rest) * x_q.unsqueeze(2) * cphaseKernel.unsqueeze(2)
        b = torch.fft.ifft2(prod, dim=(0,1)) 

        b_flat = b.permute(1, 0, 2).reshape(-1, Nc - q)
        inds_flat = inds.t().flatten()
        b_selected = b_flat[inds_flat, :]
        b_selected = b_selected.view(patchSize, patchSize, Nc - q).permute(1, 0, 2)
        
        PhP[:, :, q:, q] = b_selected
        if q < Nc - 1:
            PhP[:, :, q, q+1:] = torch.conj(PhP[:, :, q+1:, q].permute(1, 0, 2))

    PhP = PhP.permute(0, 2, 1, 3) 
    PhP = PhP.permute(3, 2, 1, 0).reshape(patchSize * Nc, patchSize * Nc).t()
    return PhP

def nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape):
    ChC = ChC_FFT_convolutions(kCal, kCal.shape[0], kCal.shape[1], kCal.shape[2], tau, 1, kernel_shape)
    U_svd, S, Vh = torch.linalg.svd(ChC, full_matrices=False)
    sing = torch.sqrt(torch.abs(S))
    sing = sing / sing[0]
    
    valid_idx = torch.where(sing >= threshold * sing[0])[0]
    Nvect = valid_idx[-1].item()
    
    U = Vh.conj().t()[:, Nvect+1:] 
    return U

def G_matrices(kCal, N1, N2, tau, U, kernel_shape):
    device = kCal.device
    N1_cal, N2_cal, Nc = kCal.shape
    grid_1d = torch.arange(-tau, tau + 1, device=device)
    in1, in2 = torch.meshgrid(grid_1d, grid_1d, indexing='xy')
    
    flat_in1 = in1.t().flatten()
    flat_in2 = in2.t().flatten()
    
    if kernel_shape == 0:
        ind = torch.arange(len(flat_in1), device=device)
    else:
        mask = in1**2 + in2**2 <= tau**2
        ind = torch.where(mask.t().flatten())[0]
        
    in1 = flat_in1[ind].long()
    in2 = flat_in2[ind].long()
    
    patchSize = len(in1)
    eind = torch.arange(patchSize, 0, -1, device=device) - 1
    total_size = 2 * (2 * tau + 1)
    
    G_flat = torch.zeros((total_size * total_size, Nc, Nc), dtype=torch.complex64, device=device)
    
    W = U @ U.conj().t()
    W = W.t().reshape(Nc, patchSize, Nc, patchSize).permute(3, 2, 1, 0)
    W = W.permute(0, 1, 3, 2)
    
    for s in range(patchSize):
        r0 = 2 * tau + 1 + in1[eind] + in1[s] 
        c0 = 2 * tau + 1 + in2[eind] + in2[s] 
        r0 = torch.clamp(r0, 0, total_size-1)
        c0 = torch.clamp(c0, 0, total_size-1)
        linear_idx = c0 * total_size + r0 
        G_flat[linear_idx, :, :] += W[:, :, :, s]

    G = G_flat.permute(2, 1, 0).reshape(Nc, Nc, total_size, total_size).permute(3, 2, 1, 0)
    
    N1_g, N2_g = N1, N2 
    n1 = torch.fft.fftfreq(N1_g, device=device)
    n2 = torch.fft.fftfreq(N2_g, device=device)
    n2, n1 = torch.meshgrid(n2, n1, indexing='xy')
    
    phaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * (N1_g - 2*tau - 1) + n2 * (N2_g - 2*tau - 1)))
    # [修改] 增加 norm='ortho' 保持能量守恒，匹配图像域算子的量级
    G = torch.fft.ifft2(G, s=(N1_g, N2_g), dim=(0,1), norm='forward') * phaseKernel.unsqueeze(2).unsqueeze(3)
    G = torch.fft.fftshift(G, dim=(0,1))
    
    return G
    
    

def compute_G_for_fastmri_slice(kspace_slice, cal_length=32, tau=3, threshold=0.08, kernel_shape=1):
    kData = kspace_slice.permute(1, 2, 0)
    N1, N2, Nc = kData.shape
    device = kData.device
    
    center_x = int(np.ceil(N1 / 2)) + even_pisco(N1)
    center_y = int(np.ceil(N2 / 2)) + even_pisco(N2)
    
    cal_index_x = torch.arange(center_x - int(np.floor(cal_length / 2)), 
                               center_x + int(np.floor(cal_length / 2)) - even_pisco(cal_length), device=device)
    cal_index_y = torch.arange(center_y - int(np.floor(cal_length / 2)), 
                               center_y + int(np.floor(cal_length / 2)) - even_pisco(cal_length), device=device)
    
    kCal = kData[cal_index_x.unsqueeze(1), cal_index_y, :]
    
    U = nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape)
    G_tensor = G_matrices(kCal, N1, N2, tau, U, kernel_shape)
    
    return G_tensor

# ==========================================
# 2. 基础 MRI 算子与裁剪
# ==========================================
def fft2c(img):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

def ifft2c(kspace):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

def diff_x(img):
    return torch.roll(img, shifts=-1, dims=-1) - img

def diff_y(img):
    return torch.roll(img, shifts=-1, dims=-2) - img

def adj_diff_x(img):
    return torch.roll(img, shifts=1, dims=-1) - img

def adj_diff_y(img):
    return torch.roll(img, shifts=1, dims=-2) - img

# [新增] 中心裁剪函数
def center_crop(data, shape):
    """对最后两维进行中心裁剪"""
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

# ==========================================
# 3. 可视化保存函数 (已更新：裁剪 + 纯灰度)
# ==========================================
def save_intermediate_results(u_tensor, c_tensor, iter_num, save_dir=".", crop_size=(320, 320)):
    """保存每次迭代的 u 和 c_q，自动裁剪并使用灰度图，分别存入不同子文件夹"""
    
    # [新增] 自动创建两个子文件夹
    u_dir = os.path.join(save_dir, "recon_u")
    cq_dir = os.path.join(save_dir, "coil_sensitivities")
    os.makedirs(u_dir, exist_ok=True)   # exist_ok=True 表示如果文件夹已存在则不报错
    os.makedirs(cq_dir, exist_ok=True)

    # 裁剪并转为幅值 numpy
    u_mag = torch.abs(center_crop(u_tensor, crop_size)).detach().cpu().numpy()
    c_mag = torch.abs(center_crop(c_tensor, crop_size)).detach().cpu().numpy()
    Nc = c_mag.shape[0]

    u_vmax = np.percentile(u_mag, 99.5) 

    # 1. 保存重建图像 u (灰度图)
    plt.figure(figsize=(6, 6))
    plt.imshow(u_mag, cmap='gray', vmin=0, vmax=u_vmax)
    plt.title(f"Reconstructed Image u - Iteration {iter_num}")
    plt.axis('off')
    # [修改] 存入 u_dir
    plt.savefig(os.path.join(u_dir, f"u_iter_{iter_num:02d}.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 2. 保存敏感度图 c_q (网格排布, 灰度图)
    cols = int(np.ceil(np.sqrt(Nc)))
    rows = int(np.ceil(Nc / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if Nc > 1 else [axes]
    
    for q in range(Nc):
        c_vmax = np.percentile(c_mag[q], 99.5)
        # [更新] 全部统一使用 gray 映射
        axes[q].imshow(c_mag[q], cmap='gray', vmin=0, vmax=c_vmax)
        axes[q].axis('off')
        axes[q].set_title(f"Coil {q+1}")
        
    for q in range(Nc, len(axes)):
        axes[q].axis('off')
        
    plt.suptitle(f"Coil Sensitivities - Iteration {iter_num}", fontsize=14)
    plt.tight_layout()
    # [修改] 存入 cq_dir
    plt.savefig(os.path.join(cq_dir, f"cq_iter_{iter_num:02d}.png"), bbox_inches='tight', dpi=150)
    plt.close()

def cg_solve(A_func, b, x0=None, max_iter=10, tol=1e-6):
    # 【修复】支持传入初始猜测值 x0 (Warm Start)
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        # 计算初始残差: r = b - A(x0)
        r = b - A_func(x)
        
    p = r.clone()
    rsold = torch.sum(torch.real(r.conj() * r))
    
    if rsold < 1e-10:
        return x

    for i in range(max_iter):
        Ap = A_func(p)
        alpha = rsold / (torch.sum(torch.real(p.conj() * Ap)) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.real(r.conj() * r))
        
        if torch.sqrt(rsnew) < tol:
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x

# ==========================================
# 5. 联合交替优化主算法
# ==========================================
class SenseJacobianSolver:
    def __init__(self, k_hat, mask, G_tensor, acs_lines=32, lambda_reg=0.01, beta_reg=0.01, eps=1e-6):
        self.k_hat = k_hat
        self.mask = mask
        self.G_tensor = G_tensor
        self.Nc, self.N1, self.N2 = k_hat.shape
        self.acs_lines = acs_lines # 传入 ACS 宽度用于汉明窗
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg
        self.eps = eps

    def _apply_low_pass(self, c_tensor):
        """对敏感度图进行二维汉明窗低通滤波"""
        device = c_tensor.device
        Nc, N1, N2 = c_tensor.shape
        
        # 1. 将敏感度图转换到 k 空间
        k_c = fft2c(c_tensor)
        
        # 2. 构建二维汉明窗掩膜
        cx, cy = N1 // 2, N2 // 2
        hx, hy = self.acs_lines // 2, self.acs_lines // 2
        
        window_x = torch.hamming_window(2 * hx, periodic=False, device=device)
        window_y = torch.hamming_window(2 * hy, periodic=False, device=device)
        window_2d = window_x.unsqueeze(1) * window_y.unsqueeze(0)
        
        # 3. 创建全尺寸的低频 k 空间张量，并把加窗后的中心低频部分填入
        k_low_freq = torch.zeros_like(k_c)
        k_low_freq[:, cx-hx:cx+hx, cy-hy:cy+hy] = k_c[:, cx-hx:cx+hx, cy-hy:cy+hy] * window_2d.unsqueeze(0)
        
        # 4. 逆傅里叶变换切回图像域
        return ifft2c(k_low_freq)

    def solve(self, max_outer_iter=10, cg_iter_u=10, cg_iter_c=10, tol=1e-4, save_dir="."):
        # ========================================
        # [核心优化] 规范化的 2D 汉明窗自适应联合初始化
        # ========================================
        device = self.k_hat.device
        cx, cy = self.N1 // 2, self.N2 // 2
        hx, hy = self.acs_lines // 2, self.acs_lines // 2
        
        # 1. 提取 k 空间低频中心区域，并施加 2D 汉明窗用于线圈敏感度 c_k 的初始化
        k_low_freq = torch.zeros_like(self.k_hat)
        
        window_x = torch.hamming_window(2 * hx, periodic=False, device=device)
        window_y = torch.hamming_window(2 * hy, periodic=False, device=device)
        window_2d = window_x.unsqueeze(1) * window_y.unsqueeze(0)
        
        # 严格从欠采样数据 self.k_hat 中提取中心低频，保证算法的封装性与独立性
        center_data = self.k_hat[:, cx-hx:cx+hx, cy-hy:cy+hy]
        k_low_freq[:, cx-hx:cx+hx, cy-hy:cy+hy] = center_data * window_2d.unsqueeze(0)
        
        img_low_freq = ifft2c(k_low_freq)
        u_low_freq = torch.sqrt(torch.sum(torch.abs(img_low_freq)**2, dim=0))
        
        # 计算初始敏感度图，加入小常数防止除 0
        c_k = img_low_freq / (u_low_freq + 1e-8)


        mask_threshold = 0.05 * torch.max(u_low_freq)
        spatial_mask = (u_low_freq > mask_threshold).to(self.k_hat.dtype)
        # 2. 计算正常零填充图像，用于自适应线圈合并初始化 u_k
        img_zf = ifft2c(self.k_hat)
        
        # 【核心改进】利用刚生成的 c_k 对零填充图像进行相位对齐合并，得到更清晰的初始真实图像 u_0
        numerator = torch.sum(c_k.conj() * img_zf, dim=0)
        denominator = torch.sum(torch.abs(c_k)**2, dim=0) + 1e-8
        u_k = (numerator / denominator).to(self.k_hat.dtype)

        # 保存初始状态 (Iteration 0)
        save_intermediate_results(u_k, c_k, 0, save_dir)

        # ========================================
        # 交替优化主循环
        # ========================================
        for k in range(max_outer_iter):
            print(f"--- Starting Outer Iteration {k+1}/{max_outer_iter} ---")
            u_old = u_k.clone()
            
            # 第一步: 更新 u
            dx_u = diff_x(u_old)
            dy_u = diff_y(u_old)
            W_img = 1.0 / torch.sqrt(torch.abs(dx_u)**2 + torch.abs(dy_u)**2 + self.eps**2)

            def A_u(u_var):
                term1 = torch.zeros_like(u_var)
                for q in range(self.Nc):
                    c_q = c_k[q]
                    img_q = c_q * u_var
                    k_q = self.mask * fft2c(img_q)
                    term1 += c_q.conj() * ifft2c(k_q)
                
                reg_x = adj_diff_x(W_img * diff_x(u_var))
                reg_y = adj_diff_y(W_img * diff_y(u_var))
                reg = self.lambda_reg * (reg_x + reg_y)
                
                return term1 + reg

            b_u = torch.zeros_like(u_old, dtype=self.k_hat.dtype)
            for q in range(self.Nc):
                b_u += c_k[q].conj() * ifft2c(self.k_hat[q])

            u_k = cg_solve(A_u, b_u, x0=u_old, max_iter=cg_iter_u)
            print("  Image u updated.")

            # 第二步: 更新 c_p
            c_new = torch.zeros_like(c_k)
            
            for p in range(self.Nc):
                def A_c(c_p_var):
                    img_p = u_k * c_p_var
                    k_p = self.mask * fft2c(img_p)
                    term_data = u_k.conj() * ifft2c(k_p)
                    
                    term_prior = self.beta_reg * self.G_tensor[:, :, p, p] * c_p_var
                    return term_data + term_prior
                
                v_p_data = u_k.conj() * ifft2c(self.k_hat[p])
                
                v_p_prior = torch.zeros_like(v_p_data)
                for q in range(self.Nc):
                    if q != p:
                        v_p_prior += self.G_tensor[:, :, p, q] * c_k[q]
                
                v_p = v_p_data - self.beta_reg * v_p_prior
                c_new[p] = cg_solve(A_c, v_p, x0=c_k[p], max_iter=cg_iter_c)
                
            c_filtered = self._apply_low_pass(c_new)
            
            # [修改] 直接乘上初始化时生成的实心掩膜，切掉背景且保留大脑内部
            c_k = c_filtered * spatial_mask.unsqueeze(0)
            print("  Coil sensitivities c_q updated and masked.")
            
            # 保存本次外层迭代的图像结果
            save_intermediate_results(u_k, c_k, k+1, save_dir)
            
            
            # 收敛性判断
            diff = torch.norm(u_k - u_old) / (torch.norm(u_old) + 1e-8)
            print(f"  Relative change of u: {diff:.6f}\n")
            if diff < tol:
                print("Converged early.")
                break
                
        return u_k, c_k

# ==========================================
# 6. 主程序运行逻辑
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    h5_path = "/home/liujunda/data_fastmri_brain_train/multicoil_train/file_brain_AXFLAIR_200_6002425.h5"
    
    print("Loading fastMRI data...")
    try:
        with h5py.File(h5_path, 'r') as f:
            kspace_data = f['kspace'][()] 
            
        slice_idx = 7
        kspace_slice = torch.tensor(kspace_data[slice_idx], dtype=torch.complex64).to(device)
    except Exception as e:
        print(f"Data loading failed: {e}. Exiting.")
        exit(1)

    Nc, N1, N2 = kspace_slice.shape
    print(f"Data shape: Coils={Nc}, Matrix={N1}x{N2}")

    # ===============================
    # 模拟欠采样 (生成 Mask)
    # ===============================
    acs_lines = 32
    acc_rate = 4
    mask1d = torch.zeros(N2, device=device)
    mask1d[::acc_rate] = 1
    center_start = N2 // 2 - acs_lines // 2
    mask1d[center_start : center_start + acs_lines] = 1
    mask = mask1d.unsqueeze(0).expand(N1, -1)

    k_hat = kspace_slice * mask.unsqueeze(0)
    
    img_zf_temp = ifft2c(k_hat)
    scale_factor = torch.max(torch.abs(img_zf_temp))
    kspace_slice = kspace_slice / scale_factor
    k_hat = k_hat / scale_factor
    # ===============================
    # 计算 G_tensor
    # ===============================
    print("Computing G_tensor via ESPIRiT null-space calibration...")
    G_tensor = compute_G_for_fastmri_slice(k_hat, cal_length=acs_lines, tau=3, threshold=0.08, kernel_shape=1)

    G_tensor = G_tensor / (torch.max(torch.abs(G_tensor)) + 1e-12)

    print(f"G_tensor shape: {G_tensor.shape}")

    # ===============================
    # 获取当前代码所在路径，作为保存结果的目录
    # ===============================
    save_directory = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
    print(f"Images will be saved to: {save_directory}")

    # ===============================
    # 运行交替优化 SENSE 重建
    # ===============================
    print("Starting Alternating Optimization...")
    # 传入 acs_lines，供汉明窗初始化使用
    solver = SenseJacobianSolver(k_hat, mask, G_tensor, acs_lines=acs_lines, lambda_reg=0.05, beta_reg=0.001)
    
    u_recon, c_recon = solver.solve(
        max_outer_iter=10, 
        cg_iter_u=20, 
        cg_iter_c=10, 
        save_dir=save_directory
    )

    print("Reconstruction finished successfully!")
