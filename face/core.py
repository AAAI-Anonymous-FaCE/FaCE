# face/core.py
# FaCE core algorithm and helpers (paper-faithful, with optional numerics).

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

torch.set_num_threads(1)
device = torch.device('cpu')

# ---------------- FFT & Filters ----------------
def _fftcoords(h: int, w: int):
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    V, U = np.meshgrid(fy, fx, indexing='ij')
    R = np.sqrt(U*U + V*V) + 1e-8
    return U.astype(np.float32), V.astype(np.float32), R.astype(np.float32)

def _lowpass_binary(R: np.ndarray, ratio: float) -> np.ndarray:
    r0 = float(ratio) * float(R.max())
    return (R <= r0).astype(np.float32)

def _lowpass_cosine(R: np.ndarray, ratio: float, width: float) -> np.ndarray:
    if width <= 0.0:
        return _lowpass_binary(R, ratio)
    r0 = float(ratio) * float(R.max())
    r1 = r0 * (1.0 - width)
    LP = np.zeros_like(R, np.float32)
    core = (R <= r1)
    ring = (R > r1) & (R <= r0)
    LP[core] = 1.0
    if np.any(ring):
        LP[ring] = 0.5 * (1 + np.cos(np.pi * (R[ring]-r1) / max(r0-r1, 1e-8)))
    return LP

def _inverse_mft(F_mod: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(F_mod)).real.astype(np.float32)

# ---------------- Spatial operators ----------------
def _laplacian_numpy(x: np.ndarray) -> np.ndarray:
    k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

def _torch_laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    x = x.unsqueeze(0).unsqueeze(0)
    y = torch.nn.functional.conv2d(x, k, padding=1)
    return y.squeeze(0).squeeze(0)

# ---------------- Clustering on MR ----------------
def _kmeans_on_MR(MR: np.ndarray, K: int, down: int, seed: int) -> np.ndarray:
    h,w = MR.shape
    hs, ws = max(h//max(down,1),1), max(w//max(down,1),1)
    MR_s = cv2.resize(MR, (ws,hs), interpolation=cv2.INTER_AREA)
    feats = MR_s.reshape(-1,1)
    K_eff = int(min(max(K,1), feats.shape[0]))
    km = KMeans(n_clusters=K_eff, n_init=20, random_state=seed)
    lab_s = km.fit_predict(feats).reshape(hs,ws)
    labels = cv2.resize(lab_s.astype(np.int32), (w,h), interpolation=cv2.INTER_NEAREST)
    return labels

# ---------------- Optimize alpha, beta ----------------
def _optimize_alpha_beta_LBFGS(gray, Ia, Ib, tau, lam_sc, lam_exp, lam_reg, iters):
    g   = torch.from_numpy(gray.astype(np.float32)).to(device)
    IaT = torch.from_numpy(Ia.astype(np.float32)).to(device)
    IbT = torch.from_numpy(Ib.astype(np.float32)).to(device)
    alpha = torch.tensor(1.0, requires_grad=True, device=device)
    beta  = torch.tensor(0.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([alpha,beta], lr=1.0, max_iter=int(iters), line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        L = alpha*IaT + beta*IbT
        loss_sc  = lam_sc  * torch.sum((_torch_laplacian(L) - _torch_laplacian(g)) ** 2)
        loss_exp = lam_exp * (L.mean() - float(tau)) ** 2
        loss_reg = lam_reg * (alpha**2 + beta**2)
        (loss_sc + loss_exp + loss_reg).backward()
        return loss_sc + loss_exp + loss_reg
    opt.step(closure)
    a,b = float(alpha.detach().cpu()), float(beta.detach().cpu())
    if not np.isfinite(a) or not np.isfinite(b):
        raise FloatingPointError('LBFGS produced non-finite alpha/beta')
    return a,b

def _solve_alpha_beta_closed_form(gray, Ia, Ib, tau, lam_sc, lam_exp, lam_reg):
    La = _laplacian_numpy(Ia); Lb = _laplacian_numpy(Ib); Lg = _laplacian_numpy(gray)
    mu_a, mu_b = float(Ia.mean()), float(Ib.mean())
    A11 = lam_sc * float((La*La).sum()) + lam_exp * (mu_a**2) + lam_reg
    A22 = lam_sc * float((Lb*Lb).sum()) + lam_exp * (mu_b**2) + lam_reg
    A12 = lam_sc * float((La*Lb).sum()) + lam_exp * (mu_a*mu_b)
    b1  = lam_sc * float((La*Lg).sum()) + lam_exp * (mu_a*tau)
    b2  = lam_sc * float((Lb*Lg).sum()) + lam_exp * (mu_b*tau)
    det = A11*A22 - A12*A12
    if abs(det) < 1e-12: return 1.0, 0.0
    alpha = (b1*A22 - b2*A12)/det
    beta  = (-b1*A12 + b2*A11)/det
    return float(alpha), float(beta)

# ---------------- Main algorithm ----------------
def PreFaCE_opt(
    img_rgb: np.ndarray, cluster_num=17, lp_ratio=0.50, tau=0.54,
    lam_sc=8.0, lam_exp=1.0, lam_reg=1e-3, down=4, opt_iters=200,
    trim_rate=0.05, cosine_width=0.08, auto_tau_eta=0.35,
    robust_norm=(1.0, 99.0), l_gamma=0.95, reflect_cap=1.6,
    use_bt709=True, retinex_eps='auto', save_debug=False, debug_prefix='',
    seed=42
):
    """Method-preserving FaCE with optional numerics. Returns (RGB uint8, alpha, beta)."""
    img_rgb = np.ascontiguousarray(img_rgb)
    img_f = img_rgb.astype(np.float32)/255.0

    # Gray
    if use_bt709:
        r,g,b = img_f[...,0], img_f[...,1], img_f[...,2]
        gray = (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)
    else:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    h,w = gray.shape

    # FFT & MFT
    F = np.fft.fftshift(np.fft.fft2(gray))
    U,V,R = _fftcoords(h,w)
    Kfac = (-1j)*(U + 1j*V)/R
    M = Kfac * F
    MR = np.log1p(np.abs(M)).astype(np.float32)

    # clustering
    labels = _kmeans_on_MR(MR, K=int(cluster_num), down=int(down), seed=seed)

    # ΔM with (optional) trimmed mean
    m_glo = float(MR.mean())
    dM = np.zeros_like(MR, dtype=np.float32)
    trim = float(trim_rate) if (trim_rate is not None) else 0.0
    for i in range(int(cluster_num)):
        mask = (labels==i)
        if not np.any(mask): continue
        vals = MR[mask].ravel()
        if trim > 0.0 and vals.size > 20:
            lo, hi = np.percentile(vals, [100*trim, 100*(1.0-trim)])
            vals = np.clip(vals, lo, hi)
        dM[mask] = float(vals.mean()) - m_glo

    # Low-pass
    LP = _lowpass_cosine(R, lp_ratio, float(cosine_width)) if (cosine_width and cosine_width>0) \
         else _lowpass_binary(R, lp_ratio)

    # Bases (only L optimized)
    F_alpha = F * LP
    F_beta  = F * dM * LP
    I_alpha = _inverse_mft(F_alpha)
    I_beta  = _inverse_mft(F_beta)

    # tau auto tuning (optional)
    tau_eff = float(tau)
    if auto_tau_eta and auto_tau_eta > 0.0:
        Ia = I_alpha
        Ia_min, Ia_max = float(Ia.min()), float(Ia.max())
        Ia_n = (Ia - Ia_min) / max(Ia_max - Ia_min, 1e-8)
        mean_mid = float(Ia_n.mean())
        tau_target = np.clip(0.56*0.7 + mean_mid*0.3, 0.50, 0.60)
        tau_eff = (1.0 - float(auto_tau_eta)) * float(tau) + float(auto_tau_eta) * tau_target

    # optimize alpha,beta
    try:
        alpha,beta = _optimize_alpha_beta_LBFGS(gray, I_alpha, I_beta,
                                                tau=tau_eff, lam_sc=lam_sc, lam_exp=lam_exp,
                                                lam_reg=lam_reg, iters=opt_iters)
    except Exception:
        alpha,beta = _solve_alpha_beta_closed_form(gray, I_alpha, I_beta,
                                                   tau=tau_eff, lam_sc=lam_sc, lam_exp=lam_exp, lam_reg=lam_reg)

    # L enhance + robust norm
    L_enh = alpha*I_alpha + beta*I_beta
    if (not np.isfinite(L_enh).all()) or (L_enh.max()-L_enh.min()<1e-8):
        L_enh = I_alpha.copy()

    if robust_norm is not None:
        p1, p2 = robust_norm
        lo, hi = np.percentile(L_enh, [p1, p2])
        if hi - lo < 1e-8:
            lo, hi = float(L_enh.min()), float(L_enh.max())
    else:
        lo, hi = float(L_enh.min()), float(L_enh.max())

    L_enh = (L_enh - lo) / max(hi - lo, 1e-8)
    L_enh = np.clip(L_enh, 0.0, 1.0)

    if l_gamma and abs(l_gamma - 1.0) > 1e-6:
        L_enh = np.power(L_enh, float(l_gamma))

    # Retinex color recovery (R fixed)
    if str(retinex_eps).lower() == 'auto':
        eps = max(1e-3, np.percentile(gray,1)/1.5)
    else:
        eps = 1e-6
    gray_safe = np.maximum(gray, eps)
    reflectance = img_f / gray_safe[...,None]
    if reflect_cap is not None:
        reflectance = np.clip(reflectance, 0.0, float(reflect_cap))

    out = np.clip(reflectance * L_enh[...,None], 0.0, 1.0)
    out_u8 = (out*255.0 + 0.5).astype(np.uint8)

    # debug exports
    if save_debug and debug_prefix:
        cv2.imwrite(f'{debug_prefix}_MR.png',  (np.clip(MR / max(MR.max(),1e-8),0,1)*65535).astype(np.uint16))
        dMviz = (dM - dM.min()) / max(dM.max()-dM.min(), 1e-8)
        cv2.imwrite(f'{debug_prefix}_dM.png',  (np.clip(dMviz,0,1)*65535).astype(np.uint16))
        cv2.imwrite(f'{debug_prefix}_LP.png',  (np.clip(LP,0,1)*65535).astype(np.uint16))
        cv2.imwrite(f'{debug_prefix}_L.png',   (np.clip(L_enh,0,1)*65535).astype(np.uint16))

    return out_u8, float(alpha), float(beta)
