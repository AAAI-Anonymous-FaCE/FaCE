# Frequency-Aware Convex Enhancement (FaCE) (ICLR2026 submission)


![FaCE main figure](https://raw.githubusercontent.com/iclr26/FaCE/main/figures/main_figure.jpg)

Overall pipeline of our proposed FaCE framework. FaCE innovatively leverages the MFT to perform data-driven spectral-domain clustering, effectively capturing joint amplitude and phase spectral characteristics. The derived enhancement curve robustly reconstructs enhanced images from low-light inputs in a mathematically rigorous, unsupervised manner, significantly improving interpretability and visual quality over existing frequency-domain methods.


### Emphasize 3 points：
1）We never optimize reflectance. We only adjust illumination (via luminance/log-luminance) and rebuild the image with unchanged chroma, which is equivalent to keeping 𝑅 fixed.

2）Input low-light I (not L)  --> (Eq.7 on I) estimate W  --> apply operator T to luminance/log-luminance (illumination only)
                      keep chroma (I / Y) unchanged --> replace luminance with the enhanced one --> output I_enh

3） We use I=R⋅L purely as a modeling convention. In practice, we derive an illumination proxy from the input (luminance/log-luminance), keep chroma unchanged (thereby treating reflectance as fixed), and apply a fixed frequency-domain operator only to the illumination proxy. Eq. (7) is evaluated on the observed low-light input to estimate spectral weights; this does not imply we modify reflectance or perform intrinsic decomposition.    


### How `I` links to `L` and `R` (what the code actually does)

We use `I = R * L` as a **modeling convention** to separate appearance into:
- **reflectance-like** (color ratios / chroma), and
- **illumination-like** (brightness).

In code we derive both **from the same input `I`**:

- **Illumination proxy.**  
  `Y = rgb_to_luminance(I)` → `L_hat = log(eps + Y)` (or use `Y` directly).  
  This is the **only** branch we enhance.

- **Chroma (reflectance-like).**  
  `C = I / (Y + eps)` (element-wise).  
  We keep `C` **fixed** during reconstruction → effectively **do not modify reflectance (`R`)**.

- **Reconstruction.**  
  After enhancing illumination only, write it back with chroma unchanged:  
  `I_enh = C * Y_enh` (or `C * exp(L_hat_enh)` if you work in log space).

> **Takeaway:** We never estimate or edit a stand-alone `R`. Chroma is preserved, so reflectance is operationally fixed; we only adjust illumination.

---

### What Eq.(7) is computed on (and why)

- **Eq.(7) is computed on the observed low-light input `I_low`.**  
  Purpose: estimate **data-driven spectral weights** `W(u,v)` **from the input you want to enhance**.

- Those weights parameterize a **fixed frequency-domain operator** `T`, which we then apply **only to the illumination proxy** (`L_hat`), not to chroma.

**Short version:**  
`Eq.(7) on I_low → W → apply T to L_hat only → rebuild with fixed chroma → I_enh`.

---

### Minimal code path (exact steps)

```python
# Input
# I: float tensor [H, W, 3] in [0,1]

# 1) From I → illumination proxy (L_hat) and chroma (C)
Y     = rgb_to_luminance(I)                 # illumination-like
L_hat = torch.log(eps + Y)                  # illumination proxy (log luminance)
C     = I / (Y.unsqueeze(-1) + eps)         # reflectance-like (kept fixed)

# 2) Eq.(7) on the observed low-light input I → spectral weights W(u,v)
W_alpha = build_W_from_I(I, a=a1, b=b1)     # MR = log1p(|FFT(Y)|) → mean-center → sigmoid
W_beta  = build_W_from_I(I, a=a2, b=b2)
LP      = gaussian_lowpass_like(W_alpha, sigma_frac=lp_sigma_frac)

# 3) Fixed frequency-domain operator T, applied ONLY to illumination (L_hat)
def T(X, W):
    F  = torch.fft.rfft2(X, norm="ortho")
    FX = F * W * LP                          # element-wise in frequency domain
    return torch.fft.irfft2(FX, s=None, norm="ortho")

L_enh_a = T(L_hat, W_alpha)
L_enh_b = T(L_hat, W_beta)

# 4) Write illumination back with chroma fixed (reflectance unchanged)
Y_enh_a = torch.exp(L_enh_a)                 # if you worked in log space
Y_enh_b = torch.exp(L_enh_b)
I_a     = C * Y_enh_a.unsqueeze(-1)          # chroma preserved
I_b     = C * Y_enh_b.unsqueeze(-1)

# 5) (Optional) choose a mix between two strengths
I_enh = I_a  # or small search over alpha in [0,1]: I_enh = alpha*I_a + (1-alpha)*I_b



## Repo layout
```
face/                # Python package
  core.py            # FaCE core algorithm (PreFaCE_opt) + helpers
  io.py              # Dataset & array utilities
  cli.py             # Command-line entry
scripts/             # Convenience launchers (Win/Linux)
docs/                # Alignment, privacy, governance, ethics (skeletons)
tests/               # Smoke tests
```

## Install
```bash
pip install -r requirements.txt
# (optional) editable install
pip install -e .
```

## Quick start
```bash
# Windows (opt, with debug images)
scripts\face_opt_win.bat D:\iclr26_DATA\datasets\LOL_v1\eval15\low D:\iclr26_DATA\datasets\LOL_v1\eval15\FaCE_out

# Windows (strict, paper-faithful)
scripts\face_strict_win.bat D:\iclr26_DATA\datasets\LOL_v1\eval15\low D:\iclr26_DATA\datasets\LOL_v1\eval15\FaCE_out

# Linux/Mac
./scripts/run_face.sh <input_dir> <output_dir> [strict|opt]
```

## CLI (python -m face.cli)
- `--input`, `--output` (required)
- `--k 17` (comma-separated accepted)
- `--lp 0.5`, `--cos-width 0.08`
- `--tau 0.54`, `--auto-tau 0.35`
- `--trim 0.05`, `--robust-norm 1,99`, `--l-gamma 0.95`
- `--reflect-cap 1.6`, `--bt709`, `--retinex-eps auto`
- `--strict` to disable all numerics

## Debug artifacts (saved when `--save-debug`)
- `*_MR.png`: frequency **log-Riesz magnitude** map used for clustering  
- `*_dM.png`: **cluster mean deviation** ΔM  
- `*_LP.png`: **low-pass mask** LP(u,v)  
- `*_L.png`: enhanced illumination **L_enh**

## Method-to-code alignment
See `docs/alignment-with-paper.md` for exact mapping (fill in formula numbers per final paper).

## Reproducibility
- Fixed seeds, CPU-stable defaults.
- No training: only (α, β) optimized by convex objective (LBFGS / closed-form fallback).

## “Misdiagnosis” clarification
Some readers may initially interpret our approach as a learnable network that alters reflectance. It is not. FaCE computes spectral weights from the observed image and applies a fixed frequency-domain operator to illumination only (via luminance/log-luminance). Reflectance is kept unchanged by preserving chroma during reconstruction. R is fixed; only L is optimized; clustering runs on MR=log(1+|M|) (Riesz). We surface debug maps and code-to-equation mapping to avoid confusion.

##
![FaCE main figure](https://raw.githubusercontent.com/iclr26/FaCE/main/figures/ex_result.jpg)

Overview of FaCE: Data-driven spectral weighting from the observed MFT magnitude for illumination-only enhancement and chroma-preserving reconstruction
（Pipeline nodes: low-light input, MFT magnitude, cluster-mean difference, low-pass filter, enhanced illumination, FaCE output vs. ground truth).

## Policies (skeletons)
- `docs/privacy-anonymization-policy.md`
- `docs/anti-plagiarism-policy.md`
- `docs/data-governance.md`
- `docs/ethics-statement.md`

© 2025 FaCE authors.
