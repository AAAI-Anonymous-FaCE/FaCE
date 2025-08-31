# Frequency-Aware Data-Driven Curve Estimation (FaCE) (AAAI submission)


![FaCE main figure](https://raw.githubusercontent.com/AAAI-Anonymous-FaCE/FaCE/main/figures/main_figure.jpg)

Overall pipeline of our proposed FaCE framework. FaCE innovatively leverages the MFT to perform data-driven spectral-domain clustering, effectively capturing joint amplitude and phase spectral characteristics. The derived enhancement curve robustly reconstructs enhanced images from low-light inputs in a mathematically rigorous, unsupervised manner, significantly improving interpretability and visual quality over existing frequency-domain methods.


### Emphasize two points：
1）We never optimize reflectance. We only adjust illumination (via luminance/log-luminance) and rebuild the image with unchanged chroma, which is equivalent to keeping 𝑅 fixed.

2）Input low-light I (not L)  --> (Eq.7 on I) estimate W  --> apply operator T to luminance/log-luminance (illumination only)
                      keep chroma (I / Y) unchanged --> replace luminance with the enhanced one --> output I_enh



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
scripts\face_opt_win.bat D:\AAAI26_DATA\datasets\LOL_v1\eval15\low D:\AAAI26_DATA\datasets\LOL_v1\eval15\FaCE_out

# Windows (strict, paper-faithful)
scripts\face_strict_win.bat D:\AAAI26_DATA\datasets\LOL_v1\eval15\low D:\AAAI26_DATA\datasets\LOL_v1\eval15\FaCE_out

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
![FaCE main figure](https://raw.githubusercontent.com/AAAI-Anonymous-FaCE/FaCE/main/figures/ex_result.jpg)

Overview of FaCE: Data-driven spectral weighting from the observed MFT magnitude for illumination-only enhancement and chroma-preserving reconstruction
（Pipeline nodes: low-light input, MFT magnitude, cluster-mean difference, low-pass filter, enhanced illumination, FaCE output vs. ground truth).

## Policies (skeletons)
- `docs/privacy-anonymization-policy.md`
- `docs/anti-plagiarism-policy.md`
- `docs/data-governance.md`
- `docs/ethics-statement.md`

© 2025 FaCE authors.
