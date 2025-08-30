# Frequency-Aware Data-Driven Curve Estimation (FaCE) (AAAI submission)




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

## “AI misdiagnosis” clarification
Prior LLMs misread our method as a learnable network and claimed we modified reflectance. We **do not**: R is fixed; only L is optimized; clustering runs on MR=log(1+|M|) (Riesz). We surface debug maps and code-to-equation mapping to avoid confusion.

## Policies (skeletons)
- `docs/privacy-anonymization-policy.md`
- `docs/anti-plagiarism-policy.md`
- `docs/data-governance.md`
- `docs/ethics-statement.md`

© 2025 FaCE authors.
