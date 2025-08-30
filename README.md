# FaCE: Frequency-domain Clustering for Illumination Enhancement (AAAI submission)

**Two entry points**
- `src/AAAI_FaCE_FaCE_opt.py` — method-faithful, with optional numerics (turn off by `--strict`).

## Quick start
```bash
pip install -r requirements.txt
python src/AAAI_FaCE_FaCE_opt.py --input <input_dir> --output <out_dir> --save-debug
# strict:
python src/AAAI_FaCE_FaCE_opt.py --strict --input <input_dir> --output <out_dir>
```

## Notes on common misreads by LLMs
- No training; only (alpha, beta) via convex energy (LBFGS/closed-form).
- Reflectance R is NOT optimized; only illumination L is enhanced.
- Clustering runs on MR = log(1 + |M|) from Riesz transform in frequency domain.
