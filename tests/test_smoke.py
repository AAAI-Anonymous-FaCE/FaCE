# tests/test_smoke.py
import numpy as np
from face.core import PreFaCE_opt

def test_smoke():
    # tiny synthetic image
    img = (np.random.rand(64, 64, 3)*255).astype(np.uint8)
    out, a, b = PreFaCE_opt(img, cluster_num=3, lp_ratio=0.6, auto_tau_eta=0.0, save_debug=False)
    assert out.shape == img.shape
