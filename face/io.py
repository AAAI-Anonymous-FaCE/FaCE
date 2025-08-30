# face/io.py
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderRGB(Dataset):
    def __init__(self, root: str):
        self.paths = [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        if not self.paths:
            raise FileNotFoundError(f'No images found in: {root}')
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return np.asarray(img), p

def to_rgb_uint8(x):
    import numpy as np
    if hasattr(x, 'numpy'):
        x = x.detach().cpu().numpy()
    x = np.array(x)
    if x.ndim == 4: x = x[0]
    if x.ndim == 3 and x.shape[0] in (1,3) and x.shape[-1] not in (1,3):
        x = np.transpose(x, (1,2,0))
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    if x.dtype != np.uint8:
        mx = float(x.max()) if x.size else 1.0
        x = (x*255.0 if mx <= 1.0 else x).clip(0,255).astype(np.uint8)
    return x
