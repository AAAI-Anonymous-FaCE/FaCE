# face/cli.py
import os, argparse
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from .io import ImageFolderRGB, to_rgb_uint8
from .core import PreFaCE_opt

def build_parser():
    p = argparse.ArgumentParser(description='FaCE optimized (method-preserving)')
    p.add_argument('--input',  type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--k',      type=str, default='17')
    p.add_argument('--lp',     type=float, default=0.50)
    p.add_argument('--tau',    type=float, default=0.54)
    p.add_argument('--lam-sc', type=float, default=8.0)
    p.add_argument('--lam-exp',type=float, default=1.0)
    p.add_argument('--lam-reg',type=float, default=1e-3)
    p.add_argument('--down',   type=int,   default=4)
    p.add_argument('--iters',  type=int,   default=200)
    # tricks
    p.add_argument('--trim',   type=float, default=0.05)
    p.add_argument('--cos-width', type=float, default=0.08)
    p.add_argument('--auto-tau',  type=float, default=0.35)
    p.add_argument('--robust-norm', type=str, default='1,99')
    p.add_argument('--l-gamma', type=float, default=0.95)
    p.add_argument('--reflect-cap', type=float, default=1.6)
    p.add_argument('--bt709', action='store_true')
    p.add_argument('--retinex-eps', type=str, choices=['auto','fixed'], default='auto')
    p.add_argument('--save-debug', action='store_true')
    # strict
    p.add_argument('--strict', action='store_true', help='turn OFF all tricks to match paper strictly')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--batch',   type=int, default=1)
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # parse K list
    try:
        k_list = [int(x) for x in str(args.k).split(',') if str(x).strip()!='']
    except Exception:
        k_list = [17]

    # robust norm
    robust_norm = None
    if not args.strict and isinstance(args.robust_norm, str) and args.robust_norm.lower() != 'none':
        try:
            p1, p2 = [float(v) for v in args.robust_norm.split(',')]
            robust_norm = (p1, p2)
        except Exception:
            robust_norm = (1.0, 99.0)

    if args.strict:
        cfg = dict(lp_ratio=args.lp, tau=args.tau, lam_sc=args.lam_sc, lam_exp=args.lam_exp, lam_reg=args.lam_reg,
                   down=args.down, opt_iters=args.iters, trim_rate=0.0, cosine_width=0.0, auto_tau_eta=0.0,
                   robust_norm=None, l_gamma=1.0, reflect_cap=None, use_bt709=False, retinex_eps='fixed',
                   save_debug=False)
        mode_tag = 'strict'
    else:
        cfg = dict(lp_ratio=args.lp, tau=args.tau, lam_sc=args.lam_sc, lam_exp=args.lam_exp, lam_reg=args.lam_reg,
                   down=args.down, opt_iters=args.iters, trim_rate=max(0.0, float(args.trim)),
                   cosine_width=max(0.0, float(args.cos_width)), auto_tau_eta=max(0.0, float(args.auto_tau)),
                   robust_norm=robust_norm, l_gamma=float(args.l_gamma),
                   reflect_cap=(None if float(args.reflect_cap) < 0 else float(args.reflect_cap)),
                   use_bt709=bool(args.bt709), retinex_eps=args.retinex_eps,
                   save_debug=bool(args.save_debug))
        mode_tag = 'opt'

    os.makedirs(args.output, exist_ok=True)
    ds = ImageFolderRGB(args.input)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    print(f"[FaCE-{mode_tag}] k={k_list}, lp={args.lp}, tau={args.tau}, lam_sc={args.lam_sc}, iters={args.iters}; "
          f"trim={cfg['trim_rate']}, cosw={cfg['cosine_width']}, auto_tau={cfg['auto_tau_eta']}, "
          f"robust={cfg['robust_norm']}, gamma={cfg['l_gamma']}, cap={cfg['reflect_cap']}, "
          f"bt709={cfg['use_bt709']}, eps={cfg['retinex_eps']}")

    pbar = tqdm(total=len(ds), desc=f'FaCE {mode_tag}')
    for img, path in dl:
        import numpy as np
        img_np = to_rgb_uint8(img)
        base = os.path.splitext(os.path.basename(path))[-2] if isinstance(path, str) else os.path.basename(path[0]).split('.')[0]
        for K in k_list:
            try:
                out_rgb, a, b = PreFaCE_opt(img_np, cluster_num=int(K), **cfg,
                                            debug_prefix=os.path.join(args.output, f'{base}_K{K}_{mode_tag}'))
                save_path = os.path.join(args.output, f'{base}_FaCE_{mode_tag}_K{K}.png')
                cv2.imwrite(save_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f'[WARN] {base} (K={K}) failed: {e}')
        pbar.update(1)
    pbar.close()
    print('Done. Results saved to:', args.output)
