import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os, argparse, numpy as np, torch
from utils.vis import imshow2
from utils.geo import elastic_deform_triplet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z_path', required=True)
    ap.add_argument('--workdir', default='./exp_sg_cunet')
    ap.add_argument('--alpha', type=float, default=1500.0)
    ap.add_argument('--sigma', type=float, default=100.0)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    Z = np.load(args.z_path).astype(np.float32)
    Zm, Zs = Z.mean(), Z.std()+1e-8
    Zn = (Z - Zm)/Zs

    Z_show = torch.from_numpy(Zn[:400,:500]).float().unsqueeze(0)  # (1,T,X)
    Z_aug, _, _ = elastic_deform_triplet(Z_show, Z_show, Z_show, args.alpha, args.sigma)
    imshow2(Z_show.squeeze(0).numpy(), Z_aug.squeeze(0).numpy(),
            titles=("原始阻抗(规范化)","扩展阻抗(规范化)"),
            out=os.path.join(args.workdir, "fig3_aug_impedance.png"))
    print("Saved fig3 to", os.path.join(args.workdir, "fig3_aug_impedance.png"))

if __name__ == "__main__":
    main()
