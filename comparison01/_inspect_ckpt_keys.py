import os
import torch

root = r"D:\SEISMIC_CODING\comparison01"

for name in [
    "marmousi_cnn_bilstm_supervised.pth",
    "marmousi_cnn_bilstm_semi.pth",
    "marmousi_cnn_bilstm.pth",
]:
    path = os.path.join(root, name)
    if not os.path.exists(path):
        continue
    sd = torch.load(path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    keys = list(sd.keys())
    print(name)
    print("  num_keys:", len(keys))
    print("  has_convs:", any(k.startswith("convs.") for k in keys))
    print("  has_conv1:", any(k.startswith("conv1.") for k in keys))
    print("  fc.weight:", tuple(sd["fc.weight"].shape) if "fc.weight" in sd else None)
    print("  sample_keys:")
    for k in keys[:25]:
        print("   ", k)
    print("  ...")
    for k in keys[-25:]:
        print("   ", k)
    print()
