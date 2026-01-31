import os
import torch

p = r"D:\SEISMIC_CODING\comparison01\marmousi_cnn_bilstm.pth"

sd = torch.load(p, map_location="cpu")
sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd

print("convs0", tuple(sd["convs.0.0.weight"].shape))
print("convs1", tuple(sd["convs.1.0.weight"].shape))
print("convs2", tuple(sd["convs.2.0.weight"].shape))
print("bilstm.weight_ih_l0", tuple(sd["bilstm.weight_ih_l0"].shape))
print("bilstm.weight_ih_l2", tuple(sd["bilstm.weight_ih_l2"].shape))
print("lstm_reg.weight_ih_l0", tuple(sd["lstm_reg.weight_ih_l0"].shape))
print("fc.weight", tuple(sd["fc.weight"].shape))
