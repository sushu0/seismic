import torch
import numpy
import scipy
import matplotlib

print("="*60)
print("环境检查 - Environment Check")
print("="*60)
print(f"torch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("GPU设备: None (CPU only)")
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print("="*60)
