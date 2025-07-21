import sys
import torch
import numpy
import pandas
import matplotlib

print("Python version:", sys.version)

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Number of available GPUs:", torch.cuda.device_count())
print("GPU names:")
for i in range(torch.cuda.device_count()):
    print(f"  {i}: {torch.cuda.get_device_name(i)}")

print("numpy version:", numpy.__version__)
print("pandas version:", pandas.__version__)
print("matplotlib version:", matplotlib.__version__)
print("torch version:", torch.__version__)

