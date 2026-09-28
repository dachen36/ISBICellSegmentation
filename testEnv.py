import torch
import cv2
import numpy as np

print(f"PyTorch 版本: {torch.__version__}")
print(f"OpenCV 版本: {cv2.__version__}")
print(f"GPU 是否可用: {torch.cuda.is_available()}") # 如果没有显卡会显示 False，没关系