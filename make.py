import os
import argparse
from tqdm import tqdm
import torch
import math
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from models import Generator, GConfig
from operation import get_dir, InfiniteSamplerWrapper

import torch
import os
import math
from torchvision.utils import save_image
from models import Generator  # 确保你的 models.py 在当前路径或 PYTHONPATH 中

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 模型参数 (必须与训练时完全一致)
ngf = 64
nz = 100        # <--- 注意这里是 100
im_size = 256

# 3. 初始化模型
netG = Generator(ngf=ngf, nz=nz, im_size=im_size, use_grassmann=True)

# 4. 加载权重
ckpt_path = "./all_240000.pth"
# 建议先加载到 cpu，然后再整体移动到 target device，这样更灵活
ckpt = torch.load(ckpt_path, map_location="cpu") 

# 加载状态字典
# strict=False 允许部分键不匹配，但最好确保 key 存在
if "g" in ckpt:
    netG.load_state_dict(ckpt["g"], strict=True) 
else:
    # 如果保存的文件直接就是 state_dict 而不是 dict{'g': ...}
    netG.load_state_dict(ckpt, strict=True)

netG.to(device)
netG.eval()

aiter = ckpt.get("iter", 25000)
output_dir = f"{aiter}_out"
os.makedirs(output_dir, exist_ok=True)

try:
    batch_input = input("batch size: ")
    batch = int(batch_input)
except ValueError:
    print("Invalid number, using default batch=16")
    batch = 16

print(f"Starting generation for {5000} batches...")

with torch.no_grad(): 
    for i in range(5065):
        z = torch.randn(batch, nz, device=device)
        
        out = netG(z)
        
        pic = (out[0] + 1) / 2
        pic = torch.clamp(pic, 0, 1) # 防止数值溢出
        
        nrow = int(math.sqrt(batch))
        if nrow == 0: nrow = 1
        
        save_path = f"{output_dir}/small_{i}.jpg"
        save_image(pic, save_path, nrow=nrow)
        print(f"Saved {save_path}")

print("Done!")