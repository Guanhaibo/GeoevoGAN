
import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torchvision.utils import save_image

import os
import re          # <--- 必须加上这一行！
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch



'''
class PoemImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        raw = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.samples = []
        missing = 0

        for _, row in raw.iterrows():
            img_id = f"{int(row['id']):04d}.png"
            path = os.path.join(self.img_dir, img_id)

            if os.path.exists(path):
                self.samples.append((path, row["poem"]))
            else:
                missing += 1

        print(f"Dataset loaded: {len(self.samples)} samples, {missing} missing images skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, poem = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        #return img, poem
        return img

'''

class PoemImageDataset(Dataset):
    """
    加载 img_dir 下符合以下命名规则的图片：
      - flickr_cat_XXXXXX.ext
      - pixabay_cat_XXXXXX.ext
    支持扩展名: png, jpg, jpeg, webp (不区分大小写)
    自动按数字 ID 排序。
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 支持的扩展名
        exts = ["png", "jpg", "jpeg", "webp"]
        
        # 编译正则表达式：匹配 flickr 或 pixabay 开头，后跟 _cat_ 和 6位数字
        # 组1: 数字ID, 组2: 扩展名
        pattern_str = r"^(flickr|pixabay)_cat_(\d{6})\.(" + "|".join(exts) + r")$"
        self.file_pattern = re.compile(pattern_str, re.IGNORECASE)
        
        samples = []
        
        # 获取目录下所有文件
        try:
            all_files = os.listdir(img_dir)
        except FileNotFoundError:
            print(f"Error: Directory {img_dir} not found!")
            self.samples = []
            return

        for fname in all_files:
            match = self.file_pattern.match(fname)
            if match:
                # match.group(2) 是数字部分 (XXXXXX)
                num_id = int(match.group(2))
                full_path = os.path.join(img_dir, fname)
                samples.append((num_id, full_path))
        
        # 按数字 ID 排序
        samples.sort(key=lambda x: x[0])
        
        # 只保留路径
        self.samples = [path for _, path in samples]
        
        print(f"Dataset loaded: {len(self.samples)} images found in {img_dir}")
        if len(self.samples) == 0:
            print("Warning: No images matched the pattern 'flickr_cat_XXXXXX' or 'pixabay_cat_XXXXXX'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 如果图片损坏，返回一个全黑图像占位，避免 DataLoader 崩溃
            if self.transform:
                # 假设 transform 后是 Tensor，这里简单返回一个零张量 (需根据实际 transform 调整)
                return torch.zeros(3, 224, 224) 
            return Image.new("RGB", (224, 224), color='black')

# ---------------- BERT Encoder ----------------

class BertConditioner:
    def __init__(self, device):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese").to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                 return_tensors="pt").to(self.device)
        out = self.bert(**tokens).last_hidden_state
        return out[:, 0]   # CLS token (B, 768)