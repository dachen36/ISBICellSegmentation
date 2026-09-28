import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob


# ==========================================
# 1. 模型结构 (针对 1050Ti 优化的轻量版)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.pool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(32, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = DoubleConv(64 + 32, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        out = self.up(x2)
        out = torch.cat([x1, out], dim=1)
        out = self.up_conv(out)
        return self.outc(out)


# ==========================================
# 2. 损失函数 (解决 Loss 0.24 问题的 Mixed Loss)
# ==========================================
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = 1 - (2. * intersection + smooth) / (union + smooth)
        return self.alpha * bce + (1 - self.alpha) * dice


# ==========================================
# 3. 数据集读取 (解决 255 像素问题)
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], 0)
        img = self.clahe.apply(cv2.resize(img, (256, 256)))
        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, (256, 256))

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        # 修正 255 像素：除以 255 后二值化，确保真值是 0 或 1
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


# ==========================================
# 4. 验证函数 (修复了你提到的 Argument 错误)
# ==========================================
def verify(model_path, img_dir, mask_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功，正在生成验证图...")
    else:
        print("❌ 找不到模型文件，请先训练！")
        return

    model.eval()
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not img_list: return

    # 取第一张进行验证
    raw_img = cv2.imread(img_list[0], 0)
    raw_mask = cv2.imread(mask_list[0], 0)

    img_p = cv2.resize(raw_img, (256, 256))
    img_t = torch.from_numpy(img_p).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(img_t)
        pred = (torch.sigmoid(output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 横向拼接：原图 | 真值 | 预测
    mask_r = cv2.resize(raw_mask, (256, 256))
    combined = np.hstack((img_p, mask_r, pred))
    cv2.imwrite('final_comparison.png', combined)
    print("✨ 验证完成！请查看 final_comparison.png")


# ==========================================
# 5. 训练主程序
# ==========================================
if __name__ == "__main__":
    img_dir = "images"
    mask_dir = "masks"

    # 如果文件夹不存在则报错提醒
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print("❌ 请确保项目目录下有 'images' 和 'masks' 文件夹！")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyUNet().to(device)
        criterion = MixedLoss(alpha=0.6)
        optimizer = optim.Adam(model.parameters(), lr=2e-4)

        dataset = SimpleDataset(img_dir, mask_dir)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        print(f"🚀 开始训练 (设备: {device})...")
        model.train()
        for epoch in range(1, 101):  # 跑 30 次看看效果
            total_loss = 0
            for imgs, masks in dataloader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/30, Loss: {total_loss / len(dataloader):.4f}")

        # 保存并验证
        torch.save(model.state_dict(), "unet_model.pth")
        verify("unet_model.pth", img_dir, mask_dir)