import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob


# ==========================================
# 1. 模型结构 (增强版：起始通道 64，解决容量不足)
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


class UNetPro(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)  # 提高到64，增强特征提取能力
        self.pool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(64, 128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        out = self.up(x2)
        out = torch.cat([x1, out], dim=1)  # 跳跃连接
        out = self.up_conv(out)
        return self.outc(out)


# ==========================================
# 2. 混合损失函数 (BCE + Dice)
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
# 3. 数据处理 (强制归一化 0/1)
# ==========================================
class MedicalDataset(Dataset):
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
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()  # 彻底解决 255 像素问题
        return img_t, mask_t


# ==========================================
# 4. 验证逻辑 (生成叠加对比图)
# ==========================================
def verify_and_overlay(model, dataset, device, epoch):
    model.eval()
    img_t, mask_t = dataset[0]  # 取第一张验证
    input_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_t)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_bin = (pred_prob > 0.5).astype(np.uint8) * 255

    # 制作叠加图 (Overlay)
    raw_img = (img_t.squeeze().numpy() * 255).astype(np.uint8)
    img_color = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    mask_overlay = np.zeros_like(img_color)
    mask_overlay[:, :, 2] = pred_bin  # 红色遮罩

    # 融合图像
    overlay_res = cv2.addWeighted(img_color, 0.7, mask_overlay, 0.3, 0)

    # 拼接：原图 | 真值 | 预测 | 叠加图
    true_mask = (mask_t.squeeze().numpy() * 255).astype(np.uint8)
    true_mask_3ch = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
    pred_3ch = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)

    final_strip = np.hstack((img_color, true_mask_3ch, pred_3ch, overlay_res))
    cv2.imwrite(f'epoch_{epoch}_val.png', final_strip)
    model.train()


# ==========================================
# 5. 主训练流程 (含动态学习率)
# ==========================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir, mask_dir = "images", "masks"

    if not os.path.exists(img_dir):
        print("❌ 文件夹不存在！");
        return

    dataset = MedicalDataset(img_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNetPro().to(device)
    criterion = MixedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 动态学习率调度器：4轮不降则折半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)

    print("🚀 1050Ti 深度调优启动...")
    for epoch in range(1, 101):  # 建议跑 100 轮
        model.train()
        total_loss = 0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)  # 触发动态调整

        print(f"Epoch [{epoch}/100], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch % 10 == 0:
            verify_and_overlay(model, dataset, device, epoch)

    torch.save(model.state_dict(), "unet_pro_final.pth")
    print("✅ 训练完成！结果已保存。")


if __name__ == "__main__":
    run_training()