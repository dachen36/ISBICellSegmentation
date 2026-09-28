import torch
import cv2
import numpy as np
import os

# 1. 环境准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 1080Ti 已就位: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. 读取模拟数据
img_path = 'test.jpg'
if not os.path.exists(img_path):
    print("❌ 找不到 test.jpg，请先放一张图进来！")
else:
    # 3. 执行预处理 (这就是你 PPT 的核心：CLAHE)
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    # 4. 模拟模型推理 (这一步是为了证明你的 GPU 能跑)
    # 将图像转为张量并送入 GPU
    tensor_img = torch.from_numpy(img_enhanced).float().to(device).unsqueeze(0).unsqueeze(0) / 255.0

    # 随便建立一个简单的卷积层模拟 U-Net 的处理
    model = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1).to(device)
    with torch.no_grad():
        output = model(tensor_img)
        # 模拟二值化分割结果
        pred = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 5. 生成 PPT 用的三联图：原图 | 增强图 | 模拟分割图
    # 统一尺寸
    h, w = img.shape
    img_resized = cv2.resize(img, (w, h))
    enhanced_resized = cv2.resize(img_enhanced, (w, h))
    pred_resized = cv2.resize(pred, (w, h))

    # 横向拼接
    combined = np.hstack((img_resized, enhanced_resized, pred_resized))
    cv2.imwrite('interview_result.jpg', combined)

    print("🎉 奇迹发生了！")
    print("1. 预处理完成 (CLAHE)")
    print("2. GPU 推理完成")
    print("3. 已生成 'interview_result.jpg'，这就是你 PPT 里的核心成果图！")