import subprocess
import os
import cv2
import torch
import numpy as np
import pandas as pd
from utils import calc_psnr, calc_ssim

# 定义 K 和 disp_focus 的值
K_values = [10, 20, 30, 40, 50]

disp_focus_values = np.linspace(0, 1, 10)


def calculate_metrics(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    print(img2.shape)
    if img1.shape != img2.shape:
        raise ValueError("两张图片的尺寸不一致，请确保图片尺寸相同。")
    img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).unsqueeze(0)
    img2 = torch.from_numpy(img2.astype(np.float32) / 255.0).unsqueeze(0)
    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2).item()
    return psnr, ssim


# 用于存储所有的 PSNR 和 SSIM 值
all_psnr_values = []
all_ssim_values = []

num = 278

# 遍历不同的 K 值
for idx_k, K in enumerate(K_values):
    # 遍历不同的 disp_focus 值
    for idx_d, disp_focus in enumerate(disp_focus_values):
        output_dir = f"./outputs_plus/evaluation/{num}"
        os.makedirs(output_dir, exist_ok=True)
        command = [
            "python",
            "/home/kmxu/Bokehme_plus/demo_evaluation.py",
            "--K", str(K),
            "--disp_focus", str(disp_focus),
            "--save_root", output_dir
        ]
        subprocess.run(command)

        path1 = f'/home/kmxu/Bokehme_plus/outputs_plus/evaluation/{num}/K{K}_disp{disp_focus}_bokeh_pred.jpg'
        path2 = f'/home/kmxu/Bokehme_plus/Bokehme_evaluate_data/data/278/bokeh_0{idx_k}_0{idx_d}.jpg'
        print(path1)
        print(path2)        
        try:
            psnr, ssim = calculate_metrics(path1, path2)
            all_psnr_values.append(psnr)
            all_ssim_values.append(ssim)
            print(f"K: {K}, disp_focus: {disp_focus:.6f}, PSNR: {psnr}, SSIM: {ssim}")
        except ValueError as e:
            print(f"K: {K}, disp_focus: {disp_focus:.6f}, 出现错误: {e}")

# 将 PSNR 和 SSIM 值转换为 5x10 的二维数组
psnr_array = np.array(all_psnr_values).reshape(5, 10)
ssim_array = np.array(all_ssim_values).reshape(5, 10)

# 创建 DataFrame
psnr_df = pd.DataFrame(psnr_array, index=K_values, columns=disp_focus_values)
ssim_df = pd.DataFrame(ssim_array, index=K_values, columns=disp_focus_values)

# 创建 ExcelWriter 对象
with pd.ExcelWriter('metrics_results_pred_plus.xlsx') as writer:
    # 将 PSNR 和 SSIM 数据分别写入不同的表
    psnr_df.to_excel(writer, sheet_name='PSNR')
    ssim_df.to_excel(writer, sheet_name='SSIM')

print("数据已成功保存到 metrics_results_pred_plus.xlsx 文件中。")