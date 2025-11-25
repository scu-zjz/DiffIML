import torch
import os
import argparse
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# 假设你的 VAE 和 LightVAE 类都在这个文件里
from IMDLBenCo.model_zoo.diffiml.diffiml import VAE, LightVAE 
# 假设你的数据集加载逻辑在这里
from IMDLBenCo.datasets.balanced_dataset import BalancedDataset 

def visualize_comparison(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print("Loading models...")
    teacher_vae = VAE(vae_path=args.original_vae_path).to(device).eval()
    student_vae = LightVAE(latent_dim=args.latent_dim, base_channels=args.base_channels).to(device).eval()

    print(f"Loading student weights from: {args.light_vae_weights}")
    checkpoint = torch.load(args.light_vae_weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        student_vae.load_state_dict(checkpoint['model'])
    else:
        student_vae.load_state_dict(checkpoint) # 兼容旧格式
    print("Models loaded.")

    # 2. 加载数据集 (只需要几张图片)
    print("Loading dataset...")
    # 注意：这里的参数需要和训练时一致
    dataset_kwargs = {
        'output_size': (args.image_size, args.image_size),
        'is_resizing': args.if_resizing,
        'edge_width': args.edge_mask_width,
        'is_padding': False 
    }
    dataset = BalancedDataset(path=args.data_path, **dataset_kwargs)

    # 只取前 N 张图片进行对比
    num_samples = min(args.num_samples, len(dataset)) 
    indices = np.random.choice(len(dataset), num_samples, replace=False) # 随机选 N 张

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving comparison images to: {args.output_dir}")

    # 3. 逐个对比并保存图片
    for i, idx in enumerate(indices):
        batch = dataset[idx] # 获取单个样本字典
        mask = batch['mask'].unsqueeze(0).to(device).float() # (1, 1, H, W)

        with torch.no_grad():
            teacher_recon, _ = teacher_vae(mask) # teacher_vae 也返回 tuple
            student_recon, _ = student_vae(mask) # student_vae 返回 tuple

        # 将 Tensor 保存为图片 (0-1范围)
        # 确保 mask 也在 0-1 范围 (如果原始 mask 是 -1 到 1 需要转换)
        mask_to_save = (mask + 1) / 2 if mask.min() < 0 else mask 

        comparison = torch.cat([mask_to_save.cpu(), teacher_recon.cpu(), student_recon.cpu()], dim=3) # 横向拼接
        save_path = os.path.join(args.output_dir, f"comparison_{i:03d}.png")
        save_image(comparison, save_path, nrow=1) # nrow=1 保证三张图在一行

    print("Visual comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LightVAE Reconstruction Quality")
    parser.add_argument('--original_vae_path', type=str, required=True, help='Path to original VAE')
    parser.add_argument('--light_vae_weights', type=str, required=True, help='Path to trained LightVAE weights (.pth)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset json for getting sample masks')
    parser.add_argument('--output_dir', type=str, default='./vae_comparison_output', help='Directory to save comparison images')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to compare')

    # --- LightVAE 和 Dataset 参数 (需要和训练时匹配) ---
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--if_resizing', action='store_true', default=True) # 假设默认是 True
    parser.add_argument('--edge_mask_width', type=int, default=7)

    args = parser.parse_args()
    visualize_comparison(args)


    # python evaluate_light_vae.py --original_vae_path /mnt/data0/yunfei/workspace/model/diffusion/stable_diff/pretrained/vae \
    # --light_vae_weights /mnt/data0/yunfei/workspace/IMDLBenCo/log/train_light_vae/checkpoints/light_vae_weights.pth \
    # --data_path /mnt/data0/yunfei/workspace/IMDLBenCo/runs/balanced_dataset.json