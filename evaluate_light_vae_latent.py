import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from tqdm import tqdm

from IMDLBenCo.model_zoo.diffiml.diffiml import VAE, LightVAE

try:
    from IMDLBenCo.datasets.balanced_dataset import BalancedDataset
except ImportError:
    print("Error: Could not import BalancedDataset. Make sure this script is in the project root.")
    exit()

def evaluate_latent_space(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_vae = VAE(vae_path=args.original_vae_path).to(device).eval()
    student_vae = LightVAE(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        norm_layer_type=args.norm_layer_type
    ).to(device).eval()

    print(f"Loading student weights from: {args.light_vae_weights}")
    checkpoint = torch.load(args.light_vae_weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Loading weights from 'model' key in checkpoint (new format).")
    else:
        state_dict = checkpoint
        print("Loading weights directly (old format checkpoint).")
    student_vae.load_state_dict(state_dict)
    print("Models loaded.")

    print("Loading dataset...")
    dataset_kwargs = {
        'output_size': (args.image_size, args.image_size),
        'is_resizing': args.if_resizing,
        'edge_width': args.edge_mask_width,
        'is_padding': False
    }
    dataset = BalancedDataset(path=args.data_path, **dataset_kwargs)

    num_samples = min(args.num_samples, len(dataset))
    indices = list(range(num_samples))
    # indices = np.random.choice(len(dataset), num_samples, replace=False) # 随机选 N 个

    print(f"Evaluating latent space similarity on {num_samples} samples...")

    total_mse = 0.0
    total_cosine_sim = 0.0

    for idx in tqdm(indices):
        batch = dataset[idx]
        mask = batch['mask'].unsqueeze(0).to(device).float()

        with torch.no_grad():
            target_latent_scaled = teacher_vae.encode_mask(mask)

            pred_latent_scaled = student_vae.encode_mask(mask)
            mse = F.mse_loss(pred_latent_scaled, target_latent_scaled).item()
            total_mse += mse

            pred_flat = pred_latent_scaled.view(pred_latent_scaled.size(0), -1)
            target_flat = target_latent_scaled.view(target_latent_scaled.size(0), -1)
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item() # 计算 batch 内平均
            total_cosine_sim += cosine_sim
    avg_mse = total_mse / num_samples
    avg_cosine_sim = total_cosine_sim / num_samples

    print("\n--- Latent Space Evaluation Results ---")
    print(f"Average MSE between scaled latents: {avg_mse:.6f}")
    print(f"Average Cosine Similarity between scaled latents: {avg_cosine_sim:.6f}")
    print("---------------------------------------")

    if avg_mse < 1e-3 and avg_cosine_sim > 0.99:
         print("Evaluation suggests LightVAE latent space is very similar to the original VAE.")
    elif avg_mse < 5e-3 and avg_cosine_sim > 0.95:
         print("Evaluation suggests LightVAE latent space is reasonably similar, but noticeable differences exist.")
    else:
         print("Warning: Significant differences detected between LightVAE and original VAE latent spaces. This might affect DiffIML performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LightVAE Latent Space Quality")
    parser.add_argument('--original_vae_path', type=str, required=True, help='Path to original VAE directory')
    parser.add_argument('--light_vae_weights', type=str, required=True, help='Path to trained LightVAE weights (.pth)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset json for getting sample masks')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to evaluate') # 评估更多样本更可靠

    parser.add_argument('--latent_dim', type=int, default=4, help='Latent dimension used during LightVAE training')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels used during LightVAE training') # <-- 确认默认值
    parser.add_argument('--norm_layer_type', type=str, default='BatchNorm', choices=['BatchNorm', 'GroupNorm'], help='Norm layer used')
    parser.add_argument('--image_size', type=int, default=512, help='Image size used')
    parser.add_argument('--if_resizing', action='store_true', default=True, help='If resizing was used') # 假设默认是 True
    parser.add_argument('--edge_mask_width', type=int, default=7, help='Edge width used')

    args = parser.parse_args()
    evaluate_latent_space(args)


    # Example usage:
    # python evaluate_light_vae_latent.py \
    #   --original_vae_path ./pretrained/sd-vae-ft-mse \
    #   --light_vae_weights ./log/train_light_vae/checkpoints/light_vae_weights.pth \
    #   --data_path ./runs/balanced_dataset.json \