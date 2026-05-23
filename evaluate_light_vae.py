import torch
import os
import argparse
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from IMDLBenCo.model_zoo.diffiml.diffiml import VAE, LightVAE
from IMDLBenCo.datasets.balanced_dataset import BalancedDataset 

def visualize_comparison(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    teacher_vae = VAE(vae_path=args.original_vae_path).to(device).eval()
    student_vae = LightVAE(latent_dim=args.latent_dim, base_channels=args.base_channels).to(device).eval()

    print(f"Loading student weights from: {args.light_vae_weights}")
    checkpoint = torch.load(args.light_vae_weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        student_vae.load_state_dict(checkpoint['model'])
    else:
        student_vae.load_state_dict(checkpoint)
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
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving comparison images to: {args.output_dir}")

    for i, idx in enumerate(indices):
        batch = dataset[idx]
        mask = batch['mask'].unsqueeze(0).to(device).float() # (1, 1, H, W)

        with torch.no_grad():
            teacher_recon, _ = teacher_vae(mask)
            student_recon, _ = student_vae(mask)

        mask_to_save = (mask + 1) / 2 if mask.min() < 0 else mask 

        comparison = torch.cat([mask_to_save.cpu(), teacher_recon.cpu(), student_recon.cpu()], dim=3)
        save_path = os.path.join(args.output_dir, f"comparison_{i:03d}.png")
        save_image(comparison, save_path, nrow=1)

    print("Visual comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LightVAE Reconstruction Quality")
    parser.add_argument('--original_vae_path', type=str, required=True, help='Path to original VAE')
    parser.add_argument('--light_vae_weights', type=str, required=True, help='Path to trained LightVAE weights (.pth)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset json for getting sample masks')
    parser.add_argument('--output_dir', type=str, default='./vae_comparison_output', help='Directory to save comparison images')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to compare')

    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--if_resizing', action='store_true', default=True)
    parser.add_argument('--edge_mask_width', type=int, default=7)

    args = parser.parse_args()
    visualize_comparison(args)


    # Example usage:
    # python evaluate_light_vae.py --original_vae_path ./pretrained/sd-vae-ft-mse \
    #   --light_vae_weights ./log/train_light_vae/checkpoints/light_vae_weights.pth \
    #   --data_path ./runs/balanced_dataset.json