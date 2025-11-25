import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from IMDLBenCo.registry import DATASETS
from IMDLBenCo.datasets.balanced_dataset import BalancedDataset
from IMDLBenCo.datasets.iml_datasets import JsonDataset, ManiDataset
from IMDLBenCo.transforms import iml_transforms
from IMDLBenCo.model_zoo.diffiml.diffiml import VAE, LightVAE
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
import torch.autograd
def train_distillation(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f"cuda:{local_rank}")
        print(f"DDP setup: Rank {local_rank}/{world_size} on device {device}")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Single GPU setup: Rank {local_rank} on device {device}")
    torch.autograd.set_detect_anomaly(True)

    # --- 4.1. 准备数据集 (不变) ---
    if local_rank == 0:
        print("Setting up dataset...")
    
    dataset_kwargs = {
        'output_size': (args.image_size, args.image_size),
        'is_resizing': args.if_resizing,
        'edge_width': args.edge_mask_width,
        'is_padding': False,
        'common_transforms': None
    }
    
    dataset = BalancedDataset(
        path=args.data_path, 
        **dataset_kwargs
    )
    if len(dataset) == 0:
        if local_rank == 0:
            print(f"Error: Dataset at {args.data_path} is empty or could not be loaded.")
        return

    if local_rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )
    if local_rank == 0:
        print("DataLoader created.")

    # --- 4.2. 准备模型 (不变) ---
    if local_rank == 0:
        print("Loading models...")
    
    teacher_vae = VAE(vae_path=args.original_vae_path).to(device)
    teacher_vae.eval()
    for param in teacher_vae.parameters():
        param.requires_grad = False
    if local_rank == 0:
        print("Teacher VAE (Original) is frozen.")

    act_fn_name = args.activation_fn.lower()
    if act_fn_name == 'relu':
        act_fn = nn.ReLU(inplace=True)
        print("Using ReLU activation")
    elif act_fn_name == 'silu':
         act_fn = nn.SiLU(inplace=True)
         print("Using SiLU activation")
    else:
         act_fn = nn.ReLU(inplace=True)
         print(f"Warning: Unknown activation_fn '{act_fn_name}'. Using ReLU.")
    
    student_vae = LightVAE(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels, # <--- 使用 base_channels
        norm_layer_type=args.norm_layer_type, # <--- 使用 norm_layer_type
        activation_fn=act_fn,
        latent_scale_factor=0.18215 # <--- 确保使用缩放
    ).to(device)

    # DDP 包装
    if world_size > 1:
        student_vae = nn.parallel.DistributedDataParallel(student_vae, device_ids=[local_rank], output_device=local_rank)
        
    student_vae.train()
    if local_rank == 0:
        print("Student VAE (Light) is in training mode.")
    
    # (打印参数量 - 不变)
    if local_rank == 0:
        model_to_check = student_vae.module if world_size > 1 else student_vae 
        teacher_params = sum(p.numel() for p in teacher_vae.parameters())
        student_params = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
        print(f"Teacher VAE Parameters: {teacher_params / 1e6:.2f} M")
        print(f"Student VAE Parameters: {student_params / 1e6:.2f} M")
        print(f"Parameter reduction: {100 * (1 - student_params / teacher_params):.2f}%")

    # --- 4.3. 准备优化器和损失函数 (不变) ---
    model_to_optimize = student_vae.module if world_size > 1 else student_vae
    optimizer = optim.Adam(model_to_optimize.parameters(), lr=args.lr)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.L1Loss().to(device)
    if local_rank == 0:
        print(f"Optimizer: Adam (lr={args.lr}), Loss: MSELoss")
    start_epoch = 0
    # (resume 逻辑 - 不变)

    # --- 4.4. 训练循环 (!! 关键修改 !!) ---
    if local_rank == 0:
        print("Starting distillation training...")
        
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
            
        student_vae.train()
        total_loss = 0.0
        
        # (只在 rank 0 上显示 TQDM)
        progress_bar = data_loader
        # if local_rank == 0:
        #     progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        for i, batch in enumerate(progress_bar):
            
            # 1. 同时加载 mask 和 edge_mask
            masks_01 = batch['mask'].to(device).float()
            edge_masks_01 = batch['edge_mask'].to(device).float()
            masks = torch.where(masks_01 == 0, -1., 1.)
            edge_masks = torch.where(edge_masks_01 == 0, -1., 1.)
            all_inputs = torch.cat((masks, edge_masks), dim=0)

            # 3. 对合并后的 all_inputs 执行蒸馏
            with torch.no_grad():
                # target_masks, _ = teacher_vae(all_inputs)
                target_latent = teacher_vae.encode_mask(all_inputs)

            # 学生模型生成预测
            pred_masks, pred_latent = student_vae(all_inputs)
            
            # 1. 重建损失
            # reconstruction_loss = criterion(pred_masks, target_masks)
            reconstruction_loss = criterion(pred_masks, all_inputs)

            # 2. 潜空间损失
            latent_loss = F.mse_loss(pred_latent, target_latent)

            # 3. 总损失
            loss = reconstruction_loss + args.latent_loss_weight * latent_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # # (TQDM 日志)
            # if local_rank == 0:
            #     progress_bar.set_postfix({
            #         "Loss": loss.item(),
            #         "Recon": reconstruction_loss.item(),
            #         "Latent": latent_loss.item()
            #     })

        avg_loss = total_loss / len(data_loader)
        
        # DDP下同步损失
        if world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG) # (!! 同步点 2 !!)
            avg_loss = loss_tensor.item()
            
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}")

        # 只有 Rank 0 才保存模型
        if local_rank == 0 and (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True) 
            save_path = os.path.join(args.output_dir, f"light_vae_epoch_{epoch+1}.pth")
            model_to_save = student_vae.module if world_size > 1 else student_vae
            checkpoint_data = {
                'model': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_data, save_path)
            print(f"Saved checkpoint to {save_path}")

    # --- 4.5. 保存最终模型 (不变) ---
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        final_save_path = os.path.join(args.output_dir, args.output_filename)
        model_to_save = student_vae.module if world_size > 1 else student_vae
        final_checkpoint_data = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs - 1 
        }
        torch.save(final_checkpoint_data, final_save_path)
        print(f"\nTraining complete. Final LightVAE weights saved to: {final_save_path}")

    if world_size > 1:
        torch.distributed.destroy_process_group()


# --- 3. 参数解析 (启动入口) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightVAE Distillation Training Script")

    # --- 路径参数 ---
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/data0/yunfei/workspace/IMDLBenCo/runs/balanced_dataset.json',
                        help='Path to the balanced_dataset.json file for training.')
    parser.add_argument('--original_vae_path', type=str, 
                        default='/mnt/data0/yunfei/workspace/model/diffusion/stable_diff/pretrained/vae',
                        help='Path to the directory of the pre-trained Original VAE (teacher).')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save the trained LightVAE weights.')
    parser.add_argument('--output_filename', type=str, default='light_vae_weights.pth',
                        help='Filename for the final saved weights.')

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (per GPU).')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for DataLoader.')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size to resize masks to (must match DiffIML config).')
    parser.add_argument('--if_resizing', action='store_true', 
                        help='If resize images (required by AbstractDataset).')
    parser.add_argument('--edge_mask_width', type=int, default=7,
                        help='Width of the edge mask (required by AbstractDataset).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='(Deprecated by torchrun) GPU ID to use for training.') 
    
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save a checkpoint every N epochs.')

    # --- LightVAE 模型参数 ---
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent dimension for LightVAE.')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count for LightVAE (controls size).')
    parser.add_argument('--norm_layer_type', type=str, default='BatchNorm', choices=['BatchNorm', 'GroupNorm'],
                        help='Normalization layer type for LightVAE.')
    parser.add_argument('--latent_loss_weight', type=float, default=0.1,
                        help='Weight for the latent space MSE loss during distillation.')
    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'silu'],
                        help="Activation function for LightVAE ('relu' for silu=false).")
    parser.add_argument('--layers_per_block', type=int, default=2,
                        help='ResNet 块中的层数')
    # parser.add_argument('--resume', type=str, default=None,
    #                     help='resume from checkpoint path')

    args = parser.parse_args()

    # 只在主进程 (rank 0) 打印设置
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Starting LightVAE distillation training with the following settings:")
        print("-" * 30)
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("-" * 30)

    train_distillation(args)