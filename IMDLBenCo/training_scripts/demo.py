import torch
import sys
import os
import torch.nn as nn
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from IMDLBenCo.model_zoo.diffiml.diffiml import VAE, LightVAE


@torch.no_grad()
def benchmark_vae(model, input_tensor, device, num_warmup=20, num_runs=100):
    """
    在 GPU 上对 VAE 模型进行预热和精确计时。
    测量 "forward" (编码+解码) 的完整时间。
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    for _ in range(num_warmup):
        _ = model(input_tensor)
    
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(num_runs):
        _ = model(input_tensor)
    end_event.record()

    torch.cuda.synchronize(device)

    execution_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = execution_time_ms / num_runs
    
    return avg_time_ms


if __name__ == '__main__':
    ORIGINAL_VAE_PATH = './pretrained/sd-vae-ft-mse'
    SLIM_VAE_WEIGHTS_PATH = './log/train_light_vae/checkpoints/light_vae_weights.pth'
    
    # SLIM_VAE_CONFIG = {
    #     'latent_dim': 4,
    #     'base_channels': 64,
    #     'layers_per_block': 2,
    #     'activation_fn': nn.ReLU(inplace=True) 
    #     # 'activation_fn': nn.SiLU(inplace=True)
    # }
    LIGHT_VAE_CONFIG = {
        'latent_dim': 4,
        'base_channels': 32,
        'activation_fn': nn.ReLU(inplace=True),
        'norm_layer_type': 'BatchNorm'
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告：正在使用 CPU 运行，计时结果不准确。")

    print(f"Loading Original VAE from: {ORIGINAL_VAE_PATH}")
    try:
        model_original = VAE(vae_path=ORIGINAL_VAE_PATH)
    except Exception as e:
        print(f"加载 VAE 失败: {e}")
        print("请检查你的 diffusers 和 transformers 库是否已安装。")
        sys.exit(1)

    print(f"Loading LightVAE from: {SLIM_VAE_WEIGHTS_PATH}")
    
    # model_slim = SlimVAE(
    #     latent_dim=SLIM_VAE_CONFIG['latent_dim'],
    #     block_out_channels=[SLIM_VAE_CONFIG['base_channels'] * m for m in [1, 2, 4, 8]],
    #     layers_per_block=SLIM_VAE_CONFIG['layers_per_block'],
    #     activation_fn=SLIM_VAE_CONFIG['activation_fn']
    # )
    model_light = LightVAE(
        latent_dim=LIGHT_VAE_CONFIG['latent_dim'],
        base_channels=LIGHT_VAE_CONFIG['base_channels'],
        activation_fn=LIGHT_VAE_CONFIG['activation_fn'],
        norm_layer_type=LIGHT_VAE_CONFIG['norm_layer_type']
    )
    
    checkpoint = torch.load(SLIM_VAE_WEIGHTS_PATH, map_location='cpu')
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        
    model_light.load_state_dict(state_dict)
    print("LightVAE权重加载成功。")

    dummy_input = (torch.rand(1, 1, 512, 512) * 2.0 - 1.0)
    print(f"Using device: {device}")
    print(f"Input tensor size: {dummy_input.shape}")

    print("\n" + "="*30)
    print("VAE 速度:")
    time_original = benchmark_vae(model_original, dummy_input, device)
    print(f"平均耗时: {time_original:.4f} 毫秒")

    print("\n" + "="*30)
    print("lightVAE速度:")
    time_slim = benchmark_vae(model_light, dummy_input, device)
    print(f"平均耗时: {time_slim:.4f} 毫秒")

    print("\n" + "="*40)
    print("--- 最终 VAE 组件速度对比 ---")
    print(f"Original VAE (教师): {time_original:.4f} 毫秒 / 每次推理")
    print(f"LightVAE (学生)   : {time_slim:.4f} 毫秒 / 每次推理")
    
    if time_original > 0 and time_slim > 0:
        speedup = time_original / time_slim
        percentage = (time_original - time_slim) / time_original
        print(f"\n结论: LightVAE 比 Original VAE 快 {speedup:.2f} 倍 (节省 {percentage:.1%} 的时间)")
    print("="*40)