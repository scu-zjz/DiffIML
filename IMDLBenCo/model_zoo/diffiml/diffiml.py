import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchinfo
from itertools import chain
from matplotlib import pyplot as plt
from torchinfo import summary
from torchvision import models
import torchvision.models.resnet
from diffusers.schedulers import DDIMScheduler
from diffusers.models import UNet2DModel
from diffusers.models import AutoencoderKL, UNet2DModel
from .segformer import get_mit_b2, get_mit_b3, get_mit_b4, get_mit_b5
from .resnet import ResNet101, ResNet50
from IMDLBenCo.registry import MODELS
from tqdm import tqdm
import os
        
# 1. 先定义 SRM 滤波器 (放在文件开头)
class SRMConv(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义3个经典的SRM滤波器核
        kernels = [
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ]
        weights = []
        for k in kernels:
            k = np.array(k, dtype=np.float32)
            k = k / (np.sum(np.abs(k)) + 1e-6)
            weights.append(k)
        self.srm_weights = torch.from_numpy(np.stack(weights, axis=0)).unsqueeze(1).float()
        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3, bias=False)
        self.conv.weight.data = self.srm_weights
        for param in self.conv.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 输入 x 必须是 [0, 1] 范围
        return self.conv(x) * 30.0 # 放大特征，防止被忽略

# 2. 定义独立的噪声提取分支 (Tiny CNN)
class NoiseBranch(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.srm = SRMConv()
        self.net = nn.Sequential(
            # 下采样 1 (256)
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2),
            # 下采样 2 (128)
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            # 下采样 3 (64) - 对齐 SegFormer 输出
            nn.Conv2d(32, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        # 确保输入给 SRM 的是 [0, 1]
        if x.min() < 0: x_srm = (x + 1.0) / 2.0
        else: x_srm = x
        
        noise = self.srm(x_srm)
        return self.net(noise)

# 3. 修改后的 ConditionEncoder
class ConditionEncoder(nn.Module):
    def __init__(self, backbone='segformer_b3', pretrain=True, out_ch=128, out_shape=64):
        super(ConditionEncoder, self).__init__()
        
        # --- 主路：SegFormer (负责结构/RGB) ---
        if backbone == 'segformer_b3':
            self.backbone = get_mit_b3(pretrain)
            # 注意：我们将 SegFormer 的输出通道减少一点，或者保持不变
            # 这里假设我们要让最终输出是 out_ch (128)
            # 我们分配 96 给 RGB，32 给 噪声
            self.rgb_dim = out_ch - 32
            self.conv = nn.Conv2d(512, self.rgb_dim, 3, stride=1, padding=1, bias=False)
        # ... (其他 backbone 同理修改) ...
        
        # --- 辅路：NoiseBranch (负责 AutoSplice) ---
        self.noise_branch = NoiseBranch(out_channels=32)
        
        self.out_shape = out_shape

    def forward(self, image):
        # 1. 主路处理 (RGB)
        features = self.backbone(image)
        fea_rgb = features[-1]
        fea_rgb = F.interpolate(fea_rgb, self.out_shape, mode='bilinear')
        fea_rgb = self.conv(fea_rgb) # [B, 96, 64, 64]
        
        # 2. 辅路处理 (Noise)
        fea_noise = self.noise_branch(image) # [B, 32, 64, 64]
        
        # 3. 晚期融合 (Concat)
        # 结果是 [B, 128, 64, 64]，完美兼容 Unet
        fea_final = torch.cat([fea_rgb, fea_noise], dim=1)
        
        return fea_final

class Unet(nn.Module):
    def __init__(self, ch=128):
        super(Unet, self).__init__()
        self.unet = UNet2DModel(
            in_channels=8,  # (mask + edge)
            out_channels=8, # 预测 8 通道 (mask + edge)
            block_out_channels=[ch], # 浅层: 只有 1 个 block
            down_block_types=(
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
            ),
            layers_per_block=2
        )
        self.concat = nn.Conv2d(ch + 128, ch, 3, stride=1, padding=1, bias=False)

    def forward(self, sample, timestep, feature):
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.unet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.unet.conv_in(sample)
        sample = self.concat(torch.cat([sample, feature], dim=1))

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.unet.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                # sample = self.concat_up_conv[idx](torch.cat([feas[idx], sample], dim=1).to(sample.device))
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        return sample


def pred_dict(pred_mask):
    output_dict = {
        # loss for backward
        "backward_loss": None,
        # predicted mask, will calculate for metrics automatically
        "pred_mask": pred_mask,
        # predicted binaray label, will calculate for metrics automatically
        "pred_label": None,

        # ----values below is for visualization----
        # automatically visualize with the key-value pairs
        "visual_loss": {
            "seg_loss": None,
            "edg_loss": None,
            "combined_loss": None
        },
        "visual_image": {
            "pred_mask": pred_mask,
        }
        # -----------------------------------------
    }
    return output_dict


@MODELS.register_module()
class DiffIML(nn.Module):
    def __init__(self,
                 backbone: str = 'segformer_b3',
                 ch: int = 128,
                 pretrain: str = True,
                 seg_weight: float = 0.2,
                 prior_rate: float = 0.0,
                 infer_time: int = 5,
                 num_inference_steps: int = 10,
                 light_vae_weights: str = '/mnt/data0/yunfei/workspace/IMDLBenCo/log/train_light_vae/checkpoints/light_vae_weights.pth',
                 latent_dim: int = 4,
                 base_channels: int = 32,
                 norm_layer_type: str = 'BatchNorm',
                #  layers_per_block: int = 2,
                #  base_channels_list: list = [64, 128, 256, 512],
                 activation_fn_name: str = 'relu'
                 ):
        super(DiffIML, self).__init__()
        pretrain_b = True if pretrain == 'True' else False
        self.extractor = ConditionEncoder(backbone, out_ch=ch, pretrain=pretrain_b)
        
        # --- 使用浅层 Unet ---
        self.unet = Unet(ch) 
        
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="sample")

        if activation_fn_name == 'relu':
             act_fn = nn.ReLU(inplace=True)
        else:
             act_fn = nn.SiLU(inplace=True)

        # self.vae = SlimVAE(
        #     latent_dim=latent_dim,
        #     block_out_channels=base_channels_list,
        #     activation_fn=act_fn,
        #     layers_per_block=layers_per_block
        # )
        self.vae = LightVAE(
            latent_dim=latent_dim,
            base_channels=base_channels,
            activation_fn=act_fn,
            norm_layer_type=norm_layer_type
        )
        if light_vae_weights and os.path.exists(light_vae_weights):
            print(f"Loading LightVAE weights from: {light_vae_weights}")
            checkpoint = torch.load(light_vae_weights, map_location='cpu')
            
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
                
            self.vae.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: LightVAE weights not provided or path invalid. Using randomly initialized LightVAE.")

        self.vae.requires_grad_(False)
        print("LightVAE (SlimVAE) frozen for DiffIML training.")
        
        self.infer_time = infer_time
        self.num_inference_steps = num_inference_steps
        self.prior_rate = prior_rate
        self.seg_weight = seg_weight
        self.latent_dim = latent_dim

    def forward(self, image, mask=None, edge_mask=None, *args, **kwargs):
        if self.training:
            # ---
            # 3. 修改：训练分支改为 4 通道扩散
            # ---
            image, mask, edge_mask = image.float(), mask.float(), edge_mask.float()
            
            # 3a. 恢复 [-1, 1] 映射
            mask, edge_mask = torch.where(mask == 0, -1., 1.), torch.where(edge_mask == 0, -1., 1.)
            
            latent_mask = self.vae.encode_mask(mask) # [B, 4, H/8, W/8]
            latent_edge = self.vae.encode_mask(edge_mask) # [B, 4, H/8, W/8]
            features = self.extractor(image)
            
            y_start = torch.cat([latent_mask, latent_edge], dim=1)

            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (image.shape[0],), device=image.device).long()
            
            noise = torch.randn_like(y_start).to(image.device)

            y_start = y_start + self.prior_rate * torch.randn_like(y_start).to(image.device)

            y_t = self.scheduler.add_noise(y_start, noise, t)
            
            # 3d. Unet(4ch) -> 8ch
            out = self.unet(y_t, t, features)  # [B, 8, H/8, W/8]
            
            # 3e. 损失函数：Unet 必须从 4ch 输入 预测 8ch 输出
            seg_loss = F.mse_loss(out[:, 0:self.latent_dim, ...], latent_mask)
            edg_loss = F.mse_loss(out[:, self.latent_dim:, ...], latent_edge)
            
            combined_loss = self.seg_weight * seg_loss + (1 - self.seg_weight) * edg_loss

            output_dict = {
                "backward_loss": combined_loss,
                "pred_mask": mask, # 注意：这里返回的 mask 是 [-1, 1] 范围的
                "pred_label": None,
                "visual_loss": {
                    "seg_loss": seg_loss,
                    "edg_loss": edg_loss,
                    "combined_loss": combined_loss
                },
                "visual_image": {
                    "pred_mask": mask,
                }
            }
            return output_dict

        else:
            # ---
            # 4. 修改：推理分支改为 4 通道扩散
            # ---
            with torch.no_grad():
                image = image.float()
                features = self.extractor(image)# [B, ch, H/8, W/8]
                outs = []
                self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
                timesteps = self.scheduler.timesteps
                
                # 4a. 潜空间形状为 4 通道
                latent_shape = (image.shape[0], self.latent_dim * 2, image.shape[2] // 8, image.shape[3] // 8)
                for _ in range(self.infer_time):
                    # 4b. 从 4 通道噪声开始
                    y_t = torch.randn(latent_shape, device=image.device)
                    
                    iterable = tqdm(enumerate(timesteps), total=len(timesteps), desc="DiffIML Inference") if self.infer_time == 1 else enumerate(timesteps)
                    for i, t in iterable:
                        # 4c. Unet(4ch) -> 8ch
                        model_output = self.unet(y_t, t, features) # [B, 8, H/8, W/8]
                        
                        # 4d. 只取 4 通道 mask 预测
                        model_output_mask = model_output[:, 0:self.latent_dim, ...]
                        
                        # 4e. 使用 4 通道预测 和 4 通道 y_t 进行步进
                        step_output = self.scheduler.step(model_output, t, y_t) # <-- 正确
                        y_t = step_output.prev_sample
                    outs.append(y_t)
                
                stacked_outs = torch.stack(outs, dim=0)
                
                # 4f. out 是最终的 4 通道 latent mask
                out = torch.mean(stacked_outs, dim=0) 
                
                assert out.shape == latent_shape
                out_mask_latent = out[:, 0:self.latent_dim, ...] # [B, 4, ...]
        
                # 4g. 解码器现在接收正确的 4 通道输入
                pred_mask = self.vae.decode_mask(out_mask_latent)
                
                # 4h. 保留关键的 Bug 修复：[-1, 1] -> [0, 1]
                pred_mask = torch.clamp((pred_mask + 1.0) / 2.0, 0.0, 1.0)
                
                output_dict = pred_dict(pred_mask)

            return output_dict


# ---
# 保留 F1=0.8 版本的教师 VAE 定义
# ---
class VAE(nn.Module):
    def __init__(self, vae_path='/mnt/data0/yunfei/workspace/model/diffusion/stable_diff/pretrained/vae'):
        super(VAE, self).__init__()
        # 允许路径被覆盖，但提供一个默认值
        if not os.path.isdir(vae_path):
             # 尝试使用 F1=0.8 版本中的硬编码路径
             vae_path_fallback = '/mnt/data0/yunfei/workspace/model/diffusion/stable_diff/pretrained/vae'
             print(f"Warning: VAE path {vae_path} not found. Trying fallback {vae_path_fallback}")
             vae_path = vae_path_fallback
             
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.mask_latent_scale_factor = 0.18215

    def encode_mask(self, mask): # mask 期望是 [-1, 1]
        image = torch.cat([mask, mask, mask], dim=1)
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.mask_latent_scale_factor
        return rgb_latent

    def decode_mask(self, mask_latent):
        # scale latent
        mask_latent = mask_latent / self.mask_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(mask_latent)
        stacked = self.vae.decoder(z) # 输出 [-1, 1]
        # mean of output channels
        mask_mean = stacked.mean(dim=1, keepdim=True) # 输出 [-1, 1]
        return mask_mean
    
    def forward(self, mask): # mask 期望是 [-1, 1]
        latent_representation = self.encode_mask(mask)
        reconstructed_mask = self.decode_mask(latent_representation)
        return reconstructed_mask, latent_representation


def f1_score(y_true, y_pred, threshold=0.5):
    # 将预测值转换为二值（1 或 0）
    y_pred = (y_pred > threshold).float()

    # 计算TP、FP、FN
    tp = (y_true * y_pred).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # 计算F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1.item()

class LightVAE(nn.Module):
    def __init__(self,
                 latent_dim=4,
                 base_channels=32,
                 latent_scale_factor=0.18215,
                 activation_fn = nn.SiLU(inplace=False),
                 norm_layer_type = "BatchNorm"):
        super(LightVAE, self).__init__()
        self.latent_scale_factor = latent_scale_factor
        self.latent_dim = latent_dim

        if norm_layer_type == "BatchNorm":
            norm_layer = lambda channels: nn.BatchNorm2d(channels)
        elif norm_layer_type == "GroupNorm":
            num_groups = 8
            norm_layer = lambda channels: nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        else:
            raise ValueError("Unsupported norm_layer_type")

        self.encoder = nn.Sequential(
            # Block 1: 512 -> 256
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1, bias=False), 
            norm_layer(base_channels),
            activation_fn,
            
            # Block 2: 256 -> 128
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(base_channels*2),
            activation_fn,
            
            # Block 3: 128 -> 64
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(base_channels*4),
            activation_fn,
            
            # Final Conv to latent dim
            nn.Conv2d(base_channels*4, latent_dim, kernel_size=1) 
        )

        self.decoder = nn.Sequential(
            # Initial Conv from latent dim
            nn.Conv2d(latent_dim, base_channels*4, kernel_size=1), 
            
            # Block 1: 64 -> 128
            norm_layer(base_channels*4),
            activation_fn,
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),

            # Block 2: 128 -> 256
            norm_layer(base_channels*2),
            activation_fn,
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),

            # Block 3: 256 -> 512
            norm_layer(base_channels),
            activation_fn,
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),

            # Final Conv to 3 channels (模仿 VAE 输出)
            norm_layer(base_channels),
            activation_fn,
            nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1) # 输出 1 通道图像
        )
        print(f"Initialized Improved LightVAE with latent_dim={latent_dim}, base_channels={base_channels}, norm={norm_layer_type}")

    # encode_mask, decode_mask, forward 方法保持不变 (因为它们处理 3 通道复制和 scale_factor)
    def encode_mask(self, mask): # mask 期望是 [-1, 1]
        if mask.shape[1] == 1:
            image = mask # 直接使用 1 通道 mask
            
        latent_mean = self.encoder(image)
        latent_scaled = latent_mean * self.latent_scale_factor
        return latent_scaled

    def decode_mask(self, mask_latent):
        mask_latent = mask_latent / self.latent_scale_factor
        stacked = self.decoder(mask_latent) # 输出 [B, 1, H, W]
        mask_mean = stacked
        
        return mask_mean

    def forward(self, mask): # mask 期望是 [-1, 1]
        if mask.shape[1] == 1:
            image = mask
        else:
            image = mask.mean(dim=1, keepdim=True)
        
        latent_mean = self.encoder(image) 
        latent_scaled = latent_mean * self.latent_scale_factor
        stacked = self.decoder(latent_mean)
        mask_mean = stacked
        
        return mask_mean, latent_scaled
    
class SlimVAE(nn.Module):
    def __init__(self,
                 block_out_channels=[64, 128, 256, 512],
                 layers_per_block=2,
                 latent_dim=4,
                 norm_num_groups=32,
                 activation_fn=nn.ReLU(inplace=True)
                 ):
        super(SlimVAE, self).__init__()
        
        act_fn_name = activation_fn.__class__.__name__.lower()
        if 'relu' in act_fn_name:
            act_fn_str = 'relu'
        elif 'silu' in act_fn_name:
            act_fn_str = 'silu'
        else:
            act_fn_str = 'relu' # default
            
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D", "DownEncoderBlock2D",
                "DownEncoderBlock2D", "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D", "UpDecoderBlock2D",
                "UpDecoderBlock2D", "UpDecoderBlock2D",
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            latent_channels=latent_dim,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn_str
        )
        
        self.mask_latent_scale_factor = 0.18215
        self.latent_dim = latent_dim
        
        params = sum(p.numel() for p in self.vae.parameters()) / 1e6
        print(f"Initialized SlimVAE with channels: {block_out_channels}, layers: {layers_per_block}, act_fn: {act_fn_str}")
        print(f"SlimVAE Parameters: {params:.2f} M")

    def encode(self, x): # x 期望是 [-1, 1]
        """返回 *未缩放* 的潜变量 (mean)"""
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean

    def decode(self, z):
        """解码 *未缩放* 的潜变量"""
        z = self.vae.post_quant_conv(z)
        stacked = self.vae.decoder(z) # 输出 [-1, 1]
        return stacked

    def encode_mask(self, mask): # mask 期望是 [-1, 1]
        # DiffIML 的推理和训练调用此函数
        if mask.shape[1] == 1:
            image = torch.cat([mask, mask, mask], dim=1)
        else:
            image = mask
            
        latent_mean = self.encode(image)
        latent_scaled = latent_mean * self.mask_latent_scale_factor # 缩放！
        return latent_scaled

    def decode_mask(self, mask_latent):
        # DiffIML 的推理调用此函数
        mask_latent = mask_latent / self.mask_latent_scale_factor # 反向缩放！
        stacked = self.decode(mask_latent) # 输出 [-1, 1]
        mask_mean = stacked.mean(dim=1, keepdim=True)
        return mask_mean

    def forward(self, mask): # mask 期望是 [-1, 1]
        # 蒸馏训练 (train_light_vae.py) 调用此函数
        if mask.shape[1] == 1:
            image = torch.cat([mask, mask, mask], dim=1)
        else:
            image = mask
        
        latent_mean = self.encode(image) # 获取未缩放的 latent
        
        # --- 兼容 0.18215 缩放的蒸馏 ---
        latent_scaled = latent_mean * self.mask_latent_scale_factor
        
        stacked = self.decode(latent_mean) # 解码器仍然解码未缩放的
        mask_mean = stacked.mean(dim=1, keepdim=True)
        
        return mask_mean, latent_scaled