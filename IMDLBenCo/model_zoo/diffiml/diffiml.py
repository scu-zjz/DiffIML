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


class ConditionEncoder(nn.Module):
    def __init__(self, backbone='segformer_b3', pretrain=True, out_ch=128, out_shape=64):
        super(ConditionEncoder, self).__init__()

        if backbone == 'segformer_b3':
            self.backbone = get_mit_b3(pretrain)
            in_channels = [64, 128, 320, 512]
        elif backbone == 'segformer_b2':
            self.backbone = get_mit_b2(pretrain)
            in_channels = [64, 128, 320, 512]
        elif backbone == 'segformer_b4':
            self.backbone = get_mit_b4(pretrain)
            in_channels = [64, 128, 320, 512]
        elif backbone == 'segformer_b5':
            self.backbone = get_mit_b5(pretrain)
            in_channels = [64, 128, 320, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        embedding_dim = out_ch

        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.out_shape = out_shape

    def forward(self, image):
        c1, c2, c3, c4 = self.backbone(image)

        target_size = (self.out_shape, self.out_shape)

        _c4 = F.interpolate(self.linear_c4(c4), size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=target_size, mode= 'bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=target_size, mode='bilinear', align_corners=False)
        _c1 = F.interpolate(self.linear_c1(c1), size=target_size, mode='bilinear', align_corners=False)

        fea_rgb = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return fea_rgb

class Unet(nn.Module):
    def __init__(self, ch=128):
        super(Unet, self).__init__()
        self.unet = UNet2DModel(
            in_channels=8,
            out_channels=8,
            block_out_channels=[ch],
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

        skip_sample = sample
        sample = self.unet.conv_in(sample)
        sample = self.concat(torch.cat([sample, feature], dim=1))

        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        sample = self.unet.mid_block(sample, emb)

        skip_sample = None
        for upsample_block in self.unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                # sample = self.concat_up_conv[idx](torch.cat([feas[idx], sample], dim=1).to(sample.device))
                sample = upsample_block(sample, res_samples, emb)

        # post-process
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
        "backward_loss": None,
        "pred_mask": pred_mask,
        "pred_label": None,

        "visual_loss": {
            "seg_loss": None,
            "edg_loss": None,
            "combined_loss": None
        },
        "visual_image": {
            "pred_mask": pred_mask,
        }
    }
    return output_dict


@MODELS.register_module()
class DiffIML(nn.Module):
    def __init__(self,
                 backbone: str = 'segformer_b3',
                 ch: int = 128,
                 pretrain: bool = True,
                 seg_weight: float = 0.2,
                 prior_rate: float = 0.0,
                 infer_time: int = 5,
                 num_inference_steps: int = 10,
                 light_vae_weights: str = './log/train_light_vae/checkpoints/light_vae_weights.pth',
                 latent_dim: int = 4,
                 base_channels: int = 32,
                 norm_layer_type: str = 'BatchNorm',
                #  layers_per_block: int = 2,
                #  base_channels_list: list = [64, 128, 256, 512],
                 activation_fn_name: str = 'relu'
                 ):
        super(DiffIML, self).__init__()
        pretrain_b = str(pretrain).lower() == 'true'
        self.extractor = ConditionEncoder(backbone, out_ch=ch, pretrain=pretrain_b)
        
        self.unet = Unet(ch) 
        
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="sample")

        if activation_fn_name == 'relu':
             act_fn = nn.ReLU(inplace=True)
        else:
             act_fn = nn.SiLU(inplace=True)

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
            image, mask, edge_mask = image.float(), mask.float(), edge_mask.float()
            
            mask, edge_mask = torch.where(mask == 0, -1., 1.), torch.where(edge_mask == 0, -1., 1.)
            
            latent_mask = self.vae.encode_mask(mask) # [B, 4, H/8, W/8]
            latent_edge = self.vae.encode_mask(edge_mask) # [B, 4, H/8, W/8]
            features = self.extractor(image)
            
            y_start = torch.cat([latent_mask, latent_edge], dim=1)

            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (image.shape[0],), device=image.device).long()
            
            noise = torch.randn_like(y_start).to(image.device)

            y_start = y_start + self.prior_rate * torch.randn_like(y_start).to(image.device)

            y_t = self.scheduler.add_noise(y_start, noise, t)
            
            out = self.unet(y_t, t, features)  # [B, 8, H/8, W/8]
            
            seg_loss = F.mse_loss(out[:, 0:self.latent_dim, ...], latent_mask)
            edg_loss = F.mse_loss(out[:, self.latent_dim:, ...], latent_edge)
            
            combined_loss = self.seg_weight * seg_loss + (1 - self.seg_weight) * edg_loss

            output_dict = {
                "backward_loss": combined_loss,
                "pred_mask": mask,
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
        # else:
        #     # 关掉 TTA
        #     with torch.no_grad():
        #         image = image.float()
                
        #         features = self.extractor(image)
                
        #         outs = []
        #         self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        #         timesteps = self.scheduler.timesteps
                
        #         latent_shape = (image.shape[0], self.latent_dim * 2, image.shape[2] // 8, image.shape[3] // 8)
                
        #         for _ in range(self.infer_time):
        #             y_t = torch.randn(latent_shape, device=image.device)
        #             for i, t in enumerate(timesteps):
        #                 model_output = self.unet(y_t, t, features)
        #                 step_output = self.scheduler.step(model_output, t, y_t)
        #                 y_t = step_output.prev_sample
        #             outs.append(y_t)
                
        #         stacked_outs = torch.stack(outs, dim=0)
        #         out = torch.mean(stacked_outs, dim=0) # [B, 8, H, W]
                
        #         out_mask_latent = out[:, 0:self.latent_dim, ...]
        #         pred_mask_raw = self.vae.decode_mask(out_mask_latent)
        #         pred_prob = torch.sigmoid(pred_mask_raw)
                
        #         return pred_dict(pred_prob)
        else:
            # 加入 TTA (Flip)
            with torch.no_grad():
                image = image.float()
                
                # [原图, 翻转图]
                image_flip = torch.flip(image, dims=[3])
                image_concat = torch.cat([image, image_flip], dim=0)
                
                features = self.extractor(image_concat)
                
                outs = []
                self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
                timesteps = self.scheduler.timesteps
                
                latent_shape = (image_concat.shape[0], self.latent_dim * 2, image_concat.shape[2] // 8, image_concat.shape[3] // 8)
                
                for _ in range(self.infer_time):
                    y_t = torch.randn(latent_shape, device=image.device)
                    
                    for i, t in enumerate(timesteps):
                        model_output = self.unet(y_t, t, features)
                        step_output = self.scheduler.step(model_output, t, y_t)
                        y_t = step_output.prev_sample
                    outs.append(y_t)
                
                stacked_outs = torch.stack(outs, dim=0)
                out = torch.mean(stacked_outs, dim=0)
                
                out_mask_latent = out[:, 0:self.latent_dim, ...]
                out_edge_latent = out[:, self.latent_dim:, ...]
                pred_mask_raw = self.vae.decode_mask(out_mask_latent)
                pred_prob = torch.sigmoid(pred_mask_raw)
                
                pred_edge_raw = self.vae.decode_mask(out_edge_latent)
                pred_edge_prob = torch.sigmoid(pred_edge_raw)

                pred_normal, pred_flip_res = torch.chunk(pred_prob, 2, dim=0)
                pred_flip_back = torch.flip(pred_flip_res, dims=[3])
                final_pred_mask = torch.max(pred_normal, pred_flip_back)

                edge_normal, edge_flip_res = torch.chunk(pred_edge_prob, 2, dim=0)
                edge_flip_back = torch.flip(edge_flip_res, dims=[3])
                final_pred_edge = torch.max(edge_normal, edge_flip_back)
                
                output_dict = pred_dict(final_pred_mask)
                output_dict["visual_image"]["pred_edge"] = final_pred_edge

            return output_dict

class VAE(nn.Module):
    def __init__(self, vae_path='./pretrained/sd-vae-ft-mse'):
        super(VAE, self).__init__()
        if not os.path.isdir(vae_path):
             # Fallback: try to load from HuggingFace hub
             vae_path_fallback = 'stabilityai/sd-vae-ft-mse'
             print(f"Warning: VAE path {vae_path} not found. Trying fallback {vae_path_fallback}")
             vae_path = vae_path_fallback
             
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.mask_latent_scale_factor = 0.18215

    def encode_mask(self, mask):
        image = torch.cat([mask, mask, mask], dim=1)
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.mask_latent_scale_factor
        return rgb_latent

    def decode_mask(self, mask_latent):
        mask_latent = mask_latent / self.mask_latent_scale_factor
        z = self.vae.post_quant_conv(mask_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        mask_mean = stacked.mean(dim=1, keepdim=True)
        return mask_mean
    
    def forward(self, mask):
        latent_representation = self.encode_mask(mask)
        reconstructed_mask = self.decode_mask(latent_representation)
        return reconstructed_mask, latent_representation


def f1_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()

    tp = (y_true * y_pred).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

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
            
            nn.Conv2d(base_channels*4, latent_dim, kernel_size=1) 
        )

        self.decoder = nn.Sequential(
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

            norm_layer(base_channels),
            activation_fn,
            nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        print(f"Initialized Improved LightVAE with latent_dim={latent_dim}, base_channels={base_channels}, norm={norm_layer_type}")

    def encode_mask(self, mask):
        if mask.shape[1] == 1:
            image = mask
            
        latent_mean = self.encoder(image)
        latent_scaled = latent_mean * self.latent_scale_factor
        return latent_scaled

    def decode_mask(self, mask_latent):
        mask_latent = mask_latent / self.latent_scale_factor
        stacked = self.decoder(mask_latent) # [B, 1, H, W]
        mask_mean = stacked
        
        return mask_mean

    def forward(self, mask):
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
            act_fn_str = 'relu'
            
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

    def encode(self, x):
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean

    def decode(self, z):
        z = self.vae.post_quant_conv(z)
        stacked = self.vae.decoder(z)
        return stacked

    def encode_mask(self, mask):
        if mask.shape[1] == 1:
            image = torch.cat([mask, mask, mask], dim=1)
        else:
            image = mask
            
        latent_mean = self.encode(image)
        latent_scaled = latent_mean * self.mask_latent_scale_factor
        return latent_scaled

    def decode_mask(self, mask_latent):
        mask_latent = mask_latent / self.mask_latent_scale_factor
        stacked = self.decode(mask_latent)
        mask_mean = stacked.mean(dim=1, keepdim=True)
        return mask_mean

    def forward(self, mask):
        if mask.shape[1] == 1:
            image = torch.cat([mask, mask, mask], dim=1)
        else:
            image = mask
        
        latent_mean = self.encode(image) # 获取未缩放的 latent
        
        latent_scaled = latent_mean * self.mask_latent_scale_factor
        
        stacked = self.decode(latent_mean) # 解码器仍然解码未缩放的
        mask_mean = stacked.mean(dim=1, keepdim=True)
        
        return mask_mean, latent_scaled