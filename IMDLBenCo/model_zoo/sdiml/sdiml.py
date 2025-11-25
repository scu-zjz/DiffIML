from torch import nn
import torch
from .segformer import get_mit_b3
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from .controlnet import ControlNetModel
from IMDLBenCo.registry import MODELS


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
class SDIML(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = get_mit_b3()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [1, 256, 16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [1, 128, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [1, 64, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [1, 32, 128, 128]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # [1, 1, 256, 256]
        )
        self.processor = nn.Sequential(
            nn.ConvTranspose2d(512, 480, kernel_size=4, stride=2, padding=1),  # [1, 480, 32, 32]
            nn.ConvTranspose2d(480, 320, kernel_size=4, stride=2, padding=1),  # [1, 320, 64, 64]
        )
        self.vae = AutoencoderKL.from_pretrained('/mnt/data0/dubo/workspace/stable-diffusion-2-1-base/vae',
                                                 use_safetensors=False)
        self.vae.requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained('/mnt/data0/dubo/workspace/stable-diffusion-2-1-base/unet',
                                                         use_safetensors=False)
        self.unet.requires_grad_(False)
        self.controlnet = ControlNetModel.from_unet(self.unet)
        self.scheduler = PNDMScheduler.from_pretrained('/mnt/data0/dubo/workspace/stable-diffusion-2-1-base/scheduler')
        self.empty_text_embed = torch.load(
            '/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/sdiml/empty_encoder_hidden_states.pt')

    def forward(self, image, mask=None, edge_mask=None, *args, **kwargs):
        if self.training:
            image, mask, edge_mask = image.float(), mask.float(), edge_mask.float()
            feature = self.feature_extractor(image)[-1]  # [1, 512, 16, 16]

            input_feature = self.processor(feature)  # [1, 4, 64, 64]

            # imagenet归一化转sd归一化
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
            sd_mean = torch.tensor([0.5, 0.5, 0.5], device=image.device)
            sd_std = torch.tensor([0.5, 0.5, 0.5], device=image.device)
            image = image * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
            image = (image - sd_mean[:, None, None]) / sd_std[:, None, None]

            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents, device=image.device)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (image.shape[0],),
                                      device=image.device)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            empty_text_embed = self.empty_text_embed.repeat(
                (image.shape[0], 1, 1)
            ).to(image.device)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                controlnet_cond_feature=input_feature,
                encoder_hidden_states=empty_text_embed,
                return_dict=False,
            )
            # mid_block_res_sample [1, 1280, 8, 8]
            aug_feature = mid_block_res_sample.clone()

            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                down_block_additional_residuals=[
                    sample for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample,
                encoder_hidden_states=empty_text_embed,
                return_dict=False,
            )[0]

            # Get the target for loss depending on the prediction type
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

            pred_mask = self.decoder(aug_feature)
            pred_mask = F.interpolate(pred_mask, size=(image.shape[2], image.shape[3]), mode='bilinear',
                                      align_corners=False)  # [1, 1, 512, 512]

            loss_rec = F.mse_loss(model_pred, target, reduction="mean")
            # loss_seg = F.binary_cross_entropy_with_logits(pred_mask, mask)
            # loss_edg = F.binary_cross_entropy_with_logits(pred_mask, mask, edge_mask)
            loss_bce = F.binary_cross_entropy_with_logits(pred_mask, mask)

            # combine_loss = loss_rec + loss_bce
            combine_loss = loss_bce
            pred_mask = torch.sigmoid(pred_mask)
            output_dict = {
                "backward_loss": combine_loss,
                "pred_mask": pred_mask,
                "pred_label": None,
                "visual_loss": {
                    "predict_loss": combine_loss,
                    "bce_loss": loss_bce,
                    "rec_loss": loss_rec
                },
                "visual_image": {
                    "pred_mask": pred_mask,
                }
            }
            return output_dict
        else:
            feature = self.feature_extractor(image)[-1]  # [1, 512, 16, 16]
            input_feature = self.processor(feature)  # [1, 4, 64, 64]
            # imagenet归一化转sd归一化
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
            sd_mean = torch.tensor([0.5, 0.5, 0.5], device=image.device)
            sd_std = torch.tensor([0.5, 0.5, 0.5], device=image.device)
            image = image * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
            image = (image - sd_mean[:, None, None]) / sd_std[:, None, None]

            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents, device=image.device)
            timesteps = torch.randint(50, 50 + 1, (image.shape[0],),
                                      device=image.device)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            empty_text_embed = self.empty_text_embed.repeat(
                (image.shape[0], 1, 1)
            ).to(image.device)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                controlnet_cond_feature=input_feature,
                encoder_hidden_states=empty_text_embed,
                return_dict=False,
            )

            pred_mask = self.decoder(mid_block_res_sample)
            pred_mask = F.interpolate(pred_mask, size=(image.shape[2], image.shape[3]), mode='bilinear',
                                      align_corners=False)  # [1, 1, 512, 512]
            pred_mask = torch.sigmoid(pred_mask)
            return pred_dict(pred_mask)
