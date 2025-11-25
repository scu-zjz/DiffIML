from torch import nn
from .segformer import get_mit_b3, get_one_channel_mit_b0, count_model_parameters, get_mit_b4, get_mit_b5
import torch.nn.functional as F
import torch
from diffusers import PNDMScheduler
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
class LatentIML(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = get_mit_b3()
        self.rc = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [1, 128, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [1, 64, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [1, 32, 128, 128]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # [1, 1, 256, 256]
        )
        self.scheduler = PNDMScheduler.from_pretrained('/mnt/data0/dubo/workspace/stable-diffusion-2-1-base/scheduler')

    def forward(self, image, mask=None, edge_mask=None, *args, **kwargs):
        if self.training:
            image, mask, edge_mask = image.float(), mask.float(), edge_mask.float()

            # add noise here
            # noise = torch.randn_like(image, device=image.device)
            # timesteps = torch.full((image.shape[0],), 49, dtype=torch.int, device=image.device)
            # image = self.scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

            z_x = self.image_encoder(image)[-1]
            z_x = self.rc(z_x)
            y_pred = self.mask_decoder(z_x)
            y_pred = F.interpolate(y_pred, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

            loss_seg = F.binary_cross_entropy_with_logits(y_pred, mask)
            loss_edg = F.binary_cross_entropy_with_logits(y_pred, mask, edge_mask)
            # loss_bce = 20 * loss_edg + loss_seg
            loss_bce = loss_seg
            combine_loss = loss_bce

            y_pred = torch.sigmoid(y_pred)
            output_dict = {
                "backward_loss": combine_loss,
                "pred_mask": y_pred,
                "pred_label": None,
                "visual_loss": {
                    "predict_loss": combine_loss,
                    "bce_loss": loss_bce,
                },
                "visual_image": {
                    "pred_mask": y_pred,
                }
            }

            return output_dict
        else:
            z = self.image_encoder(image)[-1]
            z = self.rc(z)
            y = self.mask_decoder(z)
            y = F.interpolate(y, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
            y = torch.sigmoid(y)
            return pred_dict(y)
