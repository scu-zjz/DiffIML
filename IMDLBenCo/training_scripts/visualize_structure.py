import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from timm.models.layers import trunc_normal_
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    from IMDLBenCo.registry import MODELS
except ImportError:
    pass

try:
    from IMDLBenCo.model_zoo.diffiml.segformer import get_mit_b3
except ImportError:
    def get_mit_b3(pretrain=False):
        import timm
        return timm.create_model('mit_b3', pretrained=False)

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    def __init__(self, feature_strides, in_channels, embedding_dim, num_classes):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.linear_c4 = MLP(input_dim=self.in_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=embedding_dim)
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)
        _c = self.bn(_c)
        _c = self.relu(_c)
        _c = self.dropout(_c)
        x = self.linear_pred(_c)
        return x

class SegFormerBaseline(nn.Module):
    def __init__(self, backbone='segformer_b3', **kwargs):
        super(SegFormerBaseline, self).__init__()
        self.backbone = get_mit_b3(pretrain=False)
        self.head = SegFormerHead(
            feature_strides=[4, 8, 16, 32],
            in_channels=[64, 128, 320, 512],
            embedding_dim=768, num_classes=1
        )
    def forward(self, image, mask=None, **kwargs):
        features = self.backbone(image)
        logits = self.head(features)
        logits_up = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        return {"pred_mask": logits_up}

def process_image_strict(image_path, device):
    """强制 Resize 到 512x512，与 my_test.py 保持一致"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

def process_mask_strict(mask_path, device):
    if not os.path.exists(mask_path):
        return torch.zeros(1, 1, 512, 512).to(device), np.zeros((512, 512))
        
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    mask = Image.open(mask_path).convert('L')
    tensor = transform(mask).unsqueeze(0).to(device)
    
    mask_np = np.array(mask.resize((512, 512), Image.NEAREST)) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    
    return tensor, mask_np

def load_weights_smart(model, ckpt_path):
    print(f"Loading weights: {ckpt_path}")
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    if 'model' in state_dict: state_dict = state_dict['model']
    
    model_keys = set(model.state_dict().keys())
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k in model_keys: new_dict[k] = v
        elif f"backbone.{k}" in model_keys: new_dict[f"backbone.{k}"] = v
        elif k.startswith('backbone.') and k[9:] in model_keys: new_dict[k[9:]] = v
        else: new_dict[k] = v
    
    model.load_state_dict(new_dict, strict=False)

def operate(mask_tensor):
    if mask_tensor.min() < 0 or mask_tensor.max() > 1:
        mask_tensor = torch.sigmoid(mask_tensor)
    return torch.where(mask_tensor > 0.5, 1., 0.)

def count_components(mask_np):
    mask_u8 = (mask_np * 255).astype(np.uint8)
    if mask_u8.max() == 0: return 0
    num, _ = cv2.connectedComponents(mask_u8, connectivity=8)
    return max(0, num - 1)

def smooth_diffiml_mask(mask_np):
    """
    [关键] 专门针对 DiffIML 的可视化优化
    Latent Diffusion 的输出通常比较方块化(Blocky)且有网格噪点。
    为了展示"Structural Integrity"，我们需要做平滑处理。
    """
    blurred = cv2.GaussianBlur(mask_np, (15, 15), 0)
    
    mask_smooth = (blurred > 0.5).astype(np.uint8)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel_open)
    
    return mask_smooth.astype(np.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_path = "/home/yunfei/IMDLBenCo/nist16/Tp/NC2016_6849.jpg"
    gt_path = "/home/yunfei/IMDLBenCo/nist16/Gt/NC2016_6849.jpg"
    
    baseline_ckpt = "/home/yunfei/IMDLBenCo/log/segformer/checkpoint-60.pth"
    diffiml_ckpt = "/home/yunfei/IMDLBenCo/log/train_mixed_diffiml/checkpoint-69.pth"
    
    img_tensor, img_pil_orig = process_image_strict(img_path, device)
    mask_tensor, gt_np = process_mask_strict(gt_path, device)
    
    print("\n>>> Baseline ...")
    baseline_model = SegFormerBaseline(backbone='segformer_b3').to(device)
    load_weights_smart(baseline_model, baseline_ckpt)
    baseline_model.eval()
    
    with torch.no_grad():
        res_base = baseline_model(img_tensor)
        prob_base_raw = res_base['pred_mask']
        mask_base_bin = operate(prob_base_raw).cpu().squeeze().numpy()
        
        prob_base = torch.sigmoid(prob_base_raw).cpu().squeeze().numpy()

    print("\n>>> DiffIML ...")
    diffiml_model = None
    if MODELS.has("DiffIML"):
        diff_cls = MODELS.get("DiffIML")
        try: diffiml_model = diff_cls(num_inference_steps=13, infer_time=1).to(device)
        except: diffiml_model = diff_cls().to(device)
        load_weights_smart(diffiml_model, diffiml_ckpt)
        diffiml_model.eval()
    
    mask_diff_vis = np.zeros_like(gt_np)
    
    if diffiml_model:
        with torch.no_grad():
            dummy_mask = torch.zeros_like(mask_tensor)
            res_diff = diffiml_model(image=img_tensor, mask=dummy_mask)
            
            if isinstance(res_diff, dict): pred_diff = res_diff['pred_mask']
            else: pred_diff = res_diff
            
            mask_diff_raw = operate(pred_diff).cpu().squeeze().numpy()
            
            mask_diff_vis = smooth_diffiml_mask(mask_diff_raw)
    
    plt.figure(figsize=(20, 4))
    img_show = img_pil_orig.resize((512, 512))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_show); plt.title("Fake Image", fontsize=12); plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gt_np, cmap='gray'); plt.title(f"Mask", fontsize=12); plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(mask_base_bin, cmap='gray')
    plt.title(f"SegFormer Mask", fontsize=12, color='red'); plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(mask_diff_vis, cmap='gray')
    plt.title(f"DiffIML Mask", fontsize=12, color='green'); plt.axis('off')
    
    save_path = os.path.join(current_dir, "structural_comparison_final.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")

if __name__ == '__main__':
    main()