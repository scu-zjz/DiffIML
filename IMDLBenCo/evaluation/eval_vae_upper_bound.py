"""
Evaluate VAE-only upper bound (no diffusion):
- Given a dataloader yielding masks, compute best F1 by encode->decode only.
- Works for both Teacher (SD AutoencoderKL wrapper) and Student (LightVAE) variants.
- This version is modified to use a FIXED threshold of 0.5 for fair comparison.
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image


@torch.no_grad()
def default_mask_getter(batch: Any) -> torch.Tensor:
    """Extract mask tensor (B,1,H,W) from a batch.
    Adapt this function if your batch structure differs.
    """
    if isinstance(batch, (list, tuple)):
        # Common: (image, mask, ...) -> take second element
        mask = batch[1]
    elif isinstance(batch, dict):
        # Common: {"image":..., "mask":...}
        mask = batch.get("mask", None)
    else:
        raise ValueError("Unsupported batch format for mask extraction")

    if mask is None:
        raise ValueError("Mask not found in batch. Customize default_mask_getter for your loader.")

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Expected mask 3D/4D tensor, got shape {tuple(mask.shape)}")
    return mask


def _ensure_single_channel(x: torch.Tensor) -> torch.Tensor:
    """Map (B,C,H,W) to (B,1,H,W):
    - if C==1: return as-is
    - if C==3: return mean over channel
    - else: take first channel
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
    c = x.size(1)
    if c == 1:
        return x
    if c == 3:
        return x.mean(dim=1, keepdim=True)
    return x[:, :1, :, :]


@torch.no_grad()
def evaluate_vae_upper_bound(
    dataloader: Iterable,
    device: torch.device,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    mask_getter: Callable[[Any], torch.Tensor] = default_mask_getter,
    binarize_input_mask: bool = True,
    limit_batches: Optional[int] = None,
    return_tensors: bool = False,
) -> Dict[str, Any]:
    """Evaluate VAE-only upper bound by encode->decode without diffusion.

    --- MODIFIED: This version uses a FIXED threshold of 0.5 ---

    Args:
        dataloader: Iterable yielding batches that contain a mask tensor.
        device: torch.device for compute.
        encode_fn: Callable mapping mask01 (B,1,H,W in {0,1}) -> latent (B,C,h,w).
        decode_fn: Callable mapping latent -> logits/probs (B,1,H,W) or (B,3,H,W).
        mask_getter: Function extracting mask from a batch.
        binarize_input_mask: If True, threshold input mask at 0.5 to {0,1}.
        limit_batches: If set, limit number of batches to speed up quick checks.

    Returns:
        dict with keys: f1 (at 0.5), threshold, tp, fp, fn
    """
    threshold = 0.5

    tp_sum = torch.zeros(1, device=device, dtype=torch.float64)
    fp_sum = torch.zeros(1, device=device, dtype=torch.float64)
    fn_sum = torch.zeros(1, device=device, dtype=torch.float64)

    for bidx, batch in enumerate(dataloader):
        if limit_batches is not None and bidx >= limit_batches:
            break

        mask = mask_getter(batch).to(device=device, dtype=torch.float32)  # (B,1,H,W)
        if binarize_input_mask:
            mask = (mask > 0.5).float()

        # encode -> decode
        latent = encode_fn(mask)  # (B,C,h,w)
        recon = decode_fn(latent)
        recon = _ensure_single_channel(recon)

        prob = torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0)

        pred = (prob > threshold).float()
        tp = (pred * mask).sum(dtype=torch.float64)
        fp = (pred * (1.0 - mask)).sum(dtype=torch.float64)
        fn = ((1.0 - pred) * mask).sum(dtype=torch.float64)
        
        tp_sum[0] += tp
        fp_sum[0] += fp
        fn_sum[0] += fn

    f1 = (2.0 * tp_sum) / (2.0 * tp_sum + fp_sum + fn_sum + 1e-8)
    
    f1_val = float(f1.item())

    out = {
        "f1": f1_val,
        "threshold": threshold,
        "tp": tp_sum.detach().cpu().numpy()[0],
        "fp": fp_sum.detach().cpu().numpy()[0],
        "fn": fn_sum.detach().cpu().numpy()[0],
    }
    if return_tensors:
        out.update({
            "tp_torch": tp_sum,
            "fp_torch": fp_sum,
            "fn_torch": fn_sum,
        })
    return out


@torch.no_grad()
def evaluate_latent_distribution(
    dataloader: Iterable,
    device: torch.device,
    teacher_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    student_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    mask_getter: Callable[[Any], torch.Tensor] = default_mask_getter,
    num_batches: int = 4,
) -> Dict[str, Any]:
    """Compare teacher vs student latent distributions on a few batches.

    Returns mean/std per channel (averaged across batches) and average cosine similarity.
    """
    means_t, stds_t = [], []
    means_s, stds_s = [], []
    cos_sims = []

    for bidx, batch in enumerate(dataloader):
        if bidx >= num_batches:
            break
        mask = mask_getter(batch).to(device=device, dtype=torch.float32)
        mask = (mask > 0.5).float()

        lt = teacher_encode_fn(mask)  # (B,C,h,w)
        ls = student_encode_fn(mask)

        # Compute per-channel mean/std over spatial dims and batch
        def ch_stats(x: torch.Tensor):
            B, C, H, W = x.shape
            x_flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)  # (C, B*H*W)
            mu = x_flat.mean(dim=1)  # (C,)
            sd = x_flat.std(dim=1, unbiased=False)  # (C,)
            return mu, sd

        mu_t, sd_t = ch_stats(lt)
        mu_s, sd_s = ch_stats(ls)
        means_t.append(mu_t)
        stds_t.append(sd_t)
        means_s.append(mu_s)
        stds_s.append(sd_s)

        # Cosine similarity per-sample (flatten) then averaged
        B, C, H, W = lt.shape
        flat_t = lt.view(B, -1)
        flat_s = ls.view(B, -1)
        # add tiny eps to avoid zero norms
        eps = 1e-8
        cos = (flat_t * flat_s).sum(dim=1) / (
            flat_t.norm(dim=1).clamp_min(eps) * flat_s.norm(dim=1).clamp_min(eps)
        )  # (B,)
        cos_sims.append(cos.mean())

    def stack_mean(lst):
        if not lst:
            return None
        return torch.stack(lst, dim=0).mean(dim=0).detach().cpu().numpy()

    stats = {
        "teacher_mean": stack_mean(means_t),
        "teacher_std": stack_mean(stds_t),
        "student_mean": stack_mean(means_s),
        "student_std": stack_mean(stds_s),
        "avg_cosine": (torch.stack(cos_sims).mean().item() if cos_sims else None),
    }
    return stats


if __name__ == "__main__":
    import argparse
    try:
        from diffusers.models import AutoencoderKL
    except ImportError:
        raise SystemExit("Please install diffusers: pip install diffusers transformers")

    parser = argparse.ArgumentParser(description="Evaluate VAE-only upper bound (encode->decode) for Teacher/Student")

    parser.add_argument("--data_path", type=str, required=True, help="Path to a JSON dataset describing image/mask pairs or masks")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--if_resizing", action="store_true", help="Resize masks to image_size")
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--mask_key", type=str, default=None, help="JSON key (or comma-separated keys or dotted paths) to locate mask path, e.g. 'mask' or 'meta.mask' or 'gt_path,mask_path'")
    parser.add_argument("--mask_root", type=str, default=None, help="Optional root dir to prepend to mask relative paths")

    parser.add_argument("--original_vae_path", type=str, default=None, help="Path to SD VAE (folder for AutoencoderKL.from_pretrained)")

    parser.add_argument("--student_class", type=str, default=None, help="Python path to LightVAE class, e.g. IMDLBenCo.model_zoo.diffiml.diffiml.SlimVAE")
    parser.add_argument("--student_weights", type=str, default=None, help="Path to student weights (state_dict)")

    parser.add_argument("--student_init", type=str, default=None, help="JSON string of kwargs for student VAE constructor. Required if ctor differs from defaults.")

    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--norm_layer_type", type=str, default="BatchNorm")


    parser.add_argument("--mode", type=str, default="both", choices=["teacher", "student", "both"], help="Which tokenizer(s) to evaluate")

    # DDP
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel evaluation (use with torchrun)")
    parser.add_argument("--dist_backend", type=str, default="nccl")

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = args.ddp and (world_size > 1)

    if use_ddp:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend=args.dist_backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank = 0

    class JsonMaskDataset(Dataset):
        def __init__(self, json_path: str, image_size: int = 512, if_resizing: bool = True, mask_key: Optional[str] = None, mask_root: Optional[str] = None):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept list or dict with key 'data'
            if isinstance(data, dict) and "data" in data:
                self.items = data["data"]
            elif isinstance(data, list):
                self.items = data
            else:
                raise ValueError("Unsupported JSON format. Expect list or dict with key 'data'.")
            self.image_size = image_size
            self.if_resizing = if_resizing
            self.json_dir = os.path.dirname(os.path.abspath(json_path))
            self.mask_key = mask_key
            self.mask_root = mask_root

        def __len__(self):
            return len(self.items)

        def _get_by_dotted_path(self, obj: Any, dotted: str) -> Optional[Any]:
            cur = obj
            for part in dotted.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur

        def _search_mask_recursively(self, obj: Any) -> Optional[str]:
            # breadth-first search keys containing typical substrings
            queue = [obj]
            while queue:
                x = queue.pop(0)
                if isinstance(x, dict):
                    for k, v in x.items():
                        lk = k.lower()
                        if any(t in lk for t in ["mask", "gt", "label", "ann"]) and isinstance(v, (str,)):
                            return v
                        if isinstance(v, (dict, list)):
                            queue.append(v)
                elif isinstance(x, list):
                    queue.extend(x)
            return None

        def _resolve_path(self, p: str) -> str:
            # Prepend mask_root if provided, else resolve relative to json_dir
            if os.path.isabs(p):
                return p
            if self.mask_root:
                return os.path.normpath(os.path.join(self.mask_root, p))
            return os.path.normpath(os.path.join(self.json_dir, p))

        def _find_mask_path(self, item: Any) -> str:
            if isinstance(item, str):
                return self._resolve_path(item)

            if self.mask_key and isinstance(item, dict):
                for key in [k.strip() for k in self.mask_key.split(',') if k.strip()]:
                    val = self._get_by_dotted_path(item, key) if '.' in key else item.get(key, None)
                    if isinstance(val, str) and len(val) > 0:
                        return self._resolve_path(val)

            if isinstance(item, dict):
                for k in ["mask", "gt", "gt_path", "mask_path", "label_path", "ann", "ann_path"]:
                    if k in item and isinstance(item[k], str):
                        return self._resolve_path(item[k])
                if "meta" in item and isinstance(item["meta"], dict):
                    for k in ["mask", "gt", "gt_path", "mask_path"]:
                        if k in item["meta"] and isinstance(item["meta"][k], str):
                            return self._resolve_path(item["meta"][k])

            if isinstance(item, (dict, list)):
                found = self._search_mask_recursively(item)
                if isinstance(found, str):
                    return self._resolve_path(found)

            avail_keys = list(item.keys()) if isinstance(item, dict) else type(item).__name__
            raise KeyError(f"Mask path not found in JSON item. Please set --mask_key. Available top-level keys: {avail_keys}")

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            item = self.items[idx]
            mask_path = self._find_mask_path(item)
            
            m = Image.open(mask_path).convert("L")
            if self.if_resizing:
                m = m.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            m_t = torch.from_numpy(np.array(m)).float() / 255.0  # [0,1]
            m_t = (m_t > 0.5).float().unsqueeze(0)  # (1,H,W)
            # Return (dummy image tensor for compatibility, mask)
            img_dummy = torch.zeros(3, m_t.shape[1], m_t.shape[2], dtype=torch.float32)
            return img_dummy, m_t

    def _extract_mask_paths_from_json_obj(obj: Any, json_dir: str, mask_key: Optional[str], mask_root: Optional[str]) -> list[str]:
        # Reuse the same key-searching logic as JsonMaskDataset
        helper = JsonMaskDataset.__new__(JsonMaskDataset)
        helper.json_dir = json_dir
        helper.mask_key = mask_key
        helper.mask_root = mask_root

        paths: list[str] = []
        def add_item(x: Any):
            try:
                p = helper._find_mask_path(x)
                paths.append(p)
            except Exception:
                pass

        if isinstance(obj, dict) and "data" in obj:
            for it in obj["data"]:
                add_item(it)
        elif isinstance(obj, list):
            for it in obj:
                add_item(it)
        else:
            add_item(obj)
        return sorted(list(dict.fromkeys(paths)))

    def _extract_mask_paths_from_json_file(path: str, mask_key: Optional[str], mask_root: Optional[str]) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        json_dir = os.path.dirname(os.path.abspath(path))
        return _extract_mask_paths_from_json_obj(obj, json_dir, mask_key, mask_root)

    def _collect_mask_paths_from_mani_dir(root: str, globs: list[str]) -> list[str]:
        # Search common mask naming patterns recursively
        out: list[str] = []
        for pat in globs:
            out.extend(glob.glob(os.path.join(root, pat), recursive=True))
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        out = [p for p in out if os.path.splitext(p)[1].lower() in exts]
        return sorted(list(dict.fromkeys(out)))

    def _parse_mask_sources(spec_path: str, mask_key: Optional[str], mask_root: Optional[str], mani_mask_globs: list[str], max_files_per_source: Optional[int] = None) -> list[str]:
        with open(spec_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list) and obj and isinstance(obj[0], list) and len(obj[0]) == 2 and isinstance(obj[0][0], str):
            all_paths: list[str] = []
            for entry in obj:
                dtype, path = entry
                if dtype == "JsonDataset":
                    paths = _extract_mask_paths_from_json_file(path, mask_key, mask_root)
                elif dtype == "ManiDataset":
                    paths = _collect_mask_paths_from_mani_dir(path, mani_mask_globs)
                else:
                    print(f"[WARN] Unknown dataset type '{dtype}', skipping {path}")
                    paths = []
                if max_files_per_source is not None and len(paths) > max_files_per_source:
                    paths = paths[:max_files_per_source]
                all_paths.extend(paths)
            all_paths = sorted(list(dict.fromkeys(all_paths)))
            return all_paths
        else:
            return _extract_mask_paths_from_json_file(spec_path, mask_key, mask_root)

    class PathMaskDataset(Dataset):
        def __init__(self, mask_paths: list[str], image_size: int = 512, if_resizing: bool = True):
            self.mask_paths = mask_paths
            self.image_size = image_size
            self.if_resizing = if_resizing

        def __len__(self):
            return len(self.mask_paths)

        def __getitem__(self, idx: int):
            p = self.mask_paths[idx]
            try:
                m = Image.open(p).convert("L")
            except Exception as e:
                print(f"Warning: Failed to load mask {p}. Returning empty mask. Error: {e}")
                m_t = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
                img_dummy = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
                return img_dummy, m_t

            if self.if_resizing:
                m = m.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            m_t = torch.from_numpy(np.array(m)).float() / 255.0
            m_t = (m_t > 0.5).float().unsqueeze(0)
            img_dummy = torch.zeros(3, m_t.shape[1], m_t.shape[2], dtype=torch.float32)
            return img_dummy, m_t

    mani_globs = [
        "**/*mask*.*",
        "**/*_mask.*",
        "**/*gt*.*",
        "**/*_gt.*",
        "**/GT/*.*",
        "**/gt/*.*",
    ]
    mask_paths = _parse_mask_sources(
        spec_path=args.data_path,
        mask_key=args.mask_key,
        mask_root=args.mask_root,
        mani_mask_globs=mani_globs,
        max_files_per_source=None,
    )
    if not mask_paths:
        raise SystemExit("No mask paths found. Please adjust --mask_key / --mask_root or verify the dataset spec.")
    if rank == 0:
        print(f"[INFO] Collected {len(mask_paths)} mask files from spec: {args.data_path}")

    dataset = PathMaskDataset(mask_paths=mask_paths, image_size=args.image_size, if_resizing=args.if_resizing)
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, shuffle=False if sampler is None else False)

    @dataclass
    class TeacherMaskTokenizer:
        vae: Any 
        scale: float = 0.18215

        @torch.no_grad()
        def encode(self, mask01: torch.Tensor) -> torch.Tensor:
            m = torch.where(mask01 > 0.5, 1.0, -1.0)
            m3 = m.repeat(1, 3, 1, 1)
            h = self.vae.encoder(m3)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            latent = mean * self.scale
            return latent

        @torch.no_grad()
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            z = self.vae.post_quant_conv(latent / self.scale)
            recon = self.vae.decoder(z) 
            return recon.mean(dim=1, keepdim=True) # (B,1,H,W), in [-1, 1]

    @dataclass
    class StudentMaskTokenizer:
        vae: Any

        @torch.no_grad()
        def encode(self, mask01: torch.Tensor) -> torch.Tensor:
            m = torch.where(mask01 > 0.5, 1.0, -1.0)
            latent_scaled = self.vae.encode_mask(m) 
            return latent_scaled

        @torch.no_grad()
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            recon = self.vae.decode_mask(latent) 
            return recon


    def load_teacher_tokenizer(path: str, device: torch.device) -> TeacherMaskTokenizer:
        if path is None:
            raise SystemExit("--original_vae_path is None; please provide a local folder path to SD VAE or set --mode student")
        local_dir = os.path.isdir(path)
        try:
            vae = AutoencoderKL.from_pretrained(path, local_files_only=local_dir)
        except Exception as e:
            raise SystemExit(f"Failed to load teacher VAE from '{path}'. If you are offline, ensure it's a local folder with config.json. Error: {e}")
        vae.to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return TeacherMaskTokenizer(vae=vae)

    def load_class_from_string(qualname: str):
        mod_name, cls_name = qualname.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)

    def load_student_tokenizer(cls_path: str, weights: str, device: torch.device, init_overrides: Optional[Dict[str, Any]] = None, **fallback_kwargs) -> StudentMaskTokenizer:
        VAE_cls = load_class_from_string(cls_path)
        student_model = None
        if init_overrides is not None:
            try:
                student_model = VAE_cls(**init_overrides)
                if rank == 0:
                    print(f"[Student] Initialized {cls_path} with --student_init args.")
            except Exception as e:
                if rank == 0:
                    print(f"[Student] Failed to init {cls_path} with overrides {init_overrides}: {e}. Falling back...")
        
        if student_model is None:
            try:
                student_model = VAE_cls(
                    latent_dim=fallback_kwargs.get("latent_dim", 4),
                    base_channels=fallback_kwargs.get("base_channels", 32),
                    norm_layer_type=fallback_kwargs.get("norm_layer_type", "BatchNorm")
                )
                if rank == 0:
                    print(f"[Student] Initialized {cls_path} with fallback args (LightVAE-style).")
            except Exception as e:
                if rank == 0:
                    print(f"[Student] Fallback ctor (LightVAE-style) failed: {e}. Trying no-arg ctor.")
                try:
                    student_model = VAE_cls()
                    if rank == 0:
                        print(f"[Student] Initialized {cls_path} with no-arg ctor.")
                except Exception as e2:
                    raise SystemExit(f"[Student] Could not construct {cls_path}: {e2}")

        ckpt = torch.load(weights, map_location="cpu")
        state = ckpt
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))

        # try strict=False to be robust against key mismatches
        missing, unexpected = student_model.load_state_dict(state, strict=False)
        if rank == 0:
            if missing:
                print("[Student] Missing keys:", missing)
            if unexpected:
                print("[Student] Unexpected keys:", unexpected)
        
        student_model.to(device)
        student_model.eval()
        for p in student_model.parameters():
            p.requires_grad_(False)
            
        student_vae_backend = student_model
        if hasattr(student_model, "vae") and isinstance(getattr(student_model, "vae"), AutoencoderKL):
            if rank == 0:
                print(f"[Student] Detected {cls_path} is a wrapper, using its .vae attribute as backend.")
            student_vae_backend = student_model.vae
        elif rank == 0:
            print(f"[Student] Using {cls_path} instance directly as backend.")

        return StudentMaskTokenizer(vae=student_vae_backend)

    teacher_tok = None
    student_tok = None
    if args.mode in ("teacher", "both"):
        if not args.original_vae_path:
            raise SystemExit("--original_vae_path is required for mode 'teacher' or 'both'")
        teacher_tok = load_teacher_tokenizer(args.original_vae_path, device=device)
        if rank == 0:
            print(f"Teacher tokenizer loaded from {args.original_vae_path}")

    if args.mode in ("student", "both"):
        if not (args.student_class and args.student_weights):
            raise SystemExit("--student_class and --student_weights are required for mode 'student' or 'both'")
        
        init_overrides = None
        if hasattr(args, "student_init") and args.student_init:
            try:
                init_overrides = json.loads(args.student_init)
            except Exception as e:
                print(f"[WARN] Failed to parse --student_init JSON: {e}")
        
        student_tok = load_student_tokenizer(
            cls_path=args.student_class,
            weights=args.student_weights,
            device=device,
            init_overrides=init_overrides,
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            norm_layer_type=args.norm_layer_type,
        )
        if rank == 0:
            print(f"Student tokenizer loaded from {args.student_weights}")
    if teacher_tok is not None:
        if rank == 0:
            print("Evaluating Teacher VAE...")
        res_t = evaluate_vae_upper_bound(
            dataloader=loader,
            device=device,
            encode_fn=teacher_tok.encode,
            decode_fn=teacher_tok.decode,
            limit_batches=args.limit_batches,
            return_tensors=True,
        )
        if use_ddp:
            import torch.distributed as dist
            for k in ["tp_torch", "fp_torch", "fn_torch"]:
                if res_t.get(k) is not None:
                    dist.all_reduce(res_t[k], op=dist.ReduceOp.SUM)
            tp_sum_t = res_t["tp_torch"]
            fp_sum_t = res_t["fp_torch"]
            fn_sum_t = res_t["fn_torch"]
            f1_t = (2.0 * tp_sum_t) / (2.0 * tp_sum_t + fp_sum_t + fn_sum_t + 1e-8)
            if rank == 0:
                print(f"[Teacher] F1={f1_t.item():.4f} @ thr=0.50")
        else:
            print(f"[Teacher] F1={res_t['f1']:.4f} @ thr={res_t['threshold']:.2f}")

    if student_tok is not None:
        if rank == 0:
            print("Evaluating Student VAE...")
        res_s = evaluate_vae_upper_bound(
            dataloader=loader,
            device=device,
            encode_fn=student_tok.encode,
            decode_fn=student_tok.decode,
            limit_batches=args.limit_batches,
            return_tensors=True, 
        )
        if use_ddp:
            import torch.distributed as dist
            for k in ["tp_torch", "fp_torch", "fn_torch"]:
                if res_s.get(k) is not None:
                    dist.all_reduce(res_s[k], op=dist.ReduceOp.SUM)
            tp_sum_s = res_s["tp_torch"]
            fp_sum_s = res_s["fp_torch"]
            fn_sum_s = res_s["fn_torch"]
            f1_s = (2.0 * tp_sum_s) / (2.0 * tp_sum_s + fp_sum_s + fn_sum_s + 1e-8)
            if rank == 0:
                print(f"[Student] F1={f1_s.item():.4f} @ thr=0.50")
        else:
            print(f"[Student] F1={res_s['f1']:.4f} @ thr={res_s['threshold']:.2f}")

    if teacher_tok is not None and student_tok is not None:
        if rank == 0:
            print("Evaluating Latent Distribution Stats...")
        stats = evaluate_latent_distribution(
            dataloader=loader,
            device=device,
            teacher_encode_fn=teacher_tok.encode,
            student_encode_fn=student_tok.encode,
            num_batches=4,
        )
        if (not use_ddp) or rank == 0:
            print("Latent stats (avg over a few batches):")
            for k, v in stats.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")
                else:
                    print(f"  {k}: {v}")



# Example launch:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=2 \
#   IMDLBenCo/evaluation/eval_vae_upper_bound.py \
#   --ddp \
#   --mode student \
#   --data_path ./runs/balanced_dataset.json \
#   --student_class IMDLBenCo.model_zoo.diffiml.diffiml.LightVAE \
#   --student_weights ./log/train_light_vae/checkpoints/light_vae_weights.pth \
#   --batch_size 16 \
#   --image_size 512 \
#   --if_resizing