"""
CNN Architecture and Feature Extraction (PatchDentalNet: VGG19 + ResNet152 + PatchAttention)
======================================================================================
The VGG19Branch / ResNet152Branch / PatchAttention / PatchDentalNet classes in
this file were VERIFIED against the state_dict keys of the trained .pth files
(the 989 layer names inside seed_42/patchdentalnet.pth match the classes
defined here one-to-one - vgg.features.*, vgg.head.*, resnet.stem/layer1-4/head.*,
attn.attention.*, classifier.*).

Preprocessing (CLAHE + Unsharp Masking): written to match EXACTLY the
clahe_usm() / extract_features_tta() functions from the original
SonKez1002.py (v7) reference script (CLAHE clip=2.0, tile=8x8,
USM sigma=1.0, strength=1.5, downscale to 1024px width after CLAHE+USM,
then patch extraction - same order as preprocess_and_cache()).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# Config (the subset of the training Config class needed for inference)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    CROP_ALPHA: float = 0.20
    CROP_BETA: float = 0.80
    CROP_GAMMA: float = 0.50
    CROP_DELTA: float = 0.80

    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
    USM_SIGMA: float = 1.0
    USM_STRENGTH: float = 1.5
    CACHE_MAX_WIDTH: int = 1024

    IMG_SIZE: Tuple[int, int] = (224, 224)
    PATCH_ROWS: int = 2
    PATCH_COLS: int = 8  # 2x8 = 16 local patches + 1 global = 17

    FEATURE_DIM: int = 512
    VGG_FEATURE_DIM: int = 256
    RESNET_FEATURE_DIM: int = 256
    NUM_CLASSES: int = 4
    DROPOUT: float = 0.50
    PATCH_CHUNK: int = 8


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing: CLAHE + Unsharp Masking
# ──────────────────────────────────────────────────────────────────────────────
def clahe_usm(img_bgr: np.ndarray, cfg: InferenceConfig) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP_LIMIT, tileGridSize=cfg.CLAHE_TILE_GRID)
    l_clahe = clahe.apply(l_ch)

    blurred = cv2.GaussianBlur(l_clahe, ksize=(0, 0), sigmaX=cfg.USM_SIGMA)
    l_sharp = cv2.addWeighted(l_clahe, 1 + cfg.USM_STRENGTH, blurred, -cfg.USM_STRENGTH, 0)

    merged = cv2.merge([l_sharp, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def auto_crop(img_bgr: np.ndarray, cfg: InferenceConfig) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1 = int(cfg.CROP_ALPHA * w)
    x2 = int(cfg.CROP_BETA * w)
    y1 = int(cfg.CROP_GAMMA * h)
    y2 = int(cfg.CROP_DELTA * h)
    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return img_bgr
    return cropped


def extract_patches(img_bgr: np.ndarray, cfg: InferenceConfig) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    n_rows, n_cols = cfg.PATCH_ROWS, cfg.PATCH_COLS
    ph, pw = h // n_rows, w // n_cols
    patches = [cv2.resize(img_bgr, cfg.IMG_SIZE)]  # index 0: global context

    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * ph
            y1 = (r + 1) * ph if r < n_rows - 1 else h
            x0 = c * pw
            x1 = (c + 1) * pw if c < n_cols - 1 else w
            patch = img_bgr[y0:y1, x0:x1]
            if patch.size == 0:
                patch = np.zeros((*cfg.IMG_SIZE[::-1], 3), dtype=np.uint8)
            patches.append(cv2.resize(patch, cfg.IMG_SIZE))
    return patches  # length = n_rows*n_cols + 1 (default: 17)


_eval_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def image_to_patch_tensor(pil_image, cfg: InferenceConfig) -> torch.Tensor:
    """
    PIL.Image (RGB) -> patch tensor [1, N_patches, 3, 224, 224].

    NOTE: auto_crop is INTENTIONALLY not called. Images in the training
    dataset had already been pre-cropped to the bimaxillary ROI by an expert
    (1002.py docstring: "Auto-cropping is disabled in this pipeline") and
    DentalDataset never applied any cropping. The image uploaded to the DSS
    is likewise expected to already be a cropped panoramic radiograph
    (focused on the upper+lower jaw region) - not a full skull/neck image.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bgr = clahe_usm(bgr, cfg)

    # Same as preprocess_and_cache(): downscale if width > 1024 after CLAHE+USM
    h, w = bgr.shape[:2]
    if w > cfg.CACHE_MAX_WIDTH:
        scale = cfg.CACHE_MAX_WIDTH / w
        bgr = cv2.resize(bgr, (cfg.CACHE_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

    patches_bgr = extract_patches(bgr, cfg)
    tensors = []
    for patch in patches_bgr:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensors.append(_eval_transform(patch_rgb))

    patches_tensor = torch.stack(tensors)  # [N, 3, 224, 224]
    return patches_tensor.unsqueeze(0)  # [1, N, 3, 224, 224]


# ──────────────────────────────────────────────────────────────────────────────
# Model architecture (identical to PatchDentalNet in 1002.py - .pth keys verified)
# ──────────────────────────────────────────────────────────────────────────────
class PatchAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = (weights * x).sum(dim=1)
        return context, weights.squeeze(-1)


class VGG19Branch(nn.Module):
    def __init__(self, output_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        vgg = models.vgg19(weights=None)
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class ResNet152Branch(nn.Module):
    def __init__(self, output_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        resnet = models.resnet152(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class PatchDentalNet(nn.Module):
    def __init__(self, cfg: InferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.vgg = VGG19Branch(cfg.VGG_FEATURE_DIM, cfg.DROPOUT)
        self.resnet = ResNet152Branch(cfg.RESNET_FEATURE_DIM, cfg.DROPOUT)
        self.attn = PatchAttention(cfg.FEATURE_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.FEATURE_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, cfg.NUM_CLASSES),
        )

    def forward(self, patches: torch.Tensor):
        B, N, C, H, W = patches.shape
        x = patches.view(B * N, C, H, W)

        chunk = self.cfg.PATCH_CHUNK
        f_vgg_list, f_res_list = [], []
        for i in range(0, B * N, chunk):
            x_chunk = x[i:i + chunk]
            f_vgg_list.append(self.vgg(x_chunk))
            f_res_list.append(self.resnet(x_chunk))

        f_vgg = torch.cat(f_vgg_list, dim=0)
        f_resnet = torch.cat(f_res_list, dim=0)
        f_fused = torch.cat([f_vgg, f_resnet], dim=1)     # [B*N, 512]
        f_fused = f_fused.view(B, N, -1)                  # [B, N, 512]

        context, weights = self.attn(f_fused)             # [B,512], [B,N]
        logits = self.classifier(context)                 # [B,4]
        return logits, weights, f_fused


# ──────────────────────────────────────────────────────────────────────────────
# 1557-dim feature extraction: attn(mean) 512 + std 512 + max 512 + attn_wts(17) + softmax(4)
# Same composition as X_trainval.npy / X_test.npy (features/ folder) -
# verified that the last 4 columns are softmax probabilities (sum=1.0).
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_feature_vector(model: PatchDentalNet, patches_tensor: torch.Tensor, device) -> np.ndarray:
    model.eval()
    patches_tensor = patches_tensor.to(device)
    logits, weights, f_fused = model(patches_tensor)

    probs = torch.softmax(logits, dim=1)          # [1, 4]
    attn_mean = (weights.unsqueeze(-1) * f_fused).sum(dim=1)  # [1, 512] == context
    patch_std = f_fused.std(dim=1)                 # [1, 512]
    patch_max = f_fused.max(dim=1).values           # [1, 512]

    feature = torch.cat([attn_mean, patch_std, patch_max, weights, probs], dim=1)  # [1, 1557]
    return feature.squeeze(0).cpu().numpy()
