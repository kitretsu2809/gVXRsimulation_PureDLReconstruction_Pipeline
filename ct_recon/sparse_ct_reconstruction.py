from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_torch_or_exit():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError:
        raise SystemExit(
            "Missing PyTorch in this environment.\n"
            "Install it with:\n"
            "  ./.venv/bin/pip install torch torchvision"
        )
    return torch, nn, F


@dataclass
class SparseSinogramDatasetMetadata:
    sparse_step: int
    dense_angle_count: int
    sparse_angle_count: int
    detector_count: int
    image_size: int
    downsample_factor: int
    row_start: int
    row_stop: int
    slice_count: int
    sinogram_scale: float
    image_min: float
    image_max: float
    target_volume_path: str
    settings_path: str


def resize_2d_array(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = output_shape
    src_h, src_w = image.shape
    if (src_h, src_w) == (out_h, out_w):
        return image.astype(np.float32, copy=False)

    x_src = np.arange(src_w, dtype=np.float32)
    x_dst = np.linspace(0.0, src_w - 1, out_w, dtype=np.float32)
    row_resized = np.empty((src_h, out_w), dtype=np.float32)
    for row_idx in range(src_h):
        row_resized[row_idx] = np.interp(x_dst, x_src, image[row_idx].astype(np.float32, copy=False))

    y_src = np.arange(src_h, dtype=np.float32)
    y_dst = np.linspace(0.0, src_h - 1, out_h, dtype=np.float32)
    resized = np.empty((out_h, out_w), dtype=np.float32)
    for col_idx in range(out_w):
        resized[:, col_idx] = np.interp(y_dst, y_src, row_resized[:, col_idx])
    return resized


def load_sparse_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, SparseSinogramDatasetMetadata]:
    dataset_path = Path(dataset_path)
    payload = np.load(dataset_path)
    metadata = SparseSinogramDatasetMetadata(**json.loads(payload["metadata_json"].item()))
    return (
        payload["input_sinograms"].astype(np.float32),
        payload["target_sinograms"].astype(np.float32),
        payload["target_images"].astype(np.float32),
        metadata,
    )


def psnr_np(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def save_history(history: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_history.json"
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


class ResidualConvBlock:
    def __new__(cls, channels: int):
        torch, nn, _ = _import_torch_or_exit()
        
        class _ResidualConvBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                )
                self.act = nn.ReLU(inplace=True)
            
            def forward(self, x):
                return self.act(x + self.net(x))
        
        return _ResidualConvBlock(channels)


class DoubleConv:
    def __new__(cls, in_channels: int, out_channels: int):
        torch, nn, _ = _import_torch_or_exit()
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class SinogramToImageDecoder:
    def __new__(cls, image_size: int, base_features: int = 32):
        torch, nn, F = _import_torch_or_exit()

        class _SinogramToImageDecoder(nn.Module):
            def __init__(self, image_size=image_size, base_features=base_features):
                super().__init__()
                self.image_size = int(image_size)
                self.latent_size = max(8, self.image_size // 8)

                self.encoder = nn.Sequential(
                    nn.Conv2d(1, base_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features),
                    nn.Conv2d(base_features, base_features * 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features * 2),
                    nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features * 4),
                    nn.Conv2d(base_features * 4, base_features * 8, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features * 8),
                )

                self.bridge = nn.Sequential(
                    nn.AdaptiveAvgPool2d((self.latent_size, self.latent_size)),
                    nn.Conv2d(base_features * 8, base_features * 8, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features * 4),
                    nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features * 2),
                    nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualConvBlock(base_features),
                    nn.Conv2d(base_features, 1, kernel_size=3, padding=1),
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.bridge(x)
                x = self.decoder(x)
                if x.shape[-2:] != (self.image_size, self.image_size):
                    x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                return torch.sigmoid(x)

        return _SinogramToImageDecoder()


def SparseCTReconstructionModel(
    sparse_angle_count: int,
    dense_angle_count: int,
    detector_count: int,
    image_size: int,
):
    torch, nn, _ = _import_torch_or_exit()

    class _SparseCTReconstructionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.reconstruction_net = SinogramToImageDecoder(image_size=image_size)

        def forward(self, sparse_sinogram):
            return {
                "reconstruction": self.reconstruction_net(sparse_sinogram),
            }

    return _SparseCTReconstructionModel()
