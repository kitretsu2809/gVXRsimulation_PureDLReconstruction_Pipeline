import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle mismatch in dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SinogramUNet(nn.Module):
    """
    Stage 1: Sinogram Denoising
    3-level U-Net with residual skip connection.
    Levels: 32 -> 64 -> 128 -> 64 -> 32
    """
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128 + 64, 64, bilinear=True)
        self.up2 = Up(64 + 32, 32, bilinear=True)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits + identity # Residual connection

class UpSep(nn.Module):
    """Upscaling then depthwise separable double conv, tailored for DomainTransformNet"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                DepthwiseSeparableConv(in_channels, in_channels // 2),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(in_channels // 2, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if dimension mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DomainTransformNet(nn.Module):
    """
    Stage 2: Pure DL Radon Inversion
    Maps sinogram space -> image space.
    """
    def __init__(self, target_size=256):
        super().__init__()
        self.target_size = target_size
        
        # Encoder: 64 -> 128 -> 256 -> 512
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck: Dilated convolutions and Channel Attention
        self.bottleneck_dilated = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.se_block = ChannelAttention(512)
        
        # Decoder: 512 -> 256 -> 128 -> 64
        self.up1 = UpSep(512 + 256, 256, bilinear=True)
        self.up2 = UpSep(256 + 128, 128, bilinear=True)
        self.up3 = UpSep(128 + 64, 64, bilinear=True)
        
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 1. Spatially interpolate sinogram to (target_size, target_size)
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
        # 2. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 3. Bottleneck
        b_out = self.bottleneck_dilated(x4)
        b_out = self.se_block(b_out)
        
        # 4. Decoder
        x = self.up1(b_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return self.outc(x)

class ImageUNet(nn.Module):
    """
    Stage 3: Image Refinement
    3-level U-Net with residual skip connection.
    Levels: 48 -> 96 -> 192 -> 96 -> 48
    """
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.up1 = Up(192 + 96, 96, bilinear=True)
        self.up2 = Up(96 + 48, 48, bilinear=True)
        self.outc = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits + identity # Residual connection

class PureDLPipeline(nn.Module):
    """
    Complete Pure Deep Learning CT Reconstruction Pipeline.
    Contains three stages:
    1. SinogramUNet: Denoises the sinogram.
    2. DomainTransformNet: Pure DL Radon inversion mapping sinogram to image.
    3. ImageUNet: Refines the reconstructed CT image.
    """
    def __init__(self, target_image_size=256):
        super().__init__()
        self.stage1 = SinogramUNet()
        self.stage2 = DomainTransformNet(target_size=target_image_size)
        self.stage3 = ImageUNet()

    def forward(self, noisy_sinogram):
        """
        Forward pass for the pipeline.
        
        Args:
            noisy_sinogram: Tensor of shape [B, 1, Angles, Detectors]
            
        Returns:
            final_image: Refined CT image.
            clean_sinogram: Denoised sinogram from Stage 1.
            rough_image: Initial reconstruction from Stage 2.
        """
        clean_sinogram = self.stage1(noisy_sinogram)
        rough_image = self.stage2(clean_sinogram)
        final_image = self.stage3(rough_image)
        return final_image, clean_sinogram, rough_image


if __name__ == '__main__':
    # Test block to instantiate model and run dummy pass
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Instantiate the model
    target_size = 256
    model = PureDLPipeline(target_image_size=target_size).to(device)
    
    # Calculate total parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Dummy data
    B = 2
    Angles = 360
    Detectors = 720
    dummy_noisy_sinogram = torch.randn(B, 1, Angles, Detectors).to(device)
    
    print(f"\nRunning dummy forward pass...")
    print(f"Input sinogram shape: {dummy_noisy_sinogram.shape}")
    
    model.eval() # Use eval mode for standard testing
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            start_time = time.time()
            final_img, clean_sino, rough_img = model(dummy_noisy_sinogram)
            end_time = time.time()
            
    print(f"\nOutputs:")
    print(f"Clean Sinogram shape: {clean_sino.shape}")
    print(f"Rough Image shape: {rough_img.shape}")
    print(f"Final Image shape: {final_img.shape}")
    print(f"Forward pass time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Validate output shapes
    assert clean_sino.shape == dummy_noisy_sinogram.shape, "Clean sinogram shape mismatch"
    assert rough_img.shape == (B, 1, target_size, target_size), "Rough image shape mismatch"
    assert final_img.shape == (B, 1, target_size, target_size), "Final image shape mismatch"
    
    print("\nAll shape checks passed successfully!")
