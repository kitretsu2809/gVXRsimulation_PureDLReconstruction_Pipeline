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

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution to save parameters."""
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpSep(nn.Module):
    """Upscaling then depthwise separable double conv"""
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==============================================================================
# NOVEL ATTENTION MECHANISMS FOR PURE DL RADON INVERSION
# ==============================================================================

class CrossAttentionBridge(nn.Module):
    """
    Learns the inverse Radon transform by computing attention between the 
    spatial image grid (Queries) and the sinogram measurements (Keys/Values).
    Uses PyTorch FlashAttention / Scaled Dot-Product Attention for O(1) memory.
    """
    def __init__(self, sino_channels, img_channels, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Conv2d(img_channels, embed_dim, 1)
        self.key_proj = nn.Conv2d(sino_channels, embed_dim, 1)
        self.value_proj = nn.Conv2d(sino_channels, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, img_channels, 1)
        
        # Heavy pooling on sinogram to reduce sequence length and fit in VRAM
        self.sino_pool = nn.AdaptiveAvgPool2d((30, 32))
        
    def forward(self, sino_features, img_queries):
        # Pool sinogram: [B, C_s, 30, 32] -> 960 tokens
        sino_pooled = self.sino_pool(sino_features)
        
        B, C_i, H, W = img_queries.shape
        _, C_s, A_p, D_p = sino_pooled.shape
        
        # Projections reshaped for FlashAttention: [B, heads, seq_len, head_dim]
        Q = self.query_proj(img_queries).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        K = self.key_proj(sino_pooled).view(B, self.num_heads, self.head_dim, A_p * D_p).transpose(-2, -1)
        V = self.value_proj(sino_pooled).view(B, self.num_heads, self.head_dim, A_p * D_p).transpose(-2, -1)
        
        # Memory-efficient Scaled Dot-Product Attention (FlashAttention)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(-2, -1).contiguous().view(B, self.embed_dim, H, W)
        
        return img_queries + self.out_proj(out)

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) block for global channel-wise feature re-weighting."""
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

# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

class SinogramUNet(nn.Module):
    """Stage 1: Sinogram Denoising"""
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
        return self.outc(x) + identity


class DomainTransformNet(nn.Module):
    """
    Stage 2: Pure DL Radon Inversion (Novel Cross-Attention Architecture)
    Maps sinogram space -> image space.
    """
    def __init__(self, target_size=256):
        super().__init__()
        self.target_size = target_size
        
        # Sinogram feature extractor
        self.sino_encoder = nn.Sequential(
            DoubleConv(1, 32),
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )
        
        # Image Grid Encoder: 64 -> 128 -> 256 -> 512
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck blocks
        self.bottleneck_dilated = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Cross-Attention bridges domains; SE block refines channel dependencies
        self.cross_attn = CrossAttentionBridge(sino_channels=64, img_channels=512, embed_dim=256)
        self.se_block = ChannelAttention(512)
        
        # Decoder: 512 -> 256 -> 128 -> 64
        self.up1 = UpSep(512 + 256, 256, bilinear=True)
        self.up2 = UpSep(256 + 128, 128, bilinear=True)
        self.up3 = UpSep(128 + 64, 64, bilinear=True)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, sino):
        # 1. Extract structural features from sinogram
        sino_feats = self.sino_encoder(sino)
        
        # 2. Provide a structural prior via bilinear interpolation
        img_prior = F.interpolate(sino, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
        # 3. Image Grid Encoder
        x1 = self.inc(img_prior)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 4. Bottleneck with Cross-Attention + Squeeze-and-Excitation
        b_out = self.bottleneck_dilated(x4)
        b_out = self.cross_attn(sino_feats, b_out) # Gather sinogram information!
        b_out = self.se_block(b_out)                # Squeeze-and-Excitation channel gating
        
        # 5. Image Grid Decoder
        x = self.up1(b_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return self.outc(x)


class ImageUNet(nn.Module):
    """Stage 3: Image Refinement"""
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
        return self.outc(x) + identity


class PureDLPipeline(nn.Module):
    """
    Complete Pure Deep Learning CT Reconstruction Pipeline.
    Novelty: Replaces non-differentiable FBP with Cross-Attention mapping.
    """
    def __init__(self, target_image_size=256):
        super().__init__()
        self.stage1 = SinogramUNet()
        self.stage2 = DomainTransformNet(target_size=target_image_size)
        self.stage3 = ImageUNet()

    def forward(self, noisy_sinogram):
        clean_sinogram = self.stage1(noisy_sinogram)
        rough_image = self.stage2(clean_sinogram)
        final_image = self.stage3(rough_image)
        return final_image, clean_sinogram, rough_image


if __name__ == '__main__':
    # VRAM and Architecture Sanity Check
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    target_size = 256
    model = PureDLPipeline(target_image_size=target_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Simulating RTX 3050 constraints: batch=2
    B, Angles, Detectors = 2, 360, 512
    dummy = torch.randn(B, 1, Angles, Detectors).to(device)
    
    print(f"\nRunning test pass (Batch={B})...")
    model.eval()
    with torch.no_grad():
        start = time.time()
        final_img, clean_sino, rough_img = model(dummy)
        end = time.time()
            
    print(f"Output shapes:")
    print(f" - Clean Sino: {clean_sino.shape}")
    print(f" - Rough Img:  {rough_img.shape}")
    print(f" - Final Img:  {final_img.shape}")
    print(f"Time: {(end - start) * 1000:.1f} ms")
    
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM used for inference: {peak:.1f} MB")
