import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class SinogramUNet(nn.Module):
    """
    Stage 1: Sensor Domain Rectification
    Cleans up noise in the raw sinogram.
    """
    def __init__(self, channels=32):
        super().__init__()
        self.inc = DoubleConv(1, channels)
        self.down = DoubleConv(channels, channels * 2)
        self.up = DoubleConv(channels * 2, channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down(F.max_pool2d(x1, 2))
        x_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        # Handle padding if dimensions don't perfectly match
        diffY = x1.size()[2] - x_up.size()[2]
        diffX = x1.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x3 = self.up(x_up + x1)
        return self.outc(x3) + x

class ConvolutionalDomainTransform(nn.Module):
    """
    Stage 2: The Domain Transformer (Physics Replacement)
    Maps a 2D Sinogram (Angles x Detectors) into a 2D Image slice (W x H)
    using a Fully Convolutional approach to maintain resolution independence.
    """
    def __init__(self, target_image_size=256, channels=64):
        super().__init__()
        self.target_image_size = target_image_size
        
        # We use a bottleneck architecture with dilated convolutions to force a massive 
        # receptive field, allowing the network to learn the global "unscrambling" of the Radon transform.
        self.initial_conv = DoubleConv(1, channels)
        
        self.encoder1 = DoubleConv(channels, channels * 2)
        self.encoder2 = DoubleConv(channels * 2, channels * 4)
        
        # The bottleneck acts as the global transform mechanism
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        
        self.decoder2 = DoubleConv(channels * 4, channels * 2)
        self.decoder1 = DoubleConv(channels * 2, channels)
        
        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, sinogram):
        # 1. Spatially interpolate the sinogram tensor to match the target image grid.
        # This is resolution-independent.
        x = F.interpolate(sinogram, size=(self.target_image_size, self.target_image_size), mode='bilinear', align_corners=True)
        
        # 2. Extract features
        x1 = self.initial_conv(x)
        x2 = self.encoder1(F.max_pool2d(x1, 2))
        x3 = self.encoder2(F.max_pool2d(x2, 2))
        
        # 3. Global Receptive Transform
        x_b = self.bottleneck(x3)
        
        # 4. Reconstruct spatial image
        x_up2 = F.interpolate(x_b, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = self.decoder2(x_up2 + x2)
        
        x_up1 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x5 = self.decoder1(x_up1 + x1)
        
        return self.final_conv(x5)

class ImageUNet(nn.Module):
    """
    Stage 3: Image Domain Enhancement
    Refines the output of the Domain Transform to sharpen industrial edges.
    """
    def __init__(self, channels=32):
        super().__init__()
        self.inc = DoubleConv(1, channels)
        self.down = DoubleConv(channels, channels * 2)
        self.up = DoubleConv(channels * 2, channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down(F.max_pool2d(x1, 2))
        x_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        
        diffY = x1.size()[2] - x_up.size()[2]
        diffX = x1.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x3 = self.up(x_up + x1)
        return self.outc(x3) + x

class PureDLPipeline(nn.Module):
    """
    3-Stage Pure Deep Learning Pipeline mapping Sinograms to 2D CT Slices.
    """
    def __init__(self, target_image_size=256):
        super().__init__()
        self.target_image_size = target_image_size
        
        self.stage1_sinogram_net = SinogramUNet()
        self.stage2_domain_transform = ConvolutionalDomainTransform(target_image_size=target_image_size)
        self.stage3_image_net = ImageUNet()

    def forward(self, noisy_sinogram):
        # Stage 1: Rectify
        clean_sinogram = self.stage1_sinogram_net(noisy_sinogram)
        
        # Stage 2: Domain Transform
        rough_image = self.stage2_domain_transform(clean_sinogram)
        
        # Stage 3: Enhance
        final_image = self.stage3_image_net(rough_image)
        
        return final_image, clean_sinogram, rough_image

if __name__ == "__main__":
    # Dry-run test
    print("Testing PureDLPipeline...")
    model = PureDLPipeline(target_image_size=256)
    
    # Batch=2, Channels=1, Angles=360, Detectors=512
    dummy_sino = torch.randn(2, 1, 360, 512)
    final_img, clean_sino, rough_img = model(dummy_sino)
    
    print(f"Input Sino: {dummy_sino.shape}")
    print(f"Clean Sino: {clean_sino.shape}")
    print(f"Rough Img:  {rough_img.shape}")
    print(f"Final Img:  {final_img.shape}")
    print("Test passed! Shapes align correctly.")
