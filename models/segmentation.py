"""
Modelo de Segmentación UNet para detección de manos.

Arquitectura UNet clásica con encoder pre-entrenado (MobileNetV2/ResNet).
Requiere: segmentation_models_pytorch (pip install segmentation-models-pytorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys

# Intentar importar segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("AVISO: segmentation_models_pytorch no instalado. Usando UNet custom.")

sys.path.append('..')
from config import SEGMENTATION_CONFIG


class DoubleConv(nn.Module):
    """Bloque de doble convolución para UNet."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
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


class Down(nn.Module):
    """Bloque de downsampling: maxpool + doble conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bloque de upsampling: upsample + concat + doble conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ajustar tamaños si es necesario
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Capa de salida."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Implementación de UNet para segmentación de manos.
    
    Arquitectura clásica con skip connections.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True,
                 base_features: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        
        self.outc = OutConv(base_features, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder con skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción con softmax aplicado."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class SegmentationModel(nn.Module):
    """
    Wrapper que puede usar SMP (si está disponible) o UNet custom.
    
    Proporciona interfaz unificada para diferentes backends.
    """
    
    def __init__(self, 
                 encoder_name: str = None,
                 encoder_weights: str = None,
                 in_channels: int = None,
                 classes: int = None,
                 use_smp: bool = True):
        super().__init__()
        
        encoder_name = encoder_name or SEGMENTATION_CONFIG["encoder_name"]
        encoder_weights = encoder_weights or SEGMENTATION_CONFIG["encoder_weights"]
        in_channels = in_channels or SEGMENTATION_CONFIG["in_channels"]
        classes = classes or SEGMENTATION_CONFIG["classes"]
        
        if use_smp and SMP_AVAILABLE:
            print(f"Usando SMP UNet con encoder {encoder_name}")
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None  # Aplicamos softmax en predict
            )
            self.using_smp = True
        else:
            print("Usando UNet custom")
            self.model = UNet(n_channels=in_channels, n_classes=classes)
            self.using_smp = False
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Realiza predicción y devuelve máscara binaria.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
            threshold: Umbral para binarización
        
        Returns:
            Máscara binaria (B, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            # Clase 1 es la mano
            mask = (probs[:, 1, :, :] > threshold).float()
            return mask
    
    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae features del encoder (útil para el clasificador)."""
        if self.using_smp:
            features = self.model.encoder(x)
            return features[-1]  # Último nivel de features
        else:
            # Para UNet custom, pasar por encoder
            x1 = self.model.inc(x)
            x2 = self.model.down1(x1)
            x3 = self.model.down2(x2)
            x4 = self.model.down3(x3)
            x5 = self.model.down4(x4)
            return x5


def get_segmentation_model(pretrained: bool = True) -> SegmentationModel:
    """Factory function para crear modelo de segmentación."""
    return SegmentationModel(
        encoder_weights="imagenet" if pretrained else None
    )


if __name__ == "__main__":
    # Test del modelo
    print("Testing UNet...")
    
    # Test UNet custom
    model = UNet(n_channels=3, n_classes=2)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"UNet custom - Input: {x.shape}, Output: {y.shape}")
    
    # Test SegmentationModel
    seg_model = SegmentationModel()
    y = seg_model(x)
    print(f"SegmentationModel - Input: {x.shape}, Output: {y.shape}")
    
    # Test predict
    mask = seg_model.predict(x)
    print(f"Predicted mask shape: {mask.shape}")
    
    # Contar parámetros
    params = sum(p.numel() for p in seg_model.parameters())
    print(f"Total parameters: {params:,}")
