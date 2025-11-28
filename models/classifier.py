"""
Clasificador CNN para gestos de manos.

Utiliza arquitecturas pre-entrenadas (ResNet18, MobileNetV3) con transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import sys

# Intentar importar timm para más opciones de modelos
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

sys.path.append('..')
from config import CLASSIFIER_CONFIG, NUM_CLASSES


class GestureClassifier(nn.Module):
    """
    Clasificador de gestos basado en CNN pre-entrenada.
    
    Soporta múltiples backbones: ResNet18, MobileNetV3, EfficientNet.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 num_classes: int = None,
                 pretrained: bool = None,
                 dropout: float = None,
                 freeze_backbone: bool = False):
        super().__init__()
        
        model_name = model_name or CLASSIFIER_CONFIG["model_name"]
        num_classes = num_classes or CLASSIFIER_CONFIG["num_classes"]
        pretrained = pretrained if pretrained is not None else CLASSIFIER_CONFIG["pretrained"]
        dropout = dropout or CLASSIFIER_CONFIG["dropout"]
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_dim = None
        
        # Crear backbone según el modelo especificado
        if model_name == "resnet18":
            self.backbone, self.feature_dim = self._create_resnet18(pretrained)
        elif model_name == "resnet34":
            self.backbone, self.feature_dim = self._create_resnet34(pretrained)
        elif model_name == "mobilenetv3_large":
            self.backbone, self.feature_dim = self._create_mobilenetv3_large(pretrained)
        elif model_name == "mobilenetv3_small":
            self.backbone, self.feature_dim = self._create_mobilenetv3_small(pretrained)
        elif model_name == "efficientnet_b0" and TIMM_AVAILABLE:
            self.backbone, self.feature_dim = self._create_efficientnet_b0(pretrained)
        else:
            # Default: ResNet18
            print(f"Modelo {model_name} no soportado, usando ResNet18")
            self.backbone, self.feature_dim = self._create_resnet18(pretrained)
        
        # Congelar backbone si se especifica
        if freeze_backbone:
            self._freeze_backbone()
        
        # Clasificador personalizado
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(256, num_classes)
        )
        
        # Inicialización de pesos del clasificador
        self._init_classifier()
    
    def _create_resnet18(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea backbone ResNet18."""
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        
        # Remover capa FC original
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        
        return model, feature_dim
    
    def _create_resnet34(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea backbone ResNet34."""
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        
        return model, feature_dim
    
    def _create_mobilenetv3_large(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea backbone MobileNetV3-Large."""
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        feature_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
        
        return model, feature_dim
    
    def _create_mobilenetv3_small(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea backbone MobileNetV3-Small."""
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        feature_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
        
        return model, feature_dim
    
    def _create_efficientnet_b0(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea backbone EfficientNet-B0 usando timm."""
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        feature_dim = model.num_features
        return model, feature_dim
    
    def _freeze_backbone(self):
        """Congela los parámetros del backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone congelado")
    
    def unfreeze_backbone(self, unfreeze_layers: int = -1):
        """
        Descongela el backbone.
        
        Args:
            unfreeze_layers: Número de capas a descongelar desde el final.
                            -1 para descongelar todo.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        if unfreeze_layers > 0:
            # Congelar las primeras capas
            layers = list(self.backbone.children())
            freeze_until = len(layers) - unfreeze_layers
            for i, layer in enumerate(layers):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        print(f"Backbone descongelado (últimas {unfreeze_layers} capas)" if unfreeze_layers > 0 else "Backbone completamente descongelado")
    
    def _init_classifier(self):
        """Inicializa pesos del clasificador."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
        
        Returns:
            Logits de clasificación (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae features del backbone (útil para la red temporal).
        
        Args:
            x: Tensor de entrada (B, C, H, W)
        
        Returns:
            Features (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza predicción con probabilidades.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
        
        Returns:
            Tuple de (predicciones, probabilidades)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class GestureClassifierWithLandmarks(nn.Module):
    """
    Clasificador que combina CNN features con landmarks de MediaPipe.
    
    Fusiona información visual con información geométrica de la mano.
    """
    
    def __init__(self,
                 model_name: str = None,
                 num_classes: int = None,
                 pretrained: bool = True,
                 landmark_dim: int = 63,  # 21 landmarks * 3 coordenadas
                 dropout: float = 0.3):
        super().__init__()
        
        num_classes = num_classes or NUM_CLASSES
        
        # CNN backbone
        self.cnn = GestureClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        
        # Obtener dimensión de features del CNN
        cnn_feature_dim = self.cnn.feature_dim
        
        # Reemplazar clasificador del CNN
        self.cnn.classifier = nn.Identity()
        
        # Procesador de landmarks
        self.landmark_processor = nn.Sequential(
            nn.Linear(landmark_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Clasificador de fusión
        self.fusion_classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con imagen y landmarks.
        
        Args:
            image: Tensor de imagen (B, C, H, W)
            landmarks: Tensor de landmarks (B, 63)
        
        Returns:
            Logits de clasificación (B, num_classes)
        """
        # Features de CNN
        cnn_features = self.cnn.backbone(image)
        
        # Features de landmarks
        landmark_features = self.landmark_processor(landmarks)
        
        # Fusión
        combined = torch.cat([cnn_features, landmark_features], dim=1)
        logits = self.fusion_classifier(combined)
        
        return logits
    
    def predict(self, image: torch.Tensor, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Realiza predicción con probabilidades."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(image, landmarks)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


def get_classifier(model_name: str = None, pretrained: bool = True, 
                   with_landmarks: bool = False) -> nn.Module:
    """
    Factory function para crear clasificador.
    
    Args:
        model_name: Nombre del backbone
        pretrained: Si usar pesos pre-entrenados
        with_landmarks: Si incluir procesamiento de landmarks
    
    Returns:
        Modelo clasificador
    """
    if with_landmarks:
        return GestureClassifierWithLandmarks(
            model_name=model_name,
            pretrained=pretrained
        )
    else:
        return GestureClassifier(
            model_name=model_name,
            pretrained=pretrained
        )


if __name__ == "__main__":
    # Test de modelos
    print("Testing GestureClassifier...")
    
    # Test ResNet18
    model = GestureClassifier(model_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"ResNet18 - Input: {x.shape}, Output: {y.shape}")
    
    # Test MobileNetV3
    model = GestureClassifier(model_name="mobilenetv3_small", pretrained=False)
    y = model(x)
    print(f"MobileNetV3-Small - Input: {x.shape}, Output: {y.shape}")
    
    # Test extracción de features
    features = model.extract_features(x)
    print(f"Extracted features shape: {features.shape}")
    
    # Test predicción
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}, Probs shape: {probs.shape}")
    
    # Test con landmarks
    print("\nTesting GestureClassifierWithLandmarks...")
    model = GestureClassifierWithLandmarks(pretrained=False)
    landmarks = torch.randn(2, 63)
    y = model(x, landmarks)
    print(f"With landmarks - Input image: {x.shape}, landmarks: {landmarks.shape}, Output: {y.shape}")
    
    # Contar parámetros
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
