"""
Red Temporal (GRU/LSTM) para análisis de secuencias de gestos.

Procesa secuencias de frames para:
- Suavizar predicciones
- Detectar velocidad/intensidad del gesto
- Mejorar robustez temporal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List
import sys

sys.path.append('..')
from config import TEMPORAL_CONFIG, NUM_CLASSES, CLASSIFIER_CONFIG


class TemporalGRU(nn.Module):
    """
    Red GRU para análisis temporal de gestos.
    
    Recibe secuencias de features (CNN + landmarks) y produce:
    - Clasificación del gesto
    - Intensidad/velocidad del movimiento
    """
    
    def __init__(self,
                 input_size: int = None,
                 hidden_size: int = None,
                 num_layers: int = None,
                 num_classes: int = None,
                 dropout: float = None,
                 bidirectional: bool = None):
        super().__init__()
        
        # Configuración
        cnn_features = TEMPORAL_CONFIG["cnn_features"]
        landmark_features = TEMPORAL_CONFIG["landmark_features"]
        input_size = input_size or (cnn_features + landmark_features)
        hidden_size = hidden_size or TEMPORAL_CONFIG["hidden_size"]
        num_layers = num_layers or TEMPORAL_CONFIG["num_layers"]
        num_classes = num_classes or NUM_CLASSES
        dropout = dropout if dropout is not None else TEMPORAL_CONFIG["dropout"]
        bidirectional = bidirectional if bidirectional is not None else TEMPORAL_CONFIG["bidirectional"]
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Proyección de entrada (opcional, para ajustar dimensiones)
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Capas de salida
        gru_output_size = hidden_size * self.num_directions
        
        # Clasificador de gestos
        self.gesture_classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Predictor de intensidad (0-1)
        self.intensity_predictor = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism (opcional)
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, 
                return_intensity: bool = True,
                use_attention: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Secuencia de features (B, seq_len, input_size)
            return_intensity: Si devolver predicción de intensidad
            use_attention: Si usar mecanismo de atención
        
        Returns:
            Tuple de (logits, intensity) o solo logits
        """
        batch_size = x.size(0)
        
        # Proyección de entrada
        x = self.input_projection(x)
        
        # GRU
        gru_out, hidden = self.gru(x)
        # gru_out: (B, seq_len, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, B, hidden_size)
        
        if use_attention:
            # Attention sobre todos los timesteps
            attention_weights = self.attention(gru_out)  # (B, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * gru_out, dim=1)  # (B, hidden_size * num_directions)
        else:
            # Usar último hidden state
            if self.bidirectional:
                # Concatenar últimos estados de ambas direcciones
                hidden_fwd = hidden[-2]  # (B, hidden_size)
                hidden_bwd = hidden[-1]  # (B, hidden_size)
                context = torch.cat([hidden_fwd, hidden_bwd], dim=1)
            else:
                context = hidden[-1]  # (B, hidden_size)
        
        # Clasificación
        logits = self.gesture_classifier(context)
        
        if return_intensity:
            intensity = self.intensity_predictor(context)
            return logits, intensity.squeeze(-1)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Realiza predicción completa.
        
        Returns:
            Tuple de (predicciones, probabilidades, intensidades)
        """
        self.eval()
        with torch.no_grad():
            logits, intensity = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs, intensity


class GestureSequenceModel(nn.Module):
    """
    Modelo completo que combina CNN + GRU para secuencias de gestos.
    
    Procesa secuencias de frames directamente (sin pre-extraer features).
    """
    
    def __init__(self,
                 cnn_backbone: str = None,
                 pretrained: bool = True,
                 hidden_size: int = None,
                 num_layers: int = None,
                 num_classes: int = None,
                 dropout: float = None,
                 freeze_cnn: bool = True):
        super().__init__()
        
        cnn_backbone = cnn_backbone or CLASSIFIER_CONFIG["model_name"]
        hidden_size = hidden_size or TEMPORAL_CONFIG["hidden_size"]
        num_layers = num_layers or TEMPORAL_CONFIG["num_layers"]
        num_classes = num_classes or NUM_CLASSES
        dropout = dropout if dropout is not None else TEMPORAL_CONFIG["dropout"]
        
        # CNN Backbone para extracción de features
        self.cnn, self.cnn_feature_dim = self._create_cnn(cnn_backbone, pretrained)
        
        if freeze_cnn:
            self._freeze_cnn()
        
        # Proyección de landmarks
        self.landmark_dim = TEMPORAL_CONFIG["landmark_features"]
        self.landmark_proj = nn.Linear(self.landmark_dim, 64)
        
        # Dimensión total de entrada al GRU
        total_input_dim = self.cnn_feature_dim + 64
        
        # GRU temporal
        self.temporal = TemporalGRU(
            input_size=total_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def _create_cnn(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crea el backbone CNN."""
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone == "mobilenetv3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        elif backbone == "mobilenetv3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        else:
            # Default: MobileNetV3-Small (más rápido)
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        
        return model, feature_dim
    
    def _freeze_cnn(self):
        """Congela el CNN backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = False
    
    def unfreeze_cnn(self):
        """Descongela el CNN backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = True
    
    def forward(self, frames: torch.Tensor, 
                landmarks: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Secuencia de frames (B, seq_len, C, H, W)
            landmarks: Secuencia de landmarks (B, seq_len, 63), opcional
        
        Returns:
            Tuple de (logits, intensity)
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Extraer features CNN para cada frame
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(frames_flat)  # (B*seq_len, cnn_feature_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # (B, seq_len, cnn_feature_dim)
        
        # Procesar landmarks si están disponibles
        if landmarks is not None:
            landmark_features = self.landmark_proj(landmarks)  # (B, seq_len, 64)
            combined_features = torch.cat([cnn_features, landmark_features], dim=2)
        else:
            # Usar zeros si no hay landmarks
            zeros = torch.zeros(batch_size, seq_len, 64, device=frames.device)
            combined_features = torch.cat([cnn_features, zeros], dim=2)
        
        # Pasar por GRU temporal
        logits, intensity = self.temporal(combined_features)
        
        return logits, intensity
    
    def predict(self, frames: torch.Tensor, 
                landmarks: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Realiza predicción completa.
        
        Returns:
            Tuple de (predicciones, probabilidades, intensidades)
        """
        self.eval()
        with torch.no_grad():
            logits, intensity = self.forward(frames, landmarks)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs, intensity
    
    def extract_temporal_features(self, frames: torch.Tensor,
                                   landmarks: torch.Tensor = None) -> torch.Tensor:
        """
        Extrae features temporales (útil para debugging/visualización).
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(frames_flat)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        if landmarks is not None:
            landmark_features = self.landmark_proj(landmarks)
            combined_features = torch.cat([cnn_features, landmark_features], dim=2)
        else:
            zeros = torch.zeros(batch_size, seq_len, 64, device=frames.device)
            combined_features = torch.cat([cnn_features, zeros], dim=2)
        
        return combined_features


class OnlineTemporalProcessor:
    """
    Procesador temporal para inferencia online (tiempo real).
    
    Mantiene un buffer de frames y produce predicciones suavizadas.
    """
    
    def __init__(self, model: GestureSequenceModel, 
                 sequence_length: int = None,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sequence_length = sequence_length or TEMPORAL_CONFIG["sequence_length"]
        
        # Buffers
        self.frame_buffer: List[torch.Tensor] = []
        self.landmark_buffer: List[torch.Tensor] = []
        
        # Historial de predicciones para suavizado
        self.prediction_history: List[int] = []
        self.confidence_history: List[float] = []
    
    def reset(self):
        """Reinicia los buffers."""
        self.frame_buffer.clear()
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
    
    def add_frame(self, frame: torch.Tensor, landmarks: torch.Tensor = None):
        """
        Añade un frame al buffer.
        
        Args:
            frame: Frame preprocesado (C, H, W)
            landmarks: Landmarks (63,), opcional
        """
        self.frame_buffer.append(frame)
        
        if landmarks is not None:
            self.landmark_buffer.append(landmarks)
        else:
            self.landmark_buffer.append(torch.zeros(63))
        
        # Mantener tamaño del buffer
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
            self.landmark_buffer.pop(0)
    
    def predict(self) -> Tuple[int, float, float]:
        """
        Realiza predicción con el buffer actual.
        
        Returns:
            Tuple de (clase_predicha, confianza, intensidad)
        """
        if len(self.frame_buffer) < self.sequence_length:
            # Buffer no lleno, usar padding
            padding_needed = self.sequence_length - len(self.frame_buffer)
            frames = [self.frame_buffer[0]] * padding_needed + self.frame_buffer
            landmarks = [self.landmark_buffer[0]] * padding_needed + self.landmark_buffer
        else:
            frames = self.frame_buffer
            landmarks = self.landmark_buffer
        
        # Preparar tensores
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        landmarks_tensor = torch.stack(landmarks).unsqueeze(0).to(self.device)
        
        # Predicción
        with torch.no_grad():
            preds, probs, intensity = self.model.predict(frames_tensor, landmarks_tensor)
        
        pred_class = preds[0].item()
        confidence = probs[0, pred_class].item()
        intensity_val = intensity[0].item()
        
        # Actualizar historial
        self.prediction_history.append(pred_class)
        self.confidence_history.append(confidence)
        
        # Mantener historial limitado
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
        
        return pred_class, confidence, intensity_val
    
    def get_smoothed_prediction(self, window: int = 5) -> Tuple[int, float]:
        """
        Obtiene predicción suavizada por votación.
        
        Args:
            window: Tamaño de ventana para votación
        
        Returns:
            Tuple de (clase_más_votada, confianza_promedio)
        """
        if not self.prediction_history:
            return -1, 0.0
        
        recent = self.prediction_history[-window:]
        recent_conf = self.confidence_history[-window:]
        
        # Votación por mayoría
        from collections import Counter
        votes = Counter(recent)
        most_common = votes.most_common(1)[0]
        
        # Confianza promedio para la clase ganadora
        indices = [i for i, p in enumerate(recent) if p == most_common[0]]
        avg_conf = sum(recent_conf[i] for i in indices) / len(indices)
        
        return most_common[0], avg_conf


def get_temporal_model(pretrained: bool = True, freeze_cnn: bool = True) -> GestureSequenceModel:
    """Factory function para crear modelo temporal."""
    return GestureSequenceModel(
        pretrained=pretrained,
        freeze_cnn=freeze_cnn
    )


if __name__ == "__main__":
    # Test de modelos
    print("Testing TemporalGRU...")
    
    # Test GRU solo
    gru = TemporalGRU(input_size=575, hidden_size=256, num_layers=2)
    x = torch.randn(2, 15, 575)  # (batch, seq_len, features)
    logits, intensity = gru(x)
    print(f"GRU - Input: {x.shape}, Logits: {logits.shape}, Intensity: {intensity.shape}")
    
    # Test modelo completo
    print("\nTesting GestureSequenceModel...")
    model = GestureSequenceModel(pretrained=False, freeze_cnn=True)
    frames = torch.randn(2, 15, 3, 224, 224)  # (batch, seq_len, C, H, W)
    landmarks = torch.randn(2, 15, 63)  # (batch, seq_len, landmarks)
    
    logits, intensity = model(frames, landmarks)
    print(f"Full model - Frames: {frames.shape}, Logits: {logits.shape}, Intensity: {intensity.shape}")
    
    # Test predicción
    preds, probs, intensities = model.predict(frames, landmarks)
    print(f"Predictions: {preds}, Intensities: {intensities}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test procesador online
    print("\nTesting OnlineTemporalProcessor...")
    processor = OnlineTemporalProcessor(model, device='cpu')
    
    for i in range(20):
        frame = torch.randn(3, 224, 224)
        landmarks = torch.randn(63)
        processor.add_frame(frame, landmarks)
        
        if i >= 5:
            pred, conf, intensity = processor.predict()
            smoothed_pred, smoothed_conf = processor.get_smoothed_prediction()
            print(f"Frame {i}: pred={pred}, conf={conf:.3f}, smoothed={smoothed_pred}")
