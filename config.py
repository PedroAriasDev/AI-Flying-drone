"""
Configuración global del proyecto de control de dron con gestos.
"""

import os
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Crear directorios si no existen
for dir_path in [DATA_DIR, DATASET_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASES DE GESTOS
# =============================================================================
GESTURE_CLASSES = {
    0: "PITCH_FORWARD",      # Palma abierta hacia adelante
    1: "PITCH_BACKWARD",     # Palma abierta vertical (stop)
    2: "ROLL_RIGHT",         # Índice + Medio hacia derecha
    3: "ROLL_LEFT",          # Índice + Medio hacia izquierda
    4: "THROTTLE_UP",        # Pulgar arriba
    5: "THROTTLE_DOWN",      # Pulgar abajo
    6: "YAW_RIGHT",          # Shaka hacia derecha
    7: "YAW_LEFT",           # Shaka hacia izquierda
    8: "HOVER",              # Puño cerrado
    9: "EMERGENCY_STOP",     # Vulcano salute
    10: "NO_GESTURE"         # Sin gesto (background)
}

NUM_CLASSES = len(GESTURE_CLASSES)
CLASS_TO_IDX = {v: k for k, v in GESTURE_CLASSES.items()}

# Comandos de dron asociados a cada gesto
GESTURE_TO_COMMAND = {
    "PITCH_FORWARD": {"pitch": 1.0},
    "PITCH_BACKWARD": {"pitch": -1.0},
    "ROLL_RIGHT": {"roll": 1.0},
    "ROLL_LEFT": {"roll": -1.0},
    "THROTTLE_UP": {"throttle": 1.0},
    "THROTTLE_DOWN": {"throttle": -1.0},
    "YAW_RIGHT": {"yaw": 1.0},
    "YAW_LEFT": {"yaw": -1.0},
    "HOVER": {"hover": True},
    "EMERGENCY_STOP": {"emergency": True},
    "NO_GESTURE": {}
}

# =============================================================================
# CONFIGURACIÓN DE CÁMARA Y CAPTURA
# =============================================================================
CAMERA_CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
}

# =============================================================================
# CONFIGURACIÓN DE MEDIAPIPE
# =============================================================================
MEDIAPIPE_CONFIG = {
    "model_complexity": 0,  # 0=lite (más rápido), 1=full
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
}

# =============================================================================
# CONFIGURACIÓN DE MODELOS
# =============================================================================

# NOTA: Red de Segmentación UNet YA NO SE UTILIZA en este proyecto
# MediaPipe se encarga de la detección y segmentación de manos
# Se mantiene la configuración comentada solo por compatibilidad con código legacy
# SEGMENTATION_CONFIG = {
#     "encoder_name": "mobilenet_v2",
#     "encoder_weights": "imagenet",
#     "in_channels": 3,
#     "classes": 2,  # background, hand
#     "input_size": (256, 256),
# }

# Clasificador CNN (ResNet18 o MobileNetV3)
CLASSIFIER_CONFIG = {
    "model_name": "resnet18",  # opciones: resnet18, mobilenetv3_large, mobilenetv3_small
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "input_size": (224, 224),
    "dropout": 0.3,
}

# Red Temporal (GRU)
TEMPORAL_CONFIG = {
    "sequence_length": 30,  # frames en la ventana temporal (30 para análisis de velocidad)
    "cnn_features": 512,    # features del CNN
    "landmark_features": 63, # 21 landmarks * 3 coordenadas
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,  # False = unidireccional para baja latencia
}

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================
TRAINING_CONFIG = {
    # General
    "device": "cuda",  # cuda o cpu
    "seed": 42,

    # NOTA: Segmentación eliminada - MediaPipe maneja la detección de manos

    # Clasificador - Optimizado para PC local
    "cls_epochs": 100,
    "cls_batch_size": 256,  # Batch grande para PC local
    "cls_lr": 1e-3,  # Learning rate inicial (será variable con scheduler)
    "cls_lr_min": 1e-6,  # LR mínimo para scheduler
    "cls_weight_decay": 1e-5,

    # Temporal - Optimizado para secuencias de 30 frames
    "temp_epochs": 100,
    "temp_batch_size": 32,
    "temp_lr": 1e-3,
    "temp_lr_min": 1e-6,
    "temp_weight_decay": 1e-5,

    # Data augmentation
    "augmentation": True,

    # Early stopping (más paciencia para 100 épocas)
    "patience": 20,

    # Splits
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,

    # Modelos a comparar para clasificador
    "classifier_models": ["resnet18", "resnet34", "mobilenetv3_large", "mobilenetv3_small"],
}

# =============================================================================
# CONFIGURACIÓN DE INFERENCIA
# =============================================================================
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "smoothing_window": 5,  # frames para suavizado
    "gesture_hold_frames": 3,  # frames mínimos para confirmar gesto
}

# =============================================================================
# CONFIGURACIÓN DEL SIMULADOR 3D
# =============================================================================
SIMULATOR_CONFIG = {
    "window_width": 1280,
    "window_height": 720,
    "fov": 60,
    "near_plane": 0.1,
    "far_plane": 1000.0,
    "drone_speed": 5.0,
    "drone_rotation_speed": 90.0,  # grados por segundo
    "gravity": 9.81,
}
