# 🚁 Drone Gesture Control System

Sistema de control de dron mediante gestos de manos utilizando visión por computadora y redes neuronales profundas.

**Proyecto Final - Inteligencia Artificial**

---

## 📋 Descripción

Este proyecto implementa un sistema completo para controlar un dron virtual (simulador 3D) usando gestos de manos capturados por webcam. El sistema utiliza:

- **MediaPipe Hands**: Detección de manos y extracción de landmarks (21 puntos)
- **Red de Segmentación (UNet)**: Segmenta la mano del fondo
- **Red Clasificadora (CNN)**: Clasifica el gesto entre 11 clases
- **Red Temporal (GRU)**: Analiza secuencias para suavizado y detección de intensidad

---

## 🎮 Gestos Soportados

| Gesto | Comando | Descripción |
|-------|---------|-------------|
| ✋ Palma adelante | PITCH_FORWARD | Mover dron hacia adelante |
| 🖐️ Palma vertical | PITCH_BACKWARD | Mover dron hacia atrás |
| ✌️ V-dedos derecha | ROLL_RIGHT | Mover lateralmente a la derecha |
| ✌️ V-dedos izquierda | ROLL_LEFT | Mover lateralmente a la izquierda |
| 👍 Pulgar arriba | THROTTLE_UP | Subir altitud |
| 👎 Pulgar abajo | THROTTLE_DOWN | Bajar altitud |
| 🤙 Shaka derecha | YAW_RIGHT | Rotar en sentido horario |
| 🤙 Shaka izquierda | YAW_LEFT | Rotar en sentido antihorario |
| ✊ Puño cerrado | HOVER | Mantener posición |
| 🖖 Vulcano | EMERGENCY_STOP | Parada de emergencia |

---

## 📁 Estructura del Proyecto

```
drone_gesture_control/
├── config.py                 # Configuración global
├── main.py                   # Script principal
├── dataset_recorder.py       # Grabador de dataset
├── inference.py              # Sistema de inferencia en tiempo real
├── drone_simulator.py        # Simulador 3D del dron
├── datasets.py               # Clases de Dataset PyTorch
├── training_utils.py         # Utilidades de entrenamiento
├── train_classifier.py       # Entrenamiento de CNN
├── train_segmentation.py     # Entrenamiento de UNet
├── train_temporal.py         # Entrenamiento de GRU
├── train_colab.ipynb         # Notebook para Google Colab
├── requirements.txt          # Dependencias
├── models/
│   ├── __init__.py
│   ├── classifier.py         # Modelo clasificador CNN
│   ├── segmentation.py       # Modelo UNet
│   └── temporal.py           # Modelo GRU temporal
├── data/
│   └── dataset/              # Dataset de gestos
├── checkpoints/              # Modelos entrenados
├── results/                  # Resultados y gráficos
└── logs/                     # Logs de entrenamiento
```

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd drone_gesture_control
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
# Para GPU NVIDIA (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Resto de dependencias
pip install -r requirements.txt
```

---

## 📊 Flujo de Trabajo

### Paso 1: Grabar Dataset

```bash
python main.py --mode record
```

**Controles del grabador:**
- `0-9`: Seleccionar clase de gesto
- `ESPACIO`: Iniciar/Pausar grabación
- `S`: Guardar estadísticas
- `T`: Guardar secuencia temporal
- `Q`: Salir

**Recomendaciones de grabación:**
- 8-10 minutos por gesto
- Variar iluminación (natural, artificial)
- Variar distancia a la cámara (50cm, 1m, 1.5m)
- Variar ángulos de la mano
- Total: ~1.5-2 horas de grabación

### Paso 2: Entrenar Modelos

#### Opción A: Local (GPU NVIDIA)
```bash
# Entrenar clasificador CNN
python train_classifier.py --epochs 30 --batch_size 32

# Entrenar red de segmentación
python train_segmentation.py --epochs 50

# Entrenar red temporal
python train_temporal.py --epochs 50
```

#### Opción B: Google Colab
1. Subir dataset a Google Drive
2. Abrir `train_colab.ipynb` en Colab
3. Ejecutar todas las celdas
4. Descargar checkpoints

### Paso 3: Ejecutar Sistema

```bash
# Solo demo de inferencia
python main.py --mode demo

# Solo simulador (control por teclado)
python main.py --mode simulator

# Sistema integrado completo
python main.py --mode integrated
```

---

## 🎯 Modos de Ejecución

### Demo Mode
```bash
python main.py --mode demo
```
Muestra la detección de gestos en tiempo real con la webcam. Útil para probar el sistema de inferencia.

### Simulator Mode
```bash
python main.py --mode simulator
```
Ejecuta el simulador 3D del dron con control por teclado:
- `W/S`: Pitch (adelante/atrás)
- `A/D`: Roll (izquierda/derecha)
- `Q/E`: Yaw (rotación)
- `ESPACIO/SHIFT`: Throttle (subir/bajar)
- `H`: Hover
- `X`: Emergencia
- `R`: Reset
- `ESC`: Salir

### Integrated Mode
```bash
python main.py --mode integrated
```
Sistema completo: la webcam captura gestos que controlan el dron en el simulador 3D.

---

## 📈 Arquitectura de Redes

### Red Clasificadora (CNN)
- **Backbone**: ResNet18 pre-entrenado en ImageNet
- **Entrada**: Imágenes 224x224
- **Salida**: 11 clases de gestos
- **Transfer Learning**: Fine-tuning de todas las capas

### Red de Segmentación (UNet)
- **Encoder**: MobileNetV2 pre-entrenado
- **Entrada**: Imágenes 256x256
- **Salida**: Máscara binaria (mano/fondo)
- **Loss**: Dice + BCE combinado

### Red Temporal (GRU)
- **Entrada**: Secuencia de 15 frames (CNN features + landmarks)
- **Hidden Size**: 256
- **Layers**: 2
- **Salidas**: 
  - Clasificación de gesto
  - Intensidad del movimiento (0-1)

---

## 📊 Métricas Objetivo

| Modelo | Métrica | Objetivo |
|--------|---------|----------|
| Clasificador CNN | Accuracy | >95% |
| Segmentación UNet | IoU | >90% |
| Red Temporal GRU | Accuracy | >90% |
| Sistema completo | Latencia | <100ms |
| Sistema completo | FPS | ≥20 |

---

## 🛠️ Configuración

Editar `config.py` para ajustar:

```python
# Configuración de cámara
CAMERA_CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "device": "cuda",
    "cls_epochs": 30,
    "cls_batch_size": 32,
    "cls_lr": 1e-4,
    # ...
}

# Configuración de inferencia
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "smoothing_window": 5,
    "gesture_hold_frames": 3,
}
```

---

## 🔧 Solución de Problemas

### Error: CUDA not available
```bash
# Verificar instalación de CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: No camera found
```python
# Cambiar ID de cámara en config.py
CAMERA_CONFIG["camera_id"] = 1  # Probar 0, 1, 2...
```

### Error: OpenGL not working
```bash
# Usar modo 2D alternativo
python drone_simulator.py --2d
```

### Baja precisión de gestos
- Aumentar cantidad de datos de entrenamiento
- Variar condiciones de grabación
- Ajustar `confidence_threshold` en config.py

---

## 📚 Referencias

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [HaGRID Dataset](https://github.com/hukenovs/hagrid)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 👤 Autor

**Pedro** - Proyecto Final de Inteligencia Artificial

---

## 📄 Licencia

Este proyecto es para uso académico.
