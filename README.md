# 🚁 Drone Gesture Control System

Sistema de control de dron mediante gestos de manos utilizando visión por computadora y redes neuronales profundas.

**Proyecto Final - Inteligencia Artificial**

---

## 📋 Descripción

Este proyecto implementa un sistema completo para controlar un dron virtual (simulador 3D) usando gestos de manos capturados por webcam. El sistema utiliza:

- **MediaPipe Hands**: Detección de manos, segmentación automática y extracción de landmarks (21 puntos)
- **Red Clasificadora (CNN)**: Clasifica el gesto entre 11 clases. Se comparan 4 arquitecturas: ResNet18, ResNet34, MobileNetV3-Large y MobileNetV3-Small
- **Red Temporal (GRU)**: Analiza secuencias de 30 frames para suavizado y detección de intensidad del gesto

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
AI-Flying-drone/
├── config.py                      # Configuración global
├── main.py                        # Script principal
├── dataset_recorder.py            # Grabador de dataset
├── inference.py                   # Sistema de inferencia en tiempo real
├── drone_simulator.py             # Simulador 3D mejorado con paisaje
├── datasets.py                    # Clases de Dataset PyTorch
├── training_utils.py              # Utilidades de entrenamiento
├── train_classifier.py            # Entrenamiento de CNN individual
├── train_classifier_compare.py   # 🆕 Entrenamiento comparativo de 4 modelos
├── train_temporal.py              # Entrenamiento de GRU (30 frames)
├── visualize_architectures.py    # 🆕 Visualización de arquitecturas
├── CAMBIOS_Y_MEJORAS.md           # 🆕 Documentación de cambios
├── GUIA_RAPIDA.md                 # 🆕 Guía de uso rápida
├── requirements.txt               # Dependencias
├── models/
│   ├── __init__.py
│   ├── classifier.py              # 4 modelos CNN comparables
│   ├── temporal.py                # Modelo GRU temporal (30 frames)
│   └── segmentation.py            # (NO USADO - solo legacy)
├── data/
│   └── dataset/                   # Dataset de gestos
├── checkpoints/                   # Modelos entrenados
├── results/                       # Resultados y gráficos comparativos
└── logs/                          # Logs de entrenamiento
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

### Paso 2: Entrenar Modelos (PC Local - Optimizado)

```bash
# 1. Entrenar y comparar 4 clasificadores CNN automáticamente (100 épocas)
python train_classifier_compare.py

# 2. Entrenar red temporal GRU (30 frames, 100 épocas)
python train_temporal.py --epochs 100 --batch_size 32

# 3. (Opcional) Visualizar arquitecturas de las redes
python visualize_architectures.py
```

**Características del nuevo sistema:**
- ✅ Entrenamiento optimizado para GPU local (no Colab)
- ✅ 100 épocas con batch size 256
- ✅ Comparación automática de 4 modelos
- ✅ Selección del mejor modelo basada en métricas
- ✅ Gráficos y análisis generados automáticamente

**Tiempo estimado** (GPU RTX 3060/3070):
- Clasificadores (4 modelos): ~4-5 horas
- Red temporal: ~1-2 horas
- **Total: ~6-7 horas**

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

### Red Clasificadora (CNN) - 4 Modelos Comparados

Se entrenan y comparan automáticamente 4 arquitecturas:

1. **ResNet18**
   - Parámetros: ~11M
   - Rápido y eficiente
   - Baseline sólido

2. **ResNet34**
   - Parámetros: ~21M
   - Más profundo, mejor capacidad
   - Mayor accuracy potencial

3. **MobileNetV3-Large**
   - Parámetros: ~5.5M
   - Optimizado para móviles
   - Buen balance velocidad/accuracy

4. **MobileNetV3-Small**
   - Parámetros: ~2.5M
   - Muy ligero y rápido
   - Ideal para inferencia en tiempo real

**Configuración común:**
- Entrada: 224x224x3
- Salida: 11 clases de gestos
- Transfer Learning: Pre-entrenado en ImageNet
- Fine-tuning: Todas las capas

### Red Temporal (GRU Unidireccional)

- **Secuencia**: 30 frames (1 segundo @ 30fps)
- **Arquitectura**: Unidireccional (baja latencia)
- **Input**: CNN features (512D) + MediaPipe landmarks (63D)
- **Hidden Size**: 256
- **Layers**: 2
- **Outputs**:
  - Clasificación de gesto (11 clases)
  - Intensidad del movimiento (0-1) - detecta gestos bruscos vs suaves
- **Attention**: Mecanismo de atención sobre secuencia temporal

---

## 📊 Métricas Objetivo

| Modelo | Métrica | Objetivo Actualizado |
|--------|---------|---------------------|
| Clasificador CNN | Test Accuracy | >97% |
| Clasificador CNN | Overfitting Score | <0.05 |
| Red Temporal GRU | Test Accuracy | >93% |
| Sistema completo | Latencia | <100ms |
| Sistema completo | FPS | ≥20 |

**Nuevas métricas implementadas:**
- **Overfitting Score**: Diferencia entre train y val accuracy (menor es mejor)
- **Performance Score**: Combina val_acc (40%) + test_acc (60%) para selección del mejor modelo

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
