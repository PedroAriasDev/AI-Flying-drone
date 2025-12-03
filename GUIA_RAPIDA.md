# 🚀 Guía Rápida de Uso - Control de Dron con Gestos

## ⚡ Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <tu-repositorio>
cd AI-Flying-drone

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install torchsummary  # Para visualización de arquitecturas
```

### 2. Verificar Instalación

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## 📊 Flujo Completo del Proyecto

### Paso 1: Grabar Dataset (Si no tienes datos)

```bash
python main.py --mode record
```

**Controles:**
- `0-9`: Seleccionar clase de gesto
- `ESPACIO`: Iniciar/Pausar grabación
- `S`: Guardar estadísticas
- `Q`: Salir

**Recomendaciones:**
- Grabar 8-10 minutos por gesto
- Variar iluminación, distancia y ángulos
- Total: ~1.5-2 horas

---

### Paso 2: Visualizar Arquitecturas (Opcional)

```bash
# Ver todas las arquitecturas
python visualize_architectures.py

# Ver arquitectura específica
python visualize_architectures.py --model resnet18
python visualize_architectures.py --model temporal
```

**Outputs:**
- Diagramas PNG en `/results/architectures/`
- Resumen en texto: `architectures_summary.txt`

---

### Paso 3: Entrenar y Comparar Clasificadores

```bash
# Entrenamiento completo (100 épocas, ~4-5 horas)
python train_classifier_compare.py

# Modo rápido para pruebas (10 épocas, ~30 min)
python train_classifier_compare.py --quick

# Entrenar solo modelos específicos
python train_classifier_compare.py --models resnet18 mobilenetv3_small
```

**Lo que hace:**
1. Entrena 4 modelos: ResNet18, ResNet34, MobileNetV3-Large, MobileNetV3-Small
2. Registra métricas completas (train/val/test) por época
3. Calcula Overfitting Score y Performance Score
4. Selecciona automáticamente el mejor modelo
5. Genera gráficos comparativos

**Outputs:**
- Checkpoints: `/checkpoints/classifier_<modelo>_best.pt`
- Métricas: `/results/classifier_<modelo>_*_metrics.json`
- Comparación: `/results/model_comparison_results.json`
- Gráficos:
  - `comparison_training_curves.png`
  - `comparison_final_metrics.png`

**Cómo interpretar resultados:**
```json
{
  "best_model": {
    "model": "resnet18",              // Mejor modelo seleccionado
    "test_acc": 0.9650,               // Accuracy en test
    "overfitting_score": 0.0234,      // Diferencia train-val (menor mejor)
    "performance_score": 0.9584       // Score combinado (mayor mejor)
  }
}
```

---

### Paso 4: Entrenar Red Temporal GRU

```bash
# Usar el mejor clasificador del paso anterior
python train_temporal.py --epochs 100 --batch_size 32
```

**Configuración:**
- 30 frames por secuencia (1 segundo @ 30fps)
- GRU unidireccional
- Outputs: Gesto + Intensidad (0-1)

**Outputs:**
- Checkpoint: `/checkpoints/temporal_gru_best.pt`
- Métricas: `/results/temporal_gru_metrics.json`

---

### Paso 5: Probar el Sistema

#### Opción A: Solo Simulador (control por teclado)

```bash
python main.py --mode simulator
```

**Controles:**
- `W/S`: Adelante/Atrás
- `A/D`: Izquierda/Derecha
- `Q/E`: Rotar
- `SPACE/SHIFT`: Subir/Bajar
- `H`: Hover
- `X`: Emergencia
- `R`: Reset
- `ESC`: Salir

#### Opción B: Demo de Inferencia (gestos con webcam)

```bash
python main.py --mode demo
```

Muestra detección de gestos en tiempo real.

#### Opción C: Sistema Integrado (gestos → dron)

```bash
python main.py --mode integrated
```

Controla el dron con gestos de mano capturados por webcam.

---

## 🎮 Gestos Disponibles

| Gesto | Comando | Acción |
|-------|---------|--------|
| ✋ Palma adelante | PITCH_FORWARD | Adelante |
| 🖐️ Palma vertical | PITCH_BACKWARD | Atrás |
| ✌️ V-dedos derecha | ROLL_RIGHT | Derecha |
| ✌️ V-dedos izquierda | ROLL_LEFT | Izquierda |
| 👍 Pulgar arriba | THROTTLE_UP | Subir |
| 👎 Pulgar abajo | THROTTLE_DOWN | Bajar |
| 🤙 Shaka derecha | YAW_RIGHT | Rotar derecha |
| 🤙 Shaka izquierda | YAW_LEFT | Rotar izquierda |
| ✊ Puño cerrado | HOVER | Mantener posición |
| 🖖 Vulcano | EMERGENCY_STOP | Emergencia |

---

## 📈 Monitoreo de Entrenamiento

### Durante el Entrenamiento

Terminal muestra:
```
Época 1/100
----------------------------------------
Training: 100%|████████| 45/45 [00:23<00:00]  loss: 1.2345, acc: 0.8234
Validating: 100%|████████| 12/12 [00:03<00:00]
  Train - Loss: 1.2345, Acc: 0.8234
  Val   - Loss: 0.9876, Acc: 0.8567
  LR: 0.001000
```

### Después del Entrenamiento

Revisar:
1. **Logs**: `/logs/`
2. **Checkpoints**: `/checkpoints/`
3. **Resultados**: `/results/`
4. **Gráficos**: Abrir PNG en `/results/`

---

## 🎨 Nuevo Simulador Mejorado

### Características Visuales

1. **Paisaje**:
   - Cielo celeste
   - Montañas de fondo
   - Suelo de césped verde

2. **Entorno**:
   - 12 árboles distribuidos
   - 3 edificios/torres de control
   - Ejes de coordenadas con flechas

3. **Dron**:
   - Orientación corregida (frente hacia adelante)
   - Marcadores rojos en el frente
   - Hélices animadas

### Cámara

- Sigue al dron automáticamente
- Vista desde atrás y arriba
- Distancia: 15 unidades
- Altura: 8 unidades

---

## ⚙️ Configuración Personalizada

Editar `config.py`:

```python
# Entrenamiento
TRAINING_CONFIG = {
    "cls_epochs": 100,        # Número de épocas
    "cls_batch_size": 256,    # Tamaño de batch (reducir si OOM)
    "cls_lr": 1e-3,           # Learning rate inicial
    "patience": 20,           # Early stopping
}

# Temporal
TEMPORAL_CONFIG = {
    "sequence_length": 30,    # Frames por secuencia
    "hidden_size": 256,       # Tamaño del GRU
    "bidirectional": False,   # Unidireccional
}

# Cámara
CAMERA_CONFIG = {
    "camera_id": 0,           # ID de webcam
    "frame_width": 640,
    "frame_height": 480,
}
```

---

## 🐛 Problemas Comunes

### CUDA out of memory

**Síntoma:** `RuntimeError: CUDA out of memory`

**Solución:**
```python
# En config.py
TRAINING_CONFIG = {
    "cls_batch_size": 128,  # O 64
}
```

### Webcam no funciona

**Síntoma:** `No camera found`

**Solución:**
```python
# En config.py
CAMERA_CONFIG = {
    "camera_id": 1,  # Probar 0, 1, 2...
}
```

### OpenGL no funciona

**Síntoma:** Ventana negra o error de OpenGL

**Solución:**
```bash
python drone_simulator.py --2d
```

### Baja precisión de gestos

**Solución:**
1. Grabar más datos (8-10 min por gesto)
2. Variar condiciones (luz, distancia, ángulo)
3. Aumentar épocas de entrenamiento
4. Ajustar `confidence_threshold` en config.py

---

## 📊 Métricas de Éxito

### Clasificador

✅ **Excelente**: Test Acc > 95%, Overfitting < 0.05
✅ **Bueno**: Test Acc > 90%, Overfitting < 0.10
⚠️ **Mejorable**: Test Acc < 90% o Overfitting > 0.10

### Red Temporal

✅ **Excelente**: Test Acc > 93%
✅ **Bueno**: Test Acc > 88%
⚠️ **Mejorable**: Test Acc < 88%

### Sistema Completo

✅ **Excelente**: FPS ≥ 25, Latencia < 80ms
✅ **Bueno**: FPS ≥ 20, Latencia < 100ms
⚠️ **Mejorable**: FPS < 20 o Latencia > 100ms

---

## 🎯 Tips para Mejores Resultados

### Dataset

1. **Calidad sobre cantidad**: Mejor 5 min de datos variados que 30 min monótonos
2. **Iluminación variada**: Natural, artificial, mixta
3. **Distancias**: 50cm, 1m, 1.5m
4. **Ángulos**: Frontal, ligeramente lateral
5. **Fondos**: Limpios y variados

### Entrenamiento

1. **Monitorear overfitting**: Si train >> val, necesitas más datos o regularización
2. **Learning rate**: Si no converge, reducir LR inicial
3. **Early stopping**: Si se activa muy pronto, aumentar paciencia
4. **Batch size**: Mayor batch = entrenamiento más estable pero más lento

### Inferencia

1. **Iluminación**: Consistente, evitar sombras fuertes
2. **Fondo**: Lo más limpio posible
3. **Distancia**: 0.8-1.2m de la cámara
4. **Mano completa**: Asegurarse que toda la mano sea visible

---

## 📁 Estructura de Archivos Clave

```
AI-Flying-drone/
├── config.py                        # ⚙️ Configuración principal
├── train_classifier_compare.py     # 🆕 Entrenamiento comparativo
├── visualize_architectures.py      # 🆕 Visualización de arquitecturas
├── train_temporal.py                # Entrenamiento GRU
├── main.py                          # Punto de entrada principal
├── drone_simulator.py               # 🆕 Simulador mejorado
├── checkpoints/                     # Modelos entrenados
│   ├── classifier_resnet18_best.pt
│   ├── classifier_mobilenetv3_small_best.pt
│   └── temporal_gru_best.pt
├── results/                         # Gráficos y métricas
│   ├── model_comparison_results.json
│   ├── comparison_training_curves.png
│   └── architectures/
└── data/
    └── dataset/                     # Dataset grabado
```

---

## ⏱️ Tiempo Estimado por Tarea

| Tarea | Tiempo Estimado |
|-------|----------------|
| Grabar dataset completo | 1.5-2 horas |
| Entrenar 4 clasificadores (100 épocas) | 4-5 horas |
| Entrenar red temporal (100 épocas) | 1-2 horas |
| Visualizar arquitecturas | 5 minutos |
| Probar sistema | Variable |
| **Total (sin contar pruebas)** | **~7-9 horas** |

---

## 🚀 Comando Único para Entrenamiento Completo

```bash
# Entrenar clasificadores y visualizar (dejar corriendo overnight)
python train_classifier_compare.py && \
python train_temporal.py --epochs 100 && \
python visualize_architectures.py && \
echo "✅ Entrenamiento completado!"
```

---

## 📞 Siguiente Paso

Después de completar el entrenamiento:

1. ✅ Revisar `model_comparison_results.json`
2. ✅ Analizar gráficos en `/results`
3. ✅ Probar mejor modelo con `python main.py --mode integrated`
4. ✅ Documentar resultados para proyecto final

---

## 🎓 Para el Proyecto Final

Incluir en tu reporte:

1. **Comparación de modelos**: Tabla de `model_comparison_results.json`
2. **Gráficos**: `comparison_training_curves.png`, `comparison_final_metrics.png`
3. **Arquitecturas**: Diagramas de `/results/architectures/`
4. **Métricas**: Accuracy, Overfitting Score, Performance Score
5. **Análisis**: ¿Por qué elegiste el mejor modelo?
6. **Demo**: Video del sistema funcionando

---

**¡Buena suerte con tu proyecto final! 🚁🎮**
