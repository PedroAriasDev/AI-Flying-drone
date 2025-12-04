"""
Sistema de Inferencia en Tiempo Real para control de dron con gestos.

Este módulo integra todos los modelos (segmentación, clasificación, temporal)
para producir comandos de control del dron en tiempo real.

Uso:
    python inference.py --mode demo
    python inference.py --mode production
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp
from collections import deque
from typing import Dict, Tuple, Optional, List
import time
from pathlib import Path
from dataclasses import dataclass

from config import (
    GESTURE_CLASSES, GESTURE_TO_COMMAND, CAMERA_CONFIG, 
    MEDIAPIPE_CONFIG, INFERENCE_CONFIG, CLASSIFIER_CONFIG,
    TEMPORAL_CONFIG, CHECKPOINTS_DIR
)
from models.classifier import GestureClassifier
from models.temporal import GestureSequenceModel, OnlineTemporalProcessor
from models.segmentation import SegmentationModel


@dataclass
class DroneCommand:
    """Comando de control para el dron."""
    gesture: str
    gesture_id: int
    confidence: float
    intensity: float
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    hover: bool = False
    emergency: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'gesture': self.gesture,
            'gesture_id': self.gesture_id,
            'confidence': self.confidence,
            'intensity': self.intensity,
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
            'throttle': self.throttle,
            'hover': self.hover,
            'emergency': self.emergency
        }


class GestureInferenceSystem:
    """
    Sistema completo de inferencia para control de dron.
    
    Integra:
    - MediaPipe para detección de manos
    - CNN para clasificación de gestos
    - GRU para análisis temporal
    - Lógica de suavizado y control
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 use_temporal: bool = True,
                 use_segmentation: bool = False,
                 classifier_path: Path = None,
                 temporal_path: Path = None,
                 segmentation_path: Path = None,
                 debug: bool = False):
        """
        Inicializa el sistema de inferencia.

        Args:
            device: Dispositivo para inferencia (cuda/cpu)
            use_temporal: Si usar modelo temporal para suavizado
            use_segmentation: Si usar modelo de segmentación
            classifier_path: Ruta al checkpoint del clasificador
            temporal_path: Ruta al checkpoint del modelo temporal
            segmentation_path: Ruta al checkpoint del modelo de segmentación
            debug: Si activar modo debug (muestra outputs de la red)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_temporal = use_temporal
        self.use_segmentation = use_segmentation
        self.debug = debug

        print(f"Inicializando sistema de inferencia en {self.device}...")
        if debug:
            print("🔍 MODO DEBUG ACTIVADO - Se mostrarán outputs de las redes")

        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)

        # Transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(CLASSIFIER_CONFIG["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Cargar modelos
        self._load_models(classifier_path, temporal_path, segmentation_path)

        # Buffers para procesamiento temporal
        self.frame_buffer = deque(maxlen=TEMPORAL_CONFIG["sequence_length"])
        self.landmark_buffer = deque(maxlen=TEMPORAL_CONFIG["sequence_length"])
        self.prediction_buffer = deque(maxlen=INFERENCE_CONFIG["smoothing_window"])
        self.confidence_buffer = deque(maxlen=INFERENCE_CONFIG["smoothing_window"])

        # Estado actual
        self.current_gesture = "NO_GESTURE"
        self.current_confidence = 0.0
        self.gesture_hold_counter = 0

        # Métricas de rendimiento
        self.fps_counter = deque(maxlen=30)
        self.last_inference_time = 0

        # Debug info
        self.debug_info = {
            'logits': None,
            'probs': None,
            'top5_classes': [],
            'top5_probs': [],
            'temporal_output': None,
            'hand_detected': False
        }

        print("Sistema de inferencia inicializado correctamente.")
    
    def _load_models(self, classifier_path, temporal_path, segmentation_path):
        """Carga los modelos desde checkpoints."""
        
        # Clasificador CNN
        self.classifier = GestureClassifier(
            model_name=CLASSIFIER_CONFIG["model_name"],
            pretrained=False
        )
        
        classifier_path = classifier_path or CHECKPOINTS_DIR / f"classifier_{CLASSIFIER_CONFIG['model_name']}_best.pt"
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"Clasificador cargado desde: {classifier_path}")
        else:
            print(f"AVISO: No se encontró checkpoint del clasificador en {classifier_path}")
            print("Usando modelo sin entrenar (solo para pruebas)")
        
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Modelo temporal
        if self.use_temporal:
            self.temporal_model = GestureSequenceModel(
                pretrained=False,
                freeze_cnn=True
            )
            
            temporal_path = temporal_path or CHECKPOINTS_DIR / "temporal_gru_best.pt"
            if temporal_path.exists():
                checkpoint = torch.load(temporal_path, map_location=self.device, weights_only=False)
                self.temporal_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Modelo temporal cargado desde: {temporal_path}")
            else:
                print(f"AVISO: No se encontró checkpoint temporal en {temporal_path}")
            
            self.temporal_model = self.temporal_model.to(self.device)
            self.temporal_model.eval()
            
            # Procesador online
            self.temporal_processor = OnlineTemporalProcessor(
                self.temporal_model, device=self.device
            )
        
        # Modelo de segmentación
        if self.use_segmentation:
            self.segmentation = SegmentationModel()
            
            segmentation_path = segmentation_path or CHECKPOINTS_DIR / "segmentation_unet_best.pt"
            if segmentation_path.exists():
                checkpoint = torch.load(segmentation_path, map_location=self.device, weights_only=False)
                self.segmentation.load_state_dict(checkpoint['model_state_dict'])
                print(f"Modelo de segmentación cargado desde: {segmentation_path}")
            
            self.segmentation = self.segmentation.to(self.device)
            self.segmentation.eval()
    
    def extract_hand_roi(self, frame: np.ndarray, landmarks) -> Tuple[np.ndarray, Tuple]:
        """Extrae la región de interés de la mano."""
        h, w = frame.shape[:2]
        
        # Obtener bounding box
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        padding = 0.2
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        
        x_min = max(0, int(x_min - pad_x))
        x_max = min(w, int(x_max + pad_x))
        y_min = max(0, int(y_min - pad_y))
        y_max = min(h, int(y_max + pad_y))
        
        roi = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
        
        return roi, bbox
    
    def landmarks_to_tensor(self, landmarks) -> torch.Tensor:
        """Convierte landmarks a tensor."""
        arr = []
        for lm in landmarks.landmark:
            arr.extend([lm.x, lm.y, lm.z])
        return torch.tensor(arr, dtype=torch.float32)
    
    def classify_gesture(self, roi: np.ndarray) -> Tuple[int, float]:
        """Clasifica el gesto usando el CNN."""
        if roi.size == 0:
            return 10, 0.0  # NO_GESTURE

        # Preprocesar
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            logits = self.classifier(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

            # Guardar info de debug
            if self.debug:
                self.debug_info['logits'] = logits[0].cpu().numpy()
                self.debug_info['probs'] = probs[0].cpu().numpy()

                # Top 5 predicciones
                top5_probs, top5_indices = torch.topk(probs[0], min(5, len(probs[0])))
                self.debug_info['top5_classes'] = [(GESTURE_CLASSES.get(idx.item(), f"Class_{idx.item()}"),
                                                    prob.item())
                                                   for idx, prob in zip(top5_indices, top5_probs)]

        return pred_class.item(), confidence.item()
    
    def process_temporal(self, frame_tensor: torch.Tensor, 
                         landmarks: torch.Tensor) -> Tuple[int, float, float]:
        """Procesa con el modelo temporal."""
        self.temporal_processor.add_frame(frame_tensor, landmarks)
        pred, conf, intensity = self.temporal_processor.predict()
        return pred, conf, intensity
    
    def smooth_prediction(self, gesture_id: int, confidence: float) -> Tuple[int, float]:
        """Aplica suavizado a las predicciones."""
        self.prediction_buffer.append(gesture_id)
        self.confidence_buffer.append(confidence)
        
        if len(self.prediction_buffer) < 3:
            return gesture_id, confidence
        
        # Votación por mayoría
        from collections import Counter
        votes = Counter(self.prediction_buffer)
        most_common = votes.most_common(1)[0]
        
        # Confianza promedio para el gesto ganador
        indices = [i for i, p in enumerate(self.prediction_buffer) if p == most_common[0]]
        avg_conf = sum(list(self.confidence_buffer)[i] for i in indices) / len(indices)
        
        return most_common[0], avg_conf
    
    def gesture_to_command(self, gesture_id: int, confidence: float, 
                           intensity: float = 1.0) -> DroneCommand:
        """Convierte un gesto a comando de dron."""
        gesture_name = GESTURE_CLASSES.get(gesture_id, "NO_GESTURE")
        command_map = GESTURE_TO_COMMAND.get(gesture_name, {})
        
        cmd = DroneCommand(
            gesture=gesture_name,
            gesture_id=gesture_id,
            confidence=confidence,
            intensity=intensity
        )
        
        # Aplicar intensidad a los comandos
        if "pitch" in command_map:
            cmd.pitch = command_map["pitch"] * intensity
        if "roll" in command_map:
            cmd.roll = command_map["roll"] * intensity
        if "yaw" in command_map:
            cmd.yaw = command_map["yaw"] * intensity
        if "throttle" in command_map:
            cmd.throttle = command_map["throttle"] * intensity
        if "hover" in command_map:
            cmd.hover = command_map["hover"]
        if "emergency" in command_map:
            cmd.emergency = command_map["emergency"]
        
        return cmd
    
    def process_frame(self, frame: np.ndarray) -> Tuple[DroneCommand, np.ndarray]:
        """
        Procesa un frame y devuelve el comando de dron.
        
        Args:
            frame: Frame BGR de OpenCV
        
        Returns:
            Tuple de (DroneCommand, frame_anotado)
        """
        start_time = time.time()
        
        # Flip para efecto espejo
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos con MediaPipe
        results = self.hands.process(frame_rgb)

        gesture_id = 10  # NO_GESTURE por defecto
        confidence = 0.0
        intensity = 1.0

        # Actualizar debug info
        if self.debug:
            self.debug_info['hand_detected'] = results.multi_hand_landmarks is not None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extraer ROI
                roi, bbox = self.extract_hand_roi(frame, hand_landmarks)
                
                if roi.size > 0:
                    # Clasificación CNN
                    gesture_id, confidence = self.classify_gesture(roi)
                    
                    # Procesar con modelo temporal si está activo
                    if self.use_temporal:
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        frame_tensor = self.transform(roi_rgb)
                        landmarks_tensor = self.landmarks_to_tensor(hand_landmarks)
                        
                        gesture_id, confidence, intensity = self.process_temporal(
                            frame_tensor, landmarks_tensor
                        )
                    
                    # Suavizado
                    gesture_id, confidence = self.smooth_prediction(gesture_id, confidence)
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                 (0, 255, 0), 2)
                
                break  # Solo procesar primera mano
        
        # Filtrar por umbral de confianza (reducido para ser menos restrictivo)
        if confidence < INFERENCE_CONFIG["confidence_threshold"]:
            gesture_id = 10  # NO_GESTURE
            confidence = 0.0

        # SIMPLIFICADO: Tomar directamente el gesto más probable sin tanto suavizado
        # Solo aplicar hold si el usuario tiene configuración conservadora
        if gesture_id == self.current_gesture:
            self.gesture_hold_counter += 1
        else:
            self.gesture_hold_counter = 0
            # Cambiar inmediatamente al nuevo gesto si supera umbral mínimo
            # Reducimos de gesture_hold_frames a solo 1 frame para respuesta más rápida
            if confidence > INFERENCE_CONFIG["confidence_threshold"]:
                self.current_gesture = gesture_id
                self.current_confidence = confidence

        # Si el hold counter es suficiente, actualizar
        if self.gesture_hold_counter >= 1:  # Reducido de gesture_hold_frames a 1
            self.current_gesture = gesture_id
            self.current_confidence = confidence
        
        # Crear comando
        command = self.gesture_to_command(
            self.current_gesture if isinstance(self.current_gesture, int) else 10,
            self.current_confidence,
            intensity
        )
        
        # Calcular FPS
        inference_time = time.time() - start_time
        self.fps_counter.append(1.0 / max(inference_time, 0.001))
        fps = np.mean(self.fps_counter)
        
        # Anotar frame
        frame = self._annotate_frame(frame, command, fps)
        
        return command, frame
    
    def _draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja información de debug en el frame."""
        h, w = frame.shape[:2]

        # Panel de debug más grande
        debug_panel_width = 350
        cv2.rectangle(frame, (0, 90), (debug_panel_width, h - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 90), (debug_panel_width, h - 10), (100, 100, 100), 2)

        y_offset = 115
        line_height = 20

        # Título
        cv2.putText(frame, "DEBUG MODE", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)
        y_offset += 30

        # Estado de detección
        hand_status = "SI" if self.debug_info['hand_detected'] else "NO"
        hand_color = (0, 255, 0) if self.debug_info['hand_detected'] else (0, 0, 255)
        cv2.putText(frame, f"Mano detectada: {hand_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
        y_offset += line_height + 10

        # Estado de buffers
        cv2.putText(frame, "Buffers:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 0), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Pred: {len(self.prediction_buffer)}/{self.prediction_buffer.maxlen}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Conf: {len(self.confidence_buffer)}/{self.confidence_buffer.maxlen}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Hold: {self.gesture_hold_counter} frames",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height + 10

        # Top 5 predicciones
        if self.debug_info['top5_classes']:
            cv2.putText(frame, "Top 5 Predicciones:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height

            for i, (gesture_name, prob) in enumerate(self.debug_info['top5_classes'][:5]):
                # Barra de probabilidad
                bar_width = int(prob * 200)
                bar_color = (0, 255, 0) if i == 0 else (100, 150, 255)
                cv2.rectangle(frame, (120, y_offset - 12), (120 + bar_width, y_offset - 4),
                             bar_color, -1)

                # Texto
                text = f"{i+1}. {gesture_name[:12]}: {prob*100:.1f}%"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (200, 200, 200), 1)
                y_offset += line_height

        # Probabilidades raw (solo si hay)
        if self.debug_info['probs'] is not None:
            y_offset += 10
            cv2.putText(frame, "Todas las Probs:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height

            probs = self.debug_info['probs']
            for class_id, prob in enumerate(probs):
                if prob > 0.01:  # Solo mostrar si prob > 1%
                    gesture_name = GESTURE_CLASSES.get(class_id, f"C{class_id}")
                    text = f"  {gesture_name[:10]}: {prob*100:.1f}%"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.35, (150, 150, 150), 1)
                    y_offset += 15
                    if y_offset > h - 30:  # No salir del frame
                        break

        return frame

    def _annotate_frame(self, frame: np.ndarray, command: DroneCommand,
                        fps: float) -> np.ndarray:
        """Añade anotaciones al frame."""
        h, w = frame.shape[:2]
        
        # Panel de información
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        
        # Gesto detectado
        gesture_text = f"Gesto: {command.gesture}"
        color = (0, 255, 0) if command.confidence > 0.8 else (0, 255, 255)
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, color, 2)
        
        # Confianza e intensidad
        conf_text = f"Conf: {command.confidence:.2f} | Int: {command.intensity:.2f}"
        cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (200, 200, 200), 1)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)
        
        # Panel de comandos
        cv2.rectangle(frame, (w - 200, 90), (w, h - 10), (30, 30, 30), -1)
        cv2.putText(frame, "COMANDOS", (w - 190, 115), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        y_offset = 140
        commands = [
            ("Pitch", command.pitch),
            ("Roll", command.roll),
            ("Yaw", command.yaw),
            ("Throttle", command.throttle),
        ]
        
        for name, value in commands:
            # Barra de valor
            bar_color = (0, 255, 0) if value > 0 else (0, 0, 255) if value < 0 else (100, 100, 100)
            bar_width = int(abs(value) * 80)
            bar_x = w - 100 if value >= 0 else w - 100 - bar_width
            cv2.rectangle(frame, (bar_x, y_offset - 10), (bar_x + bar_width, y_offset + 5), 
                         bar_color, -1)
            
            cv2.putText(frame, f"{name}: {value:+.2f}", (w - 190, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 25
        
        # Estados especiales
        if command.hover:
            cv2.putText(frame, "HOVER", (w - 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 0), 2)
        if command.emergency:
            cv2.putText(frame, "EMERGENCY!", (w - 180, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)

        # Agregar debug overlay si está activado
        if self.debug:
            frame = self._draw_debug_overlay(frame)

        return frame
    
    def run_demo(self):
        """Ejecuta demo de inferencia con webcam."""
        print("\nIniciando demo de inferencia...")
        print("Presiona 'Q' para salir\n")
        
        cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        cv2.namedWindow("Drone Gesture Control", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                command, annotated_frame = self.process_frame(frame)
                
                # Mostrar
                cv2.imshow("Drone Gesture Control", annotated_frame)
                
                # Salir con Q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
    
    def cleanup(self):
        """Libera recursos."""
        self.hands.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de inferencia para control de dron")
    parser.add_argument('--mode', type=str, default='demo', 
                        choices=['demo', 'production'],
                        help='Modo de ejecución')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Dispositivo (cuda/cpu)')
    parser.add_argument('--no-temporal', action='store_true',
                        help='Desactivar modelo temporal')
    parser.add_argument('--segmentation', action='store_true',
                        help='Activar modelo de segmentación')
    parser.add_argument('--debug', action='store_true',
                        help='Activar modo debug (muestra outputs de la red)')

    args = parser.parse_args()

    system = GestureInferenceSystem(
        device=args.device,
        use_temporal=not args.no_temporal,
        use_segmentation=args.segmentation,
        debug=args.debug
    )
    
    if args.mode == 'demo':
        system.run_demo()
    else:
        print("Modo producción: Conectar con simulador de dron")
        system.run_demo()  # Por ahora igual que demo


if __name__ == "__main__":
    main()
