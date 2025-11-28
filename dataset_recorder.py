"""
Grabador de Dataset para gestos de control de dron.

Uso:
    python dataset_recorder.py

Controles:
    - Números 0-9: Seleccionar clase de gesto
    - ESPACIO: Iniciar/Pausar grabación
    - S: Guardar sesión actual
    - Q: Salir
    - R: Reiniciar contador de frames
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import threading

from config import (
    GESTURE_CLASSES, DATASET_DIR, CAMERA_CONFIG, 
    MEDIAPIPE_CONFIG, SEGMENTATION_CONFIG
)


class DatasetRecorder:
    """Grabador interactivo de dataset para gestos de manos."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or DATASET_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)
        
        # Estado de grabación
        self.current_class = 0
        self.is_recording = False
        self.frame_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Buffer para secuencias temporales
        self.frame_buffer = deque(maxlen=30)
        self.landmark_buffer = deque(maxlen=30)
        
        # Estadísticas
        self.class_counts = {i: 0 for i in GESTURE_CLASSES.keys()}
        self.load_existing_stats()
        
        # Configuración de cámara
        self.cap = None
        
    def load_existing_stats(self):
        """Carga estadísticas existentes del dataset."""
        stats_file = self.output_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.class_counts = {int(k): v for k, v in stats.get('class_counts', {}).items()}
    
    def save_stats(self):
        """Guarda estadísticas del dataset."""
        stats_file = self.output_dir / "dataset_stats.json"
        stats = {
            'class_counts': self.class_counts,
            'total_frames': sum(self.class_counts.values()),
            'last_updated': datetime.now().isoformat()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def create_class_dirs(self):
        """Crea directorios para cada clase de gesto."""
        for class_id, class_name in GESTURE_CLASSES.items():
            class_dir = self.output_dir / "images" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Directorio para máscaras de segmentación
            mask_dir = self.output_dir / "masks" / class_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            
            # Directorio para landmarks
            landmark_dir = self.output_dir / "landmarks" / class_name
            landmark_dir.mkdir(parents=True, exist_ok=True)
            
            # Directorio para secuencias temporales
            seq_dir = self.output_dir / "sequences" / class_name
            seq_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_hand_roi(self, frame, landmarks, padding=0.2):
        """Extrae la región de interés de la mano con padding."""
        h, w = frame.shape[:2]
        
        # Obtener bounding box de landmarks
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Añadir padding
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        
        x_min = max(0, int(x_min - pad_x))
        x_max = min(w, int(x_max + pad_x))
        y_min = max(0, int(y_min - pad_y))
        y_max = min(h, int(y_max + pad_y))
        
        roi = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
        
        return roi, bbox
    
    def create_hand_mask(self, frame, landmarks):
        """Crea una máscara binaria de la mano usando convex hull."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Obtener puntos de landmarks
        points = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Crear convex hull
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Dilatar para cubrir mejor la mano
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def landmarks_to_array(self, landmarks):
        """Convierte landmarks a array numpy normalizado."""
        arr = []
        for lm in landmarks.landmark:
            arr.extend([lm.x, lm.y, lm.z])
        return np.array(arr, dtype=np.float32)
    
    def save_frame_data(self, frame, landmarks, class_id):
        """Guarda un frame con todos sus datos asociados."""
        class_name = GESTURE_CLASSES[class_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Extraer ROI de la mano
        roi, bbox = self.extract_hand_roi(frame, landmarks)
        if roi.size == 0:
            return False
        
        # Redimensionar ROI
        input_size = SEGMENTATION_CONFIG["input_size"]
        roi_resized = cv2.resize(roi, input_size)
        
        # Crear máscara
        mask = self.create_hand_mask(frame, landmarks)
        mask_roi = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if mask_roi.size == 0:
            return False
        mask_resized = cv2.resize(mask_roi, input_size)
        
        # Guardar imagen
        img_path = self.output_dir / "images" / class_name / f"{timestamp}.jpg"
        cv2.imwrite(str(img_path), roi_resized)
        
        # Guardar máscara
        mask_path = self.output_dir / "masks" / class_name / f"{timestamp}.png"
        cv2.imwrite(str(mask_path), mask_resized)
        
        # Guardar landmarks
        landmarks_arr = self.landmarks_to_array(landmarks)
        landmark_path = self.output_dir / "landmarks" / class_name / f"{timestamp}.npy"
        np.save(str(landmark_path), landmarks_arr)
        
        # Actualizar contadores
        self.class_counts[class_id] += 1
        self.frame_count += 1
        
        return True
    
    def save_sequence(self, class_id):
        """Guarda una secuencia temporal de frames."""
        if len(self.frame_buffer) < 15:
            return False
        
        class_name = GESTURE_CLASSES[class_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Guardar secuencia de frames
        seq_dir = self.output_dir / "sequences" / class_name / timestamp
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (frame, landmarks) in enumerate(zip(self.frame_buffer, self.landmark_buffer)):
            if landmarks is not None:
                frame_path = seq_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                landmark_path = seq_dir / f"landmarks_{i:03d}.npy"
                np.save(str(landmark_path), landmarks)
        
        # Guardar metadata
        metadata = {
            'class_id': class_id,
            'class_name': class_name,
            'num_frames': len(self.frame_buffer),
            'timestamp': timestamp
        }
        meta_path = seq_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def draw_ui(self, frame):
        """Dibuja la interfaz de usuario sobre el frame."""
        h, w = frame.shape[:2]
        
        # Panel superior
        cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
        
        # Título
        title = "DRONE GESTURE DATASET RECORDER"
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        
        # Estado de grabación
        status = "GRABANDO" if self.is_recording else "PAUSADO"
        color = (0, 255, 0) if self.is_recording else (0, 0, 255)
        cv2.circle(frame, (w - 100, 25), 10, color, -1)
        cv2.putText(frame, status, (w - 85, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
        
        # Clase actual
        class_name = GESTURE_CLASSES[self.current_class]
        cv2.putText(frame, f"Clase: [{self.current_class}] {class_name}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Contador de frames
        cv2.putText(frame, f"Frames: {self.frame_count} | Total clase: {self.class_counts[self.current_class]}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Panel lateral con estadísticas
        panel_x = w - 250
        cv2.rectangle(frame, (panel_x, 110), (w, h - 50), (30, 30, 30), -1)
        cv2.putText(frame, "ESTADISTICAS", (panel_x + 10, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = 160
        for class_id, count in self.class_counts.items():
            name = GESTURE_CLASSES[class_id][:12]
            text = f"{class_id}: {name}: {count}"
            color = (0, 255, 0) if class_id == self.current_class else (150, 150, 150)
            cv2.putText(frame, text, (panel_x + 10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 20
        
        # Instrucciones
        cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)
        instructions = "[0-9]: Clase | ESPACIO: Grabar | S: Guardar | Q: Salir | T: Secuencia"
        cv2.putText(frame, instructions, (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Ejecuta el grabador de dataset."""
        print("Iniciando grabador de dataset...")
        print("Creando directorios de clases...")
        self.create_class_dirs()
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("\n" + "="*50)
        print("CONTROLES:")
        print("  0-9: Seleccionar clase de gesto")
        print("  ESPACIO: Iniciar/Pausar grabación")
        print("  S: Guardar estadísticas")
        print("  T: Guardar secuencia temporal")
        print("  R: Reiniciar contador")
        print("  Q: Salir")
        print("="*50 + "\n")
        
        cv2.namedWindow("Dataset Recorder", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error leyendo frame")
                    break
                
                # Flip horizontal para efecto espejo
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Procesar con MediaPipe
                results = self.hands.process(frame_rgb)
                
                # Dibujar landmarks si se detecta mano
                landmarks_arr = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Si está grabando, guardar datos
                        if self.is_recording:
                            success = self.save_frame_data(
                                frame_rgb, hand_landmarks, self.current_class)
                            if success:
                                # Indicador visual de captura
                                cv2.circle(frame, (50, 150), 20, (0, 255, 0), -1)
                        
                        # Actualizar buffer temporal
                        landmarks_arr = self.landmarks_to_array(hand_landmarks)
                
                # Actualizar buffers
                self.frame_buffer.append(frame_rgb.copy())
                self.landmark_buffer.append(landmarks_arr)
                
                # Dibujar UI
                display_frame = self.draw_ui(frame.copy())
                cv2.imshow("Dataset Recorder", display_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.is_recording = not self.is_recording
                    status = "iniciada" if self.is_recording else "pausada"
                    print(f"Grabación {status}")
                elif key == ord('s'):
                    self.save_stats()
                    print("Estadísticas guardadas")
                elif key == ord('r'):
                    self.frame_count = 0
                    print("Contador reiniciado")
                elif key == ord('t'):
                    if self.save_sequence(self.current_class):
                        print(f"Secuencia guardada para clase {GESTURE_CLASSES[self.current_class]}")
                elif ord('0') <= key <= ord('9'):
                    new_class = key - ord('0')
                    if new_class in GESTURE_CLASSES:
                        self.current_class = new_class
                        print(f"Clase cambiada a: {GESTURE_CLASSES[self.current_class]}")
                elif key == ord('-'):  # Para clase 10
                    self.current_class = 10
                    print(f"Clase cambiada a: {GESTURE_CLASSES[self.current_class]}")
        
        finally:
            self.save_stats()
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("\nEstadísticas finales:")
            for class_id, count in self.class_counts.items():
                print(f"  {GESTURE_CLASSES[class_id]}: {count} frames")
            print(f"Total: {sum(self.class_counts.values())} frames")


if __name__ == "__main__":
    recorder = DatasetRecorder()
    recorder.run()
