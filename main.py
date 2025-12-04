"""
Sistema Integrado de Control de Dron con Gestos.

Este script integra:
- Captura de webcam
- Detección de gestos con MediaPipe + CNN + GRU
- Simulador 3D del dron

Uso:
    python main.py --mode demo          # Solo inferencia con visualización
    python main.py --mode simulator     # Solo simulador con teclado
    python main.py --mode integrated    # Sistema completo integrado
    python main.py --mode record        # Grabar dataset
"""

import cv2
import numpy as np
import torch
import threading
import time
from queue import Queue
import argparse
import sys
from pathlib import Path

# Importar módulos del proyecto
from config import CAMERA_CONFIG, GESTURE_CLASSES, CHECKPOINTS_DIR
from inference import GestureInferenceSystem, DroneCommand
from drone_simulator import DroneSimulator, DroneState


class IntegratedSystem:
    """
    Sistema integrado que conecta inferencia de gestos con simulador de dron.
    
    Ejecuta la captura de video e inferencia en un thread separado,
    mientras el simulador corre en el thread principal (para OpenGL).
    """
    
    def __init__(self,
                 use_temporal: bool = True,
                 device: str = 'cuda',
                 debug: bool = False):
        """
        Inicializa el sistema integrado.

        Args:
            use_temporal: Si usar modelo temporal para suavizado
            device: Dispositivo para inferencia (cuda/cpu)
            debug: Si activar modo debug
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_temporal = use_temporal
        self.debug = debug
        
        # Cola para comunicación entre threads
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        
        # Flags de control
        self.running = False
        self.inference_thread = None
        
        # Sistema de inferencia (se inicializa en el thread)
        self.inference_system = None
        
        # Simulador (se ejecuta en main thread)
        self.simulator = None
        
        # Estadísticas
        self.fps_inference = 0
        self.last_command = None
    
    def _inference_loop(self):
        """Bucle de inferencia que corre en thread separado."""
        print("Iniciando thread de inferencia...")
        
        # Inicializar sistema de inferencia
        self.inference_system = GestureInferenceSystem(
            device=self.device,
            use_temporal=self.use_temporal,
            debug=self.debug
        )
        
        # Abrir cámara
        cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            self.running = False
            return
        
        # Crear ventana de video
        cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Control", 640, 480)
        
        fps_counter = []
        
        try:
            while self.running:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Procesar frame
                command, annotated_frame = self.inference_system.process_frame(frame)

                # Enviar comando al simulador
                cmd_dict = command.to_dict()
                self.command_queue.put(cmd_dict)
                self.last_command = command

                # Debug: mostrar comandos que se están enviando
                if self.debug and command.gesture != "NO_GESTURE":
                    print(f"[DEBUG] Enviando comando: {command.gesture} "
                          f"(conf: {command.confidence:.2f}, "
                          f"pitch: {command.pitch:.2f}, roll: {command.roll:.2f}, "
                          f"yaw: {command.yaw:.2f}, throttle: {command.throttle:.2f})")
                
                # Mostrar frame anotado
                cv2.imshow("Gesture Control", annotated_frame)
                
                # Calcular FPS
                elapsed = time.time() - start_time
                fps_counter.append(1.0 / max(elapsed, 0.001))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                self.fps_inference = np.mean(fps_counter)
                
                # Salir con Q o ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.inference_system:
                self.inference_system.cleanup()
            print("Thread de inferencia terminado")
    
    def _simulator_loop(self):
        """Bucle del simulador que corre en el thread principal."""
        print("Iniciando simulador...")

        # Crear simulador
        self.simulator = DroneSimulator(use_3d=True, debug=self.debug)
        
        import pygame
        from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
        
        clock = pygame.time.Clock()
        
        while self.running:
            dt = clock.tick(60) / 1000.0
            
            # Procesar eventos de pygame
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
            
            # Obtener comandos de la cola y actualizar el simulador
            while not self.command_queue.empty():
                cmd = self.command_queue.get()
                # Actualizar directamente el current_command del simulador
                self.simulator.current_command = {
                    'pitch': cmd.get('pitch', 0.0),
                    'roll': cmd.get('roll', 0.0),
                    'yaw': cmd.get('yaw', 0.0),
                    'throttle': cmd.get('throttle', 0.0),
                    'hover': cmd.get('hover', False),
                    'emergency': cmd.get('emergency', False)
                }
                self.simulator.gesture_name = cmd.get('gesture', 'NO_GESTURE')

                # Debug: mostrar comandos recibidos por el simulador
                if self.debug and self.simulator.gesture_name != "NO_GESTURE":
                    print(f"[SIMULATOR] Aplicando comando: {self.simulator.gesture_name} -> "
                          f"pitch={self.simulator.current_command['pitch']:.2f}, "
                          f"roll={self.simulator.current_command['roll']:.2f}, "
                          f"yaw={self.simulator.current_command['yaw']:.2f}, "
                          f"throttle={self.simulator.current_command['throttle']:.2f}")
            
            # Actualizar física
            self.simulator.state = self.simulator.physics.update(
                self.simulator.state,
                self.simulator.current_command,
                dt
            )
            
            # Renderizar
            self.simulator.renderer.render(self.simulator.state)
            
            # HUD
            if not self.simulator.use_3d:
                hud = self.simulator._draw_pygame_hud()
                self.simulator.screen.blit(hud, (10, 10))
            
            pygame.display.flip()
        
        pygame.quit()
        print("Simulador terminado")
    
    def run(self):
        """Ejecuta el sistema integrado."""
        print("\n" + "="*60)
        print("SISTEMA INTEGRADO DE CONTROL DE DRON CON GESTOS")
        print("="*60)
        print(f"Dispositivo: {self.device}")
        print(f"Modelo temporal: {'Activado' if self.use_temporal else 'Desactivado'}")
        print("="*60)
        print("\nPresiona Q en la ventana de video o ESC en el simulador para salir")
        print("="*60 + "\n")
        
        self.running = True
        
        # Iniciar thread de inferencia
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.start()
        
        # Esperar un momento para que inicie la inferencia
        time.sleep(2)
        
        # Ejecutar simulador en main thread (necesario para OpenGL)
        if self.running:
            self._simulator_loop()
        
        # Esperar a que termine el thread de inferencia
        self.running = False
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2)
        
        print("\nSistema terminado correctamente")


def run_demo(debug=False):
    """Ejecuta solo la demo de inferencia (sin simulador)."""
    print("\n=== MODO DEMO: Solo Inferencia ===\n")

    system = GestureInferenceSystem(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_temporal=True,
        debug=debug
    )
    system.run_demo()


def run_simulator():
    """Ejecuta solo el simulador (control por teclado)."""
    print("\n=== MODO SIMULADOR: Control por Teclado ===\n")
    
    simulator = DroneSimulator(use_3d=True)
    simulator.run(gesture_mode=False)


def run_integrated(debug=False):
    """Ejecuta el sistema integrado completo."""
    print("\n=== MODO INTEGRADO: Gestos + Simulador ===\n")

    system = IntegratedSystem(
        use_temporal=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=debug
    )
    system.run()


def run_recorder():
    """Ejecuta el grabador de dataset."""
    print("\n=== MODO GRABACIÓN: Dataset Recorder ===\n")
    
    from dataset_recorder import DatasetRecorder
    recorder = DatasetRecorder()
    recorder.run()


def check_models():
    """Verifica si los modelos están entrenados."""
    models = [
        ("Clasificador CNN", CHECKPOINTS_DIR / "classifier_resnet18_best.pt"),
        ("Modelo Temporal", CHECKPOINTS_DIR / "temporal_gru_best.pt"),
        ("Segmentación UNet", CHECKPOINTS_DIR / "segmentation_unet_best.pt"),
    ]
    
    print("\n=== Estado de Modelos ===")
    all_present = True
    for name, path in models:
        exists = path.exists()
        status = "✓ Encontrado" if exists else "✗ No encontrado"
        print(f"  {name}: {status}")
        if not exists:
            all_present = False
    print()
    
    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Control de Dron con Gestos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos disponibles:
  demo        - Solo inferencia de gestos con visualización
  simulator   - Solo simulador 3D con control por teclado
  integrated  - Sistema completo: gestos + simulador
  record      - Grabar dataset de gestos
  check       - Verificar estado de modelos entrenados

Ejemplos:
  python main.py --mode demo
  python main.py --mode integrated --device cuda
  python main.py --mode record
        """
    )
    
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'simulator', 'integrated', 'record', 'check'],
                        help='Modo de ejecución')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Dispositivo para inferencia (cuda/cpu)')
    parser.add_argument('--no-temporal', action='store_true',
                        help='Desactivar modelo temporal')
    parser.add_argument('--debug', action='store_true',
                        help='Activar modo debug (muestra outputs de la red)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  DRONE GESTURE CONTROL SYSTEM")
    print("  Proyecto Final - Inteligencia Artificial")
    print("="*60)

    # Verificar modelos si es necesario
    if args.mode in ['demo', 'integrated']:
        if not check_models():
            print("AVISO: Algunos modelos no están entrenados.")
            print("El sistema funcionará pero con capacidad reducida.")
            print("Ejecuta los scripts de entrenamiento primero:\n")
            print("  python train_classifier.py")
            print("  python train_temporal.py")
            print("  python train_segmentation.py")
            print()

    # Ejecutar modo seleccionado
    if args.mode == 'demo':
        run_demo(debug=args.debug)
    elif args.mode == 'simulator':
        run_simulator()
    elif args.mode == 'integrated':
        run_integrated(debug=args.debug)
    elif args.mode == 'record':
        run_recorder()
    elif args.mode == 'check':
        check_models()
        print("Para entrenar modelos, ejecuta:")
        print("  python train_classifier.py")
        print("  python train_temporal.py")
        print("  python train_segmentation.py")


if __name__ == "__main__":
    main()
