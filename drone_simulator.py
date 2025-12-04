"""
Simulador 3D de Dron para control con gestos.

Utiliza PyGame + PyOpenGL para renderizar un dron en 3D
que responde a los comandos generados por el sistema de gestos.

Uso:
    python drone_simulator.py --mode standalone
    python drone_simulator.py --mode integrated

Controles manuales (modo standalone):
    W/S: Pitch (adelante/atrás)
    A/D: Roll (izquierda/derecha)
    Q/E: Yaw (rotación)
    ESPACIO/SHIFT: Throttle (subir/bajar)
    H: Hover
    ESC: Salir
"""

import pygame
from pygame.locals import *
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional
import threading
from queue import Queue

# OpenGL imports
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("AVISO: PyOpenGL no instalado. Usando renderizado 2D alternativo.")

from config import SIMULATOR_CONFIG, GESTURE_CLASSES


@dataclass
class DroneState:
    """Estado del dron en el simulador."""
    # Posición
    x: float = 0.0
    y: float = 5.0  # Altura inicial
    z: float = 0.0
    
    # Velocidad
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Rotación (en grados)
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    
    # Velocidad angular
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    yaw_rate: float = 0.0
    
    # Estado de motores
    throttle: float = 0.5  # 0-1
    motor_speeds: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    
    # Flags
    is_flying: bool = True
    is_hovering: bool = False
    emergency_stop: bool = False


class DronePhysics:
    """Simula la física del dron."""
    
    def __init__(self):
        self.gravity = SIMULATOR_CONFIG["gravity"]
        self.mass = 1.0  # kg
        self.drag = 0.1
        self.max_speed = 10.0
        self.max_rotation_speed = 180.0  # grados/s
        
        # Constantes de control
        self.pitch_response = 30.0  # grados de inclinación por unidad de input
        self.roll_response = 30.0
        self.yaw_response = 90.0  # grados/s por unidad de input
        self.throttle_response = 15.0  # m/s² por unidad de input
    
    def update(self, state: DroneState, command: dict, dt: float) -> DroneState:
        """
        Actualiza el estado del dron basado en comandos.
        
        Args:
            state: Estado actual del dron
            command: Diccionario con comandos (pitch, roll, yaw, throttle, hover, emergency)
            dt: Delta time en segundos
        """
        if state.emergency_stop:
            # Aterrizaje de emergencia
            state.vy = -5.0
            state.vx *= 0.9
            state.vz *= 0.9
            state.pitch *= 0.9
            state.roll *= 0.9
            
            if state.y <= 0.1:
                state.y = 0.0
                state.vy = 0.0
                state.is_flying = False
            return state
        
        # Extraer comandos
        pitch_cmd = command.get('pitch', 0.0)
        roll_cmd = command.get('roll', 0.0)
        yaw_cmd = command.get('yaw', 0.0)
        throttle_cmd = command.get('throttle', 0.0)
        hover = command.get('hover', False)
        emergency = command.get('emergency', False)
        
        if emergency:
            state.emergency_stop = True
            return state
        
        # Modo hover
        if hover:
            state.is_hovering = True
            # Estabilizar gradualmente
            state.pitch *= 0.9
            state.roll *= 0.9
            state.vx *= 0.95
            state.vz *= 0.95
            # Mantener altura
            throttle_cmd = 0.0
        else:
            state.is_hovering = False
        
        # Actualizar rotación
        target_pitch = pitch_cmd * self.pitch_response
        target_roll = roll_cmd * self.roll_response
        
        # Suavizar transición de rotación
        state.pitch += (target_pitch - state.pitch) * 5.0 * dt
        state.roll += (target_roll - state.roll) * 5.0 * dt
        state.yaw += yaw_cmd * self.yaw_response * dt
        
        # Normalizar yaw
        state.yaw = state.yaw % 360
        
        # Calcular fuerzas
        # Thrust basado en throttle
        thrust = (state.throttle + throttle_cmd * 0.5) * self.throttle_response
        
        # Componentes de velocidad basadas en orientación
        pitch_rad = math.radians(state.pitch)
        roll_rad = math.radians(state.roll)
        yaw_rad = math.radians(state.yaw)
        
        # Aceleración
        ax = math.sin(pitch_rad) * thrust
        az = -math.sin(roll_rad) * thrust
        ay = math.cos(pitch_rad) * math.cos(roll_rad) * thrust - self.gravity
        
        # Rotar por yaw
        ax_world = ax * math.cos(yaw_rad) - az * math.sin(yaw_rad)
        az_world = ax * math.sin(yaw_rad) + az * math.cos(yaw_rad)
        
        # Actualizar velocidad
        state.vx += ax_world * dt
        state.vy += ay * dt
        state.vz += az_world * dt
        
        # Aplicar drag
        state.vx *= (1 - self.drag * dt)
        state.vy *= (1 - self.drag * dt)
        state.vz *= (1 - self.drag * dt)
        
        # Limitar velocidad
        speed = math.sqrt(state.vx**2 + state.vy**2 + state.vz**2)
        if speed > self.max_speed:
            factor = self.max_speed / speed
            state.vx *= factor
            state.vy *= factor
            state.vz *= factor
        
        # Actualizar posición
        state.x += state.vx * dt
        state.y += state.vy * dt
        state.z += state.vz * dt
        
        # Colisión con suelo
        if state.y < 0.1:
            state.y = 0.1
            state.vy = max(0, state.vy)
            if abs(state.vy) < 0.1 and throttle_cmd <= 0:
                state.is_flying = False
        else:
            state.is_flying = True
        
        # Límites del mundo
        state.x = np.clip(state.x, -50, 50)
        state.z = np.clip(state.z, -50, 50)
        state.y = np.clip(state.y, 0, 50)
        
        # Actualizar velocidades de motor (visualización)
        base_speed = 0.5 + throttle_cmd * 0.3
        state.motor_speeds = (
            base_speed + pitch_cmd * 0.1 - roll_cmd * 0.1,
            base_speed + pitch_cmd * 0.1 + roll_cmd * 0.1,
            base_speed - pitch_cmd * 0.1 + roll_cmd * 0.1,
            base_speed - pitch_cmd * 0.1 - roll_cmd * 0.1,
        )
        
        return state


class DroneRenderer3D:
    """Renderizador 3D del dron usando OpenGL."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.propeller_angle = 0
        
        # Inicializar OpenGL
        self._init_gl()
    
    def _init_gl(self):
        """Inicializa configuración de OpenGL."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Luz
        glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 10.0, 5.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        
        # Perspectiva
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(SIMULATOR_CONFIG["fov"], 
                      self.width / self.height,
                      SIMULATOR_CONFIG["near_plane"],
                      SIMULATOR_CONFIG["far_plane"])
        glMatrixMode(GL_MODELVIEW)
    
    def render(self, state: DroneState):
        """Renderiza la escena completa."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Cielo con gradiente (fondo)
        glClearColor(0.53, 0.81, 0.92, 1.0)  # Celeste

        # Cámara siguiendo al dron
        cam_distance = 15
        cam_height = 8
        cam_x = state.x - cam_distance * math.sin(math.radians(state.yaw))
        cam_z = state.z - cam_distance * math.cos(math.radians(state.yaw))

        gluLookAt(cam_x, state.y + cam_height, cam_z,
                  state.x, state.y, state.z,
                  0, 1, 0)

        # Dibujar paisaje
        self._draw_landscape()

        # Dibujar suelo mejorado
        self._draw_ground()

        # Dibujar elementos del entorno
        self._draw_environment()

        # Dibujar dron
        self._draw_drone(state)

        # Dibujar marcadores de referencia
        self._draw_markers()

        # Actualizar ángulo de hélices
        avg_motor = sum(state.motor_speeds) / 4
        self.propeller_angle += avg_motor * 50
    
    def _draw_landscape(self):
        """Dibuja montañas de fondo."""
        glDisable(GL_LIGHTING)

        # Montañas lejanas (fondo)
        mountain_positions = [
            (-40, 0, -45, 15, 8),   # x, y, z, width, height
            (-20, 0, -48, 20, 12),
            (10, 0, -47, 18, 10),
            (35, 0, -46, 12, 7),
        ]

        for mx, my, mz, mw, mh in mountain_positions:
            # Triángulos para montañas
            glBegin(GL_TRIANGLES)
            # Cara frontal (gris-azulado)
            glColor3f(0.4, 0.45, 0.5)
            glVertex3f(mx - mw/2, my, mz)
            glVertex3f(mx + mw/2, my, mz)
            glVertex3f(mx, my + mh, mz)

            # Cara izquierda (más oscura)
            glColor3f(0.3, 0.35, 0.4)
            glVertex3f(mx - mw/2, my, mz)
            glVertex3f(mx, my + mh, mz)
            glVertex3f(mx, my + mh, mz + 2)

            # Cara derecha (más oscura)
            glColor3f(0.3, 0.35, 0.4)
            glVertex3f(mx + mw/2, my, mz)
            glVertex3f(mx, my + mh, mz)
            glVertex3f(mx, my + mh, mz + 2)
            glEnd()

        glEnable(GL_LIGHTING)

    def _draw_ground(self):
        """Dibuja el suelo mejorado con textura de césped."""
        glDisable(GL_LIGHTING)

        # Suelo de césped (verde)
        glColor3f(0.2, 0.6, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-50, 0, -50)
        glVertex3f(50, 0, -50)
        glVertex3f(50, 0, 50)
        glVertex3f(-50, 0, 50)
        glEnd()

        # Cuadrícula más sutil
        glColor3f(0.25, 0.65, 0.25)
        glBegin(GL_LINES)
        for i in range(-50, 51, 5):
            glVertex3f(i, 0.01, -50)
            glVertex3f(i, 0.01, 50)
            glVertex3f(-50, 0.01, i)
            glVertex3f(50, 0.01, i)
        glEnd()

        # Ejes de referencia más visibles
        glLineWidth(3)
        # Eje X (rojo) - hacia la derecha
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0.02, 0)
        glVertex3f(5, 0.02, 0)
        glEnd()

        # Flecha
        glBegin(GL_TRIANGLES)
        glVertex3f(5, 0.02, 0)
        glVertex3f(4.5, 0.02, 0.3)
        glVertex3f(4.5, 0.02, -0.3)
        glEnd()

        # Eje Z (azul) - hacia adelante
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(0, 0.02, 0)
        glVertex3f(0, 0.02, -5)
        glEnd()

        # Flecha
        glBegin(GL_TRIANGLES)
        glVertex3f(0, 0.02, -5)
        glVertex3f(0.3, 0.02, -4.5)
        glVertex3f(-0.3, 0.02, -4.5)
        glEnd()

        # Eje Y (verde) - hacia arriba
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0.02, 0)
        glVertex3f(0, 5, 0)
        glEnd()

        glLineWidth(1)
        glEnable(GL_LIGHTING)

    def _draw_environment(self):
        """Dibuja elementos del entorno (árboles, edificios, etc.)."""
        # Árboles distribuidos por el terreno
        tree_positions = [
            (15, 0, 10), (-12, 0, 15), (20, 0, -8), (-18, 0, -12),
            (8, 0, 20), (-25, 0, 8), (30, 0, 5), (-15, 0, -20),
            (5, 0, -15), (-8, 0, 22), (25, 0, -15), (-22, 0, -5),
        ]

        for tx, ty, tz in tree_positions:
            self._draw_tree(tx, ty, tz)

        # Edificios pequeños (torres de control)
        building_positions = [
            (30, 0, 30, 2, 8, 2),      # x, y, z, width, height, depth
            (-35, 0, 25, 2.5, 10, 2.5),
            (-30, 0, -30, 2, 6, 2),
        ]

        for bx, by, bz, bw, bh, bd in building_positions:
            self._draw_building(bx, by, bz, bw, bh, bd)

    def _draw_tree(self, x, y, z, height=3.0):
        """Dibuja un árbol simple."""
        # Tronco
        glColor3f(0.4, 0.25, 0.1)
        glPushMatrix()
        glTranslatef(x, y + height * 0.3, z)
        self._draw_cylinder(0.15, height * 0.6, 8)
        glPopMatrix()

        # Copa (cono de hojas)
        glColor3f(0.1, 0.6, 0.1)
        glPushMatrix()
        glTranslatef(x, y + height * 0.6, z)

        # Dibujar cono simple con triángulos
        segments = 8
        cone_height = height * 0.8
        cone_radius = height * 0.4

        glBegin(GL_TRIANGLES)
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments

            x1 = cone_radius * math.cos(angle1)
            z1 = cone_radius * math.sin(angle1)
            x2 = cone_radius * math.cos(angle2)
            z2 = cone_radius * math.sin(angle2)

            # Cara lateral
            glVertex3f(0, cone_height, 0)  # Punta
            glVertex3f(x1, 0, z1)
            glVertex3f(x2, 0, z2)

            # Base
            glVertex3f(0, 0, 0)
            glVertex3f(x2, 0, z2)
            glVertex3f(x1, 0, z1)
        glEnd()

        glPopMatrix()

    def _draw_building(self, x, y, z, width, height, depth):
        """Dibuja un edificio simple."""
        glColor3f(0.6, 0.6, 0.65)

        glPushMatrix()
        glTranslatef(x, y + height/2, z)
        self._draw_box(0, 0, 0, width/2, height/2, depth/2)
        glPopMatrix()

        # Techo
        glColor3f(0.5, 0.2, 0.1)
        glPushMatrix()
        glTranslatef(x, y + height, z)
        self._draw_box(0, 0.1, 0, width/2 * 1.1, 0.1, depth/2 * 1.1)
        glPopMatrix()

        # Ventanas
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.8)
        num_floors = int(height / 1.5)
        for floor in range(num_floors):
            floor_y = y + 1 + floor * 1.5
            # Ventana frontal
            glBegin(GL_QUADS)
            glVertex3f(x - width/4, floor_y, z + depth/2 + 0.01)
            glVertex3f(x + width/4, floor_y, z + depth/2 + 0.01)
            glVertex3f(x + width/4, floor_y + 0.8, z + depth/2 + 0.01)
            glVertex3f(x - width/4, floor_y + 0.8, z + depth/2 + 0.01)
            glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_drone(self, state: DroneState):
        """Dibuja el dron."""
        glPushMatrix()

        # Posición
        glTranslatef(state.x, state.y, state.z)

        # Rotación del dron
        # Primero rotar el modelo 90° para que el frente apunte hacia adelante (Z negativo)
        glRotatef(90, 0, 1, 0)  # Corregir orientación base
        glRotatef(state.yaw, 0, 1, 0)
        glRotatef(state.pitch, 1, 0, 0)
        glRotatef(state.roll, 0, 0, 1)
        
        # Cuerpo central
        glColor3f(0.2, 0.2, 0.2)
        self._draw_box(0, 0, 0, 0.4, 0.1, 0.4)
        
        # Brazos
        arm_length = 0.8
        arm_positions = [
            (arm_length, 0, arm_length),
            (arm_length, 0, -arm_length),
            (-arm_length, 0, arm_length),
            (-arm_length, 0, -arm_length),
        ]
        
        # Colores de brazos (frente rojo, atrás negro)
        arm_colors = [
            (0.8, 0.2, 0.2),  # Frente derecha
            (0.8, 0.2, 0.2),  # Frente izquierda
            (0.3, 0.3, 0.3),  # Atrás derecha
            (0.3, 0.3, 0.3),  # Atrás izquierda
        ]
        
        for i, (ax, ay, az) in enumerate(arm_positions):
            # Brazo
            glColor3f(0.4, 0.4, 0.4)
            glPushMatrix()
            glTranslatef(ax/2, 0, az/2)
            self._draw_box(0, 0, 0, abs(ax)/2, 0.03, 0.03)
            glPopMatrix()
            
            # Motor
            glColor3f(*arm_colors[i])
            glPushMatrix()
            glTranslatef(ax, 0.05, az)
            self._draw_cylinder(0.1, 0.1)
            glPopMatrix()
            
            # Hélice
            glPushMatrix()
            glTranslatef(ax, 0.15, az)
            glRotatef(self.propeller_angle * (1 if i % 2 == 0 else -1), 0, 1, 0)
            glColor3f(0.6, 0.6, 0.6)
            self._draw_propeller(0.3)
            glPopMatrix()
        
        # Indicador de dirección frontal
        glColor3f(0, 1, 0)
        glPushMatrix()
        glTranslatef(0.5, 0.05, 0)
        self._draw_box(0, 0, 0, 0.1, 0.02, 0.02)
        glPopMatrix()
        
        glPopMatrix()
    
    def _draw_box(self, x, y, z, sx, sy, sz):
        """Dibuja una caja."""
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(sx, sy, sz)
        
        vertices = [
            (1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, -1), (-1, -1, 1)
        ]
        
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
        ]
        
        normals = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1)
        ]
        
        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glNormal3fv(normals[i])
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
        
        glPopMatrix()
    
    def _draw_cylinder(self, radius, height, segments=16):
        """Dibuja un cilindro."""
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            glNormal3f(math.cos(angle), 0, math.sin(angle))
            glVertex3f(x, 0, z)
            glVertex3f(x, height, z)
        glEnd()
    
    def _draw_propeller(self, length):
        """Dibuja una hélice de dos palas."""
        glBegin(GL_QUADS)
        # Pala 1
        glNormal3f(0, 1, 0)
        glVertex3f(-length, 0, -0.02)
        glVertex3f(length, 0, -0.02)
        glVertex3f(length, 0, 0.02)
        glVertex3f(-length, 0, 0.02)
        # Pala 2
        glVertex3f(-0.02, 0, -length)
        glVertex3f(0.02, 0, -length)
        glVertex3f(0.02, 0, length)
        glVertex3f(-0.02, 0, length)
        glEnd()
    
    def _draw_markers(self):
        """Dibuja marcadores de referencia en el mundo."""
        glDisable(GL_LIGHTING)
        
        # Marcadores de altura
        glColor3f(1, 1, 0)
        for h in [5, 10, 15, 20]:
            glPushMatrix()
            glTranslatef(0, h, 0)
            glBegin(GL_LINE_LOOP)
            for i in range(36):
                angle = 2 * math.pi * i / 36
                glVertex3f(2 * math.cos(angle), 0, 2 * math.sin(angle))
            glEnd()
            glPopMatrix()
        
        glEnable(GL_LIGHTING)


class DroneRenderer2D:
    """Renderizador 2D alternativo (sin OpenGL)."""
    
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.width = surface.get_width()
        self.height = surface.get_height()
    
    def render(self, state: DroneState):
        """Renderiza vista 2D del dron."""
        self.surface.fill((20, 20, 30))
        
        # Vista superior (XZ)
        self._draw_top_view(state, 50, 50, 300, 300)
        
        # Vista lateral (XY)
        self._draw_side_view(state, 400, 50, 300, 300)
        
        # Información
        self._draw_info(state, 50, 400)
    
    def _draw_top_view(self, state: DroneState, x, y, w, h):
        """Vista desde arriba."""
        pygame.draw.rect(self.surface, (40, 40, 50), (x, y, w, h))
        pygame.draw.rect(self.surface, (100, 100, 100), (x, y, w, h), 2)
        
        # Título
        font = pygame.font.SysFont('Arial', 16)
        text = font.render("Vista Superior (XZ)", True, (200, 200, 200))
        self.surface.blit(text, (x + 5, y + 5))
        
        # Convertir posición del dron a coordenadas de pantalla
        scale = w / 100  # 100 unidades = ancho de la vista
        cx = x + w // 2
        cy = y + h // 2
        
        dx = int(cx + state.x * scale)
        dy = int(cy + state.z * scale)
        
        # Dibujar dron
        drone_size = 15
        yaw_rad = math.radians(state.yaw)
        
        # Cuerpo
        pygame.draw.circle(self.surface, (50, 150, 50), (dx, dy), drone_size)
        
        # Indicador de dirección
        end_x = dx + int(drone_size * 1.5 * math.sin(yaw_rad))
        end_y = dy - int(drone_size * 1.5 * math.cos(yaw_rad))
        pygame.draw.line(self.surface, (255, 100, 100), (dx, dy), (end_x, end_y), 3)
        
        # Cuadrícula
        for i in range(-5, 6):
            gx = int(cx + i * 10 * scale)
            gy = int(cy + i * 10 * scale)
            pygame.draw.line(self.surface, (60, 60, 70), (gx, y), (gx, y + h), 1)
            pygame.draw.line(self.surface, (60, 60, 70), (x, gy), (x + w, gy), 1)
    
    def _draw_side_view(self, state: DroneState, x, y, w, h):
        """Vista lateral."""
        pygame.draw.rect(self.surface, (40, 40, 50), (x, y, w, h))
        pygame.draw.rect(self.surface, (100, 100, 100), (x, y, w, h), 2)
        
        # Título
        font = pygame.font.SysFont('Arial', 16)
        text = font.render("Vista Lateral (XY)", True, (200, 200, 200))
        self.surface.blit(text, (x + 5, y + 5))
        
        # Suelo
        ground_y = y + h - 30
        pygame.draw.line(self.surface, (100, 80, 50), (x, ground_y), (x + w, ground_y), 2)
        
        # Dron
        scale = w / 100
        dx = int(x + w // 2 + state.x * scale)
        dy = int(ground_y - state.y * scale * 5)  # Escala vertical aumentada
        
        drone_size = 15
        pitch_rad = math.radians(state.pitch)
        
        # Cuerpo inclinado
        left_x = dx - int(drone_size * math.cos(pitch_rad))
        left_y = dy + int(drone_size * math.sin(pitch_rad))
        right_x = dx + int(drone_size * math.cos(pitch_rad))
        right_y = dy - int(drone_size * math.sin(pitch_rad))
        
        pygame.draw.line(self.surface, (50, 150, 50), (left_x, left_y), (right_x, right_y), 4)
        pygame.draw.circle(self.surface, (100, 200, 100), (dx, dy), 8)
    
    def _draw_info(self, state: DroneState, x, y):
        """Información del estado."""
        font = pygame.font.SysFont('Arial', 18)
        
        info_lines = [
            f"Posición: X={state.x:.1f}  Y={state.y:.1f}  Z={state.z:.1f}",
            f"Velocidad: {math.sqrt(state.vx**2 + state.vy**2 + state.vz**2):.1f} m/s",
            f"Rotación: Pitch={state.pitch:.1f}°  Roll={state.roll:.1f}°  Yaw={state.yaw:.1f}°",
            f"Estado: {'VOLANDO' if state.is_flying else 'EN TIERRA'}" + 
            (" [HOVER]" if state.is_hovering else "") +
            (" [EMERGENCIA]" if state.emergency_stop else ""),
        ]
        
        for i, line in enumerate(info_lines):
            color = (200, 200, 200)
            if "EMERGENCIA" in line:
                color = (255, 100, 100)
            elif "HOVER" in line:
                color = (255, 255, 100)
            
            text = font.render(line, True, color)
            self.surface.blit(text, (x, y + i * 25))


class DroneSimulator:
    """Simulador principal del dron."""
    
    def __init__(self, width: int = None, height: int = None, use_3d: bool = True, debug: bool = False):
        self.width = width or SIMULATOR_CONFIG["window_width"]
        self.height = height or SIMULATOR_CONFIG["window_height"]
        self.use_3d = use_3d and OPENGL_AVAILABLE
        self.debug = debug

        # Inicializar pygame
        pygame.init()
        pygame.font.init()

        if self.use_3d:
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                DOUBLEBUF | OPENGL
            )
            self.renderer = DroneRenderer3D(self.width, self.height)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.renderer = DroneRenderer2D(self.screen)

        pygame.display.set_caption("Drone Simulator - Control por Gestos")

        # Estado y física
        self.state = DroneState()
        self.physics = DronePhysics()

        # Control
        self.command_queue = Queue()
        self.current_command = {
            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
            'throttle': 0.0, 'hover': False, 'emergency': False
        }

        # Tiempo
        self.clock = pygame.time.Clock()
        self.running = True

        # Modo
        self.gesture_mode = False
        self.gesture_name = "NO_GESTURE"
    
    def set_command(self, command: dict):
        """Establece comando desde sistema externo (gestos)."""
        self.command_queue.put(command)
    
    def _process_keyboard(self):
        """Procesa entrada de teclado (modo standalone)."""
        if self.gesture_mode:
            return
        
        keys = pygame.key.get_pressed()
        
        # Reset de comandos
        cmd = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
               'throttle': 0.0, 'hover': False, 'emergency': False}
        
        # Pitch (W/S)
        if keys[K_w]:
            cmd['pitch'] = 1.0
        elif keys[K_s]:
            cmd['pitch'] = -1.0
        
        # Roll (A/D)
        if keys[K_a]:
            cmd['roll'] = -1.0
        elif keys[K_d]:
            cmd['roll'] = 1.0
        
        # Yaw (Q/E)
        if keys[K_q]:
            cmd['yaw'] = -1.0
        elif keys[K_e]:
            cmd['yaw'] = 1.0
        
        # Throttle (Space/Shift)
        if keys[K_SPACE]:
            cmd['throttle'] = 1.0
        elif keys[K_LSHIFT]:
            cmd['throttle'] = -1.0
        
        # Hover (H)
        if keys[K_h]:
            cmd['hover'] = True
        
        # Emergency (X)
        if keys[K_x]:
            cmd['emergency'] = True
        
        self.current_command = cmd
    
    def _process_gesture_commands(self):
        """Procesa comandos de la cola de gestos."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.current_command = {
                'pitch': cmd.get('pitch', 0.0),
                'roll': cmd.get('roll', 0.0),
                'yaw': cmd.get('yaw', 0.0),
                'throttle': cmd.get('throttle', 0.0),
                'hover': cmd.get('hover', False),
                'emergency': cmd.get('emergency', False)
            }
            self.gesture_name = cmd.get('gesture', 'NO_GESTURE')

            # Debug: mostrar comandos recibidos
            if self.debug and self.gesture_name != "NO_GESTURE":
                print(f"[SIMULATOR DEBUG] Comando recibido: {self.gesture_name} -> "
                      f"pitch={self.current_command['pitch']:.2f}, "
                      f"roll={self.current_command['roll']:.2f}, "
                      f"yaw={self.current_command['yaw']:.2f}, "
                      f"throttle={self.current_command['throttle']:.2f}")
    
    def _draw_hud(self):
        """Dibuja HUD sobre la vista 3D."""
        if not self.use_3d:
            return
        
        # Cambiar a modo 2D para HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Fondo semitransparente para info
        glColor4f(0, 0, 0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(300, 10)
        glVertex2f(300, 150)
        glVertex2f(10, 150)
        glEnd()
        
        # Restaurar matrices
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _draw_pygame_hud(self):
        """Dibuja HUD usando pygame (para overlay en 3D)."""
        # Crear superficie transparente
        hud = pygame.Surface((350, 180), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 150))
        
        font = pygame.font.SysFont('Arial', 16)
        small_font = pygame.font.SysFont('Arial', 14)
        
        # Información del estado
        lines = [
            f"Posición: ({self.state.x:.1f}, {self.state.y:.1f}, {self.state.z:.1f})",
            f"Velocidad: {math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2):.1f} m/s",
            f"Altura: {self.state.y:.1f} m",
            f"Orientación: P={self.state.pitch:.0f}° R={self.state.roll:.0f}° Y={self.state.yaw:.0f}°",
            "",
            f"Gesto: {self.gesture_name}" if self.gesture_mode else "Modo: Teclado",
        ]
        
        if self.state.is_hovering:
            lines.append(">>> HOVER ACTIVO <<<")
        if self.state.emergency_stop:
            lines.append("!!! EMERGENCIA !!!")
        
        y = 10
        for line in lines:
            color = (255, 255, 255)
            if "EMERGENCIA" in line:
                color = (255, 50, 50)
            elif "HOVER" in line:
                color = (255, 255, 50)
            elif "Gesto:" in line:
                color = (50, 255, 50)
            
            text = font.render(line, True, color)
            hud.blit(text, (10, y))
            y += 22
        
        # Controles
        controls = "W/S:Pitch A/D:Roll Q/E:Yaw SPACE/SHIFT:Throttle H:Hover X:Emergency"
        ctrl_text = small_font.render(controls, True, (150, 150, 150))
        hud.blit(ctrl_text, (10, y + 10))
        
        return hud
    
    def run(self, gesture_mode: bool = False):
        """Bucle principal del simulador."""
        self.gesture_mode = gesture_mode
        
        print("\n" + "="*50)
        print("DRONE SIMULATOR")
        print("="*50)
        if gesture_mode:
            print("Modo: Control por gestos")
        else:
            print("Modo: Control por teclado")
            print("Controles:")
            print("  W/S: Pitch (adelante/atrás)")
            print("  A/D: Roll (izquierda/derecha)")
            print("  Q/E: Yaw (rotación)")
            print("  ESPACIO/SHIFT: Throttle (subir/bajar)")
            print("  H: Hover")
            print("  X: Emergencia")
            print("  R: Reset")
            print("  ESC: Salir")
        print("="*50 + "\n")
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time en segundos
            
            # Eventos
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_r:
                        # Reset
                        self.state = DroneState()
                        self.current_command = {
                            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                            'throttle': 0.0, 'hover': False, 'emergency': False
                        }
            
            # Procesar entrada
            if gesture_mode:
                self._process_gesture_commands()
            else:
                self._process_keyboard()
            
            # Actualizar física
            self.state = self.physics.update(self.state, self.current_command, dt)
            
            # Renderizar
            if self.use_3d:
                self.renderer.render(self.state)
                
                # Overlay HUD con pygame
                # (Nota: mezclar pygame y OpenGL requiere cuidado adicional)
                pygame.display.flip()
            else:
                self.renderer.render(self.state)
                
                # HUD
                hud = self._draw_pygame_hud()
                self.screen.blit(hud, (10, 10))
                
                pygame.display.flip()
        
        pygame.quit()
    
    def get_state(self) -> DroneState:
        """Retorna el estado actual del dron."""
        return self.state


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulador 3D de Dron")
    parser.add_argument('--mode', type=str, default='standalone',
                        choices=['standalone', 'integrated'],
                        help='Modo de ejecución')
    parser.add_argument('--2d', action='store_true', dest='use_2d',
                        help='Usar renderizado 2D')
    
    args = parser.parse_args()
    
    simulator = DroneSimulator(use_3d=not args.use_2d)
    
    if args.mode == 'standalone':
        simulator.run(gesture_mode=False)
    else:
        # Modo integrado: lanzar junto con sistema de inferencia
        simulator.run(gesture_mode=True)


if __name__ == "__main__":
    main()
