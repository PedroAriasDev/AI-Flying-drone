"""
Script para visualizar las arquitecturas de las redes neuronales del proyecto.

Genera diagramas y resúmenes de:
- Modelos clasificadores (ResNet18, ResNet34, MobileNetV3-Large, MobileNetV3-Small)
- Red temporal GRU
- Red de segmentación UNet

Uso:
    python visualize_architectures.py
    python visualize_architectures.py --model resnet18  # Solo un modelo específico
"""

import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from config import CLASSIFIER_CONFIG, TEMPORAL_CONFIG, RESULTS_DIR, NUM_CLASSES
from models.classifier import GestureClassifier
from models.temporal import TemporalGRU, GestureSequenceModel
from models.segmentation import UNet


def print_model_summary(model: nn.Module, input_size: tuple, model_name: str):
    """Imprime resumen detallado del modelo."""
    print(f"\n{'='*80}")
    print(f"ARQUITECTURA: {model_name.upper()}")
    print(f"{'='*80}\n")

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 ESTADÍSTICAS DEL MODELO")
    print(f"{'─'*80}")
    print(f"Parámetros totales:      {total_params:,}")
    print(f"Parámetros entrenables:  {trainable_params:,}")
    print(f"Parámetros congelados:   {total_params - trainable_params:,}")
    print(f"Tamaño en MB (~):        {total_params * 4 / (1024**2):.2f} MB")
    print()

    # Intentar mostrar summary
    try:
        print(f"📋 RESUMEN DE CAPAS")
        print(f"{'─'*80}")
        summary(model, input_size, device='cpu')
    except Exception as e:
        print(f"⚠️  No se pudo generar summary detallado: {e}")
        print("\nEstructura del modelo:")
        print(model)

    print(f"\n{'='*80}\n")


def visualize_classifier_architecture(model_name: str, save_dir: Path):
    """Visualiza la arquitectura de un clasificador CNN."""
    print(f"\n🔍 Visualizando arquitectura: {model_name}")

    # Crear modelo
    model = GestureClassifier(model_name=model_name, pretrained=False)

    # Input size para el modelo
    input_size = CLASSIFIER_CONFIG["input_size"]

    # Imprimir resumen
    print_model_summary(model, (3, *input_size), model_name)

    # Crear diagrama visual
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # Título
    fig.suptitle(f'Arquitectura del Clasificador: {model_name.upper()}',
                 fontsize=18, fontweight='bold', y=0.98)

    # Coordenadas
    y_start = 0.9
    x_center = 0.5
    box_width = 0.3
    box_height = 0.08
    spacing = 0.12

    # Función para dibujar caja
    def draw_box(ax, text, y_pos, color='lightblue', is_input=False, is_output=False):
        if is_input:
            color = '#90EE90'  # Verde claro
        elif is_output:
            color = '#FFB6C1'  # Rosa claro

        box = FancyBboxPatch(
            (x_center - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)

        ax.text(x_center, y_pos, text,
               ha='center', va='center',
               fontsize=11, fontweight='bold')

    def draw_arrow(ax, y_from, y_to):
        arrow = FancyArrowPatch(
            (x_center, y_from - box_height/2),
            (x_center, y_to + box_height/2),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='black',
            linewidth=2
        )
        ax.add_patch(arrow)

    # Dibujar arquitectura
    current_y = y_start

    # Input
    draw_box(ax, f'INPUT\n{input_size[0]}x{input_size[1]}x3', current_y, is_input=True)
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # Backbone
    if 'resnet' in model_name:
        draw_box(ax, f'{model_name.upper()} Backbone\n(Pre-trained on ImageNet)', current_y, color='#87CEEB')
        feature_dim = 512 if 'resnet18' in model_name or 'resnet34' in model_name else 2048
    elif 'mobilenet' in model_name:
        draw_box(ax, f'{model_name.upper()} Backbone\n(Pre-trained on ImageNet)', current_y, color='#87CEEB')
        feature_dim = 1280 if 'large' in model_name else 576

    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # Global Average Pooling (implícito en backbone)
    draw_box(ax, f'Features\n{feature_dim}D', current_y, color='#DDA0DD')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # Dropout
    draw_box(ax, f'Dropout ({CLASSIFIER_CONFIG["dropout"]})', current_y, color='#F0E68C')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # FC 1
    draw_box(ax, f'Linear\n{feature_dim} → 256', current_y, color='#FFD700')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # ReLU
    draw_box(ax, 'ReLU', current_y, color='#F0E68C')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # Dropout 2
    draw_box(ax, f'Dropout ({CLASSIFIER_CONFIG["dropout"]/2})', current_y, color='#F0E68C')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # FC 2 (Output)
    draw_box(ax, f'Linear\n256 → {NUM_CLASSES}', current_y, color='#FFD700')
    current_y -= spacing
    draw_arrow(ax, current_y + spacing, current_y)

    # Output
    draw_box(ax, f'OUTPUT\nLogits ({NUM_CLASSES} clases)', current_y, is_output=True)

    # Info adicional
    total_params = sum(p.numel() for p in model.parameters())
    ax.text(0.02, 0.02, f'Parámetros totales: {total_params:,}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = save_dir / f'architecture_{model_name}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Diagrama guardado: {save_path}")
    plt.close()


def visualize_temporal_architecture(save_dir: Path):
    """Visualiza la arquitectura de la red temporal GRU."""
    print(f"\n🔍 Visualizando arquitectura: Red Temporal GRU")

    # Crear modelo
    model = GestureSequenceModel(pretrained=False, freeze_cnn=True)

    # Input size
    seq_len = TEMPORAL_CONFIG["sequence_length"]
    input_size = CLASSIFIER_CONFIG["input_size"]

    # Imprimir resumen
    print(f"\n{'='*80}")
    print("ARQUITECTURA: RED TEMPORAL GRU")
    print(f"{'='*80}\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 ESTADÍSTICAS DEL MODELO")
    print(f"{'─'*80}")
    print(f"Secuencia de entrada:    {seq_len} frames")
    print(f"Tamaño de frame:         {input_size}")
    print(f"Hidden size GRU:         {TEMPORAL_CONFIG['hidden_size']}")
    print(f"Número de capas GRU:     {TEMPORAL_CONFIG['num_layers']}")
    print(f"Bidireccional:           {TEMPORAL_CONFIG['bidirectional']}")
    print(f"Parámetros totales:      {total_params:,}")
    print(f"Parámetros entrenables:  {trainable_params:,}")
    print(f"{'='*80}\n")

    # Crear diagrama visual
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle('Arquitectura de la Red Temporal GRU (Unidireccional)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Coordenadas
    y_start = 0.9
    spacing_y = 0.11

    # Dibujar secuencia de frames
    num_frames_shown = 5  # Mostrar solo algunos frames para claridad
    frame_width = 0.12
    frame_height = 0.08
    frame_spacing = 0.15
    start_x = 0.5 - (num_frames_shown * frame_spacing) / 2

    current_y = y_start

    # Título de entrada
    ax.text(0.5, current_y + 0.05, f'INPUT: Secuencia de {seq_len} frames',
            ha='center', fontsize=12, fontweight='bold')

    # Dibujar frames
    for i in range(num_frames_shown):
        x_pos = start_x + i * frame_spacing
        frame_box = FancyBboxPatch(
            (x_pos - frame_width/2, current_y - frame_height/2),
            frame_width, frame_height,
            boxstyle="round,pad=0.005",
            edgecolor='black',
            facecolor='#90EE90',
            linewidth=1.5
        )
        ax.add_patch(frame_box)

        if i == num_frames_shown - 1:
            ax.text(x_pos, current_y, f'Frame t\n224x224x3',
                   ha='center', va='center', fontsize=9)
        else:
            ax.text(x_pos, current_y, f'Frame {i}\n224x224x3',
                   ha='center', va='center', fontsize=9)

    current_y -= spacing_y

    # CNN Feature Extraction
    cnn_box_width = 0.7
    cnn_box_height = 0.08
    cnn_box = FancyBboxPatch(
        (0.5 - cnn_box_width/2, current_y - cnn_box_height/2),
        cnn_box_width, cnn_box_height,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#87CEEB',
        linewidth=2
    )
    ax.add_patch(cnn_box)
    ax.text(0.5, current_y, 'CNN Backbone (Frozen)\nFeature Extraction → 512D per frame',
           ha='center', va='center', fontsize=10, fontweight='bold')

    current_y -= spacing_y

    # Features + Landmarks Fusion
    fusion_box = FancyBboxPatch(
        (0.5 - cnn_box_width/2, current_y - cnn_box_height/2),
        cnn_box_width, cnn_box_height,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#DDA0DD',
        linewidth=2
    )
    ax.add_patch(fusion_box)
    ax.text(0.5, current_y, 'Feature Fusion\nCNN Features (512D) + Landmarks (64D) = 576D',
           ha='center', va='center', fontsize=10, fontweight='bold')

    current_y -= spacing_y

    # Input Projection
    proj_box = FancyBboxPatch(
        (0.5 - 0.4, current_y - 0.06),
        0.8, 0.06,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#F0E68C',
        linewidth=2
    )
    ax.add_patch(proj_box)
    ax.text(0.5, current_y, f'Linear Projection\n576D → {TEMPORAL_CONFIG["hidden_size"]}D',
           ha='center', va='center', fontsize=10, fontweight='bold')

    current_y -= spacing_y

    # GRU Layers
    for layer in range(TEMPORAL_CONFIG["num_layers"]):
        gru_box = FancyBboxPatch(
            (0.5 - 0.35, current_y - 0.06),
            0.7, 0.06,
            boxstyle="round,pad=0.01",
            edgecolor='black',
            facecolor='#FFB6C1',
            linewidth=2
        )
        ax.add_patch(gru_box)
        ax.text(0.5, current_y, f'GRU Layer {layer + 1}\nHidden: {TEMPORAL_CONFIG["hidden_size"]}D (Unidirectional)',
               ha='center', va='center', fontsize=10, fontweight='bold')
        current_y -= spacing_y * 0.8

    current_y -= spacing_y * 0.3

    # Attention Mechanism
    att_box = FancyBboxPatch(
        (0.5 - 0.3, current_y - 0.06),
        0.6, 0.06,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#FFDAB9',
        linewidth=2
    )
    ax.add_patch(att_box)
    ax.text(0.5, current_y, 'Attention Mechanism\nWeighted temporal aggregation',
           ha='center', va='center', fontsize=10, fontweight='bold')

    current_y -= spacing_y

    # Branches
    branch_y = current_y
    branch_spacing = 0.35

    # Gesture Classification Branch
    class_x = 0.5 - branch_spacing
    class_box = FancyBboxPatch(
        (class_x - 0.15, branch_y - 0.06),
        0.3, 0.06,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#FFD700',
        linewidth=2
    )
    ax.add_patch(class_box)
    ax.text(class_x, branch_y, f'Gesture Classifier\n{TEMPORAL_CONFIG["hidden_size"]}D → {NUM_CLASSES}',
           ha='center', va='center', fontsize=9, fontweight='bold')

    # Intensity Prediction Branch
    int_x = 0.5 + branch_spacing
    int_box = FancyBboxPatch(
        (int_x - 0.15, branch_y - 0.06),
        0.3, 0.06,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#FFD700',
        linewidth=2
    )
    ax.add_patch(int_box)
    ax.text(int_x, branch_y, f'Intensity Predictor\n{TEMPORAL_CONFIG["hidden_size"]}D → 1',
           ha='center', va='center', fontsize=9, fontweight='bold')

    current_y = branch_y - spacing_y

    # Outputs
    out_class_box = FancyBboxPatch(
        (class_x - 0.12, current_y - 0.05),
        0.24, 0.05,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#98FB98',
        linewidth=2
    )
    ax.add_patch(out_class_box)
    ax.text(class_x, current_y, f'Gesture Class\n({NUM_CLASSES} clases)',
           ha='center', va='center', fontsize=9)

    out_int_box = FancyBboxPatch(
        (int_x - 0.12, current_y - 0.05),
        0.24, 0.05,
        boxstyle="round,pad=0.01",
        edgecolor='black',
        facecolor='#98FB98',
        linewidth=2
    )
    ax.add_patch(out_int_box)
    ax.text(int_x, current_y, 'Intensity\n(0-1)',
           ha='center', va='center', fontsize=9)

    # Info adicional
    info_text = f'''Configuración:
    • Secuencia: {seq_len} frames
    • Hidden size: {TEMPORAL_CONFIG["hidden_size"]}
    • Layers: {TEMPORAL_CONFIG["num_layers"]}
    • Dropout: {TEMPORAL_CONFIG["dropout"]}
    • Parámetros: {total_params:,}
    '''

    ax.text(0.02, 0.1, info_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    save_path = save_dir / 'architecture_temporal_gru.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Diagrama guardado: {save_path}")
    plt.close()


def create_architecture_summary(save_dir: Path):
    """Crea un documento de resumen con todas las arquitecturas."""
    summary_path = save_dir / 'architectures_summary.txt'

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DE ARQUITECTURAS DE REDES NEURONALES\n")
        f.write("Proyecto: Control de Dron con Gestos\n")
        f.write("="*80 + "\n\n")

        # Clasificadores
        f.write("1. REDES CLASIFICADORAS (CNN)\n")
        f.write("-"*80 + "\n\n")

        for model_name in TRAINING_CONFIG.get("classifier_models", ["resnet18", "mobilenetv3_small"]):
            try:
                model = GestureClassifier(model_name=model_name, pretrained=False)
                total_params = sum(p.numel() for p in model.parameters())
                f.write(f"  Modelo: {model_name.upper()}\n")
                f.write(f"  - Parámetros: {total_params:,}\n")
                f.write(f"  - Input: {CLASSIFIER_CONFIG['input_size']}\n")
                f.write(f"  - Output: {NUM_CLASSES} clases\n")
                f.write(f"  - Dropout: {CLASSIFIER_CONFIG['dropout']}\n\n")
            except Exception as e:
                f.write(f"  Error con {model_name}: {e}\n\n")

        # Red Temporal
        f.write("\n2. RED TEMPORAL (GRU UNIDIRECCIONAL)\n")
        f.write("-"*80 + "\n\n")

        try:
            model = GestureSequenceModel(pretrained=False, freeze_cnn=True)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            f.write(f"  Configuración:\n")
            f.write(f"  - Longitud de secuencia: {TEMPORAL_CONFIG['sequence_length']} frames\n")
            f.write(f"  - Hidden size: {TEMPORAL_CONFIG['hidden_size']}\n")
            f.write(f"  - Número de capas: {TEMPORAL_CONFIG['num_layers']}\n")
            f.write(f"  - Bidireccional: {TEMPORAL_CONFIG['bidirectional']}\n")
            f.write(f"  - Dropout: {TEMPORAL_CONFIG['dropout']}\n")
            f.write(f"  - Parámetros totales: {total_params:,}\n")
            f.write(f"  - Parámetros entrenables: {trainable_params:,}\n")
            f.write(f"  - Outputs:\n")
            f.write(f"    • Clasificación de gesto ({NUM_CLASSES} clases)\n")
            f.write(f"    • Intensidad del movimiento (0-1)\n\n")
        except Exception as e:
            f.write(f"  Error: {e}\n\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✅ Resumen de arquitecturas guardado: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualizar arquitecturas de redes neuronales")
    parser.add_argument('--model', type=str, default='all',
                        help='Modelo a visualizar (all, resnet18, resnet34, mobilenetv3_large, mobilenetv3_small, temporal)')

    args = parser.parse_args()

    # Importar configuración necesaria
    from config import TRAINING_CONFIG

    print("\n" + "="*80)
    print("VISUALIZACIÓN DE ARQUITECTURAS DE REDES NEURONALES")
    print("="*80)

    save_dir = RESULTS_DIR / "architectures"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.model == 'all':
        # Visualizar todos los clasificadores
        for model_name in TRAINING_CONFIG.get("classifier_models", ["resnet18", "mobilenetv3_small"]):
            try:
                visualize_classifier_architecture(model_name, save_dir)
            except Exception as e:
                print(f"❌ Error visualizando {model_name}: {e}")

        # Visualizar red temporal
        try:
            visualize_temporal_architecture(save_dir)
        except Exception as e:
            print(f"❌ Error visualizando red temporal: {e}")

    elif args.model == 'temporal':
        visualize_temporal_architecture(save_dir)

    elif args.model in ['resnet18', 'resnet34', 'mobilenetv3_large', 'mobilenetv3_small']:
        visualize_classifier_architecture(args.model, save_dir)

    else:
        print(f"❌ Modelo desconocido: {args.model}")
        return

    # Crear resumen
    create_architecture_summary(save_dir)

    print("\n✅ Visualización completada. Archivos guardados en:", save_dir)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
