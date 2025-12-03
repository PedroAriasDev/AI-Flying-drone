"""
Script de entrenamiento para red Clasificadora CNN.

Uso:
    python train_classifier.py [--epochs N] [--batch_size N] [--lr LR]
    python train_classifier.py --model mobilenetv3_large --freeze_backbone
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from config import TRAINING_CONFIG, CLASSIFIER_CONFIG, GESTURE_CLASSES, NUM_CLASSES, RESULTS_DIR
from datasets import get_dataloaders, GestureClassificationDataset
from models.classifier import GestureClassifier, get_classifier
from training_utils import (
    MetricsTracker, TrainingVisualizer, CheckpointManager,
    EarlyStopping, calculate_accuracy
)


class EnhancedMetricsTracker(MetricsTracker):
    """Tracker de métricas mejorado que registra train/val/test por época."""

    def __init__(self, experiment_name: str = None):
        super().__init__(experiment_name)

        # Métricas detalladas por época
        self.epoch_train_loss = []
        self.epoch_train_acc = []
        self.epoch_val_loss = []
        self.epoch_val_acc = []
        self.learning_rates = []

    def record_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Registra todas las métricas de una época."""
        self.epoch_train_loss.append(train_loss)
        self.epoch_train_acc.append(train_acc)
        self.epoch_val_loss.append(val_loss)
        self.epoch_val_acc.append(val_acc)
        self.learning_rates.append(lr)

        self.update_train(train_loss, train_acc)
        self.update_val(val_loss, val_acc, epoch)


def freeze_backbone(model, model_name: str):
    """
    Congela el backbone del modelo y deja solo las últimas capas para entrenar.
    Esto acelera el entrenamiento significativamente.

    Args:
        model: Modelo a modificar
        model_name: Nombre del modelo (resnet18, mobilenetv3_large, etc.)
    """
    print("🔒 Congelando backbone (transfer learning)...")

    # Congelar todos los parámetros primero
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar solo las últimas capas según el modelo
    if 'resnet' in model_name.lower():
        # Para ResNet: descongelar solo la capa fc (fully connected)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif 'mobilenet' in model_name.lower():
        # Para MobileNet: descongelar solo el classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Default: intentar descongelar fc o classifier
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True

    # Contar parámetros congelados vs entrenables
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"   ✅ Parámetros congelados: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"   ✅ Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   🚀 Reducción en parámetros: {frozen_params/total_params*100:.1f}%\n")


def generate_evolution_plots(metrics: EnhancedMetricsTracker, model_name: str, save_dir: Path):
    """
    Genera gráficos de evolución de train/val/test para un modelo específico.

    Args:
        metrics: Objeto EnhancedMetricsTracker con las métricas del modelo
        model_name: Nombre del modelo
        save_dir: Directorio donde guardar los gráficos
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Evolución del Entrenamiento - {model_name.upper()}',
                 fontsize=16, fontweight='bold')

    epochs = range(1, len(metrics.epoch_train_loss) + 1)

    # 1. Loss Evolution
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics.epoch_train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, metrics.epoch_val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Evolución de Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy Evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc*100 for acc in metrics.epoch_train_acc], 'b-',
             label='Train Acc', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, [acc*100 for acc in metrics.epoch_val_acc], 'r-',
             label='Val Acc', linewidth=2, marker='s', markersize=3)
    # Línea horizontal para test accuracy
    ax2.axhline(y=metrics.test_acc*100, color='g', linestyle='--',
                label=f'Test Acc ({metrics.test_acc*100:.2f}%)', linewidth=2)
    # Línea vertical en mejor época
    ax2.axvline(x=metrics.best_epoch + 1, color='orange', linestyle=':',
                label=f'Best Epoch ({metrics.best_epoch + 1})', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Evolución de Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # 3. Overfitting Tracking (Train - Val Accuracy)
    ax3 = axes[1, 0]
    overfitting_gap = [(train - val)*100 for train, val in
                       zip(metrics.epoch_train_acc, metrics.epoch_val_acc)]
    ax3.plot(epochs, overfitting_gap, 'purple', linewidth=2, marker='d', markersize=3)
    ax3.axhline(y=5, color='orange', linestyle='--', label='Ligero overfitting (5%)', alpha=0.7)
    ax3.axhline(y=10, color='red', linestyle='--', label='Overfitting alto (10%)', alpha=0.7)
    ax3.fill_between(epochs, 0, overfitting_gap, alpha=0.3, color='purple')
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Diferencia Train-Val (%)', fontsize=12)
    ax3.set_title('Tracking de Overfitting', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Learning Rate Schedule
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics.learning_rates, 'darkgreen', linewidth=2, marker='^', markersize=3)
    ax4.set_xlabel('Época', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Guardar gráfico
    plot_filename = f'evolution_{model_name}.png'
    plot_path = save_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Gráfico de evolución guardado: {plot_filename}")
    plt.close(fig)


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> tuple:
    """Entrena una época."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Soportar datasets con y sin landmarks
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, landmarks = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Métricas
        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        
        batch_acc = (predicted == labels).float().mean().item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: str,
             return_predictions: bool = False) -> tuple:
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, landmarks = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
    
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    return epoch_loss, epoch_acc


def train_classifier(epochs: int = None, batch_size: int = None,
                     learning_rate: float = None, device: str = None,
                     model_name: str = None, freeze_backbone: bool = False,
                     resume: bool = False):
    """
    Función principal de entrenamiento del clasificador.
    
    Args:
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        device: Dispositivo (cuda/cpu)
        model_name: Nombre del backbone (resnet18, mobilenetv3_large, etc.)
        freeze_backbone: Si congelar el backbone inicialmente
        resume: Si continuar desde checkpoint
    """
    # Configuración
    epochs = epochs or TRAINING_CONFIG["cls_epochs"]
    batch_size = batch_size or TRAINING_CONFIG["cls_batch_size"]
    learning_rate = learning_rate or TRAINING_CONFIG["cls_lr"]
    device = device or TRAINING_CONFIG["device"]
    model_name = model_name or CLASSIFIER_CONFIG["model_name"]
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        device = "cpu"
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO DE RED CLASIFICADORA CNN")
    print(f"{'='*60}")
    print(f"Modelo: {model_name}")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Clases: {NUM_CLASSES}")
    print(f"{'='*60}\n")
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type='classification',
        batch_size=batch_size
    )
    
    # Crear modelo
    model = get_classifier(model_name=model_name, pretrained=True)
    model = model.to(device)

    # Congelar backbone si se especifica
    if freeze_backbone:
        freeze_backbone(model, model_name)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")

    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG["cls_weight_decay"]
    )
    # Usar CosineAnnealingLR como en train_classifier_compare.py
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=TRAINING_CONFIG["cls_lr_min"]
    )

    # Utilidades - usar EnhancedMetricsTracker para los gráficos
    experiment_name = f"classifier_{model_name}"
    metrics = EnhancedMetricsTracker(experiment_name)
    checkpoint_manager = CheckpointManager(model_name=f"classifier_{model_name}")
    early_stopping = EarlyStopping(patience=TRAINING_CONFIG["patience"], mode='max')
    
    start_epoch = 0

    # Cargar checkpoint si se especifica
    if resume:
        start_epoch, _ = checkpoint_manager.load(model, optimizer, load_best=False)

    # Entrenamiento
    print("Iniciando entrenamiento...\n")

    for epoch in range(start_epoch, epochs):
        print(f"Época {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Learning rate actual
        current_lr = optimizer.param_groups[0]['lr']

        # Registrar métricas completas de la época
        metrics.record_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        # Visualización mejorada de accuracy
        print(f"  📊 RESULTADOS DE LA ÉPOCA:")
        print(f"     Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"     Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        # Indicar si es el mejor modelo hasta ahora
        if val_acc >= metrics.best_val_acc:
            print(f"     ⭐ NUEVO MEJOR MODELO! (Anterior: {metrics.best_val_acc*100:.2f}%)")

        # Mostrar diferencia train-val (overfitting indicator)
        acc_diff = train_acc - val_acc
        if acc_diff > 0.1:
            print(f"     ⚠️  Overfitting detectado: {acc_diff*100:.2f}% diferencia")
        elif acc_diff > 0.05:
            print(f"     ⚡ Ligero overfitting: {acc_diff*100:.2f}% diferencia")
        else:
            print(f"     ✅ Buen balance: {acc_diff*100:.2f}% diferencia")

        print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduler
        scheduler.step()
        
        # Guardar checkpoint
        is_best = val_acc >= metrics.best_val_acc
        checkpoint_manager.save(model, optimizer, epoch, metrics, is_best)
        
        # Limpiar checkpoints antiguos
        if epoch % 5 == 0:
            checkpoint_manager.cleanup_old_checkpoints(keep_last=3)
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping en época {epoch + 1}")
            break
        
        print()
    
    # Evaluación final en test
    print("\n" + "="*60)
    print("EVALUACIÓN EN TEST SET")
    print("="*60)
    
    # Cargar mejor modelo
    checkpoint_manager.load(model, load_best=True)
    
    test_loss, test_acc, predictions, labels = validate(
        model, test_loader, criterion, device, return_predictions=True
    )

    # Visualización mejorada del test accuracy
    print(f"\n🎯 RESULTADOS FINALES EN TEST SET:")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n📈 RESUMEN DEL MODELO:")
    print(f"   Mejor Val Acc:  {metrics.best_val_acc*100:.2f}% (época {metrics.best_epoch + 1})")
    print(f"   Test Accuracy:  {test_acc*100:.2f}%")
    print(f"   Diferencia:     {(metrics.best_val_acc - test_acc)*100:.2f}%")

    # Indicador de performance
    if test_acc >= 0.95:
        print(f"   ⭐⭐⭐ EXCELENTE MODELO!")
    elif test_acc >= 0.90:
        print(f"   ⭐⭐ BUEN MODELO")
    elif test_acc >= 0.85:
        print(f"   ⭐ MODELO ACEPTABLE")
    else:
        print(f"   ⚠️  MODELO NECESITA MEJORA")

    metrics.update_test(test_loss, test_acc, predictions, labels)

    # Generar gráficos de evolución
    generate_evolution_plots(metrics, model_name, RESULTS_DIR)

    # Visualizar resultados
    visualizer = TrainingVisualizer()
    visualizer.generate_full_report(metrics, experiment_name)

    print("\n¡Entrenamiento completado!")
    print(f"Mejor accuracy de validación: {metrics.best_val_acc:.4f} (época {metrics.best_epoch + 1})")
    print(f"Accuracy en test: {test_acc:.4f}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrenar red clasificadora")
    parser.add_argument('--epochs', type=int, default=None, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=None, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Dispositivo (cuda/cpu)')
    parser.add_argument('--model', type=str, default=None, 
                        choices=['resnet18', 'resnet34', 'mobilenetv3_large', 'mobilenetv3_small'],
                        help='Modelo backbone')
    parser.add_argument('--freeze_backbone', action='store_true', 
                        help='Congelar backbone inicialmente')
    parser.add_argument('--resume', action='store_true', help='Continuar desde checkpoint')
    
    args = parser.parse_args()
    
    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
