"""
Script de entrenamiento comparativo para múltiples modelos clasificadores.

Entrena 4 modelos diferentes (ResNet18, ResNet34, MobileNetV3-Large, MobileNetV3-Small)
y compara sus métricas de rendimiento para seleccionar el mejor.

Uso:
    python train_classifier_compare.py
    python train_classifier_compare.py --quick  # Solo 10 épocas para prueba rápida
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import TRAINING_CONFIG, CLASSIFIER_CONFIG, GESTURE_CLASSES, NUM_CLASSES
from datasets import get_dataloaders
from models.classifier import get_classifier
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

    def calculate_overfitting_score(self):
        """
        Calcula un score de overfitting basado en la diferencia entre train y val.
        Menor score = menos overfitting
        """
        if not self.epoch_train_acc or not self.epoch_val_acc:
            return float('inf')

        # Usar las últimas 10 épocas para el cálculo
        recent_train = np.mean(self.epoch_train_acc[-10:])
        recent_val = np.mean(self.epoch_val_acc[-10:])

        # Score de overfitting: diferencia entre train y val
        overfitting_score = recent_train - recent_val

        return overfitting_score

    def get_performance_score(self):
        """
        Calcula un score de performance combinando val_acc y test_acc.
        Mayor score = mejor performance
        """
        # Combinación de mejor val accuracy y test accuracy
        score = (self.best_val_acc * 0.4) + (self.test_acc * 0.6)
        return score

    def get_full_summary(self):
        """Retorna resumen completo incluyendo overfitting y performance."""
        summary = self.get_summary()
        summary['overfitting_score'] = self.calculate_overfitting_score()
        summary['performance_score'] = self.get_performance_score()
        return summary


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


def train_single_model(model_name: str, epochs: int, batch_size: int,
                       learning_rate: float, device: str,
                       train_loader, val_loader, test_loader):
    """
    Entrena un único modelo y retorna sus métricas.
    """
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {model_name.upper()}")
    print(f"{'='*70}")

    # Crear modelo
    model = get_classifier(model_name=model_name, pretrained=True)
    model = model.to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")

    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG["cls_weight_decay"]
    )

    # Learning rate scheduler con warmup y cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=TRAINING_CONFIG["cls_lr_min"]
    )

    # Utilidades
    experiment_name = f"classifier_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = EnhancedMetricsTracker(experiment_name)
    checkpoint_manager = CheckpointManager(model_name=f"classifier_{model_name}")
    early_stopping = EarlyStopping(patience=TRAINING_CONFIG["patience"], mode='max')

    # Entrenamiento
    print("Iniciando entrenamiento...\n")

    for epoch in range(epochs):
        print(f"Época {epoch + 1}/{epochs}")
        print("-" * 50)

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

        # Registrar métricas
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

        print(f"     LR: {current_lr:.6f}")

        # Actualizar scheduler
        scheduler.step()

        # Guardar checkpoint
        is_best = val_acc >= metrics.best_val_acc
        checkpoint_manager.save(model, optimizer, epoch, metrics, is_best)

        # Limpiar checkpoints antiguos
        if epoch % 10 == 0:
            checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping en época {epoch + 1}")
            break

        print()

    # Evaluación final en test
    print("\n" + "="*50)
    print("EVALUACIÓN EN TEST SET")
    print("="*50)

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

    # Guardar métricas
    metrics.save()

    return metrics


def compare_models(all_metrics: dict, save_dir: Path):
    """
    Compara los resultados de todos los modelos y genera visualizaciones.
    """
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70 + "\n")

    # Tabla de resultados
    print(f"{'Modelo':<20} {'Best Val Acc':<15} {'Test Acc':<15} {'Overfitting':<15} {'Performance':<15}")
    print("-" * 80)

    results = []
    for model_name, metrics in all_metrics.items():
        summary = metrics.get_full_summary()
        results.append({
            'model': model_name,
            'best_val_acc': summary['best_val_acc'],
            'test_acc': summary['test_acc'],
            'overfitting_score': summary['overfitting_score'],
            'performance_score': summary['performance_score']
        })

        print(f"{model_name:<20} {summary['best_val_acc']:<15.4f} {summary['test_acc']:<15.4f} "
              f"{summary['overfitting_score']:<15.4f} {summary['performance_score']:<15.4f}")

    # Determinar el mejor modelo
    print("\n" + "="*70)
    print("SELECCIÓN DEL MEJOR MODELO")
    print("="*70)

    # Ordenar por performance score
    results_sorted = sorted(results, key=lambda x: x['performance_score'], reverse=True)
    best_model = results_sorted[0]

    print(f"\n🏆 MEJOR MODELO: {best_model['model'].upper()}")
    print(f"   - Test Accuracy: {best_model['test_acc']:.4f}")
    print(f"   - Overfitting Score: {best_model['overfitting_score']:.4f} (menor es mejor)")
    print(f"   - Performance Score: {best_model['performance_score']:.4f}")

    # Encontrar modelo con menor overfitting
    least_overfit = min(results, key=lambda x: x['overfitting_score'])
    if least_overfit['model'] != best_model['model']:
        print(f"\n📊 MODELO CON MENOR OVERFITTING: {least_overfit['model'].upper()}")
        print(f"   - Overfitting Score: {least_overfit['overfitting_score']:.4f}")

    # Guardar resultados en JSON
    results_path = save_dir / "model_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'comparison': results,
            'best_model': best_model,
            'least_overfit_model': least_overfit
        }, f, indent=2)

    print(f"\n✅ Resultados guardados en: {results_path}")

    # Generar gráficos comparativos
    generate_comparison_plots(all_metrics, save_dir)

    return best_model


def generate_comparison_plots(all_metrics: dict, save_dir: Path):
    """Genera gráficos comparativos de todos los modelos."""

    # 1. Curvas de training/validation accuracy
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación de Modelos - Curvas de Entrenamiento', fontsize=16, fontweight='bold')

    for idx, (model_name, metrics) in enumerate(all_metrics.items()):
        ax = axes[idx // 2, idx % 2]
        epochs = range(1, len(metrics.train_acc) + 1)

        ax.plot(epochs, metrics.train_acc, 'b-', label='Train Acc', linewidth=2, alpha=0.7)
        ax.plot(epochs, metrics.val_acc, 'r-', label='Val Acc', linewidth=2, alpha=0.7)
        ax.axhline(y=metrics.test_acc, color='g', linestyle='--',
                   label=f'Test Acc ({metrics.test_acc:.4f})', linewidth=2)
        ax.axvline(x=metrics.best_epoch + 1, color='orange', linestyle=':',
                   label=f'Best Epoch ({metrics.best_epoch + 1})', alpha=0.7)

        ax.set_xlabel('Época', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: comparison_training_curves.png")

    # 2. Comparación de métricas finales
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = list(all_metrics.keys())
    best_val_accs = [m.best_val_acc for m in all_metrics.values()]
    test_accs = [m.test_acc for m in all_metrics.values()]
    overfitting_scores = [m.calculate_overfitting_score() for m in all_metrics.values()]

    # Best Val Accuracy
    axes[0].bar(model_names, best_val_accs, color='#3498db')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Best Validation Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(best_val_accs):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

    # Test Accuracy
    axes[1].bar(model_names, test_accs, color='#2ecc71')
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Test Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(test_accs):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

    # Overfitting Score
    axes[2].bar(model_names, overfitting_scores, color='#e74c3c')
    axes[2].set_ylabel('Overfitting Score (menor es mejor)', fontsize=12)
    axes[2].set_title('Overfitting Score', fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(overfitting_scores):
        axes[2].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_final_metrics.png', dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: comparison_final_metrics.png")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Entrenar y comparar múltiples clasificadores")
    parser.add_argument('--quick', action='store_true', help='Modo rápido (solo 10 épocas)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Modelos específicos a entrenar (por defecto: todos)')

    args = parser.parse_args()

    # Configuración
    epochs = 10 if args.quick else TRAINING_CONFIG["cls_epochs"]
    batch_size = TRAINING_CONFIG["cls_batch_size"]
    learning_rate = TRAINING_CONFIG["cls_lr"]
    device = TRAINING_CONFIG["device"]

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA no disponible, usando CPU")
        device = "cpu"

    # Modelos a entrenar
    models_to_train = args.models or TRAINING_CONFIG["classifier_models"]

    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPARATIVO DE CLASIFICADORES")
    print(f"{'='*70}")
    print(f"Modelos: {', '.join(models_to_train)}")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*70}\n")

    # Crear dataloaders (compartidos entre todos los modelos)
    print("Cargando datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type='classification',
        batch_size=batch_size
    )

    # Entrenar cada modelo
    all_metrics = {}

    for model_name in models_to_train:
        try:
            metrics = train_single_model(
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader
            )
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"\n❌ Error entrenando {model_name}: {e}")
            continue

    # Comparar resultados
    if len(all_metrics) > 0:
        from config import RESULTS_DIR
        best_model = compare_models(all_metrics, RESULTS_DIR)

        print("\n" + "="*70)
        print("¡ENTRENAMIENTO COMPARATIVO COMPLETADO!")
        print("="*70)
    else:
        print("\n❌ No se pudo entrenar ningún modelo")


if __name__ == "__main__":
    main()
