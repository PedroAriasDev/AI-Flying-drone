"""
Script de entrenamiento para red Temporal GRU/LSTM.

Esta red analiza secuencias de frames para:
- Clasificar gestos con contexto temporal
- Predecir intensidad del movimiento
- Suavizar predicciones

Uso:
    python train_temporal.py [--epochs N] [--batch_size N] [--lr LR]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

from config import TRAINING_CONFIG, TEMPORAL_CONFIG, GESTURE_CLASSES, NUM_CLASSES
from datasets import get_dataloaders, TemporalSequenceDataset
from models.temporal import GestureSequenceModel, get_temporal_model
from training_utils import (
    MetricsTracker, TrainingVisualizer, CheckpointManager,
    EarlyStopping, calculate_accuracy
)


class TemporalLoss(nn.Module):
    """
    Pérdida combinada para clasificación e intensidad.
    """
    
    def __init__(self, classification_weight: float = 1.0,
                 intensity_weight: float = 0.3):
        super().__init__()
        self.classification_weight = classification_weight
        self.intensity_weight = intensity_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, logits: torch.Tensor, intensity_pred: torch.Tensor,
                labels: torch.Tensor, intensity_target: torch.Tensor = None) -> torch.Tensor:
        # Pérdida de clasificación
        cls_loss = self.ce_loss(logits, labels)
        
        # Pérdida de intensidad (si hay target)
        if intensity_target is not None:
            int_loss = self.mse_loss(intensity_pred, intensity_target)
        else:
            # Si no hay target de intensidad, usar pseudo-label basado en varianza
            int_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = (self.classification_weight * cls_loss + 
                     self.intensity_weight * int_loss)
        
        return total_loss, cls_loss, int_loss


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: TemporalLoss, optimizer: optim.Optimizer,
                device: str) -> tuple:
    """Entrena una época."""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_correct = 0
    running_total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for frames, landmarks, labels in pbar:
        frames = frames.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, intensity = model(frames, landmarks)
        
        # Calcular pérdida (sin target de intensidad por ahora)
        total_loss, cls_loss, _ = criterion(logits, intensity, labels)
        
        total_loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Métricas
        _, predicted = torch.max(logits, 1)
        running_loss += total_loss.item() * frames.size(0)
        running_cls_loss += cls_loss.item() * frames.size(0)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        
        batch_acc = (predicted == labels).float().mean().item()
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    epoch_loss = running_loss / running_total
    epoch_cls_loss = running_cls_loss / running_total
    epoch_acc = running_correct / running_total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: TemporalLoss, device: str,
             return_predictions: bool = False) -> tuple:
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    all_predictions = []
    all_labels = []
    all_intensities = []
    
    with torch.no_grad():
        for frames, landmarks, labels in tqdm(dataloader, desc="Validating"):
            frames = frames.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            logits, intensity = model(frames, landmarks)
            total_loss, _, _ = criterion(logits, intensity, labels)
            
            _, predicted = torch.max(logits, 1)
            running_loss += total_loss.item() * frames.size(0)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_intensities.extend(intensity.cpu().numpy().tolist())
    
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_predictions, all_labels, all_intensities
    
    return epoch_loss, epoch_acc


def train_temporal(epochs: int = None, batch_size: int = None,
                   learning_rate: float = None, device: str = None,
                   freeze_cnn: bool = True, resume: bool = False):
    """
    Función principal de entrenamiento de la red temporal.
    
    Args:
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        device: Dispositivo (cuda/cpu)
        freeze_cnn: Si congelar el backbone CNN
        resume: Si continuar desde checkpoint
    """
    # Configuración
    epochs = epochs or TRAINING_CONFIG["temp_epochs"]
    batch_size = batch_size or TRAINING_CONFIG["temp_batch_size"]
    learning_rate = learning_rate or TRAINING_CONFIG["temp_lr"]
    device = device or TRAINING_CONFIG["device"]
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        device = "cpu"
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO DE RED TEMPORAL (GRU)")
    print(f"{'='*60}")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {TEMPORAL_CONFIG['sequence_length']}")
    print(f"Hidden size: {TEMPORAL_CONFIG['hidden_size']}")
    print(f"Num layers: {TEMPORAL_CONFIG['num_layers']}")
    print(f"Freeze CNN: {freeze_cnn}")
    print(f"{'='*60}\n")
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type='temporal',
        batch_size=batch_size
    )
    
    # Crear modelo
    model = get_temporal_model(pretrained=True, freeze_cnn=freeze_cnn)
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")
    
    # Criterio y optimizador
    criterion = TemporalLoss(classification_weight=1.0, intensity_weight=0.3)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG["temp_weight_decay"]
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Utilidades
    experiment_name = "temporal_gru"
    metrics = MetricsTracker(experiment_name)
    checkpoint_manager = CheckpointManager(model_name="temporal_gru")
    early_stopping = EarlyStopping(patience=TRAINING_CONFIG["patience"], mode='max')
    
    start_epoch = 0
    unfreeze_epoch = epochs // 2  # Descongelar CNN después de la mitad
    
    # Cargar checkpoint si se especifica
    if resume:
        start_epoch, _ = checkpoint_manager.load(model, optimizer, load_best=False)
    
    # Entrenamiento
    print("Iniciando entrenamiento...\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"Época {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Descongelar CNN gradualmente
        if freeze_cnn and epoch == unfreeze_epoch:
            print("  *** Descongelando CNN backbone ***")
            model.unfreeze_cnn()
            
            # Actualizar optimizador
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate * 0.1,
                weight_decay=TRAINING_CONFIG["temp_weight_decay"]
            )
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Step scheduler (OneCycleLR necesita step por batch, pero simplificamos)
        # scheduler.step()
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Actualizar métricas
        metrics.update_train(train_loss, train_acc)
        metrics.update_val(val_loss, val_acc, epoch)
        
        # Guardar métricas adicionales
        metrics.add_epoch_metrics({
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
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
    
    test_loss, test_acc, predictions, labels, intensities = validate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    print(f"Intensidad promedio predicha: {np.mean(intensities):.4f}")
    
    metrics.update_test(test_loss, test_acc, predictions, labels)
    
    # Visualizar resultados
    visualizer = TrainingVisualizer()
    visualizer.generate_full_report(metrics, experiment_name)
    
    print("\n¡Entrenamiento completado!")
    print(f"Mejor accuracy de validación: {metrics.best_val_acc:.4f} (época {metrics.best_epoch + 1})")
    print(f"Accuracy en test: {test_acc:.4f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrenar red temporal")
    parser.add_argument('--epochs', type=int, default=None, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=None, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Dispositivo (cuda/cpu)')
    parser.add_argument('--unfreeze_cnn', action='store_true', 
                        help='No congelar el backbone CNN')
    parser.add_argument('--resume', action='store_true', help='Continuar desde checkpoint')
    
    args = parser.parse_args()
    
    train_temporal(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        freeze_cnn=not args.unfreeze_cnn,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
