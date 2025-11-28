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

from config import TRAINING_CONFIG, CLASSIFIER_CONFIG, GESTURE_CLASSES, NUM_CLASSES
from datasets import get_dataloaders, GestureClassificationDataset
from models.classifier import GestureClassifier, get_classifier
from training_utils import (
    MetricsTracker, TrainingVisualizer, CheckpointManager,
    EarlyStopping, calculate_accuracy
)


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
    
    if freeze_backbone:
        model._freeze_backbone()
    
    model = model.to(device)
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Utilidades
    experiment_name = f"classifier_{model_name}"
    metrics = MetricsTracker(experiment_name)
    checkpoint_manager = CheckpointManager(model_name=f"classifier_{model_name}")
    early_stopping = EarlyStopping(patience=TRAINING_CONFIG["patience"], mode='max')
    
    start_epoch = 0
    unfreeze_epoch = epochs // 3  # Descongelar después de 1/3 del entrenamiento
    
    # Cargar checkpoint si se especifica
    if resume:
        start_epoch, _ = checkpoint_manager.load(model, optimizer, load_best=False)
    
    # Entrenamiento
    print("Iniciando entrenamiento...\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"Época {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Descongelar backbone gradualmente
        if freeze_backbone and epoch == unfreeze_epoch:
            print("  *** Descongelando backbone ***")
            model.unfreeze_backbone()
            
            # Actualizar optimizador con todos los parámetros
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate * 0.1,  # LR más bajo para fine-tuning
                weight_decay=TRAINING_CONFIG["cls_weight_decay"]
            )
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
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
    
    print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    metrics.update_test(test_loss, test_acc, predictions, labels)
    
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
