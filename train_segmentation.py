"""
Script de entrenamiento para red de Segmentación UNet.

Uso:
    python train_segmentation.py [--epochs N] [--batch_size N] [--lr LR]

Para Google Colab, ejecutar las celdas en train_colab.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

from config import TRAINING_CONFIG, SEGMENTATION_CONFIG, CHECKPOINTS_DIR
from datasets import get_dataloaders, SegmentationDataset
from models.segmentation import SegmentationModel, get_segmentation_model
from training_utils import (
    MetricsTracker, TrainingVisualizer, CheckpointManager, 
    EarlyStopping, calculate_iou, calculate_dice
)


class DiceBCELoss(nn.Module):
    """Combina Dice Loss con BCE Loss para segmentación."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE Loss
        bce_loss = self.bce(logits[:, 1, :, :], targets.squeeze(1))
        
        # Dice Loss
        probs = torch.sigmoid(logits[:, 1, :, :])
        intersection = (probs * targets.squeeze(1)).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.squeeze(1).sum(dim=(1, 2))
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> tuple:
    """Entrena una época."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Métricas
        with torch.no_grad():
            probs = torch.sigmoid(outputs[:, 1, :, :])
            pred_masks = (probs > 0.5).float()
            
            iou = calculate_iou(pred_masks, masks.squeeze(1))
            dice = calculate_dice(pred_masks, masks.squeeze(1))
        
        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'IoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}'
        })
    
    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_dice = running_dice / num_batches
    
    return avg_loss, avg_iou, avg_dice


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: str) -> tuple:
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            probs = torch.sigmoid(outputs[:, 1, :, :])
            pred_masks = (probs > 0.5).float()
            
            iou = calculate_iou(pred_masks, masks.squeeze(1))
            dice = calculate_dice(pred_masks, masks.squeeze(1))
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            num_batches += 1
    
    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_dice = running_dice / num_batches
    
    return avg_loss, avg_iou, avg_dice


def train_segmentation(epochs: int = None, batch_size: int = None,
                       learning_rate: float = None, device: str = None,
                       resume: bool = False):
    """
    Función principal de entrenamiento de segmentación.
    
    Args:
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        device: Dispositivo (cuda/cpu)
        resume: Si continuar desde checkpoint
    """
    # Configuración
    epochs = epochs or TRAINING_CONFIG["seg_epochs"]
    batch_size = batch_size or TRAINING_CONFIG["seg_batch_size"]
    learning_rate = learning_rate or TRAINING_CONFIG["seg_lr"]
    device = device or TRAINING_CONFIG["device"]
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        device = "cpu"
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO DE RED DE SEGMENTACIÓN (UNet)")
    print(f"{'='*60}")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type='segmentation',
        batch_size=batch_size
    )
    
    # Crear modelo
    model = get_segmentation_model(pretrained=True)
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")
    
    # Criterio y optimizador
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                           weight_decay=TRAINING_CONFIG["seg_weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Utilidades
    metrics = MetricsTracker("segmentation")
    checkpoint_manager = CheckpointManager(model_name="segmentation_unet")
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
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )
        
        # Actualizar métricas (usamos IoU como accuracy principal)
        metrics.update_train(train_loss, train_iou)
        metrics.update_val(val_loss, val_iou, epoch)
        
        # Guardar métricas adicionales
        metrics.add_epoch_metrics({
            'epoch': epoch,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_iou)
        
        # Guardar checkpoint
        is_best = val_iou >= metrics.best_val_acc
        checkpoint_manager.save(model, optimizer, epoch, metrics, is_best)
        
        # Early stopping
        if early_stopping(val_iou):
            print(f"\nEarly stopping en época {epoch + 1}")
            break
        
        print()
    
    # Evaluación final en test
    print("\n" + "="*60)
    print("EVALUACIÓN EN TEST SET")
    print("="*60)
    
    # Cargar mejor modelo
    checkpoint_manager.load(model, load_best=True)
    
    test_loss, test_iou, test_dice = validate(model, test_loader, criterion, device)
    
    print(f"Test - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")
    
    metrics.test_loss = test_loss
    metrics.test_acc = test_iou
    
    # Visualizar resultados
    visualizer = TrainingVisualizer()
    visualizer.plot_training_curves(metrics, 
                                    title="Segmentación UNet - Curvas de Entrenamiento",
                                    save_name="segmentation_unet")
    
    # Guardar métricas finales
    metrics.save()
    
    print("\n¡Entrenamiento completado!")
    print(f"Mejor IoU de validación: {metrics.best_val_acc:.4f} (época {metrics.best_epoch + 1})")
    print(f"IoU en test: {test_iou:.4f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrenar red de segmentación")
    parser.add_argument('--epochs', type=int, default=None, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=None, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Dispositivo (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='Continuar desde checkpoint')
    
    args = parser.parse_args()
    
    train_segmentation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
