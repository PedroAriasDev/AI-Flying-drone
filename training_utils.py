"""
Utilidades para entrenamiento: métricas, visualización, checkpoints.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import CHECKPOINTS_DIR, RESULTS_DIR, GESTURE_CLASSES


class MetricsTracker:
    """Rastrea y almacena métricas durante el entrenamiento."""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Métricas de entrenamiento
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        
        # Métricas de validación
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        
        # Métricas de test (solo al final)
        self.test_loss: float = 0.0
        self.test_acc: float = 0.0
        self.test_predictions: List[int] = []
        self.test_labels: List[int] = []
        
        # Mejor epoch
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        
        # Métricas adicionales por época
        self.epoch_metrics: List[Dict] = []
    
    def update_train(self, loss: float, acc: float):
        """Actualiza métricas de entrenamiento."""
        self.train_loss.append(loss)
        self.train_acc.append(acc)
    
    def update_val(self, loss: float, acc: float, epoch: int):
        """Actualiza métricas de validación."""
        self.val_loss.append(loss)
        self.val_acc.append(acc)
        
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            self.best_epoch = epoch
    
    def update_test(self, loss: float, acc: float, 
                    predictions: List[int], labels: List[int]):
        """Actualiza métricas de test."""
        self.test_loss = loss
        self.test_acc = acc
        self.test_predictions = predictions
        self.test_labels = labels
    
    def add_epoch_metrics(self, metrics: Dict):
        """Añade métricas adicionales de una época."""
        self.epoch_metrics.append(metrics)
    
    def get_summary(self) -> Dict:
        """Retorna resumen de métricas."""
        return {
            'experiment_name': self.experiment_name,
            'num_epochs': len(self.train_loss),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_loss[-1] if self.train_loss else 0,
            'final_train_acc': self.train_acc[-1] if self.train_acc else 0,
            'final_val_loss': self.val_loss[-1] if self.val_loss else 0,
            'final_val_acc': self.val_acc[-1] if self.val_acc else 0,
            'test_acc': self.test_acc,
            'test_loss': self.test_loss,
        }
    
    def save(self, path: Path = None):
        """Guarda métricas a archivo JSON."""
        path = path or RESULTS_DIR / f"{self.experiment_name}_metrics.json"
        
        data = {
            'summary': self.get_summary(),
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'test_predictions': self.test_predictions,
            'test_labels': self.test_labels,
            'epoch_metrics': self.epoch_metrics
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Métricas guardadas en: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'MetricsTracker':
        """Carga métricas desde archivo."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tracker = cls(data['summary']['experiment_name'])
        tracker.train_loss = data['train_loss']
        tracker.train_acc = data['train_acc']
        tracker.val_loss = data['val_loss']
        tracker.val_acc = data['val_acc']
        tracker.test_predictions = data.get('test_predictions', [])
        tracker.test_labels = data.get('test_labels', [])
        tracker.best_val_acc = data['summary']['best_val_acc']
        tracker.best_epoch = data['summary']['best_epoch']
        
        return tracker


class TrainingVisualizer:
    """Genera visualizaciones del entrenamiento."""
    
    def __init__(self, save_dir: Path = None):
        self.save_dir = save_dir or RESULTS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, metrics: MetricsTracker, 
                             title: str = "Training Curves",
                             save_name: str = None):
        """Grafica curvas de pérdida y accuracy."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(metrics.train_loss) + 1)
        
        # Gráfico de pérdida
        ax1 = axes[0]
        ax1.plot(epochs, metrics.train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, metrics.val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.axvline(x=metrics.best_epoch + 1, color='g', linestyle='--', 
                    label=f'Best Epoch ({metrics.best_epoch + 1})')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Pérdida', fontsize=12)
        ax1.set_title('Pérdida de Entrenamiento y Validación', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de accuracy
        ax2 = axes[1]
        ax2.plot(epochs, metrics.train_acc, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, metrics.val_acc, 'r-', label='Val Acc', linewidth=2)
        ax2.axvline(x=metrics.best_epoch + 1, color='g', linestyle='--', 
                    label=f'Best Epoch ({metrics.best_epoch + 1})')
        ax2.axhline(y=metrics.best_val_acc, color='orange', linestyle=':', 
                    label=f'Best Val Acc ({metrics.best_val_acc:.4f})')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy de Entrenamiento y Validación', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, predictions: List[int], labels: List[int],
                              class_names: List[str] = None,
                              title: str = "Confusion Matrix",
                              save_name: str = None):
        """Genera matriz de confusión."""
        class_names = class_names or list(GESTURE_CLASSES.values())
        
        # Filtrar clases que realmente aparecen
        unique_labels = sorted(set(labels) | set(predictions))
        filtered_names = [class_names[i] for i in unique_labels]
        
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Matriz de confusión absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=filtered_names, yticklabels=filtered_names)
        axes[0].set_xlabel('Predicción', fontsize=12)
        axes[0].set_ylabel('Real', fontsize=12)
        axes[0].set_title('Matriz de Confusión (Absoluta)', fontsize=14)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Matriz de confusión normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=filtered_names, yticklabels=filtered_names)
        axes[1].set_xlabel('Predicción', fontsize=12)
        axes[1].set_ylabel('Real', fontsize=12)
        axes[1].set_title('Matriz de Confusión (Normalizada)', fontsize=14)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}_confusion.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")
        
        plt.show()
        return fig
    
    def plot_class_metrics(self, predictions: List[int], labels: List[int],
                           class_names: List[str] = None,
                           title: str = "Metrics per Class",
                           save_name: str = None):
        """Grafica métricas por clase (precision, recall, f1)."""
        class_names = class_names or list(GESTURE_CLASSES.values())
        
        report = classification_report(labels, predictions, 
                                       target_names=class_names,
                                       output_dict=True,
                                       zero_division=0)
        
        # Extraer métricas por clase
        classes = []
        precision = []
        recall = []
        f1 = []
        
        for class_name in class_names:
            if class_name in report:
                classes.append(class_name[:12])  # Truncar nombres largos
                precision.append(report[class_name]['precision'])
                recall.append(report[class_name]['recall'])
                f1.append(report[class_name]['f1-score'])
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Clase', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores sobre las barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}_class_metrics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Métricas por clase guardadas en: {save_path}")
        
        plt.show()
        return fig
    
    def generate_full_report(self, metrics: MetricsTracker, 
                             experiment_name: str):
        """Genera reporte completo con todas las visualizaciones."""
        print(f"\n{'='*60}")
        print(f"REPORTE DE ENTRENAMIENTO: {experiment_name}")
        print(f"{'='*60}\n")
        
        # Resumen
        summary = metrics.get_summary()
        print("RESUMEN:")
        print(f"  Épocas totales: {summary['num_epochs']}")
        print(f"  Mejor época: {summary['best_epoch'] + 1}")
        print(f"  Mejor Val Accuracy: {summary['best_val_acc']:.4f}")
        print(f"  Test Accuracy: {summary['test_acc']:.4f}")
        print()
        
        # Curvas de entrenamiento
        self.plot_training_curves(metrics, 
                                  title=f"Curvas de Entrenamiento - {experiment_name}",
                                  save_name=experiment_name)
        
        # Matriz de confusión (si hay datos de test)
        if metrics.test_predictions and metrics.test_labels:
            self.plot_confusion_matrix(metrics.test_predictions, 
                                       metrics.test_labels,
                                       title=f"Matriz de Confusión - {experiment_name}",
                                       save_name=experiment_name)
            
            self.plot_class_metrics(metrics.test_predictions,
                                    metrics.test_labels,
                                    title=f"Métricas por Clase - {experiment_name}",
                                    save_name=experiment_name)
            
            # Reporte de clasificación
            print("\nREPORTE DE CLASIFICACIÓN:")
            print(classification_report(metrics.test_labels, 
                                        metrics.test_predictions,
                                        target_names=list(GESTURE_CLASSES.values()),
                                        zero_division=0))
        
        # Guardar métricas
        metrics.save(self.save_dir / f"{experiment_name}_metrics.json")


class CheckpointManager:
    """Gestiona checkpoints de modelos."""
    
    def __init__(self, save_dir: Path = None, model_name: str = "model"):
        self.save_dir = save_dir or CHECKPOINTS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metrics: MetricsTracker, is_best: bool = False):
        """
        Guarda checkpoint del modelo.
        
        Args:
            model: Modelo a guardar
            optimizer: Optimizador
            epoch: Época actual
            metrics: Métricas de entrenamiento
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.get_summary(),
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'train_acc': metrics.train_acc,
            'val_acc': metrics.val_acc,
        }
        
        # Guardar checkpoint regular
        path = self.save_dir / f"{self.model_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        # Guardar último checkpoint
        latest_path = self.save_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Mejor modelo guardado (epoch {epoch})")
        
        return path
    
    def load(self, model: nn.Module, optimizer: torch.optim.Optimizer = None,
             checkpoint_path: Path = None, load_best: bool = True) -> Tuple[int, Dict]:
        """
        Carga checkpoint.
        
        Args:
            model: Modelo donde cargar pesos
            optimizer: Optimizador (opcional)
            checkpoint_path: Ruta específica del checkpoint
            load_best: Si cargar el mejor modelo
        
        Returns:
            Tuple de (época, métricas)
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / f"{self.model_name}_best.pt"
            else:
                checkpoint_path = self.save_dir / f"{self.model_name}_latest.pt"
        
        if not checkpoint_path.exists():
            print(f"No se encontró checkpoint en: {checkpoint_path}")
            return 0, {}
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint cargado desde: {checkpoint_path}")
        print(f"  Época: {checkpoint['epoch']}")
        print(f"  Métricas: {checkpoint['metrics']}")
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Elimina checkpoints antiguos, manteniendo los últimos N."""
        checkpoints = sorted(self.save_dir.glob(f"{self.model_name}_epoch_*.pt"),
                            key=lambda x: x.stat().st_mtime,
                            reverse=True)
        
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()
            print(f"Eliminado checkpoint antiguo: {checkpoint.name}")


class EarlyStopping:
    """Implementa early stopping para evitar overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'max'):
        """
        Args:
            patience: Épocas a esperar sin mejora
            min_delta: Cambio mínimo para considerar mejora
            mode: 'max' para accuracy, 'min' para loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Verifica si debe detener el entrenamiento.
        
        Args:
            score: Métrica actual (val_acc o val_loss)
        
        Returns:
            True si debe detener
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"  Early stopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reinicia el contador."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calcula accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, 
                  smooth: float = 1e-6) -> float:
    """Calcula IoU (Intersection over Union) para segmentación."""
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor,
                   smooth: float = 1e-6) -> float:
    """Calcula Dice coefficient para segmentación."""
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    intersection = (pred_mask * true_mask).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)
    
    return dice.item()


if __name__ == "__main__":
    # Test de visualización
    print("Testing training utilities...")
    
    # Crear métricas de ejemplo
    metrics = MetricsTracker("test_experiment")
    
    for epoch in range(20):
        train_loss = 1.0 - epoch * 0.04 + np.random.random() * 0.1
        train_acc = 0.5 + epoch * 0.02 + np.random.random() * 0.05
        val_loss = 1.0 - epoch * 0.03 + np.random.random() * 0.15
        val_acc = 0.5 + epoch * 0.018 + np.random.random() * 0.05
        
        metrics.update_train(train_loss, min(train_acc, 0.99))
        metrics.update_val(val_loss, min(val_acc, 0.98), epoch)
    
    # Generar predicciones de test falsas
    np.random.seed(42)
    test_labels = np.random.randint(0, 11, 100).tolist()
    test_preds = test_labels.copy()
    # Añadir algo de ruido a las predicciones
    for i in range(len(test_preds)):
        if np.random.random() < 0.15:
            test_preds[i] = np.random.randint(0, 11)
    
    metrics.update_test(0.3, 0.85, test_preds, test_labels)
    
    # Visualizar
    visualizer = TrainingVisualizer()
    visualizer.generate_full_report(metrics, "test_experiment")
