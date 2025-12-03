"""
Clases de Dataset para PyTorch.
Incluye datasets para clasificación y análisis temporal.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
from typing import Tuple, List, Optional, Dict
import random

from config import (
    GESTURE_CLASSES, DATASET_DIR,
    CLASSIFIER_CONFIG, TEMPORAL_CONFIG, TRAINING_CONFIG, CLASS_TO_IDX
)


# NOTA: SegmentationDataset ha sido ELIMINADO
# Ya no se utiliza red de segmentación UNet en este proyecto
# MediaPipe maneja toda la detección y segmentación de manos


class GestureClassificationDataset(Dataset):
    """Dataset para clasificación de gestos."""
    
    def __init__(self, root_dir: Path = None, transform=None, split='train', 
                 use_landmarks=False):
        self.root_dir = root_dir or DATASET_DIR
        self.transform = transform
        self.split = split
        self.use_landmarks = use_landmarks
        
        self.images_dir = self.root_dir / "images"
        self.landmarks_dir = self.root_dir / "landmarks"
        
        # Recolectar muestras
        self.samples = []
        for class_id, class_name in GESTURE_CLASSES.items():
            class_img_dir = self.images_dir / class_name
            class_lm_dir = self.landmarks_dir / class_name
            
            if not class_img_dir.exists():
                continue
            
            for img_file in class_img_dir.glob("*.jpg"):
                lm_file = class_lm_dir / f"{img_file.stem}.npy" if use_landmarks else None
                if use_landmarks and lm_file and not lm_file.exists():
                    continue
                self.samples.append((img_file, class_id, lm_file))
        
        # Shuffle y split
        random.seed(TRAINING_CONFIG["seed"])
        random.shuffle(self.samples)
        
        n = len(self.samples)
        train_end = int(n * TRAINING_CONFIG["train_split"])
        val_end = train_end + int(n * TRAINING_CONFIG["val_split"])
        
        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif split == 'test':
            self.samples = self.samples[val_end:]
        
        # Transform por defecto
        if self.transform is None:
            self.transform = self._get_default_transform(split)
    
    def _get_default_transform(self, split):
        input_size = CLASSIFIER_CONFIG["input_size"]
        if split == 'train' and TRAINING_CONFIG["augmentation"]:
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id, lm_path = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if self.use_landmarks and lm_path:
            landmarks = np.load(lm_path)
            landmarks = torch.from_numpy(landmarks).float()
            return image, class_id, landmarks
        
        return image, class_id


class TemporalSequenceDataset(Dataset):
    """Dataset para análisis temporal con GRU/LSTM."""
    
    def __init__(self, root_dir: Path = None, transform=None, split='train',
                 sequence_length: int = None):
        self.root_dir = root_dir or DATASET_DIR
        self.transform = transform
        self.split = split
        self.sequence_length = sequence_length or TEMPORAL_CONFIG["sequence_length"]
        
        self.sequences_dir = self.root_dir / "sequences"
        
        # Recolectar secuencias
        self.sequences = []
        for class_id, class_name in GESTURE_CLASSES.items():
            class_seq_dir = self.sequences_dir / class_name
            
            if not class_seq_dir.exists():
                continue
            
            for seq_dir in class_seq_dir.iterdir():
                if seq_dir.is_dir():
                    meta_file = seq_dir / "metadata.json"
                    if meta_file.exists():
                        self.sequences.append((seq_dir, class_id))
        
        # También crear secuencias sintéticas a partir de frames individuales
        self._create_synthetic_sequences()
        
        # Shuffle y split
        random.seed(TRAINING_CONFIG["seed"])
        random.shuffle(self.sequences)
        
        n = len(self.sequences)
        train_end = int(n * TRAINING_CONFIG["train_split"])
        val_end = train_end + int(n * TRAINING_CONFIG["val_split"])
        
        if split == 'train':
            self.sequences = self.sequences[:train_end]
        elif split == 'val':
            self.sequences = self.sequences[train_end:val_end]
        elif split == 'test':
            self.sequences = self.sequences[val_end:]
        
        # Transform por defecto
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _create_synthetic_sequences(self):
        """Crea secuencias sintéticas agrupando frames consecutivos."""
        images_dir = self.root_dir / "images"
        
        for class_id, class_name in GESTURE_CLASSES.items():
            class_img_dir = images_dir / class_name
            if not class_img_dir.exists():
                continue
            
            img_files = sorted(class_img_dir.glob("*.jpg"))
            
            # Crear secuencias de frames consecutivos
            for i in range(0, len(img_files) - self.sequence_length, self.sequence_length // 2):
                seq_files = img_files[i:i + self.sequence_length]
                if len(seq_files) == self.sequence_length:
                    self.sequences.append((seq_files, class_id))
    
    def _get_default_transform(self):
        input_size = CLASSIFIER_CONFIG["input_size"]
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data, class_id = self.sequences[idx]
        
        frames = []
        landmarks_seq = []
        
        if isinstance(seq_data, Path):
            # Directorio de secuencia real
            for i in range(self.sequence_length):
                frame_path = seq_data / f"frame_{i:03d}.jpg"
                lm_path = seq_data / f"landmarks_{i:03d}.npy"
                
                if frame_path.exists():
                    img = Image.open(frame_path).convert('RGB')
                    frames.append(self.transform(img))
                    
                    if lm_path.exists():
                        landmarks_seq.append(np.load(lm_path))
                    else:
                        landmarks_seq.append(np.zeros(63))
            
            # Pad si es necesario
            while len(frames) < self.sequence_length:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
                landmarks_seq.append(landmarks_seq[-1] if landmarks_seq else np.zeros(63))
        
        else:
            # Lista de archivos de imagen (secuencia sintética)
            landmarks_dir = self.root_dir / "landmarks" / GESTURE_CLASSES[class_id]
            
            for img_path in seq_data:
                img = Image.open(img_path).convert('RGB')
                frames.append(self.transform(img))
                
                lm_path = landmarks_dir / f"{img_path.stem}.npy"
                if lm_path.exists():
                    landmarks_seq.append(np.load(lm_path))
                else:
                    landmarks_seq.append(np.zeros(63))
        
        # Stack frames y landmarks
        frames = torch.stack(frames)  # (seq_len, C, H, W)
        landmarks = torch.from_numpy(np.array(landmarks_seq)).float()  # (seq_len, 63)
        
        return frames, landmarks, class_id


def get_dataloaders(dataset_type: str = 'classification',
                    batch_size: int = None,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train, val y test.

    Args:
        dataset_type: 'classification' o 'temporal' (segmentation ELIMINADO)
        batch_size: Tamaño del batch (usa default del config si es None)
        num_workers: Número de workers para carga de datos

    Returns:
        Tuple de (train_loader, val_loader, test_loader)
    """

    if dataset_type == 'classification':
        batch_size = batch_size or TRAINING_CONFIG["cls_batch_size"]
        train_dataset = GestureClassificationDataset(split='train')
        val_dataset = GestureClassificationDataset(split='val')
        test_dataset = GestureClassificationDataset(split='test')
    
    elif dataset_type == 'temporal':
        batch_size = batch_size or TRAINING_CONFIG["temp_batch_size"]
        train_dataset = TemporalSequenceDataset(split='train')
        val_dataset = TemporalSequenceDataset(split='val')
        test_dataset = TemporalSequenceDataset(split='test')
    
    else:
        raise ValueError(f"Tipo de dataset desconocido: {dataset_type}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset {dataset_type}:")
    print(f"  Train: {len(train_dataset)} muestras")
    print(f"  Val: {len(val_dataset)} muestras")
    print(f"  Test: {len(test_dataset)} muestras")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test de datasets
    print("Testeando datasets...")

    # Test clasificación
    print("\n--- Dataset de Clasificación ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders('classification')
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
            break
    except Exception as e:
        print(f"Error: {e}")

    # Test temporal
    print("\n--- Dataset Temporal ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders('temporal')
        for frames, landmarks, labels in train_loader:
            print(f"Frames shape: {frames.shape}, Landmarks: {landmarks.shape}, Labels: {labels[:5]}")
            break
    except Exception as e:
        print(f"Error: {e}")
