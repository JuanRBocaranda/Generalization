"""
factorial/class_weights.py
==========================
Cálculo de class weights para manejar el desbalance de clases durante
el entrenamiento mediante CrossEntropyLoss pesada.

Fórmula:
    weight[c] = total_samples / (num_classes * count[c])

Los pesos se calculan SOLO a partir del split de entrenamiento real de
cada dataset y semilla. No se aplican en validación ni test.
"""

import pandas as pd
import torch


NUM_CLASSES = 5


def compute_class_weights(df_train: pd.DataFrame, device: torch.device = None) -> torch.Tensor:
    """
    Calcula class weights a partir del DataFrame de entrenamiento.

    Fórmula: weight[c] = total / (num_classes * count[c])
    Clases ausentes reciben peso 0 (no contribuyen al loss).

    Args:
        df_train (pd.DataFrame): Split de entrenamiento con columna 'diagnosis'.
        device (torch.device):   Dispositivo al que mover el tensor (opcional).

    Returns:
        torch.Tensor de shape (num_classes,) con los pesos por clase.
    """
    class_counts = []
    for c in range(NUM_CLASSES):
        count = int((df_train["diagnosis"] == c).sum())
        class_counts.append(count)

    total = sum(class_counts)
    weights = []
    for c in range(NUM_CLASSES):
        if class_counts[c] > 0:
            w = total / (NUM_CLASSES * class_counts[c])
        else:
            w = 0.0  # Clase ausente → peso 0
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    if device is not None:
        weight_tensor = weight_tensor.to(device)

    return weight_tensor


def print_class_weights(weights: torch.Tensor, train_dataset: str, seed: int, architecture: str) -> None:
    """
    Imprime los class weights calculados para un experimento específico.
    """
    print(f"\n[Class Weights] Train: {train_dataset} | Seed: {seed} | Arch: {architecture}")
    for c in range(NUM_CLASSES):
        print(f"  Clase {c}: {weights[c].item():.4f}")


def compute_class_counts(df: pd.DataFrame) -> dict:
    """
    Retorna el conteo de muestras por clase como diccionario.

    Returns:
        dict {clase: count}
    """
    return {c: int((df["diagnosis"] == c).sum()) for c in range(NUM_CLASSES)}
