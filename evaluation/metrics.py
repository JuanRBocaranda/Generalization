import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

NUM_CLASSES = 5
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calcula métricas globales de clasificación.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted classes.
        y_prob (np.array): Predicted probabilities (shape: N x num_classes).
        
    Returns:
        dict: accuracy, precision, recall, f1_score, auc + métricas por clase.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Macro average para multiclase
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Métricas por clase
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    
    # ROC AUC — One-vs-Rest para multiclase
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    
    metrics = {
        'accuracy':  float(acc),
        'precision': float(precision_macro),
        'recall':    float(recall_macro),
        'f1_score':  float(f1_macro),
        'auc':       float(auc),
    }
    
    # Agregar métricas por clase
    for c in range(NUM_CLASSES):
        metrics[f'precision_class_{c}'] = float(precision_per_class[c]) if c < len(precision_per_class) else 0.0
        metrics[f'recall_class_{c}']    = float(recall_per_class[c])    if c < len(recall_per_class)    else 0.0
        metrics[f'f1_class_{c}']        = float(f1_per_class[c])        if c < len(f1_per_class)        else 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Genera y guarda opcionalmente una matriz de confusión.

    Args:
        y_true (np.array):    Ground truth labels.
        y_pred (np.array):    Predicted labels.
        class_names (list):   Nombres de las clases.
        save_path (str):      Ruta para guardar la imagen.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()


def save_metrics(metrics: dict, save_path: str) -> None:
    """
    Guarda el diccionario de métricas en un archivo JSON.

    Args:
        metrics (dict): Diccionario de métricas.
        save_path (str): Ruta del archivo de salida.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_predictions(
    image_ids: list,
    image_paths: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """
    Guarda las predicciones individuales en un CSV.

    Columnas de salida:
        image_id, image_path, true_label, predicted_label,
        prob_class_0, prob_class_1, prob_class_2, prob_class_3, prob_class_4

    Args:
        image_ids (list):      Identificadores de las imágenes.
        image_paths (list):    Rutas absolutas de las imágenes.
        y_true (np.ndarray):   Etiquetas reales.
        y_pred (np.ndarray):   Etiquetas predichas.
        y_prob (np.ndarray):   Probabilidades (N x num_classes).
        save_path (str):       Ruta de salida del CSV.
    """
    records = []
    n = len(y_true)

    for i in range(n):
        row = {
            'image_id':        image_ids[i] if i < len(image_ids) else f"img_{i}",
            'image_path':      image_paths[i] if i < len(image_paths) else "",
            'true_label':      int(y_true[i]),
            'predicted_label': int(y_pred[i]),
        }
        probs = y_prob[i] if i < len(y_prob) else [0.0] * NUM_CLASSES
        for c in range(NUM_CLASSES):
            row[f'prob_class_{c}'] = float(probs[c]) if c < len(probs) else 0.0
        records.append(row)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
