"""
factorial/reporting.py
======================
Consolidación de todos los resultados de evaluación en archivos CSV y XLSX.

Genera:
    outputs/factorial_experiments/results_summary.csv
    outputs/factorial_experiments/results_summary.xlsx

Con una fila por evaluación (54 filas = 18 modelos × 3 datasets de test).
Incluye métricas, distribuciones, class weights, rutas de archivos.
"""

import os
import json
import glob
import pandas as pd
import numpy as np

from factorial.class_weights import compute_class_weights, compute_class_counts
from factorial.splits import load_split_dataframe
from factorial.normalization import _get_project_root


NUM_CLASSES = 5


def consolidate_results(config: dict) -> pd.DataFrame:
    """
    Recorre todos los directorios de salida y consolida los resultados
    en un único DataFrame.

    Args:
        config (dict): Configuración del experimento factorial.

    Returns:
        pd.DataFrame con todas las evaluaciones (filas) y métricas (columnas).
    """
    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    if not os.path.isabs(factorial_output):
        factorial_output = os.path.join(_get_project_root(), factorial_output)

    seeds         = config["factorial_experiment"]["seeds"]
    train_datasets = config["factorial_experiment"]["train_datasets"]
    test_datasets  = config["factorial_experiment"]["test_datasets"]
    architectures  = config["factorial_experiment"]["architectures"]

    records = []

    for seed in seeds:
        for train_ds in train_datasets:
            for arch in architectures:
                for test_ds in test_datasets:
                    record = _build_record(
                        config, factorial_output,
                        seed, train_ds, arch, test_ds,
                    )
                    if record is not None:
                        records.append(record)

    if not records:
        print("[Reporting] No se encontraron resultados para consolidar.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ordenar columnas de forma lógica
    df = _reorder_columns(df)

    # Guardar CSV
    csv_path = os.path.join(factorial_output, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Reporting] [OK] results_summary.csv guardado: {csv_path}")

    # Guardar XLSX
    xlsx_path = os.path.join(factorial_output, "results_summary.xlsx")
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"[Reporting] [OK] results_summary.xlsx guardado: {xlsx_path}")
    except Exception as e:
        print(f"[Reporting] [WARN] No se pudo guardar XLSX: {e}")
        print(f"             Instala openpyxl con: pip install openpyxl")

    print(f"[Reporting] Total de evaluaciones consolidadas: {len(df)}")
    return df


def _build_record(
    config: dict,
    factorial_output: str,
    seed: int,
    train_ds: str,
    arch: str,
    test_ds: str,
) -> dict | None:
    """
    Construye un registro (fila) del DataFrame de resultados para una
    combinación (seed, train_ds, arch, test_ds).

    Returns None si no existen los archivos de métricas.
    """
    # Rutas de los archivos de esta evaluación
    model_dir  = os.path.join(factorial_output, train_ds, arch, f"seed_{seed}")
    eval_dir   = os.path.join(model_dir, "evaluations", f"test_{test_ds}")
    metrics_p  = os.path.join(eval_dir, "metrics.json")
    preds_p    = os.path.join(eval_dir, "predictions.csv")
    cm_p       = os.path.join(eval_dir, "confusion_matrix.png")
    model_p    = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(metrics_p):
        return None  # Evaluación no completada aún

    # Cargar métricas
    with open(metrics_p) as f:
        metrics = json.load(f)

    # Cargar class weights desde los splits
    try:
        df_train = load_split_dataframe(train_ds, "train", seed, config)
        df_val   = load_split_dataframe(train_ds, "val",   seed, config)
        df_test  = load_split_dataframe(test_ds,  "test",  seed, config)

        cw_tensor = compute_class_weights(df_train)
        cw = cw_tensor.numpy().tolist()

        train_counts = compute_class_counts(df_train)
        val_counts   = compute_class_counts(df_val)
        test_counts  = compute_class_counts(df_test)
    except Exception as e:
        print(f"[Reporting] ? No se pudieron cargar splits para {train_ds}/seed_{seed}: {e}")
        cw = [None] * NUM_CLASSES
        train_counts = {c: None for c in range(NUM_CLASSES)}
        val_counts   = {c: None for c in range(NUM_CLASSES)}
        test_counts  = {c: None for c in range(NUM_CLASSES)}
        df_train = pd.DataFrame()
        df_val   = pd.DataFrame()
        df_test  = pd.DataFrame()

    use_cbam = (arch == "cbam")
    imb_cfg  = config.get("imbalance_handling", {})

    record = {
        # Identificadores del experimento
        "seed":            seed,
        "train_dataset":   train_ds,
        "test_dataset":    test_ds,
        "architecture":    arch,
        "use_cbam":        use_cbam,

        # Métricas globales
        "accuracy":   metrics.get("accuracy"),
        "precision":  metrics.get("precision"),
        "recall":     metrics.get("recall"),
        "f1_score":   metrics.get("f1_score"),
        "auc":        metrics.get("auc"),
        "val_loss":   metrics.get("val_loss"),

        # Conteos por split
        "num_train_images": len(df_train),
        "num_val_images":   len(df_val),
        "num_test_images":  len(df_test),

        # Rutas de archivos
        "model_path":           model_p  if os.path.exists(model_p)  else "",
        "metrics_path":         metrics_p,
        "predictions_path":     preds_p  if os.path.exists(preds_p)  else "",
        "confusion_matrix_path": cm_p    if os.path.exists(cm_p)     else "",

        # Manejo de desbalance
        "use_class_weights":    imb_cfg.get("use_class_weights", True),
        "use_weighted_sampler": imb_cfg.get("use_weighted_sampler", False),
    }

    # Métricas por clase
    for c in range(NUM_CLASSES):
        record[f"precision_class_{c}"] = metrics.get(f"precision_class_{c}")
        record[f"recall_class_{c}"]    = metrics.get(f"recall_class_{c}")
        record[f"f1_class_{c}"]        = metrics.get(f"f1_class_{c}")

    # Distribución de train/val/test por clase
    for c in range(NUM_CLASSES):
        record[f"train_class_{c}"] = train_counts.get(c)
        record[f"val_class_{c}"]   = val_counts.get(c)
        record[f"test_class_{c}"]  = test_counts.get(c)

    # Class weights
    for c in range(NUM_CLASSES):
        record[f"class_weight_{c}"] = cw[c] if c < len(cw) else None

    return record


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena las columnas del DataFrame en un orden lógico."""
    priority = [
        "seed", "train_dataset", "test_dataset", "architecture", "use_cbam",
        "accuracy", "precision", "recall", "f1_score", "auc", "val_loss",
        "num_train_images", "num_val_images", "num_test_images",
    ]
    per_class_cols = (
        [f"precision_class_{c}" for c in range(NUM_CLASSES)] +
        [f"recall_class_{c}"    for c in range(NUM_CLASSES)] +
        [f"f1_class_{c}"        for c in range(NUM_CLASSES)] +
        [f"train_class_{c}"     for c in range(NUM_CLASSES)] +
        [f"val_class_{c}"       for c in range(NUM_CLASSES)] +
        [f"test_class_{c}"      for c in range(NUM_CLASSES)] +
        [f"class_weight_{c}"    for c in range(NUM_CLASSES)]
    )
    other = [
        "use_class_weights", "use_weighted_sampler",
        "model_path", "metrics_path", "predictions_path", "confusion_matrix_path",
    ]

    ordered = []
    for col in priority + per_class_cols + other:
        if col in df.columns:
            ordered.append(col)

    # Agregar columnas que no estén en ninguna lista
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def print_summary_table(df: pd.DataFrame) -> None:
    """Imprime una tabla resumen de resultados en consola."""
    if df.empty:
        print("[Reporting] No hay resultados para mostrar.")
        return

    print(f"\n{'='*80}")
    print(f"  RESUMEN DE RESULTADOS — {len(df)} evaluaciones")
    print(f"{'='*80}")
    print(
        f"{'Seed':>4} {'Train':>10} {'Test':>10} {'Arch':>6} "
        f"{'Acc':>7} {'F1':>7} {'AUC':>7}"
    )
    print(f"{'-'*80}")

    for _, row in df.iterrows():
        acc = f"{row['accuracy']:.4f}" if pd.notna(row.get('accuracy')) else "N/A"
        f1  = f"{row['f1_score']:.4f}" if pd.notna(row.get('f1_score')) else "N/A"
        auc = f"{row['auc']:.4f}"      if pd.notna(row.get('auc'))      else "N/A"
        print(
            f"{row['seed']:>4} {row['train_dataset']:>10} {row['test_dataset']:>10} "
            f"{row['architecture']:>6} {acc:>7} {f1:>7} {auc:>7}"
        )

    print(f"{'='*80}\n")


def load_existing_summary(config: dict) -> pd.DataFrame:
    """Carga el CSV de resultados existente si existe."""
    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    if not os.path.isabs(factorial_output):
        factorial_output = os.path.join(_get_project_root(), factorial_output)

    csv_path = os.path.join(factorial_output, "results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()
