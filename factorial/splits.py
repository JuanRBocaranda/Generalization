"""
factorial/splits.py
===================
Creación y carga de splits balanceados por clase para el diseño experimental.

El diseño garantiza:
    - Test: 205 imágenes por dataset (distribución fija: 100/30/50/15/10 por clase)
    - Train/Val: 1539 imágenes por dataset (distribución: 917/240/297/60/25 por clase)
    - Split train/val: 80% train / 20% val, estratificado por clase
    - Mismas particiones para CNN base y CNN+CBAM (dentro de cada seed)
    - Sin solapamiento entre train, val y test

Guardado en:
    outputs/factorial_experiments/splits/seed_{seed}/
        {dataset}_train.csv
        {dataset}_val.csv
        {dataset}_test.csv
"""

import os
import numpy as np
import pandas as pd

from factorial.normalization import (
    load_aptos_dataframe,
    load_messidor_dataframe,
    load_odir_dataframe,
    _get_project_root,
)


# =============================================================================
# Punto de entrada principal
# =============================================================================
def create_or_load_splits(config: dict, seed: int, force: bool = False) -> dict:
    """
    Crea o carga los splits balanceados para todos los datasets en una seed.

    Args:
        config (dict): Configuración del experimento factorial.
        seed (int):    Semilla de reproducibilidad.
        force (bool):  Si True, regenera aunque ya existan.

    Returns:
        dict: {dataset_name: {"train": df, "val": df, "test": df}}
    """
    splits_root = _get_splits_root(config)
    seed_dir = os.path.join(splits_root, f"seed_{seed}")

    # Verificar si ya existen todos los splits
    if not force and _all_splits_exist(seed_dir):
        print(f"[Splits] Cargando splits existentes para seed={seed}")
        return _load_splits_from_disk(seed_dir, seed)

    print(f"\n{'='*60}")
    print(f"[Splits] Creando splits balanceados para seed={seed}")
    print(f"{'='*60}")

    # Cargar DataFrames base
    df_aptos_tv, df_aptos_test_src = load_aptos_dataframe(config)
    df_messidor = load_messidor_dataframe(config)
    df_odir = load_odir_dataframe(config)

    # Distribuciones objetivo
    test_dist = _parse_distribution(config["factorial_experiment"]["test_distribution"])
    trainval_dist = _parse_distribution(config["factorial_experiment"]["trainval_distribution"])
    train_ratio = config["factorial_experiment"]["train_val_split"]["train"]

    # Crear RNG reproducible
    rng = np.random.RandomState(seed)

    all_splits = {}

    # Modo de datos de entrenamiento
    training_data_mode = config.get("factorial_experiment", {}).get("training_data_mode", {})
    balanced_trainval = training_data_mode.get("balanced_trainval", True)
    full_trainval = training_data_mode.get("full_trainval_with_balanced_test", False)

    # -----------------------------------------------------------------------
    # APTOS: test desde test.csv, train/val desde train - APTOS.csv
    # -----------------------------------------------------------------------
    df_aptos_test = _sample_balanced(df_aptos_test_src, test_dist, rng, "APTOS-test")
    if full_trainval and not balanced_trainval:
        df_aptos_trainval = df_aptos_tv.copy()
    else:
        df_aptos_trainval = _sample_balanced(df_aptos_tv, trainval_dist, rng, "APTOS-trainval")
    df_aptos_train, df_aptos_val = _stratified_split(df_aptos_trainval, train_ratio, rng)

    _tag_split(df_aptos_test, "test", seed, "aptos")
    _tag_split(df_aptos_train, "train", seed, "aptos")
    _tag_split(df_aptos_val, "val", seed, "aptos")

    all_splits["aptos"] = {
        "train": df_aptos_train,
        "val":   df_aptos_val,
        "test":  df_aptos_test,
    }

    # -----------------------------------------------------------------------
    # Messidor: test y train/val desde messidor_data.csv (sin solapamiento)
    # -----------------------------------------------------------------------
    df_messidor_test = _sample_balanced(df_messidor, test_dist, rng, "Messidor-test")
    # Excluir índices ya usados en test
    test_indices = set(df_messidor_test.index)
    df_messidor_remaining = df_messidor.drop(index=test_indices)
    if full_trainval and not balanced_trainval:
        df_messidor_trainval = df_messidor_remaining.copy()
    else:
        df_messidor_trainval = _sample_balanced(df_messidor_remaining, trainval_dist, rng, "Messidor-trainval")
    df_messidor_train, df_messidor_val = _stratified_split(df_messidor_trainval, train_ratio, rng)

    _tag_split(df_messidor_test, "test", seed, "messidor")
    _tag_split(df_messidor_train, "train", seed, "messidor")
    _tag_split(df_messidor_val, "val", seed, "messidor")

    all_splits["messidor"] = {
        "train": df_messidor_train,
        "val":   df_messidor_val,
        "test":  df_messidor_test,
    }

    # -----------------------------------------------------------------------
    # ODIR: test y train/val desde odir_image_level.csv (sin solapamiento)
    # -----------------------------------------------------------------------
    df_odir_test = _sample_balanced(df_odir, test_dist, rng, "ODIR-test")
    test_indices_odir = set(df_odir_test.index)
    df_odir_remaining = df_odir.drop(index=test_indices_odir)
    if full_trainval and not balanced_trainval:
        df_odir_trainval = df_odir_remaining.copy()
    else:
        df_odir_trainval = _sample_balanced(df_odir_remaining, trainval_dist, rng, "ODIR-trainval")
    df_odir_train, df_odir_val = _stratified_split(df_odir_trainval, train_ratio, rng)

    _tag_split(df_odir_test, "test", seed, "odir")
    _tag_split(df_odir_train, "train", seed, "odir")
    _tag_split(df_odir_val, "val", seed, "odir")

    all_splits["odir"] = {
        "train": df_odir_train,
        "val":   df_odir_val,
        "test":  df_odir_test,
    }

    # Guardar todos los splits en disco
    os.makedirs(seed_dir, exist_ok=True)
    for ds_name, ds_splits in all_splits.items():
        for split_name, df_split in ds_splits.items():
            csv_path = os.path.join(seed_dir, f"{ds_name}_{split_name}.csv")
            df_split.to_csv(csv_path, index=False)
            print(f"[Splits] Guardado: {csv_path}")

    return all_splits


def load_split_dataframe(dataset_name: str, split: str, seed: int, config: dict) -> pd.DataFrame:
    """
    Carga un split específico desde disco.

    Args:
        dataset_name (str): 'aptos', 'messidor' o 'odir'
        split (str):        'train', 'val' o 'test'
        seed (int):         Semilla de la partición
        config (dict):      Configuración del experimento

    Returns:
        pd.DataFrame con columnas estándar.
    """
    splits_root = _get_splits_root(config)
    csv_path = os.path.join(splits_root, f"seed_{seed}", f"{dataset_name}_{split}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[Splits] CSV no encontrado: {csv_path}\n"
            f"         Ejecuta create_or_load_splits(config, seed={seed}) primero."
        )

    return pd.read_csv(csv_path)


def get_split_path(dataset_name: str, split: str, seed: int, config: dict) -> str:
    """Retorna la ruta del CSV de un split."""
    splits_root = _get_splits_root(config)
    return os.path.join(splits_root, f"seed_{seed}", f"{dataset_name}_{split}.csv")


def print_splits_verification_table(all_splits: dict, seed: int) -> None:
    """
    Imprime la tabla de verificación de distribución de clases para todos
    los splits de una seed.
    """
    print(f"\n{'='*70}")
    print(f"  TABLA DE VERIFICACIÓN — Seed {seed}")
    print(f"{'='*70}")
    print(f"{'Dataset/Split':<25} {'Total':>6} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6}")
    print(f"{'-'*70}")

    for ds_name in ["aptos", "messidor", "odir"]:
        if ds_name not in all_splits:
            continue
        for split_name in ["train", "val", "test"]:
            df = all_splits[ds_name][split_name]
            row_label = f"{ds_name}/{split_name}"
            counts = [int((df["diagnosis"] == c).sum()) for c in range(5)]
            total = sum(counts)
            print(
                f"{row_label:<25} {total:>6} "
                + " ".join(f"{c:>6}" for c in counts)
            )
        print(f"{'-'*70}")

    print(f"{'='*70}\n")


def validate_split_feasibility(config: dict, df: pd.DataFrame, dist: dict, ds_name: str, split_name: str) -> None:
    """
    Verifica que el DataFrame tiene suficientes imágenes por clase para
    satisfacer la distribución solicitada. Lanza ValueError si no.
    """
    for cls, required in dist.items():
        available = int((df["diagnosis"] == cls).sum())
        if available < required:
            raise ValueError(
                f"\n[ERROR] No hay suficientes imágenes en {ds_name}/{split_name}.\n"
                f"  Clase {cls}: disponibles={available}, requeridas={required}.\n"
                f"  Ajusta las distribuciones en config_factorial.yaml."
            )


# =============================================================================
# Funciones internas de muestreo
# =============================================================================
def _sample_balanced(df: pd.DataFrame, dist: dict, rng: np.random.RandomState, label: str = "") -> pd.DataFrame:
    """
    Muestreo estratificado controlado: selecciona exactamente `dist[cls]`
    imágenes por clase. No es SMOTE ni generación sintética.

    Args:
        df (pd.DataFrame): DataFrame fuente con columna 'diagnosis'.
        dist (dict):       {clase: cantidad_requerida}
        rng:               RNG numpy para reproducibilidad.
        label (str):       Etiqueta de debug.

    Returns:
        pd.DataFrame con muestras seleccionadas (índice original preservado).
    """
    selected = []

    for cls, n_required in dist.items():
        cls = int(cls)
        df_cls = df[df["diagnosis"] == cls]
        available = len(df_cls)

        if available < n_required:
            raise ValueError(
                f"\n[ERROR] Dataset '{label}' — Clase {cls}: "
                f"disponibles={available}, requeridas={n_required}.\n"
                f"  Ajusta las distribuciones en config_factorial.yaml."
            )

        # Muestreo sin reemplazo
        chosen_idx = rng.choice(df_cls.index, size=n_required, replace=False)
        selected.append(df.loc[chosen_idx])

    return pd.concat(selected).reset_index(drop=False)


def _stratified_split(df: pd.DataFrame, train_ratio: float, rng: np.random.RandomState) -> tuple:
    """
    Divide un DataFrame en train/val de forma estratificada por clase.

    Returns:
        (df_train, df_val)
    """
    train_frames = []
    val_frames = []

    for cls in df["diagnosis"].unique():
        df_cls = df[df["diagnosis"] == cls].copy()
        n = len(df_cls)
        n_train = max(1, int(round(n * train_ratio)))

        # Barajar y dividir
        shuffled_idx = rng.permutation(len(df_cls))
        df_cls_reset = df_cls.reset_index(drop=True)

        train_frames.append(df_cls_reset.iloc[shuffled_idx[:n_train]])
        val_frames.append(df_cls_reset.iloc[shuffled_idx[n_train:]])

    df_train = pd.concat(train_frames).reset_index(drop=True)
    df_val = pd.concat(val_frames).reset_index(drop=True)

    return df_train, df_val


def _tag_split(df: pd.DataFrame, split: str, seed: int, source_dataset: str) -> None:
    """Agrega columnas de metadatos a un DataFrame de split (in-place)."""
    df["split"] = split
    df["seed"] = seed
    if "source_dataset" not in df.columns:
        df["source_dataset"] = source_dataset


def _parse_distribution(dist_config: dict) -> dict:
    """Convierte las claves a int (YAML puede cargarlas como int o str)."""
    return {int(k): int(v) for k, v in dist_config.items()}


def _get_splits_root(config: dict) -> str:
    """Retorna la ruta raíz de los splits."""
    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    if not os.path.isabs(factorial_output):
        factorial_output = os.path.join(_get_project_root(), factorial_output)
    return os.path.join(factorial_output, "splits")


def _all_splits_exist(seed_dir: str) -> bool:
    """Verifica que existan todos los CSV de splits para una seed."""
    datasets = ["aptos", "messidor", "odir"]
    splits = ["train", "val", "test"]
    for ds in datasets:
        for sp in splits:
            if not os.path.exists(os.path.join(seed_dir, f"{ds}_{sp}.csv")):
                return False
    return True


def _load_splits_from_disk(seed_dir: str, seed: int) -> dict:
    """Carga todos los splits desde disco para una seed."""
    result = {}
    for ds in ["aptos", "messidor", "odir"]:
        result[ds] = {}
        for sp in ["train", "val", "test"]:
            path = os.path.join(seed_dir, f"{ds}_{sp}.csv")
            result[ds][sp] = pd.read_csv(path)
    return result
