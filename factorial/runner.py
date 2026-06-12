"""
factorial/runner.py
===================
Lógica central de entrenamiento y evaluación para el diseño experimental.

Implementa:
    set_seed(seed)                  — Fija semillas para reproducibilidad total
    run_training(...)               — Entrena un modelo (18 entrenamientos reales)
                                      Para CBAM: entrenamiento en DOS FASES si
                                      cbam_training.two_phase == True en config.
    run_evaluation(...)             — Evalúa un modelo en un dataset de test (54 runs)

Dos fases para CBAM:
    Fase 1 (warm-up):  Backbone ResNet50 CONGELADO. Solo se entrenan los
                       módulos CBAM (inyectados en bottlenecks) y la capa fc.
                       Permite que los parámetros aleatorios de CBAM converjan
                       sin desestabilizar las representaciones preentrenadas.
    Fase 2 (fine-tune): Toda la red DESCONGELADA, con learning rates diferenciados:
                        backbone_lr < cbam_lr == fc_lr. Early stopping independiente.
"""

import os
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from preprocessing.transforms import get_transforms
from models import get_model
from training.engine import train_one_epoch, validate, EarlyStopping
from datasets.loaders import get_dataset_from_split, build_weighted_sampler
from evaluation.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    save_metrics,
    save_predictions,
)
from factorial.class_weights import compute_class_weights, print_class_weights
from factorial.splits import load_split_dataframe


# =============================================================================
# Control de semillas
# =============================================================================
def set_seed(seed: int) -> None:
    """
    Fija todas las semillas para garantizar reproducibilidad completa.
    Debe llamarse antes de crear splits, DataLoaders, modelo y entrenamiento.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Función de entrenamiento de un modelo
# =============================================================================
def run_training(
    train_dataset_name: str,
    architecture: str,
    seed: int,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Entrena un modelo completo para una combinación (train_dataset, architecture, seed).

    Para architecture='base': entrenamiento estándar de una sola fase.
    Para architecture='cbam': entrenamiento en dos fases si cbam_training.two_phase=True.

    Fase 1 — Warm-up (solo CBAM):
        Congela el backbone ResNet50. Entrena únicamente los módulos CBAM
        (atributo .cbam en bottlenecks de layer2/3/4) y la capa fc.

    Fase 2 — Fine-tuning (red completa):
        Descongela todo. Usa parameter groups con learning rates diferenciados:
        backbone_lr (bajo) y cbam_lr / fc_lr (normal).

    Args:
        train_dataset_name (str): 'aptos', 'messidor' o 'odir'
        architecture (str):       'base' o 'cbam'
        seed (int):               Semilla de la corrida
        config (dict):            Configuración factorial
        device (torch.device):    Dispositivo de cómputo

    Returns:
        (model, output_dir, training_history)
    """
    use_cbam = (architecture == "cbam")
    output_dir = _get_model_output_dir(config, train_dataset_name, architecture, seed)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "training_history.json")

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # skip_existing: si el modelo ya fue entrenado, cargarlo directamente
    # -----------------------------------------------------------------------
    skip_existing = config["factorial_experiment"].get("skip_existing", True)
    if skip_existing and os.path.exists(checkpoint_path):
        print(f"\n[Runner] [SKIP] Modelo ya existe, cargando: {checkpoint_path}")
        run_config = _build_run_config(config, use_cbam)
        model = get_model(run_config).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        history = {}
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

        return model, output_dir, history

    # -----------------------------------------------------------------------
    # Fijar semilla antes de crear DataLoaders y modelo
    # -----------------------------------------------------------------------
    set_seed(seed)

    # -----------------------------------------------------------------------
    # 1. Cargar splits
    # -----------------------------------------------------------------------
    train_transform = get_transforms(config, is_training=True)
    val_transform   = get_transforms(config, is_training=False)

    train_ds = get_dataset_from_split(train_dataset_name, "train", seed, train_transform, config)
    val_ds   = get_dataset_from_split(train_dataset_name, "val",   seed, val_transform,   config)

    if len(train_ds) == 0:
        raise ValueError(
            f"[Runner] Dataset de entrenamiento vacio: {train_dataset_name}/train/seed_{seed}"
        )

    # -----------------------------------------------------------------------
    # 2. Calcular class weights a partir del split de entrenamiento
    # -----------------------------------------------------------------------
    df_train = load_split_dataframe(train_dataset_name, "train", seed, config)
    class_weights = compute_class_weights(df_train, device=device)
    print_class_weights(class_weights, train_dataset_name, seed, architecture)

    # -----------------------------------------------------------------------
    # 3. Crear DataLoaders
    # -----------------------------------------------------------------------
    num_workers = config["training"].get("num_workers", 0)
    batch_size  = config["training"]["batch_size"]
    use_sampler = config.get("imbalance_handling", {}).get("use_weighted_sampler", False)
    pin_mem     = (device.type == "cuda")

    if use_sampler:
        sampler = build_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_mem)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)

    # -----------------------------------------------------------------------
    # 4. Crear modelo
    # -----------------------------------------------------------------------
    run_config = _build_run_config(config, use_cbam)
    model = get_model(run_config).to(device)

    # -----------------------------------------------------------------------
    # 5. Loss con class weights (solo entrenamiento)
    # -----------------------------------------------------------------------
    use_class_weights = config.get("imbalance_handling", {}).get("use_class_weights", True)
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Criterio sin peso para validacion (mide rendimiento neutral)
    val_criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # 6. Entrenamiento: una fase (Base) o dos fases (CBAM)
    # -----------------------------------------------------------------------
    cbam_cfg   = config.get("cbam_training", {})
    two_phase  = use_cbam and cbam_cfg.get("two_phase", False)

    if two_phase:
        history = _run_two_phase_training(
            model, train_loader, val_loader,
            criterion, val_criterion,
            checkpoint_path, history_path,
            train_dataset_name, architecture, seed,
            cbam_cfg, device,
            len(train_ds), len(val_ds),
        )
    else:
        history = _run_standard_training(
            model, train_loader, val_loader,
            criterion, val_criterion,
            optimizer=optim.Adam(model.parameters(), lr=config["training"]["learning_rate"]),
            num_epochs=config["training"]["num_epochs"],
            patience=config["training"]["patience"],
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            train_dataset_name=train_dataset_name,
            architecture=architecture,
            seed=seed,
            device=device,
            n_train=len(train_ds),
            n_val=len(val_ds),
        )

    return model, output_dir, history


# =============================================================================
# Entrenamiento estándar — una sola fase (para CNN Base y CBAM sin warm-up)
# =============================================================================
def _run_standard_training(
    model, train_loader, val_loader,
    criterion, val_criterion, optimizer,
    num_epochs, patience, checkpoint_path, history_path,
    train_dataset_name, architecture, seed, device,
    n_train, n_val,
    phase_label="",
) -> dict:
    """Bucle de entrenamiento estándar con early stopping."""
    label = f"[Fase {phase_label}] " if phase_label else ""

    print(f"\n[Runner] {label}[START] Iniciando entrenamiento:")
    print(f"         Train dataset : {train_dataset_name}")
    print(f"         Arquitectura  : {architecture}")
    print(f"         Seed          : {seed}")
    print(f"         Epocas max.   : {num_epochs}")
    print(f"         Paciencia     : {patience}")
    print(f"         Train imgs    : {n_train}")
    print(f"         Val imgs      : {n_val}")
    print(f"         Checkpoint    : {checkpoint_path}\n")

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    train_losses = []
    val_losses   = []

    for epoch in range(num_epochs):
        print(f"  [Epoch {epoch+1}/{num_epochs}]", end=" ")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _ = validate(model, val_loader, val_criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  [Early Stopping] Activado en epoca {epoch+1}")
            break

    # Cargar el mejor checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    history = {
        "train_dataset": train_dataset_name,
        "architecture":  architecture,
        "seed":          seed,
        "use_cbam":      (architecture == "cbam"),
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_val_loss": float(min(val_losses)) if val_losses else None,
        "epochs_run":    len(train_losses),
    }

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    best = history["best_val_loss"]
    print(f"\n[Runner] [DONE] Entrenamiento completado. Mejor val_loss: {best:.4f}")
    return history


# =============================================================================
# Entrenamiento en DOS FASES — solo para CNN + CBAM
# =============================================================================
def _run_two_phase_training(
    model, train_loader, val_loader,
    criterion, val_criterion,
    checkpoint_path, history_path,
    train_dataset_name, architecture, seed,
    cbam_cfg, device,
    n_train, n_val,
) -> dict:
    """
    Entrena el modelo CBAM en dos fases:

    Fase 1 — Warm-up:
        - Congela todos los parámetros del backbone ResNet50.
        - Solo activa gradientes en módulos CBAM (bottleneck.cbam) y capa fc.
        - Objetivo: estabilizar los pesos aleatorios de CBAM sin romper
          las representaciones preentrenadas de ImageNet.

    Fase 2 — Fine-tuning:
        - Descongela toda la red.
        - Usa learning rates diferenciados por grupo de parámetros:
            backbone: backbone_lr (muy pequeño, preserva features)
            CBAM:     cbam_lr    (normal, sigue ajustando atención)
            fc:       fc_lr      (normal, ajusta clasificador)
        - Early stopping independiente, guarda el mejor checkpoint global.
    """
    warmup_epochs   = cbam_cfg.get("warmup_epochs", 5)
    warmup_lr       = cbam_cfg.get("warmup_lr", 1e-4)
    warmup_patience = cbam_cfg.get("warmup_patience", 5)

    finetune_epochs   = cbam_cfg.get("finetune_epochs", 25)
    backbone_lr       = cbam_cfg.get("backbone_lr", 1e-5)
    cbam_lr           = cbam_cfg.get("cbam_lr", 1e-4)
    fc_lr             = cbam_cfg.get("fc_lr", 1e-4)
    finetune_patience = cbam_cfg.get("finetune_patience", 5)

    # Checkpoint de warm-up (temporal) y de fine-tuning (final)
    base_dir     = os.path.dirname(checkpoint_path)
    warmup_ckpt  = os.path.join(base_dir, "warmup_model.pt")
    final_ckpt   = checkpoint_path   # best_model.pt

    # =========================================================
    # FASE 1 — Warm-up: congelar backbone, entrenar CBAM + fc
    # =========================================================
    print("\n" + "="*60)
    print("  FASE 1 / 2 — Warm-up CBAM (backbone congelado)")
    print("="*60)

    # Congelar todo
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar módulos CBAM (bottleneck.cbam en layers 2/3/4)
    cbam_params = _collect_cbam_params(model)
    for param in cbam_params:
        param.requires_grad = True

    # Descongelar capa fc
    for param in model.base_model.fc.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Parametros activos: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    warmup_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=warmup_lr,
    )

    warmup_history = _run_standard_training(
        model, train_loader, val_loader,
        criterion, val_criterion,
        optimizer=warmup_optimizer,
        num_epochs=warmup_epochs,
        patience=warmup_patience,
        checkpoint_path=warmup_ckpt,
        history_path=os.path.join(base_dir, "warmup_history.json"),
        train_dataset_name=train_dataset_name,
        architecture=architecture,
        seed=seed,
        device=device,
        n_train=n_train,
        n_val=n_val,
        phase_label="1-WarmUp",
    )

    # =========================================================
    # FASE 2 — Fine-tuning: red completa, LR diferenciados
    # =========================================================
    print("\n" + "="*60)
    print("  FASE 2 / 2 — Fine-tuning completo (LR diferenciados)")
    print("="*60)

    # Descongelar todo
    for param in model.parameters():
        param.requires_grad = True

    # Separar grupos de parámetros
    cbam_param_ids = {id(p) for p in _collect_cbam_params(model)}
    fc_param_ids   = {id(p) for p in model.base_model.fc.parameters()}

    backbone_group = [p for p in model.parameters()
                      if id(p) not in cbam_param_ids and id(p) not in fc_param_ids]
    cbam_group     = list(_collect_cbam_params(model))
    fc_group       = list(model.base_model.fc.parameters())

    print(f"  Backbone params : {sum(p.numel() for p in backbone_group):,}  LR={backbone_lr}")
    print(f"  CBAM params     : {sum(p.numel() for p in cbam_group):,}  LR={cbam_lr}")
    print(f"  FC params       : {sum(p.numel() for p in fc_group):,}  LR={fc_lr}")

    finetune_optimizer = optim.Adam([
        {"params": backbone_group, "lr": backbone_lr},
        {"params": cbam_group,     "lr": cbam_lr},
        {"params": fc_group,       "lr": fc_lr},
    ])

    finetune_history = _run_standard_training(
        model, train_loader, val_loader,
        criterion, val_criterion,
        optimizer=finetune_optimizer,
        num_epochs=finetune_epochs,
        patience=finetune_patience,
        checkpoint_path=final_ckpt,
        history_path=history_path,
        train_dataset_name=train_dataset_name,
        architecture=architecture,
        seed=seed,
        device=device,
        n_train=n_train,
        n_val=n_val,
        phase_label="2-FineTune",
    )

    # Enriquecer el historial final con info de ambas fases
    finetune_history["warmup_epochs_run"]    = warmup_history.get("epochs_run", 0)
    finetune_history["warmup_best_val_loss"] = warmup_history.get("best_val_loss")
    finetune_history["two_phase"]            = True

    with open(history_path, "w") as f:
        json.dump(finetune_history, f, indent=4)

    return finetune_history


# =============================================================================
# Función de evaluación de un modelo en un dataset de test
# =============================================================================
def run_evaluation(
    model: nn.Module,
    test_dataset_name: str,
    train_dataset_name: str,
    architecture: str,
    seed: int,
    config: dict,
    device: torch.device,
    model_output_dir: str,
) -> dict:
    """
    Evalúa un modelo entrenado en el conjunto de test de un dataset.

    Guarda:
        - metrics.json
        - predictions.csv
        - confusion_matrix.png

    Args:
        model:               Modelo entrenado (ya en eval mode).
        test_dataset_name:   Dataset de test ('aptos', 'messidor', 'odir').
        train_dataset_name:  Dataset de entrenamiento (para logging).
        architecture:        'base' o 'cbam'.
        seed:                Semilla de la corrida.
        config:              Configuración factorial.
        device:              Dispositivo de cómputo.
        model_output_dir:    Directorio base del modelo entrenado.

    Returns:
        dict con todas las métricas de evaluación.
    """
    eval_dir = os.path.join(model_output_dir, "evaluations", f"test_{test_dataset_name}")
    os.makedirs(eval_dir, exist_ok=True)

    metrics_path     = os.path.join(eval_dir, "metrics.json")
    predictions_path = os.path.join(eval_dir, "predictions.csv")
    cm_path          = os.path.join(eval_dir, "confusion_matrix.png")

    print(f"\n[Runner] [EVAL] Evaluando: {train_dataset_name}/{architecture}/seed_{seed} -> test_{test_dataset_name}")

    # -----------------------------------------------------------------------
    # Cargar test dataset con return_paths=True para guardar predicciones
    # -----------------------------------------------------------------------
    test_transform = get_transforms(config, is_training=False)
    test_ds = get_dataset_from_split(
        test_dataset_name, "test", seed, test_transform, config, return_paths=True
    )

    if len(test_ds) == 0:
        print(f"[Runner] [WARN] Dataset de test vacio: {test_dataset_name}/test/seed_{seed}")
        return {}

    num_workers = config["training"].get("num_workers", 0)
    batch_size  = config["training"]["batch_size"]
    pin_mem     = (device.type == "cuda")

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )

    # -----------------------------------------------------------------------
    # Inferencia
    # -----------------------------------------------------------------------
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds  = []
    all_labels = []
    all_probs  = []
    all_paths  = []

    with torch.no_grad():
        for batch in test_loader:
            # SplitDataset con return_paths=True retorna (image, label, path)
            if len(batch) == 3:
                inputs, labels, paths = batch
                all_paths.extend(paths)
            else:
                inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # -----------------------------------------------------------------------
    # Calcular métricas
    # -----------------------------------------------------------------------
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["val_loss"] = float(criterion(
        torch.tensor(y_prob, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.long)
    ).item())
    metrics["num_test_images"] = len(y_true)

    # Distribución real del test
    for c in range(5):
        metrics[f"test_class_{c}"] = int((y_true == c).sum())

    # -----------------------------------------------------------------------
    # Guardar resultados
    # -----------------------------------------------------------------------
    save_metrics(metrics, metrics_path)

    image_ids   = test_ds.get_image_ids()
    image_paths = all_paths if all_paths else test_ds.get_image_paths()
    save_predictions(image_ids, image_paths, y_true, y_pred, y_prob, predictions_path)

    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    print(f"[Runner]   Accuracy : {metrics['accuracy']:.4f}")
    print(f"[Runner]   F1 macro : {metrics['f1_score']:.4f}")
    print(f"[Runner]   AUC      : {metrics['auc']:.4f}")
    print(f"[Runner]   Guardado : {eval_dir}")

    return metrics


# =============================================================================
# Utilidades internas
# =============================================================================
def _collect_cbam_params(model: nn.Module):
    """
    Recolecta todos los parámetros de los módulos CBAM inyectados
    en los bottlenecks de las capas layer2, layer3 y layer4 de ResNet50.

    Los módulos CBAM se almacenan como atributo `.cbam` en cada bottleneck
    por `add_cbam_to_resnet()` en models/resnet.py.
    """
    params = []
    for layer_name in ["layer2", "layer3", "layer4"]:
        layer = getattr(model.base_model, layer_name, None)
        if layer is None:
            continue
        for bottleneck in layer.children():
            cbam_module = getattr(bottleneck, "cbam", None)
            if cbam_module is not None:
                params.extend(cbam_module.parameters())
    return params


def _get_model_output_dir(config: dict, train_dataset: str, architecture: str, seed: int) -> str:
    """Retorna el directorio de salida de un modelo entrenado."""
    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    if not os.path.isabs(factorial_output):
        from factorial.normalization import _get_project_root
        factorial_output = os.path.join(_get_project_root(), factorial_output)
    return os.path.join(factorial_output, train_dataset, architecture, f"seed_{seed}")


def _build_run_config(config: dict, use_cbam: bool) -> dict:
    """
    Construye una copia de la configuración base con use_cbam actualizado
    y los parámetros de preprocessing en el nivel esperado por get_transforms().
    """
    run_config = copy.deepcopy(config)
    run_config["model"]["use_cbam"] = use_cbam

    # get_transforms() espera config['preprocessing']
    if "preprocessing" not in run_config:
        run_config["preprocessing"] = {
            "image_size": config.get("model", {}).get("image_size", 224),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std":  [0.229, 0.224, 0.225],
        }

    return run_config
