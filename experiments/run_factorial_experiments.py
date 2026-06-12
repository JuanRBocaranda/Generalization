"""
experiments/run_factorial_experiments.py
========================================
Script principal del diseño experimental multifactorial.

Ejecuta 18 entrenamientos reales y 54 evaluaciones finales comparando
CNN base contra CNN + CBAM sobre los datasets APTOS, Messidor y ODIR,
usando 3 semillas de reproducibilidad.

Uso:
    # Experimento completo
    python experiments/run_factorial_experiments.py

    # Filtrar por dataset de entrenamiento
    python experiments/run_factorial_experiments.py --train_dataset aptos

    # Filtrar por arquitectura
    python experiments/run_factorial_experiments.py --architecture cbam

    # Filtrar por semilla
    python experiments/run_factorial_experiments.py --seed 42

    # Combinación de filtros
    python experiments/run_factorial_experiments.py --train_dataset aptos --architecture cbam --seed 42

    # Prueba rápida (pocas imágenes, 1 época)
    python experiments/run_factorial_experiments.py --quick_test

    # Forzar recreación de splits aunque ya existan
    python experiments/run_factorial_experiments.py --force_splits

    # Regenerar normalización de ODIR
    python experiments/run_factorial_experiments.py --force_odir

    # Solo consolidar resultados existentes (sin entrenar)
    python experiments/run_factorial_experiments.py --only_report

    # Solo generar gráficos
    python experiments/run_factorial_experiments.py --only_plots
"""

import sys
import os
import argparse
import json
import traceback
from datetime import datetime

# Agregar la raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from factorial.normalization import normalize_odir, print_dataset_summary
from factorial.splits import (
    create_or_load_splits,
    print_splits_verification_table,
)
from factorial.runner import set_seed, run_training, run_evaluation
from factorial.reporting import consolidate_results, print_summary_table, load_existing_summary
from factorial.plots import generate_all_plots


# =============================================================================
# Carga de configuración
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_factorial_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config_factorial.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {config_path}\n"
            f"Asegúrate de que config_factorial.yaml esté en la raíz del proyecto."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolver rutas relativas al PROJECT_ROOT
    for key in ["factorial_output_dir", "odir_processed_path"]:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(PROJECT_ROOT, config[key])

    for ds_key in list(config.get("dataset_paths", {}).keys()):
        path = config["dataset_paths"][ds_key]
        if not os.path.isabs(path):
            config["dataset_paths"][ds_key] = os.path.join(PROJECT_ROOT, path)

    return config


# =============================================================================
# Aplicar configuración de quick_test
# =============================================================================
def apply_quick_test_config(config: dict) -> dict:
    """
    Sobreescribe la configuración con los parámetros de quick_test.
    """
    qt = config.get("quick_test", {})

    config["training"]["num_epochs"] = qt.get("num_epochs", 1)
    config["training"]["patience"]   = qt.get("patience",   1)
    config["training"]["batch_size"] = qt.get("batch_size",  8)

    if "test_distribution" in qt:
        config["factorial_experiment"]["test_distribution"] = qt["test_distribution"]
    if "trainval_distribution" in qt:
        config["factorial_experiment"]["trainval_distribution"] = qt["trainval_distribution"]

    print("\n[QuickTest] [!] Configuración de prueba rápida activada.")
    print(f"            Épocas: {config['training']['num_epochs']}")
    print(f"            Distribución test: {config['factorial_experiment']['test_distribution']}")
    print(f"            Distribución trainval: {config['factorial_experiment']['trainval_distribution']}")

    return config


# =============================================================================
# Tabla de verificación pre-entrenamiento
# =============================================================================
def print_verification_tables(all_splits_by_seed: dict) -> None:
    """Imprime la tabla de distribución de clases para todos los splits y seeds."""
    for seed, all_splits in all_splits_by_seed.items():
        print_splits_verification_table(all_splits, seed)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Diseño Experimental Multifactorial -- Retinopatía Diabética",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_dataset", type=str, default=None,
                        choices=["aptos", "messidor", "odir"],
                        help="Filtrar por dataset de entrenamiento")
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["base", "cbam"],
                        help="Filtrar por arquitectura")
    parser.add_argument("--seed", type=int, default=None,
                        help="Filtrar por semilla específica")
    parser.add_argument("--quick_test", action="store_true",
                        help="Ejecutar prueba rápida con pocas imágenes y 1 época")
    parser.add_argument("--force_splits", action="store_true",
                        help="Forzar recreación de splits aunque ya existan")
    parser.add_argument("--force_odir", action="store_true",
                        help="Forzar regeneración del CSV normalizado de ODIR")
    parser.add_argument("--only_report", action="store_true",
                        help="Solo consolidar resultados existentes (sin entrenar)")
    parser.add_argument("--only_plots", action="store_true",
                        help="Solo generar gráficos desde results_summary.csv existente")
    parser.add_argument("--config", type=str, default=None,
                        help="Ruta al archivo de configuración (por defecto: config_factorial.yaml)")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Cargar configuración
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  DISEÑO EXPERIMENTAL MULTIFACTORIAL -- RETINOPATÍA DIABÉTICA")
    print("  'Mejoramiento de la generalización mediante módulos CBAM'")
    print("="*70)
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = load_factorial_config(args.config)

    if args.quick_test:
        config = apply_quick_test_config(config)
        # Usar solo una semilla en quick_test para ser rápido
        config["factorial_experiment"]["seeds"] = [config["factorial_experiment"]["seeds"][0]]

    # Determinar conjuntos a ejecutar según filtros
    seeds       = config["factorial_experiment"]["seeds"]
    train_dsets = config["factorial_experiment"]["train_datasets"]
    test_dsets  = config["factorial_experiment"]["test_datasets"]
    archs       = config["factorial_experiment"]["architectures"]

    if args.seed is not None:
        seeds = [s for s in seeds if s == args.seed]
    if args.train_dataset is not None:
        train_dsets = [d for d in train_dsets if d == args.train_dataset]
    if args.architecture is not None:
        archs = [a for a in archs if a == args.architecture]

    total_trainings = len(seeds) * len(train_dsets) * len(archs)
    total_evals     = total_trainings * len(test_dsets)

    print(f"\n  Seeds             : {seeds}")
    print(f"  Train datasets    : {train_dsets}")
    print(f"  Test datasets     : {test_dsets}")
    print(f"  Arquitecturas     : {archs}")
    print(f"  Entrenamientos    : {total_trainings}")
    print(f"  Evaluaciones      : {total_evals}")
    print(f"  skip_existing     : {config['factorial_experiment'].get('skip_existing', True)}")
    print(f"  class_weights     : {config.get('imbalance_handling', {}).get('use_class_weights', True)}")
    print(f"  weighted_sampler  : {config.get('imbalance_handling', {}).get('use_weighted_sampler', False)}")
    print()

    # -----------------------------------------------------------------------
    # Modo solo-reportes o solo-gráficos
    # -----------------------------------------------------------------------
    if args.only_report or args.only_plots:
        df = load_existing_summary(config)
        if df.empty:
            df = consolidate_results(config)
        if args.only_plots and not df.empty:
            generate_all_plots(df, config)
        print_summary_table(df)
        return

    # -----------------------------------------------------------------------
    # Dispositivo
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Dispositivo: {device}")
    if device.type == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------------
    # Fase 1: Normalización ODIR
    # -----------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("[Fase 1] Normalización de ODIR")
    print(f"{'-'*70}")

    try:
        normalize_odir(config, force=args.force_odir)
    except Exception as e:
        print(f"[ERROR] Falló la normalización de ODIR: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Fase 2: Crear splits balanceados por seed
    # -----------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("[Fase 2] Creación de splits balanceados")
    print(f"{'-'*70}")

    all_splits_by_seed = {}
    for seed in seeds:
        try:
            all_splits = create_or_load_splits(config, seed, force=args.force_splits)
            all_splits_by_seed[seed] = all_splits
        except ValueError as e:
            print(f"\n[ERROR CRÍTICO] {e}")
            print("El experimento no puede continuar. Ajusta las distribuciones en config_factorial.yaml.")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Falló la creación de splits para seed={seed}: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Tabla de verificación pre-entrenamiento
    print_verification_tables(all_splits_by_seed)

    # -----------------------------------------------------------------------
    # Fase 3: Entrenamiento y evaluación
    # -----------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("[Fase 3] Entrenamiento y evaluación")
    print(f"{'-'*70}")
    print(f"         Total entrenamientos: {total_trainings}")
    print(f"         Total evaluaciones  : {total_evals}")

    run_number = 0
    failed_runs = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        for train_ds in train_dsets:
            for arch in archs:
                run_number += 1
                print(f"\n[Entrenamiento {run_number}/{total_trainings}] "
                      f"{train_ds}/{arch}/seed_{seed}")
                print("-" * 60)

                # ---- Entrenamiento ----
                try:
                    model, model_dir, history = run_training(
                        train_dataset_name=train_ds,
                        architecture=arch,
                        seed=seed,
                        config=config,
                        device=device,
                    )
                except Exception as e:
                    print(f"[ERROR] Entrenamiento fallido: {train_ds}/{arch}/seed_{seed}")
                    print(f"        {e}")
                    traceback.print_exc()
                    failed_runs.append({
                        "stage": "training",
                        "seed": seed,
                        "train_dataset": train_ds,
                        "architecture": arch,
                        "error": str(e),
                    })
                    continue

                # ---- Evaluación en cada dataset de test ----
                for test_ds in test_dsets:
                    try:
                        run_evaluation(
                            model=model,
                            test_dataset_name=test_ds,
                            train_dataset_name=train_ds,
                            architecture=arch,
                            seed=seed,
                            config=config,
                            device=device,
                            model_output_dir=model_dir,
                        )
                    except Exception as e:
                        print(f"[ERROR] Evaluación fallida: {train_ds}/{arch}/seed_{seed} -> test_{test_ds}")
                        print(f"        {e}")
                        traceback.print_exc()
                        failed_runs.append({
                            "stage": "evaluation",
                            "seed": seed,
                            "train_dataset": train_ds,
                            "architecture": arch,
                            "test_dataset": test_ds,
                            "error": str(e),
                        })

    # -----------------------------------------------------------------------
    # Fase 4: Consolidación de resultados
    # -----------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("[Fase 4] Consolidación de resultados")
    print(f"{'-'*70}")

    try:
        df_results = consolidate_results(config)
        print_summary_table(df_results)
    except Exception as e:
        print(f"[ERROR] Falló la consolidación: {e}")
        traceback.print_exc()
        df_results = None

    # -----------------------------------------------------------------------
    # Fase 5: Gráficos comparativos
    # -----------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("[Fase 5] Generación de gráficos")
    print(f"{'-'*70}")

    if df_results is not None and not df_results.empty:
        try:
            generate_all_plots(df_results, config)
        except Exception as e:
            print(f"[ERROR] Falló la generación de gráficos: {e}")
            traceback.print_exc()
    else:
        print("[Plots] Sin datos suficientes para generar gráficos.")

    # -----------------------------------------------------------------------
    # Resumen final
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  EXPERIMENTO COMPLETADO")
    print(f"  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if df_results is not None:
        print(f"\n  Evaluaciones completadas: {len(df_results)}/{total_evals}")

    if failed_runs:
        print(f"\n  [!] Corridas fallidas: {len(failed_runs)}")
        for fr in failed_runs:
            if fr["stage"] == "training":
                print(f"    - Entrenamiento: {fr['train_dataset']}/{fr['architecture']}/seed_{fr['seed']}")
            else:
                print(f"    - Evaluación: {fr['train_dataset']}/{fr['architecture']}/seed_{fr['seed']} -> test_{fr['test_dataset']}")

        # Guardar registro de fallos
        factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
        failed_path = os.path.join(factorial_output, "failed_runs.json")
        os.makedirs(factorial_output, exist_ok=True)
        with open(failed_path, "w") as f:
            json.dump(failed_runs, f, indent=2)
        print(f"\n  Registro de fallos: {failed_path}")
    else:
        print("\n  [OK] Todas las corridas completadas exitosamente.")

    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    print(f"\n  Resultados en: {factorial_output}")
    print()


if __name__ == "__main__":
    main()
