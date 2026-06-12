"""
factorial/plots.py
==================
Gráficos comparativos para el análisis del diseño experimental multifactorial.

Genera en outputs/factorial_experiments/plots/:
    accuracy_by_arch.png          — Accuracy promedio por arquitectura
    f1_by_arch.png                — F1-score promedio por arquitectura
    recall_by_arch.png            — Recall promedio por arquitectura
    comparison_by_combination.png — Base vs CBAM por combinación train-test
    heatmap_base.png              — Heatmap de generalización CNN base
    heatmap_cbam.png              — Heatmap de generalización CNN + CBAM
    heatmap_diff.png              — Diferencia CBAM ? Base
    boxplot_accuracy.png          — Boxplot de accuracy (3 semillas) por arquitectura
    boxplot_f1.png                — Boxplot de F1-score (3 semillas) por arquitectura
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para scripts
import matplotlib.pyplot as plt
import seaborn as sns

from factorial.normalization import _get_project_root

# Paleta de colores del proyecto
PALETTE = {
    "base": "#4A90D9",   # Azul institucional
    "cbam": "#E67E22",   # Naranja destacado
}

DATASET_LABELS = {
    "aptos":    "APTOS",
    "messidor": "Messidor",
    "odir":     "ODIR",
}


def generate_all_plots(df: pd.DataFrame, config: dict) -> None:
    """
    Genera todos los gráficos comparativos a partir del DataFrame consolidado.

    Args:
        df (pd.DataFrame): DataFrame de results_summary.csv.
        config (dict):     Configuración del experimento factorial.
    """
    if df.empty:
        print("[Plots] No hay datos para graficar.")
        return

    plots_dir = _get_plots_dir(config)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n[Plots] Generando gráficos en: {plots_dir}")

    _plot_metric_by_arch(df, "accuracy",  "Accuracy",   plots_dir)
    _plot_metric_by_arch(df, "f1_score",  "F1-Score",   plots_dir)
    _plot_metric_by_arch(df, "recall",    "Recall",     plots_dir)
    _plot_comparison_by_combination(df, plots_dir)
    _plot_generalization_heatmap(df, "base", plots_dir)
    _plot_generalization_heatmap(df, "cbam", plots_dir)
    _plot_heatmap_difference(df, plots_dir)
    _plot_boxplot(df, "accuracy", "Accuracy", plots_dir)
    _plot_boxplot(df, "f1_score", "F1-Score",  plots_dir)

    print(f"[Plots] ? Gráficos generados exitosamente.")


# =============================================================================
# Gráficos individuales
# =============================================================================
def _plot_metric_by_arch(df: pd.DataFrame, metric: str, metric_label: str, plots_dir: str) -> None:
    """Barras: métrica promedio (± std) por arquitectura."""
    fig, ax = plt.subplots(figsize=(6, 5))

    archs = ["base", "cbam"]
    means = [df[df["architecture"] == a][metric].mean() for a in archs]
    stds  = [df[df["architecture"] == a][metric].std()  for a in archs]
    colors = [PALETTE[a] for a in archs]

    bars = ax.bar(
        ["CNN Base", "CNN + CBAM"], means, yerr=stds,
        color=colors, edgecolor="white", linewidth=1.2,
        capsize=6, error_kw={"linewidth": 1.5}
    )

    # Anotar valores
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} Promedio por Arquitectura\n(todas las combinaciones, 3 semillas)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, f"{metric}_by_arch.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots]   ? {os.path.basename(save_path)}")


def _plot_comparison_by_combination(df: pd.DataFrame, plots_dir: str) -> None:
    """Líneas: Base vs CBAM por combinación train-test (promedio de semillas)."""
    df_agg = df.groupby(["train_dataset", "test_dataset", "architecture"])["accuracy"].mean().reset_index()
    df_agg["combination"] = df_agg["train_dataset"].str.upper() + "?" + df_agg["test_dataset"].str.upper()

    combinations = sorted(df_agg["combination"].unique())

    fig, ax = plt.subplots(figsize=(12, 5))

    for arch, style in [("base", "o-"), ("cbam", "s--")]:
        df_arch = df_agg[df_agg["architecture"] == arch]
        values = []
        for combo in combinations:
            row = df_arch[df_arch["combination"] == combo]
            values.append(row["accuracy"].values[0] if len(row) > 0 else np.nan)

        label = "CNN Base" if arch == "base" else "CNN + CBAM"
        ax.plot(range(len(combinations)), values, style,
                color=PALETTE[arch], label=label, linewidth=2, markersize=7)

    ax.set_xticks(range(len(combinations)))
    ax.set_xticklabels(combinations, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (promedio 3 semillas)", fontsize=11)
    ax.set_title("Comparación CNN Base vs CNN + CBAM\npor combinación Train ? Test", fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, "comparison_by_combination.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots]   ? {os.path.basename(save_path)}")


def _plot_generalization_heatmap(df: pd.DataFrame, architecture: str, plots_dir: str) -> None:
    """Heatmap de accuracy promedio (Train Dataset × Test Dataset) para una arquitectura."""
    df_arch = df[df["architecture"] == architecture]
    pivot = df_arch.groupby(["train_dataset", "test_dataset"])["accuracy"].mean().unstack()

    # Reordenar ejes
    order = ["aptos", "messidor", "odir"]
    pivot = pivot.reindex(index=[d for d in order if d in pivot.index],
                          columns=[d for d in order if d in pivot.columns])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"size": 11, "weight": "bold"}
    )

    arch_label = "CNN Base" if architecture == "base" else "CNN + CBAM"
    ax.set_title(f"Heatmap de Generalización — {arch_label}\n(Accuracy promedio, 3 semillas)", fontsize=11)
    ax.set_xlabel("Dataset de Test", fontsize=11)
    ax.set_ylabel("Dataset de Entrenamiento", fontsize=11)
    ax.set_xticklabels([DATASET_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], fontsize=10)
    ax.set_yticklabels([DATASET_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], fontsize=10)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, f"heatmap_{architecture}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots]   ? {os.path.basename(save_path)}")


def _plot_heatmap_difference(df: pd.DataFrame, plots_dir: str) -> None:
    """Heatmap de la diferencia de accuracy CBAM ? Base por combinación Train×Test."""
    df_base = df[df["architecture"] == "base"].groupby(
        ["train_dataset", "test_dataset"])["accuracy"].mean()
    df_cbam = df[df["architecture"] == "cbam"].groupby(
        ["train_dataset", "test_dataset"])["accuracy"].mean()

    diff = (df_cbam - df_base).unstack()

    order = ["aptos", "messidor", "odir"]
    diff = diff.reindex(index=[d for d in order if d in diff.index],
                        columns=[d for d in order if d in diff.columns])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        diff, annot=True, fmt=".3f",
        cmap="RdYlGn", center=0, vmin=-0.2, vmax=0.2,
        linewidths=0.5, ax=ax, annot_kws={"size": 11, "weight": "bold"}
    )
    ax.set_title("Diferencia de Accuracy: CBAM ? Base\n(Verde = CBAM mejor, Rojo = Base mejor)", fontsize=11)
    ax.set_xlabel("Dataset de Test", fontsize=11)
    ax.set_ylabel("Dataset de Entrenamiento", fontsize=11)
    ax.set_xticklabels([DATASET_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], fontsize=10)
    ax.set_yticklabels([DATASET_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], fontsize=10)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, "heatmap_diff.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots]   ? {os.path.basename(save_path)}")


def _plot_boxplot(df: pd.DataFrame, metric: str, metric_label: str, plots_dir: str) -> None:
    """Boxplot de una métrica por arquitectura (considerando las 3 semillas)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    df_plot = df[["architecture", metric]].copy()
    df_plot["Arquitectura"] = df_plot["architecture"].map({"base": "CNN Base", "cbam": "CNN + CBAM"})

    palette = {"CNN Base": PALETTE["base"], "CNN + CBAM": PALETTE["cbam"]}

    sns.boxplot(
        data=df_plot, x="Arquitectura", y=metric,
        palette=palette, width=0.5, ax=ax,
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        medianprops={"linewidth": 2.5, "color": "black"},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.5},
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Distribución de {metric_label} por Arquitectura\n(todas las combinaciones, 3 semillas)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, f"boxplot_{metric}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots]   ? {os.path.basename(save_path)}")


# =============================================================================
# Utilidad
# =============================================================================
def _get_plots_dir(config: dict) -> str:
    factorial_output = config.get("factorial_output_dir", "outputs/factorial_experiments")
    if not os.path.isabs(factorial_output):
        factorial_output = os.path.join(_get_project_root(), factorial_output)
    return os.path.join(factorial_output, "plots")
