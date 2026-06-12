"""
factorial/ — Módulos para el diseño experimental multifactorial.

Estructura:
    normalization.py  — Conversión ODIR paciente→imagen y carga estándar de datasets
    splits.py         — Creación de splits balanceados por seed
    class_weights.py  — Cálculo de class weights para CrossEntropyLoss
    runner.py         — Lógica de entrenamiento (18 modelos) y evaluación (54 runs)
    reporting.py      — Consolidación de resultados en CSV/XLSX
    plots.py          — Gráficos comparativos y heatmaps de generalización
"""
