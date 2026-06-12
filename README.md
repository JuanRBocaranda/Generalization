# Generalización Cruzada en Retinopatía Diabética (Módulos de Atención CBAM)

Este proyecto de investigación implementa un diseño experimental factorial exhaustivo en PyTorch para evaluar el impacto de los módulos de atención convolucional (CBAM) en la capacidad de generalización cruzada de redes neuronales (ResNet-50).

El estudio entrena, evalúa y cruza matrices de resultados utilizando tres bases de datos públicas de retinopatía diabética: **APTOS 2019**, **Messidor-2** y **ODIR-5K**.

## 📁 Estructura Final del Proyecto

*   **`config_factorial.yaml`**: Orquestador principal del proyecto. Define los hiperparámetros, datasets, particiones (splits) balanceadas de entrenamiento, y la configuración arquitectónica (Base vs. CBAM).
*   **`experiments/run_factorial_experiments.py`**: Script maestro que ejecuta las 54 combinaciones del experimento factorial (Dataset × Arquitectura × Semilla).
*   **`experiments/extract_dashboard_data.py`**: Script que extrae, consolida métricas (F1, Accuracy, AUC) y genera las imágenes GradCAM para el dashboard en PowerBI.
*   **`factorial/`**: Módulos de orquestación centralizada, utilidades de ploteo, balanceo de clases y reportes.
*   **`models/`**: Arquitecturas de red neuronal base y la inyección dinámica de capas de atención (ResNet-50 + CBAM).
*   **`interpretability/`**: Implementación de GradCAM para construir los mapas térmicos de interpretabilidad.
*   **`outputs/`**: Carpeta de salida principal. Aquí se almacenan los modelos entrenados, logs de épocas, la matriz maestra de resultados (`dashboard_data.json`) y las imágenes del visor unificadas (`dashboard_images/`).

## 🚀 Cómo Ejecutar

1. **Instalar Dependencias**
   Asegúrate de tener un entorno virtual activo y Python 3.9+ instalado.
   ```bash
   pip install -r requirements.txt
   ```

2. **Preparar los Datos**
   Asegúrate de que las imágenes de los datasets y sus archivos CSV se encuentren en la carpeta `datasets/` o `data/processed/`, respetando la estructura definida en `config_factorial.yaml`.

3. **Ejecutar el Experimento Factorial**
   Inicia el pipeline masivo de entrenamiento y validación cruzada.
   ```bash
   python experiments/run_factorial_experiments.py
   ```

4. **Generar los Datos para el Dashboard**
   Una vez terminados los 54 entrenamientos, extrae los KPIs consolidados y las proyecciones GradCAM:
   ```bash
   python experiments/extract_dashboard_data.py
   ```
   *Esto generará el archivo `outputs/dashboard_data.json` y los emparejamientos visuales en `outputs/dashboard_images/`, los cuales están listos para ser ingestados en la herramienta de Inteligencia de Negocios.*

## 📊 Integración con PowerBI

El pipeline final genera un formato tabular de resultados (`results_summary.csv`) junto a un modelo estructurado en JSON (`dashboard_data.json`). Estos elementos alimentan el archivo `.pbix` del repositorio, exponiendo de manera interactiva visualizaciones sobre sesgos, brechas de Accuracy vs F1-Macro, matrices de confusión dinámicas y un visor lado a lado de activaciones de atención visual (GradCAM).
