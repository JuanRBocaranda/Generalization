# Generalización Cruzada en Retinopatía Diabética (Módulos de Atención CBAM)

Este proyecto de investigación implementa un diseño experimental factorial exhaustivo en PyTorch para evaluar el impacto de los módulos de atención convolucional (CBAM) en la capacidad de generalización cruzada de redes neuronales (ResNet-50).

El estudio entrena, evalúa y cruza matrices de resultados utilizando tres bases de datos públicas de retinopatía diabética: **APTOS 2019**, **Messidor-2** y **ODIR-5K**.

## Fuentes de los Conjuntos de Datos (Datasets)

Para reproducir este proyecto desde cero, debes descargar las imágenes originales desde sus respectivas fuentes oficiales o competencias de Kaggle y ubicarlas en la carpeta `datasets/`:

1. **APTOS 2019 Blindness Detection**: 
   - [Página Oficial en Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)
   - Proporcionado originalmente por Aravind Eye Hospital. Contiene miles de fotografías de fondo de ojo etiquetadas en una escala del 0 al 4.

2. **Messidor-2**: 
   - [Página Oficial (ADCIS)](https://www.adcis.net/en/third-party/messidor2/) | [Enlace Alternativo (Kaggle)](https://www.kaggle.com/datasets/google-brain/messidor2)
   - Contiene 1,748 imágenes. Utilizado extensamente en la investigación de retinopatía diabética.

3. **ODIR-5K (Ocular Disease Intelligent Recognition)**: 
   - [ODIR-2019 Grand Challenge](https://odir2019.grand-challenge.org/) | [Enlace Alternativo (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
   - Colección clínica que incluye información detallada y múltiples patologías (Normal, Glaucoma, Cataratas, etc.).

## Estructura Final del Proyecto

*   **`config_factorial.yaml`**: Orquestador principal del proyecto. Define los hiperparámetros, datasets, particiones (splits) balanceadas de entrenamiento, y la configuración arquitectónica (Base vs. CBAM).
*   **`experiments/run_factorial_experiments.py`**: Script maestro que ejecuta las 54 combinaciones del experimento factorial cruzado (Dataset × Arquitectura × Semilla).
*   **`experiments/extract_dashboard_data.py`**: Script final que extrae, consolida métricas (F1, Accuracy, AUC) y genera las imágenes térmicas GradCAM para el dashboard en PowerBI.
*   **`factorial/`**: Módulos de orquestación centralizada (utilidades de gráficos, balanceo de clases y lógica del runner).
*   **`models/`**: Arquitecturas de red neuronal base y la inyección dinámica de capas de atención (ResNet-50 + CBAM).
*   **`interpretability/`**: Implementación nativa de GradCAM para construir mapas térmicos de activación basados en gradientes.
*   **`outputs/`**: Carpeta de salida principal. Alberga los modelos pesados, historial de épocas, la matriz final JSON (`dashboard_data.json`) y las imágenes estandarizadas para el dashboard (`dashboard_images/`).

## Cómo Ejecutar

1. **Instalar Dependencias**
   Asegúrate de contar con Python 3.9+ e instala los requerimientos:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el Experimento Factorial**
   Inicia el pipeline masivo de entrenamiento y validación cruzada:
   ```bash
   python experiments/run_factorial_experiments.py
   ```

3. **Generar los Datos para el Dashboard**
   Una vez terminados los 54 entrenamientos, consolida los KPIs y proyecciones GradCAM:
   ```bash
   python experiments/extract_dashboard_data.py
   ```

## 📊 Integración con PowerBI

El framework arroja métricas tabulares e imágenes que alimentan directamente el archivo `.pbix` de PowerBI incluido en este repositorio. En él se exponen visualizaciones para analizar sesgos cognitivos del modelo, brechas de Accuracy y el comparador visual dinámico (Original vs GradCAM).
