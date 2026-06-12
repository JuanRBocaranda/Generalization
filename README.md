# Diabetic Retinopathy Cross-Dataset Generalization

This is a PyTorch machine learning research project for studying cross-dataset generalization of Convolutional Neural Networks (CNNs) on diabetic retinopathy detection across the APTOS, Messidor, and ODIR datasets.

## Project Structure

*   **`config.yaml`**: Centralized configuration for datasets, hyperparameters, and models.
*   **`datasets/`**: Custom PyTorch dataset loaders that standardize labels to a 0-4 severity scale.
*   **`preprocessing/`**: torchvision pipelines for image resizing, normalization, and augmentation.
*   **`models/`**: Implementations of a custom SimpleCNN and a transfer learning ResNet50 model.
*   **`training/`**: A reusable training engine featuring an epoch loop, early stopping, and model checkpointing.
*   **`evaluation/`**: Metrics computation (Accuracy, Precision, Recall, F1, ROC AUC) and confusion matrix visualization.
*   **`interpretability/`**: Generates Grad-CAM heatmaps to visualize the target layers of the CNN.
*   **`experiments/`**: Ready-to-run scripts testing various cross-dataset generalization scenarios.

## Running Experiments

Update `config.yaml` with your local dataset paths, then execute the experiment scripts from the project root:

```bash
# Train on APTOS, evaluate on Messidor
python experiments/exp1_aptos_to_messidor.py

# Train on Messidor, evaluate on APTOS
python experiments/exp2_messidor_to_aptos.py

# Train on APTOS + Messidor, evaluate on ODIR
python experiments/exp3_both_to_odir.py
```

---

# Generalización Cruzada de Conjuntos de Datos en Retinopatía Diabética

Este es un proyecto de investigación de aprendizaje automático en PyTorch para estudiar la generalización cruzada de Redes Neuronales Convolucionales (CNNs) en la detección de retinopatía diabética a través de los conjuntos de datos APTOS, Messidor y ODIR.

## Estructura del Proyecto

*   **`config.yaml`**: Configuración centralizada para conjuntos de datos, hiperparámetros y modelos.
*   **`datasets/`**: Cargadores de datos personalizados de PyTorch que estandarizan las etiquetas a una escala de gravedad de 0 a 4.
*   **`preprocessing/`**: Pipelines de torchvision para el redimensionamiento, normalización y aumento de imágenes.
*   **`models/`**: Implementaciones de una SimpleCNN personalizada y un modelo ResNet50 de aprendizaje por transferencia.
*   **`training/`**: Un motor de entrenamiento reutilizable que incluye un bucle de épocas, parada temprana (early stopping) y guardado de puntos de control (checkpointing) del modelo.
*   **`evaluation/`**: Cálculo de métricas (Precisión, Exhaustividad, F1, ROC AUC) y visualización de matrices de confusión.
*   **`interpretability/`**: Genera mapas de calor Grad-CAM para visualizar las activaciones de las CNNs.
*   **`experiments/`**: Scripts listos para ejecutar que prueban varios escenarios de generalización cruzada.

## Ejecución de Experimentos

Actualice el archivo `config.yaml` con las rutas locales de sus conjuntos de datos y luego ejecute los scripts de experimentos desde la raíz del proyecto:

```bash
# Entrenar en APTOS, evaluar en Messidor
python experiments/exp1_aptos_to_messidor.py

# Entrenar en Messidor, evaluar en APTOS
python experiments/exp2_messidor_to_aptos.py

# Entrenar en APTOS y Messidor, evaluar en ODIR
python experiments/exp3_both_to_odir.py
```
