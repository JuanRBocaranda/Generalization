"""
factorial/normalization.py
==========================
Conversión de datasets a formato estándar para el diseño experimental.

Funciones principales:
    normalize_odir(config)        — Convierte ODIR de formato paciente a imagen/ojo.
                                    Guarda data/processed/odir_image_level.csv.
    load_aptos_dataframe(config)  — Carga APTOS train y test como DataFrames estándar.
    load_messidor_dataframe(config) — Carga Messidor como DataFrame estándar.
    load_odir_dataframe(config)   — Carga el CSV procesado de ODIR.

Columnas estándar de salida:
    image_id, image_path, diagnosis, source_dataset
    (ODIR también incluye: eye, diagnostic_keywords)
"""

import os
import pandas as pd

# =============================================================================
# Mapeo de diagnósticos ODIR ? clase DR (0–4)
# Solo se incluyen textos con severidad exacta; "diabetic retinopathy" genérico
# se EXCLUYE según el diseño experimental para evitar ambigüedad.
# =============================================================================
ODIR_KEYWORD_MAP = [
    # Orden de más específico a menos específico para evitar falsos matches
    ("severe nonproliferative retinopathy",             3),
    ("severe proliferative diabetic retinopathy",       4),
    ("proliferative diabetic retinopathy",              4),
    ("moderate non proliferative retinopathy",          2),
    ("mild nonproliferative retinopathy",               1),
    ("normal fundus",                                   0),
]

# Diagnósticos excluidos explícitamente (ambiguos o sin severidad exacta)
ODIR_EXCLUDED_KEYWORDS = [
    "diabetic retinopathy",             # genérico, sin grado
    "suspected diabetic retinopathy",   # sospecha, no confirmado
    "suspicious diabetic retinopathy",
]


def _map_odir_keywords(keywords: str) -> int:
    """
    Mapea las palabras clave de diagnóstico ODIR a una clase DR (0-4).

    Returns:
        int: Clase 0-4, o -1 si se debe excluir.
    """
    kw = str(keywords).strip().lower()

    # Mapear términos exactos primero (en orden de especificidad)
    for keyword, label in ODIR_KEYWORD_MAP:
        if keyword in kw:
            return label

    # Excluir términos ambiguos SOLO SI no hubo match positivo previo
    for excluded in ODIR_EXCLUDED_KEYWORDS:
        if excluded in kw:
            return -1

    # Cualquier otro diagnóstico no relacionado con DR
    return -1


def normalize_odir(config: dict, force: bool = False) -> str:
    """
    Convierte ODIR de formato paciente (una fila por paciente, dos ojos)
    a formato imagen/ojo (una fila por imagen).

    Pasos:
        1. Lee el Excel de anotaciones de ODIR.
        2. Genera dos filas por paciente (Left-Fundus, Right-Fundus).
        3. Mapea diagnósticos a clases 0–4 (excluye ambiguos).
        4. Verifica existencia física de cada imagen.
        5. Guarda el CSV procesado.

    Args:
        config (dict): Configuración del experimento factorial.
        force (bool):  Si True, regenera aunque ya exista el CSV.

    Returns:
        str: Ruta al CSV generado.
    """
    output_path = config.get("odir_processed_path", "data/processed/odir_image_level.csv")

    # Convertir a ruta absoluta si es relativa (relativa a PROJECT_ROOT)
    if not os.path.isabs(output_path):
        project_root = _get_project_root()
        output_path = os.path.join(project_root, output_path)

    if os.path.exists(output_path) and not force:
        print(f"[ODIR] CSV normalizado ya existe: {output_path}")
        print(f"[ODIR] Usa force=True para regenerar.")
        return output_path

    odir_dir = config["dataset_paths"]["odir"]
    if not os.path.isabs(odir_dir):
        odir_dir = os.path.join(_get_project_root(), odir_dir)

    xlsx_path = os.path.join(odir_dir, "ODIR-5K_Training_Annotations(Updated)_V2.xlsx")
    img_folder = os.path.join(odir_dir, "ODIR-5K_Training_Dataset")

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"[ODIR] Archivo de anotaciones no encontrado: {xlsx_path}")
    if not os.path.isdir(img_folder):
        raise FileNotFoundError(f"[ODIR] Carpeta de imágenes no encontrada: {img_folder}")

    print(f"[ODIR] Leyendo anotaciones: {xlsx_path}")
    df_raw = pd.read_excel(xlsx_path)

    records = []
    skipped_label = 0
    skipped_missing = 0

    for _, row in df_raw.iterrows():
        patient_id = row.get("ID", row.get("Patient ID", "unknown"))

        for side in ["Left", "Right"]:
            fundus_col = f"{side}-Fundus"
            keywords_col = f"{side}-Diagnostic Keywords"

            if fundus_col not in row or keywords_col not in row:
                continue

            img_filename = str(row[fundus_col]).strip()
            keywords = str(row[keywords_col]).strip()

            # Mapear diagnóstico a clase
            label = _map_odir_keywords(keywords)
            if label == -1:
                skipped_label += 1
                continue

            # Verificar existencia física de la imagen
            img_path = os.path.join(img_folder, img_filename)
            if not os.path.exists(img_path):
                skipped_missing += 1
                continue

            records.append({
                "image_id":           img_filename,
                "image_path":         img_path,
                "eye":                side.lower(),
                "diagnostic_keywords": keywords,
                "diagnosis":          int(label),
                "source_dataset":     "odir",
            })

    df_out = pd.DataFrame(records)

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    # Resumen
    print(f"[ODIR] Normalización completada.")
    print(f"       Registros generados : {len(df_out)}")
    print(f"       Excluidos por label  : {skipped_label}")
    print(f"       Excluidos por imagen : {skipped_missing}")
    print(f"       Distribución por clase:")
    for cls in sorted(df_out["diagnosis"].unique()):
        n = (df_out["diagnosis"] == cls).sum()
        print(f"         Clase {cls}: {n}")
    print(f"       Guardado en: {output_path}")

    return output_path


def load_aptos_dataframe(config: dict) -> tuple:
    """
    Carga los DataFrames de APTOS train y test en formato estándar.

    Returns:
        (df_trainval, df_test): Tupla de DataFrames con columnas estándar.
    """
    aptos_dir = config["dataset_paths"]["aptos"]
    if not os.path.isabs(aptos_dir):
        aptos_dir = os.path.join(_get_project_root(), aptos_dir)

    # Train (se usa "train - APTOS.csv" tal como especifica el diseño)
    trainval_csv = os.path.join(aptos_dir, "train - APTOS.csv")
    if not os.path.exists(trainval_csv):
        # Fallback a train.csv si el nombre con espacio no existe
        trainval_csv = os.path.join(aptos_dir, "train.csv")

    test_csv = os.path.join(aptos_dir, "test.csv")

    df_trainval = _load_aptos_csv(trainval_csv, aptos_dir, "train_images", split_hint="trainval")
    df_test = _load_aptos_csv(test_csv, aptos_dir, "test_images", split_hint="test")

    return df_trainval, df_test


def _load_aptos_csv(csv_path: str, base_dir: str, img_subfolder: str, split_hint: str = "") -> pd.DataFrame:
    """Carga un CSV de APTOS y lo convierte al formato estándar."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[APTOS] CSV no encontrado: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # La carpeta de imágenes puede estar en una subcarpeta con el mismo nombre
    img_folder_direct = os.path.join(base_dir, img_subfolder)
    img_folder_nested = os.path.join(base_dir, img_subfolder, img_subfolder)

    records = []
    missing = 0
    for _, row in df_raw.iterrows():
        img_id = str(row["id_code"]).strip()

        # Intentar ambas rutas posibles
        img_path = os.path.join(img_folder_nested, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_folder_direct, f"{img_id}.png")
        if not os.path.exists(img_path):
            missing += 1
            continue

        label = row.get("diagnosis", -1)
        if pd.isna(label) or int(label) < 0:
            continue

        records.append({
            "image_id":       img_id,
            "image_path":     img_path,
            "diagnosis":      int(label),
            "source_dataset": "aptos",
        })

    if missing > 0:
        print(f"[APTOS/{split_hint}] {missing} imágenes no encontradas en disco.")

    return pd.DataFrame(records)


def load_messidor_dataframe(config: dict) -> pd.DataFrame:
    """
    Carga Messidor en formato estándar.

    Returns:
        df: DataFrame con columnas estándar.
    """
    messidor_dir = config["dataset_paths"]["messidor"]
    if not os.path.isabs(messidor_dir):
        messidor_dir = os.path.join(_get_project_root(), messidor_dir)

    csv_path = os.path.join(messidor_dir, "messidor_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Messidor] CSV no encontrado: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Ruta de imágenes (estructura conocida del proyecto)
    img_folder = os.path.join(messidor_dir, "images", "messidor-2", "preprocess")

    records = []
    missing = 0
    for _, row in df_raw.iterrows():
        img_id = str(row["id_code"]).strip()
        img_name = img_id if img_id.lower().endswith((".jpg", ".png")) else img_id + ".jpg"
        img_path = os.path.join(img_folder, img_name)

        if not os.path.exists(img_path):
            missing += 1
            continue

        label = row.get("diagnosis", row.get("retinopathy_grade", -1))
        if pd.isna(label) or int(label) < 0:
            continue

        records.append({
            "image_id":       img_id,
            "image_path":     img_path,
            "diagnosis":      int(label),
            "source_dataset": "messidor",
        })

    if missing > 0:
        print(f"[Messidor] {missing} imágenes no encontradas en disco.")

    return pd.DataFrame(records)


def load_odir_dataframe(config: dict) -> pd.DataFrame:
    """
    Carga el CSV normalizado de ODIR (ya procesado por normalize_odir).

    Returns:
        df: DataFrame con columnas estándar + eye + diagnostic_keywords.
    """
    output_path = config.get("odir_processed_path", "data/processed/odir_image_level.csv")
    if not os.path.isabs(output_path):
        output_path = os.path.join(_get_project_root(), output_path)

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"[ODIR] CSV procesado no encontrado: {output_path}\n"
            f"       Ejecuta normalize_odir(config) primero."
        )

    df = pd.read_csv(output_path)

    # Verificar columnas mínimas
    required = ["image_id", "image_path", "diagnosis", "source_dataset"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[ODIR] CSV procesado falta columna: '{col}'")

    return df


# =============================================================================
# Utilidades internas
# =============================================================================
def _get_project_root() -> str:
    """Retorna la ruta absoluta del directorio raíz del proyecto."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    """Imprime un resumen de distribución de clases de un DataFrame."""
    print(f"\n[{name.upper()}] Total: {len(df)} imágenes")
    for cls in range(5):
        n = (df["diagnosis"] == cls).sum()
        print(f"  Clase {cls}: {n}")
