import os
import json
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import cv2
from PIL import Image

# Import GradCAM and model logic
import sys
# Add parent directory to sys.path to allow importing from interpretability and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpretability.gradcam import GradCAM, save_gradcam, get_target_layer
from models.resnet import ResNet50Transfer
import yaml
from torchvision import transforms

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_metrics_and_predictions(base_dir):
    data = []
    # Pattern: outputs/factorial_experiments/{train}/{arch}/seed_{seed}/evaluations/test_{test}/metrics.json
    pattern = os.path.join(base_dir, "*", "*", "seed_*", "evaluations", "test_*", "metrics.json")
    for filepath in glob.glob(pattern):
        parts = os.path.normpath(filepath).split(os.sep)
        # base_dir is part 0,1 ..
        test = parts[-2].replace("test_", "")
        seed = parts[-4].replace("seed_", "")
        arch = parts[-5]
        train = parts[-6]
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
            
        pred_path = filepath.replace('metrics.json', 'predictions.csv')
        preds_df = pd.DataFrame()
        if os.path.exists(pred_path):
            preds_df = pd.read_csv(pred_path)
        else:
            print(f"Warning: predictions.csv not found at {pred_path}")
            
        data.append({
            'train': train,
            'arch': arch,
            'seed': int(seed),
            'test': test,
            'metrics': metrics,
            'preds': preds_df,
            'eval_dir': os.path.dirname(filepath),
            'model_dir': os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
        })
    return data

def calculate_dashboard_data(data, base_dir, config):
    output = {
        "resumen_ejecutivo": {},
        "generalizacion": {
            "matriz_base": {},
            "matriz_cbam": {},
            "ganancia_cbam_pp": {}
        },
        "comparativa_modelos": {
            "f1_por_origen": {},
            "auc_por_dataset_prueba": {},
            "recall_clases_graves_3_4": {}
        },
        "clases_criticas": {
            "recall_por_clase": {"base": {}, "cbam": {}, "mejora_pp": {}},
            "clase_con_mayor_mejora": ""
        },
        "estabilidad_semillas": {},
        "visor_gradcam": {
            "casos_por_combinacion": {},
            "requires_gradcam_generation": [],
            "confianza_promedio": {"base": {}, "cbam": {}},
            "textos_clinicos": {
                "clase_0": "Sin hallazgos patológicos. Retina dentro de parámetros normales.",
                "clase_1": "Microaneurismas aislados. Seguimiento anual recomendado.",
                "clase_2": "Exudados y hemorragias presentes. Control oftalmológico en 6 meses.",
                "clase_3": "Neovascularización o hemorragias extensas. Derivación urgente.",
                "clase_4": "Retinopatía proliferativa activa. Intervención inmediata requerida."
            }
        }
    }
    
    if not data:
        print("No data found!")
        return output

    # 1. Resumen Ejecutivo
    output["resumen_ejecutivo"]["total_evaluaciones"] = len(data)
    
    # 1.4 Mejor F1 macro
    best_f1 = -1
    best_combo = {}
    for d in data:
        f1 = d['metrics'].get('f1_score', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_combo = {
                "train": d['train'],
                "test": d['test'],
                "architecture": d['arch'],
                "seed": d['seed']
            }
    output["resumen_ejecutivo"]["mejor_f1_macro"] = {
        "valor": round(best_f1 * 100, 1),
        "combinacion": best_combo
    }
    
    # 1.1 Métricas globales de arquitectura
    f1_base_list = [d['metrics'].get('f1_score', 0) for d in data if d['arch'] == 'base']
    f1_cbam_list = [d['metrics'].get('f1_score', 0) for d in data if d['arch'] == 'cbam']
    acc_base_list = [d['metrics'].get('accuracy', 0) for d in data if d['arch'] == 'base']
    acc_cbam_list = [d['metrics'].get('accuracy', 0) for d in data if d['arch'] == 'cbam']
    auc_base_list = [d['metrics'].get('auc', 0) for d in data if d['arch'] == 'base']
    auc_cbam_list = [d['metrics'].get('auc', 0) for d in data if d['arch'] == 'cbam']
    
    avg_f1_base = np.mean(f1_base_list) * 100 if f1_base_list else 0
    avg_f1_cbam = np.mean(f1_cbam_list) * 100 if f1_cbam_list else 0
    avg_acc_base = np.mean(acc_base_list) * 100 if acc_base_list else 0
    avg_acc_cbam = np.mean(acc_cbam_list) * 100 if acc_cbam_list else 0
    
    output["resumen_ejecutivo"]["f1_macro_por_arquitectura"] = {
        "base": round(avg_f1_base, 1),
        "cbam": round(avg_f1_cbam, 1),
        "mejora_cbam_pp": round(avg_f1_cbam - avg_f1_base, 1)
    }
    
    output["resumen_ejecutivo"]["accuracy_por_arquitectura"] = {
        "base": round(avg_acc_base, 1),
        "cbam": round(avg_acc_cbam, 1),
        "mejora_cbam_pp": round(avg_acc_cbam - avg_acc_base, 1),
        "brecha_vs_f1_macro": {
            "base": round(avg_acc_base - avg_f1_base, 1),
            "cbam": round(avg_acc_cbam - avg_f1_cbam, 1),
            "interpretacion": "Una brecha positiva indica sesgo hacia clases mayoritarias."
        }
    }
    
    output["resumen_ejecutivo"]["auc_por_arquitectura"] = {
        "base": round(np.mean(auc_base_list), 2) if auc_base_list else 0,
        "cbam": round(np.mean(auc_cbam_list), 2) if auc_cbam_list else 0
    }
    
    # 1.2 F1 macro promedio por dataset de entrenamiento
    for ds in ['aptos', 'messidor', 'odir']:
        f1_list = [d['metrics'].get('f1_score', 0) for d in data if d['train'] == ds]
        acc_list = [d['metrics'].get('accuracy', 0) for d in data if d['train'] == ds]
        output["resumen_ejecutivo"].setdefault("f1_macro_por_dataset_entrenamiento", {})[ds] = round(np.mean(f1_list)*100, 1) if f1_list else 0
        output["resumen_ejecutivo"].setdefault("accuracy_por_dataset_entrenamiento", {})[ds] = round(np.mean(acc_list)*100, 1) if acc_list else 0
        
    # 1.3 F1 macro promedio por semilla
    for s in [42, 123, 2025]:
        f1_list = [d['metrics'].get('f1_score', 0) for d in data if d['seed'] == s]
        acc_list = [d['metrics'].get('accuracy', 0) for d in data if d['seed'] == s]
        output["resumen_ejecutivo"].setdefault("f1_macro_por_semilla", {})[str(s)] = round(np.mean(f1_list)*100, 1) if f1_list else 0
        output["resumen_ejecutivo"].setdefault("accuracy_por_semilla", {})[str(s)] = round(np.mean(acc_list)*100, 1) if acc_list else 0
        
    seed_f1s = [output["resumen_ejecutivo"]["f1_macro_por_semilla"][str(s)] for s in [42, 123, 2025] if f1_list]
    seed_accs = [output["resumen_ejecutivo"]["accuracy_por_semilla"][str(s)] for s in [42, 123, 2025] if acc_list]
    if seed_f1s:
        output["resumen_ejecutivo"]["f1_macro_por_semilla"]["variacion_pp"] = round(max(seed_f1s) - min(seed_f1s), 1)
        output["resumen_ejecutivo"]["accuracy_por_semilla"]["variacion_pp"] = round(max(seed_accs) - min(seed_accs), 1)
    else:
        output["resumen_ejecutivo"]["f1_macro_por_semilla"]["variacion_pp"] = 0
        output["resumen_ejecutivo"]["accuracy_por_semilla"]["variacion_pp"] = 0

    # 2. Generalización (MATRIZ CRUZADA)
    output["generalizacion"]["matriz_base_acc"] = {}
    output["generalizacion"]["matriz_cbam_acc"] = {}
    output["generalizacion"]["ganancia_cbam_acc_pp"] = {}
    
    for arch, mat_name, mat_acc_name in [('base', 'matriz_base', 'matriz_base_acc'), ('cbam', 'matriz_cbam', 'matriz_cbam_acc')]:
        for train in ['aptos', 'messidor', 'odir']:
            for test in ['aptos', 'messidor', 'odir']:
                f1_list = [d['metrics'].get('f1_score', 0) for d in data if d['train'] == train and d['test'] == test and d['arch'] == arch]
                acc_list = [d['metrics'].get('accuracy', 0) for d in data if d['train'] == train and d['test'] == test and d['arch'] == arch]
                output["generalizacion"][mat_name][f"{train}_{test}"] = round(np.mean(f1_list)*100, 1) if f1_list else 0
                output["generalizacion"][mat_acc_name][f"{train}_{test}"] = round(np.mean(acc_list)*100, 1) if acc_list else 0

    for train in ['aptos', 'messidor', 'odir']:
        for test in ['aptos', 'messidor', 'odir']:
            k = f"{train}_{test}"
            base_val = output["generalizacion"]["matriz_base"][k]
            cbam_val = output["generalizacion"]["matriz_cbam"][k]
            output["generalizacion"]["ganancia_cbam_pp"][k] = round(cbam_val - base_val, 1)
            
            base_acc_val = output["generalizacion"]["matriz_base_acc"][k]
            cbam_acc_val = output["generalizacion"]["matriz_cbam_acc"][k]
            output["generalizacion"]["ganancia_cbam_acc_pp"][k] = round(cbam_acc_val - base_acc_val, 1)

    # 3. Comparativa de Modelos
    for ds in ['aptos', 'messidor', 'odir']:
        output["comparativa_modelos"]["f1_por_origen"][ds] = {}
        for arch in ['base', 'cbam']:
            f1_list = [d['metrics'].get('f1_score', 0) for d in data if d['train'] == ds and d['arch'] == arch]
            output["comparativa_modelos"]["f1_por_origen"][ds][arch] = round(np.mean(f1_list)*100, 1) if f1_list else 0

    for ds in ['aptos', 'messidor', 'odir']:
        output["comparativa_modelos"]["auc_por_dataset_prueba"][ds] = {}
        for arch in ['base', 'cbam']:
            auc_list = [d['metrics'].get('auc', 0) for d in data if d['test'] == ds and d['arch'] == arch]
            output["comparativa_modelos"]["auc_por_dataset_prueba"][ds][arch] = round(np.mean(auc_list), 2) if auc_list else 0

    # Recall clases graves
    for arch in ['base', 'cbam']:
        recall_list = []
        for d in data:
            if d['arch'] == arch:
                r3 = d['metrics'].get('recall_class_3', 0)
                r4 = d['metrics'].get('recall_class_4', 0)
                recall_list.extend([r3, r4])
        output["comparativa_modelos"]["recall_clases_graves_3_4"][arch] = round(np.mean(recall_list)*100, 1) if recall_list else 0

    # 4. Clases Críticas
    for arch in ['base', 'cbam']:
        for i in range(5):
            cls_str = str(i)
            r_list = [d['metrics'].get(f'recall_class_{i}', 0) for d in data if d['arch'] == arch]
            output["clases_criticas"]["recall_por_clase"][arch][f"clase_{i}"] = round(np.mean(r_list)*100, 1) if r_list else 0

    max_mejora = -100
    mejor_clase = ""
    for i in range(5):
        k = f"clase_{i}"
        base_v = output["clases_criticas"]["recall_por_clase"]["base"][k]
        cbam_v = output["clases_criticas"]["recall_por_clase"]["cbam"][k]
        diff = round(cbam_v - base_v, 1)
        output["clases_criticas"]["recall_por_clase"]["mejora_pp"][k] = diff
        if diff > max_mejora:
            max_mejora = diff
            mejor_clase = k
    output["clases_criticas"]["clase_con_mayor_mejora"] = mejor_clase

    # 5. Estabilidad Semillas
    for train in ['aptos', 'messidor', 'odir']:
        for arch in ['base', 'cbam']:
            key = f"{train}_{arch}"
            f1_dict = {}
            for s in [42, 123, 2025]:
                f1_list = [d['metrics'].get('f1_score', 0) for d in data if d['train'] == train and d['arch'] == arch and d['seed'] == s]
                f1_dict[f"seed_{s}"] = round(np.mean(f1_list)*100, 1) if f1_list else 0
            
            vals = list(f1_dict.values())
            output["estabilidad_semillas"][key] = {
                "seed_42": f1_dict["seed_42"],
                "seed_123": f1_dict["seed_123"],
                "seed_2025": f1_dict["seed_2025"],
                "media": round(np.mean(vals), 1) if vals else 0,
                "std": round(np.std(vals), 1) if vals else 0,
                "rango": round(max(vals) - min(vals), 1) if vals else 0
            }

    # 6. Visor GradCAM
    # Confianzas
    for arch in ['base', 'cbam']:
        corr_conf = []
        incorr_conf = []
        for d in data:
            if d['arch'] == arch and not d['preds'].empty:
                df = d['preds']
                correct = df[df['true_label'] == df['predicted_label']]
                incorrect = df[df['true_label'] != df['predicted_label']]
                
                probs = df[[c for c in df.columns if c.startswith('prob_')]].max(axis=1) * 100
                corr_conf.extend(probs[df['true_label'] == df['predicted_label']].tolist())
                incorr_conf.extend(probs[df['true_label'] != df['predicted_label']].tolist())
                
        output["visor_gradcam"]["confianza_promedio"][arch] = {
            "correctas": round(np.mean(corr_conf), 1) if corr_conf else 0,
            "incorrectas": round(np.mean(incorr_conf), 1) if incorr_conf else 0
        }

    # Selección de casos y generación de GradCAM
    for test_ds in ['aptos', 'messidor', 'odir']:
        for arch in ['base', 'cbam']:
            # Find the target run
            target_run = None
            for s in [42, 123, 2025]:
                matches = [d for d in data if d['arch'] == arch and d['test'] == test_ds and d['train'] == test_ds and d['seed'] == s]
                if matches and not matches[0]['preds'].empty:
                    target_run = matches[0]
                    break
            
            combo_key = f"{arch}_{test_ds}"
            output["visor_gradcam"]["casos_por_combinacion"][combo_key] = []
            
            if not target_run:
                output["visor_gradcam"]["requires_gradcam_generation"].append(combo_key)
                continue
                
            df = target_run['preds']
            prob_cols = [c for c in df.columns if c.startswith('prob_class_')]
            if not prob_cols:
                continue
                
            df['confidence'] = df[prob_cols].max(axis=1) * 100
            df['correct'] = df['true_label'] == df['predicted_label']
            
            selected_indices = []
            
            # 1 por clase
            for c in range(5):
                class_df = df[df['true_label'] == c]
                if class_df.empty:
                    continue
                correct_df = class_df[class_df['correct'] == True]
                if not correct_df.empty:
                    idx = correct_df['confidence'].idxmax()
                else:
                    idx = class_df['confidence'].idxmax()
                selected_indices.append(idx)
                
            # 1 grave (3 o 4) con mayor confianza no seleccionado aún
            grave_df = df[(df['true_label'].isin([3, 4])) & (~df.index.isin(selected_indices))]
            if not grave_df.empty:
                correct_grave = grave_df[grave_df['correct'] == True]
                if not correct_grave.empty:
                    idx = correct_grave['confidence'].idxmax()
                else:
                    idx = grave_df['confidence'].idxmax()
                selected_indices.append(idx)
                
            selected_df = df.loc[selected_indices]
            
            # Generate GradCAMs
            model_path = os.path.join(target_run['model_dir'], 'best_model.pt')
            gradcam_dir = os.path.join(target_run['eval_dir'], 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            
            model = None
            if os.path.exists(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cfg = {'model': {'use_cbam': arch == 'cbam', 'cbam_layers': ['layer2', 'layer3', 'layer4']}}
                model = ResNet50Transfer(num_classes=5, pretrained=False, config=cfg)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                
                # Setup GradCAM
                target_layer = get_target_layer(model, {'model': {'use_cbam': arch == 'cbam'}})
                cam_extractor = GradCAM(model, target_layer)
                
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                print(f"Model path not found: {model_path}")

            for _, row in selected_df.iterrows():
                filename = str(row['image_id'])
                img_path = str(row['image_path'])
                
                gradcam_path = None
                if model and img_path:
                    try:
                        orig_image = cv2.imread(img_path)
                        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                        orig_image = cv2.resize(orig_image, (224, 224))
                        
                        pil_img = Image.fromarray(orig_image)
                        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                        
                        heatmap = cam_extractor(input_tensor, class_idx=int(row['predicted_label']))
                        
                        gradcam_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_{row['predicted_label']}.png"
                        save_dest = os.path.join(gradcam_dir, gradcam_filename)
                        save_gradcam(input_tensor[0], heatmap, save_dest, original_image=orig_image)
                        
                        gradcam_path = os.path.relpath(save_dest).replace("\\", "/")
                    except Exception as e:
                        print(f"Failed to generate GradCAM for {filename}: {e}")
                
                case_data = {
                    "image_id": filename,
                    "dataset_origen": test_ds,
                    "clase_real": int(row['true_label']),
                    "clase_predicha": int(row['predicted_label']),
                    "correcto": bool(row['correct']),
                    "confianza": float(row['confidence']),
                    "probabilidades": {
                        "clase_0": float(row['prob_class_0']),
                        "clase_1": float(row['prob_class_1']),
                        "clase_2": float(row['prob_class_2']),
                        "clase_3": float(row['prob_class_3']),
                        "clase_4": float(row['prob_class_4'])
                    },
                    "original_image_path": os.path.relpath(img_path).replace("\\", "/") if img_path and os.path.exists(img_path) else img_path,
                    "gradcam_path": gradcam_path
                }
                output["visor_gradcam"]["casos_por_combinacion"][combo_key].append(case_data)
                
            if not any(c['gradcam_path'] for c in output["visor_gradcam"]["casos_por_combinacion"][combo_key]):
                output["visor_gradcam"]["requires_gradcam_generation"].append(combo_key)

    return output

if __name__ == "__main__":
    base_dir = "outputs/factorial_experiments"
    config = load_yaml_config("config_factorial.yaml")
    
    print("Loading metrics and predictions...")
    data = load_metrics_and_predictions(base_dir)
    print(f"Processed evaluations: {len(data)}/54")
    
    if len(data) < 54:
        print(f"WARNING: Missing data! Expected 54 evaluations, found {len(data)}")
        
    print("Calculating dashboard data and generating GradCAMs...")
    dashboard_data = calculate_dashboard_data(data, base_dir, config)
    
    out_path = "outputs/dashboard_data.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
    print(f"Dashboard data successfully saved to {out_path}")
