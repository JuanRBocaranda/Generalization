import os
import json
import pandas as pd

def main():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    experiments = [
        'exp1_aptos_messidor',
        'exp2_messidor_aptos',
        'exp3_both_odir'
    ]

    data = []

    for exp in experiments:
        baseline_dir = os.path.join(results_dir, exp)
        cbam_dir = os.path.join(results_dir, exp + '_cbam')

        # Load baseline
        baseline_metrics_path = os.path.join(baseline_dir, 'metrics.json')
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
            data.append({
                'Experiment': exp,
                'Model': 'Baseline',
                'Accuracy': baseline_metrics.get('accuracy', 0),
                'F1 Macro': baseline_metrics.get('f1_score', 0),
                'AUC ROC': baseline_metrics.get('roc_auc', 0)
            })

        # Load CBAM
        cbam_metrics_path = os.path.join(cbam_dir, 'metrics.json')
        if os.path.exists(cbam_metrics_path):
            with open(cbam_metrics_path, 'r') as f:
                cbam_metrics = json.load(f)
            data.append({
                'Experiment': exp,
                'Model': 'CBAM',
                'Accuracy': cbam_metrics.get('accuracy', 0),
                'F1 Macro': cbam_metrics.get('f1_score', 0),
                'AUC ROC': cbam_metrics.get('roc_auc', 0)
            })

    if not data:
        print("No metrics.json files found in results directories.")
        return

    df = pd.DataFrame(data)
    # Pivot for a better view
    try:
        pivot_df = df.pivot(index='Experiment', columns='Model', values=['Accuracy', 'F1 Macro', 'AUC ROC'])
        print("\n--- Comparative Results ---")
        print(pivot_df.to_string())
    except Exception as e:
        print("\n--- Comparative Results ---")
        print(df.to_string())

if __name__ == "__main__":
    # Adjust path if run from evaluation folder
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'evaluation':
        os.chdir('..')
    main()
