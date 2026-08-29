import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Volumes/TRANSCEND/Projects/VinUni/finetune_vs_retrain")
CM_CSV = BASE_DIR / "results" / "confusion_matrices" / "confusion_matrix_metrics_summary.csv"
OUT_CSV = BASE_DIR / "results" / "metrics_comparison.csv"

if CM_CSV.exists():
    df_cm = pd.read_csv(CM_CSV)
    df_out = pd.DataFrame({
        "Condition": df_cm["Condition"],
        "ROC_AUC": df_cm["ROC_AUC"],
        "PR_AUC": df_cm["PR_AUC"],
        "MCC": df_cm["MCC"],
        "F1": df_cm["F1_Score"],
        "Accuracy": df_cm["Accuracy"],
        "Tool": df_cm["Model"]
    })
    df_out.to_csv(OUT_CSV, index=False)
    print("Updated results/metrics_comparison.csv with realistic balanced test metrics:")
    print(df_out.to_string())
