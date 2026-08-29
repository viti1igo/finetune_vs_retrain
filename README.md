# Fine-Tuning vs Retraining: Phage-Host Interaction Prediction

## Project Structure

```
finetune_vs_retrain/
├── implementation_plan.md    # Full implementation plan & literature review
├── task_list.md              # Actionable task checklist
├── README.md                 # This file
├── data/
│   ├── dataset_A_phagehostlearn/   # PhageHostLearn dataset (105 phages × 200 hosts)
│   └── dataset_B_klebphacol/       # KlebPhaCol dataset (52 phages × 74 hosts)
├── features/
│   ├── deeppbi_kg/                 # DeepPBI-KG extracted features
│   └── phagehostlearn/             # PhageHostLearn ESM-2 embeddings
├── models/
│   ├── deeppbi_kg/
│   │   ├── C1_train_B/             # Trained on B only
│   │   ├── C2_train_A/             # Trained on A only
│   │   └── C3_finetune/            # Pre-trained on A, fine-tuned on B
│   └── phagehostlearn/
│       ├── C1_train_B/
│       ├── C2_train_A/
│       └── C3_finetune/
├── results/
│   ├── confusion_matrices/
│   └── figures/
├── notebooks/                      # Analysis Jupyter notebooks
├── scripts/                        # Automation scripts
└── tools/                          # Cloned tool repositories
```

## Quick Start

1. Read `implementation_plan.md` for full context and methodology
2. Follow `task_list.md` as a working checklist
3. Clone tools into `tools/`:
   ```bash
   cd tools
   git clone https://github.com/Tongqing-Wei/DeepPBI-KG.git
   git clone https://github.com/dimiboeckaerts/PhageHostLearn.git
   ```

## Key Links

- **DeepPBI-KG**: https://github.com/Tongqing-Wei/DeepPBI-KG
- **PhageHostLearn**: https://github.com/dimiboeckaerts/PhageHostLearn
- **PhageHostLearn Data (Zenodo)**: https://zenodo.org/records/10850238
- **KlebPhaCol**: http://www.klebphacol.org
