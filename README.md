# EGFR_RF_Modeling

Reproduction of a machine learning-based pipeline from Nada et al. (2023), which integrates cheminformatics and machine learning to predict potent EGFR inhibitors for breast cancer. This project showcases expertise in molecular descriptor generation (RDKit), regression modeling (Random Forest, XGBoost), and bioactivity prediction.

Nada et al. (2023) - *A Machine Learning-Based Approach to Developing Potent EGFR Inhibitors for Breast Cancer: Design, Synthesis, and Biological Evaluation* ([ACS Omega, 2023](https://doi.org/10.1021/acsomega.3c02799))

---

## ✅ TODO (Planned Enhancements)

- [x] Generate RDKit descriptors from SMILES strings
- [x] Train Random Forest regression model and evaluate performance
- [x] Add support for XGBoost model (from paper Table 1)
- [ ] Implement molecular docking pipeline (`docking.py`) using PDB: 1M17
- [ ] Compare predicted vs. experimental pIC50 for designed ligands
- [ ] Deploy a Streamlit web app for user-friendly prediction interface (optional)

---

## 📁 Repository Structure

```
EGFR_RF_Modeling/
├── data/
│   ├── egfr_04_bioactivity_data_raw.csv    # Curated dataset (from paper)
│   └── egfr_rdkit_features.csv             # Features generated with RDKit
├── scripts/
│   ├── generate_descriptors.py             # Script to compute RDKit descriptors + fingerprints
│   └── random_forest_regression.py         # Random Forest regression and performance plot
├── outputs/
│   └── rf_actual_vs_predicted.png          # Saved prediction plot (Figure 2 reproduction)
├── environment.yml                         # Conda environment with dependencies
└── README.md                               # Project overview and instructions
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/EGFR_RF_Modeling.git
cd EGFR_RF_Modeling
```

### 2. Set up environment
```bash
conda env create -f environment.yml
conda activate egfr-env
```

### 3. Generate descriptors
```bash
python scripts/generate_descriptors.py
```

### 4. Train and evaluate model
```bash
python scripts/random_forest_regression_r2.py
```

---

## 📊 Output
- `rf_actual_vs_predicted.png` — Scatterplot of predicted vs. actual pIC50 (similar to Figure 2 in the paper)

---

## 🧪 Tools Used
- RDKit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 👤 Author
Sanjeev Gautam  
Ph.D. in Computational Chemistry
