# 🩺 Predicting Diabetes Likelihood in Patients
### Machine Learning for Early Detection & Preventive Healthcare

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![AIM](https://img.shields.io/badge/AIM-PGDip%20AI%20%26%20ML-darkblue)](https://www.aim.edu)
[![GitHub](https://img.shields.io/badge/GitHub-superk314%2Fcapstone-black?logo=github)](https://github.com/superk314/capstone)

---

## 👤 Author

**Karen R. Mangilit**
Postgraduate Diploma in Artificial Intelligence and Machine Learning
Asian Institute of Management · June 2025

[![LinkedIn](https://img.shields.io/badge/LinkedIn-karenmangilit-blue?logo=linkedin&logoColor=white)](https://linkedin.com/in/karenmangilit)

---

## 📌 Project Overview

Diabetes affects **830 million adults worldwide** (WHO, 2022), yet **52.8% of cases in the Western Pacific region remain undiagnosed** until serious complications arise — including kidney failure, blindness, and limb amputation (IDF Diabetes Atlas, 10th Ed., 2021).

This capstone project demonstrates the **end-to-end machine learning lifecycle** applied to a real-world healthcare problem: predicting diabetes likelihood in patients using routine clinical measurements — enabling early intervention before symptoms escalate.

---

## 🎯 Problem Statement

**Business Problem:**
Healthcare providers need a reliable, scalable, and explainable way to identify high-risk diabetic patients at the point of triage — before they present with late-stage complications.

**Task Type:** Binary Classification
- Target variable: `Outcome` (1 = Diabetic, 0 = Not Diabetic)
- Input: 8 routine clinical measurements (no specialist tests required)

**Success Metrics:**

| Metric | Type | Rationale |
|---|---|---|
| AUC-ROC | Primary (ML) | Robust to class imbalance |
| Recall | Clinical Priority | Minimises missed diagnoses |
| F1-Score | Secondary (ML) | Balances precision and recall |
| Precision | Secondary (ML) | Minimises unnecessary false alarms |

> **Why Recall?** A missed diabetes diagnosis (False Negative) is far more clinically dangerous than a false alarm (False Positive). A false alarm results in a follow-up consultation. A missed diagnosis can result in years of undetected disease progression.

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Name** | Pima Indians Diabetes Database |
| **Source** | UCI Machine Learning Repository / Kaggle |
| **URL** | [kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Population** | Female patients of Pima Indian heritage, aged 21+ |
| **Size** | 768 patients · 8 clinical features · 1 target variable |
| **License** | Public Domain |

### Data Dictionary

| Feature | Type | Description | Unit |
|---|---|---|---|
| Pregnancies | int | Number of times pregnant | Count |
| Glucose | int | Plasma glucose (2hr oral test) | mg/dL |
| BloodPressure | int | Diastolic blood pressure | mm Hg |
| SkinThickness | int | Triceps skinfold thickness | mm |
| Insulin | int | 2-hour serum insulin | µU/mL |
| BMI | float | Body mass index | kg/m² |
| DiabetesPedigreeFunction | float | Genetic risk score | 0.0–2.5 |
| Age | int | Patient age | Years |
| **Outcome** | int | **Target: Diabetes (1) / No Diabetes (0)** | Binary |

---

## 🗂️ Repository Structure

```
diabetes-prediction/
│
├── 📓 notebooks/
│   ├── KarenMangilit-diabetes_eda-Final.ipynb              # EDA & Feature Engineering
│   ├── KarenMangilit-diabetes_modelling-Final.ipynb        # Model Training & Evaluation
│   └── KarenMangilit-diabetes_bias_fairness-Final.ipynb    # Ethical AI & Bias Auditing
│
├── 📁 data/
│   └── diabetes.csv                       # Pima Indians Diabetes Dataset
│
├── 📁 models/
│   ├── xgboost.pkl                        # Best model (XGBoost)
│   ├── xgboost_tuned.pkl                  # Hyperparameter-tuned XGBoost
│   ├── random_forest.pkl                  # Random Forest
│   ├── decision_tree.pkl                  # Decision Tree
│   ├── logistic_regression.pkl            # Logistic Regression (baseline)
│   ├── scaler.pkl                         # StandardScaler (fit on train)
│   ├── config.json                        # Feature list & run config
│   ├── best_params.json                   # Best hyperparameters (tuning)
│   └── model_metrics.csv                  # Full metrics comparison
│
├── 📁 src/
│  
│
├── 📁 presentation decks/
│   ├── Karen Mangilit - Technical Presentation.pdf      # Technical deck (for peers)
│   └── Karen Mangilit - Business Presentation.pdf   # Business deck (for executives)
│
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # This file
```

---

## 🔬 Methodology — ML Lifecycle

### Step 1 · Problem Framing
Defined as binary classification with AUC-ROC as primary metric and Recall as clinical priority metric. Business KPIs established around early detection rate improvement and cost reduction.

### Step 2 · Data Collection & Understanding
Pima Indians Diabetes Dataset sourced from UCI/Kaggle. Full data dictionary documented. Feature types, missing values, outliers, and class imbalance identified.

### Step 3 · EDA & Feature Engineering
- **Zero-masking resolved** — biologically impossible zeros in 5 columns replaced with NaN and imputed using column median
- **Duplicate check** — confirmed no duplicate rows
- **Class imbalance identified** — 65.1% vs 34.9% (Non-Diabetic vs Diabetic)
- **5 engineered features** created: `AgeGroup`, `BMICategory`, `GlucoseRisk`, `InsulinGlucoseRatio`, `HighRisk`
- **Feature scaling** — StandardScaler applied (mean=0, std=1)
- **Feature selection** — ANOVA F-Score (filter) + Random Forest importances (embedded)
- **Dimensionality reduction** — PCA for variance analysis and visualisation

### Step 4 · Model Implementation
Four models trained on 80/20 stratified split with SMOTE applied to training data only:

| Model | Accuracy | AUC-ROC | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| **XGBoost ✓** | **0.8766** | **0.9467** | **0.8288** | **0.8070** | **0.8519** |
| Random Forest | 0.8766 | 0.9447 | 0.8257 | 0.8182 | 0.8333 |
| Decision Tree | 0.8636 | 0.8944 | 0.8142 | 0.7797 | 0.8519 |
| Logistic Regression | 0.7468 | 0.8161 | 0.6929 | 0.6027 | 0.8148 |

**Hyperparameter tuning** via RandomizedSearchCV (50 combinations × 5-fold CV = 250 fits).
**5-fold stratified cross-validation** confirmed stability: XGBoost 0.869 ± 0.019.

### Step 5 · Ethical AI & Bias Auditing
- **SHAP** — global and local explainability (beeswarm + bar plots)
- **PDP + ICE plots** — feature effect visualisation per individual patient
- **Fairness audit** across Age Group, BMI Category, and Pregnancy Group
- **Fairness metrics** — Demographic Parity, Equalised Odds, Disparate Impact
- **Threshold sensitivity analysis** — clinical threshold set to 0.40 (vs default 0.50)
- **Data leakage assessment** — documented residual risks and mitigations
- **Known limitations** — population scope, missing sensitive attributes, imputation order

### Fairness Audit Results

| Sensitive Group | Weakest Subgroup | Recall | Status |
|---|---|---|---|
| Age Group | 21–30 | 0.878 | ⚠️ Monitor |
| BMI Category | Normal BMI | 0.857 | ✅ Pass |
| Pregnancy Group | Low (1–3) | 0.880 | ✅ Pass |

---

## 🏆 Key Results

```
╔══════════════════════════════════════════════════════════╗
║                 BEST MODEL: XGBoost                      ║
╠══════════════════════════════════════════════════════════╣
║  AUC-ROC   0.9467  →  94.7% discrimination ability      ║
║  Recall    0.8519  →  85 in 100 diabetic patients caught ║
║  Precision 0.8070  →  80.7% of alarms are genuine        ║
║  F1-Score  0.8288  →  Strong precision-recall balance    ║
╠══════════════════════════════════════════════════════════╣
║  Top Predictors (SHAP): Insulin · Glucose · Age · BMI   ║
║  Clinical Threshold: 0.40 (conservative for safety)      ║
╚══════════════════════════════════════════════════════════╝
```

> ⚕️ **Clinical Note:** This model must be used as a **decision support tool only** — not as an autonomous diagnostic system. All predictions must be reviewed by a qualified clinician.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/superk314/capstone.git
cd capstone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the `data/` folder.

### 4. Run the Notebooks

Launch Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

| Order | Notebook | Description |
|---|---|---|
| 1 | `notebooks/KarenMangilit-diabetes_eda-Final.ipynb` | EDA, preprocessing & feature engineering |
| 2 | `notebooks/KarenMangilit-diabetes_modelling-Final.ipynb` | Model training, tuning & evaluation |
| 3 | `notebooks/KarenMangilit-diabetes_bias_fairness-Final.ipynb` | Explainability & fairness audit |

> **Important:** Run notebooks in order. Notebook 2 depends on the cleaned dataset from Notebook 1. Notebook 3 depends on the saved models from Notebook 2.

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0
joblib>=1.2.0
scipy>=1.9.0
jupyter>=1.0.0
```

See `requirements.txt` for pinned versions.

---

## 🔑 Reproducibility Checklist

- ✅ `random_state=42` set on all models, splits, and SMOTE
- ✅ `StandardScaler` fitted on training data only — test data transformed using training statistics
- ✅ `SMOTE` applied to training data only — test set reflects real-world class distribution
- ✅ Stratified 5-fold cross-validation used throughout
- ✅ All trained models saved as `.pkl` artefacts in `models/`
- ✅ Hyperparameters documented in `models/config.json` and `models/best_params.json`
- ✅ Full metrics comparison saved to `models/model_metrics.csv`

---

## ⚖️ Ethical Considerations

This project includes a dedicated bias and fairness audit (Notebook 3) covering:

- **Explainability** — SHAP values, PDP, and ICE plots for every prediction
- **Fairness audit** — performance evaluated across Age Group, BMI Category, and Pregnancy Group
- **Disparate Impact** — computed using the 80% rule threshold
- **Threshold analysis** — clinical threshold lowered to 0.40 to prioritise patient safety (Recall)
- **Known limitations** — race, ethnicity, gender, and socioeconomic status absent from dataset; population limited to Pima Indian women aged 21+
- **Data leakage assessment** — documented and mitigated

**Deployment recommendation:** Clinical decision support tool only. Clinician review required for all flagged patients. External validation on local patient population mandatory before any real-world deployment.

---

## 📚 References

**Dataset & Epidemiology**

International Diabetes Federation. (2021). *IDF Diabetes Atlas* (10th ed.). https://diabetesatlas.org

Sun, H., Saeedi, P., Karuranga, S., Pinkepank, M., Ogurtsova, K., Duncan, B. B., ... & Magliano, D. J. (2022). IDF Diabetes Atlas: Global, regional and country-level diabetes prevalence estimates for 2021 and projections for 2045. *Diabetes Research and Clinical Practice, 183*, 109119. https://doi.org/10.1016/j.diabres.2021.109119

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
> *Original paper that introduced and published the Pima Indians Diabetes Dataset.*

---

**Algorithms & Methods**

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
> *Originating paper for the Random Forest algorithm used as an ensemble baseline model.*

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
> *Originating paper for XGBoost — the best-performing model in this project.*

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953
> *Originating paper for SMOTE, applied to the training set to address class imbalance (65/35 split).*

---

**Explainability**

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
> *Originating paper for SHAP (SHapley Additive exPlanations), used for global and local model explainability throughout this project.*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Asian Institute of Management** — for the Postgraduate Diploma in AI & ML programme
- **UCI Machine Learning Repository** — for the Pima Indians Diabetes Dataset
- **International Diabetes Federation** — for global diabetes epidemiology data
- The open-source community behind scikit-learn, XGBoost, SHAP, and the Python data science ecosystem

---

<p align="center">
  <i>Built as part of the Capstone Project requirement for the<br>
  Postgraduate Diploma in Artificial Intelligence and Machine Learning<br>
  Asian Institute of Management · Class of 2025 (Cohort 3)</i>
</p>
