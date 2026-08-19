# Diabetes Prediction — Mini Project (IT41033)

This repo contains a complete scaffold to reproduce your mini project:
- Data preprocessing & EDA
- Model training with multiple algorithms + cross-validation
- Evaluation (confusion matrix, ROC curve, metrics table)
- Streamlit app for inference
- Report template (3–4 pages) to fill and export to PDF

## 1) Setup
```bash
# (Recommended) Create a fresh virtual environment
python -m venv .venv
# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2) Get the dataset
Download **Pima Indians Diabetes** dataset (Kaggle) and place the CSV as:
```
data/raw/diabetes.csv
```
The file should have these columns (common Kaggle version):
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

## 3) Exploratory Data Analysis (optional but recommended)
This will save figures under `reports/figures/`:
```bash
python src/eda.py --csv data/raw/diabetes.csv
```

## 4) Train & evaluate models
This runs cross-validation across Logistic Regression, SVM, kNN, Decision Tree, Random Forest,
selects the best by ROC AUC, evaluates on a held-out test set, and saves:
- `models/best_pipeline.joblib`
- `reports/metrics.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
- `reports/figures/feature_importance.png` (when available)
```bash
python src/train.py --csv data/raw/diabetes.csv
```

## 5) Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
The app loads `models/best_pipeline.joblib` and allows manual input to predict diabetes risk.

## 6) Write your report
Fill in `REPORT_TEMPLATE.md` with your findings, then export to PDF (e.g., VS Code Markdown PDF extension or Pandoc).

---

### Notes
- Zeros in clinical features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are treated as missing and imputed with the median.
- Scaling is applied where appropriate; class imbalance is mitigated with `class_weight='balanced'` for supported models.
- Random seeds are fixed for reproducibility.
