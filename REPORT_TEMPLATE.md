# Diabetes Prediction from Health Data Using Machine Learning Techniques

**Course:** IT41033 — Nature Inspired Algorithms  
**Group Members:**  
- K.M.L.Sampath — ITBIN-2211-0277  
- M.N.S.K.Bandra — ITBIN-2211-0152  
- G.G.S.A.Ananda — ITBIN-2211-0139  

## 1. Dataset description and selection rationale
- Source: Pima Indians Diabetes Dataset (Kaggle, UCI-origin).  
- Instances: 768, Attributes: 9 (all numeric).  
- Target: `Outcome` — 0 (non-diabetic), 1 (diabetic).  
- Rationale: Classic, well-curated, widely studied medical dataset suitable for teaching and benchmarking.

## 2. Data preprocessing steps
- Treated zeros in {Glucose, BloodPressure, SkinThickness, Insulin, BMI} as missing; imputed medians.
- Checked/handled outliers (IQR capping if needed).
- Split data (stratified) into train/test (e.g., 80/20).  
- Standardized features for distance- and margin-based models (LR, SVM, kNN).

## 3. Model selection and evaluation process
- Models: Logistic Regression, Decision Tree, Random Forest, k-NN, SVM.  
- Hyperparameters tuned via GridSearchCV (5-fold Stratified CV) using ROC AUC as primary scoring.  
- Class imbalance handled with `class_weight='balanced'` when supported.  
- Final evaluation on the test set with: Accuracy, Precision, Recall, F1-score, ROC AUC.  
- Included Confusion Matrix and ROC Curve plots.

## 4. Streamlit app design decisions
- Saved the **entire pipeline** (preprocessing + best model) to `best_pipeline.joblib` for simple, robust inference.  
- Simple form UI with domain-aware input ranges and validation.  
- Probability output with interpretation guidance and disclaimer (not medical advice).

## 5. Deployment process and challenges
- Local deployment with `streamlit run`.  
- (Optional) Streamlit Community Cloud: push repo to GitHub, connect app, set Python version and `requirements.txt`.  
- Common issues: missing model file, path errors, incompatible package versions.

## 6. Screenshots of the application
- Insert screenshots of the input form and prediction results here (from `streamlit run`).

## 7. Reflection on learning outcomes
- What worked well, what didn’t, model trade-offs, and potential future work (e.g., calibration, SHAP, threshold tuning, additional features).
