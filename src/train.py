import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
NUM_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

def treat_zeros_as_nan(df):
    zero_as_nan = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)
    return df

def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_pipeline, NUM_COLS)],
        remainder='drop'
    )
    return preprocessor

def get_models_and_grids():
    models = {
        'logreg': (LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),
                   {'model__C':[0.01,0.1,1,10], 'model__penalty':['l1','l2']}),
        'svm': (SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE),
                {'model__C':[0.1,1,10], 'model__kernel':['rbf','linear'], 'model__gamma':['scale','auto']}),
        'knn': (KNeighborsClassifier(),
                {'model__n_neighbors':[3,5,7,9,11], 'model__weights':['uniform','distance']}),
        'dt': (DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
               {'model__max_depth':[None,3,5,7,9], 'model__min_samples_split':[2,5,10]}),
        'rf': (RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
               {'model__n_estimators':[200,500], 'model__max_depth':[None,5,10], 'model__max_features':['sqrt','log2']})
    }
    return models

def evaluate_and_plots(best_pipeline, X_test, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_prob = best_pipeline.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }
    # Save metrics
    pd.DataFrame([metrics]).to_csv(out_dir.parent / 'metrics.csv', index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f'{val}', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png')
    plt.close()

    # Feature importance/coefs if available
    try:
        # Try to extract names after preprocessing
        feature_names = NUM_COLS
        model = best_pipeline.named_steps['model']
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).ravel()
        if importances is not None:
            order = np.argsort(importances)[::-1]
            plt.figure()
            plt.bar(range(len(importances)), importances[order])
            plt.xticks(range(len(importances)), [feature_names[i] for i in order], rotation=45, ha='right')
            plt.title('Feature Importance / Coefficients')
            plt.tight_layout()
            plt.savefig(out_dir / 'feature_importance.png')
            plt.close()
    except Exception as e:
        # Don't fail if we can't plot importances
        pass

    return metrics

def main(csv_path: str):
    out_models = Path('models')
    out_figs = Path('reports/figures')
    out_models.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = treat_zeros_as_nan(df)

    X = df[NUM_COLS].copy()
    y = df['Outcome'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    models = get_models_and_grids()

    best_score = -np.inf
    best_name = None
    best_estimator = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results_rows = []
    for name, (estimator, grid) in models.items():
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', estimator)])
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)
        mean_score = gs.best_score_
        results_rows.append({'model': name, 'cv_roc_auc': mean_score, 'best_params': str(gs.best_params_)})
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_estimator = gs.best_estimator_

    # Save CV results table
    pd.DataFrame(results_rows).to_csv('reports/cv_results.csv', index=False)

    # Fit best on full training set (already fitted by GridSearchCV, but refit-safe)
    best_estimator.fit(X_train, y_train)

    # Evaluate and plot
    metrics = evaluate_and_plots(best_estimator, X_test, y_test, out_figs)

    # Persist the full pipeline for the app
    joblib.dump(best_estimator, out_models / 'best_pipeline.joblib')

    # Write summary JSON
    summary = {
        'best_model': best_name,
        'best_cv_roc_auc': float(best_score),
        'test_metrics': metrics
    }
    Path('reports').mkdir(exist_ok=True, parents=True)
    (Path('reports') / 'summary.json').write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to diabetes.csv')
    args = parser.parse_args()
    main(args.csv)
