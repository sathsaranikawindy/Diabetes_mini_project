import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_STATE = 42
NUM_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

def treat_zeros_as_nan(df):
    zero_as_nan = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)
    return df

def main(csv_path: str):
    out_dir = Path('reports/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = treat_zeros_as_nan(df)

    # Summary stats
    desc = df.describe(include='all')
    desc.to_csv('reports/eda_describe.csv', index=True)

    # Histograms
    for col in NUM_COLS:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col); plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(out_dir / f'hist_{col}.png')
        plt.close()

    # Correlation heatmap (matplotlib only)
    plt.figure()
    corr = df[NUM_COLS + ['Outcome']].corr()
    im = plt.imshow(corr, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_matrix.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    args = parser.parse_args()
    main(args.csv)
