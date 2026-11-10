import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dataset import train_val_split  # Leveraging your existing group-safe split logic


def load_and_preprocess_metadata(csv_path, output_dir="../data/processed/", classification_mode="binary"):
    df = pd.read_csv(csv_path)

    # 1. Handle Missing Values Explicitly
    df['age_is_missing'] = df['age_approx'].isna().astype(int)
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())

    df['sex'] = df['sex'].fillna('unknown')
    df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')

    # 2. One-Hot Encode Categorical Variables
    categorical_cols = ['sex', 'anatom_site_general']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)

    # 3. Determine target column
    target_col = "BinaryTarget" if classification_mode == "binary" else "target"

    # 4. Perform group-aware split
    stratify_col = "risk_level" if classification_mode == "binary" else "diagnosis"
    train_df, val_df = train_val_split(df_encoded, val_frac=0.2, stratify_col=stratify_col)

    # 5. Scale Numerical Features (Fit on TRAIN ONLY)
    scaler = StandardScaler()
    train_df['age_approx'] = scaler.fit_transform(train_df[['age_approx']])
    val_df['age_approx'] = scaler.transform(val_df[['age_approx']])

    # 6. Save separate processed train and val CSVs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = out_dir / "train_metadata.csv"
    val_out_path = out_dir / "val_metadata.csv"

    train_df.to_csv(train_out_path, index=False)
    val_df.to_csv(val_out_path, index=False)

    encoded_cat_cols = [col for col in df_encoded.columns if col.startswith(('sex_', 'anatom_site_general_'))]
    feature_cols = ['age_approx', 'age_is_missing'] + encoded_cat_cols

    print(f"Metadata preprocessing complete!")
    print(f"Saved train set to: {train_out_path}")
    print(f"Saved val set to: {val_out_path}")
    print(f"Total features extracted: {len(feature_cols)}")

    return train_df, val_df, feature_cols, target_col

if __name__ == "__main__":
    data_path = Path('../data/processed/df_merged_cleaned.csv')
    train_df, val_df, features, target = load_and_preprocess_metadata(data_path)
    print("\nSample feature columns:", features[:5])
    print("Sample target column:", target)