import pandas as pd
from pathlib import Path

def make_label_df(ground_truth_csv):
    df = pd.read_csv(ground_truth_csv)

    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    # Extract the string label name for each row by finding the column with the 1.00
    df['diagnosis'] = df[label_cols].idxmax(axis=1)

    # Create a dictionary mapping class names to integer IDs (0 to 7)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(label_cols)}
    df['target'] = df['diagnosis'].map(class_to_idx)

    df = df.drop(columns=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])

    return df

def make_metadata_df(metadata_csv):
    df = pd.read_csv(metadata_csv)
    return df

def create_df():
    labels = make_label_df('../data/raw/train/ISIC_2019_Training_GroundTruth.csv')
    metadata = make_metadata_df('../data/raw/train/ISIC_2019_Training_Metadata.csv')

    df_merged = metadata.merge(labels, on='image', how='inner')

    assert len(df_merged) == len(labels), (
        f"Merge dropped rows: {len(labels)} labels -> {len(df_merged)} merged"
    )

    # Map classes to binary risk levels
    risk_mapping = {
        'NV': 'Benign',
        'BKL': 'Benign',
        'DF': 'Benign',
        'VASC': 'Benign',
        'AK': 'Malignant/Pre-Malignant',
        'MEL': 'Malignant/Pre-Malignant',
        'BCC': 'Malignant/Pre-Malignant',
        'SCC': 'Malignant/Pre-Malignant'
    }

    df_merged['risk_level'] = df_merged['diagnosis'].map(risk_mapping)

    # Fill in lesion_id for the NaN ones which are the lesions with only one image
    df_merged['lesion_id_filled'] = df_merged['lesion_id'].fillna(df_merged['image'])
    return df_merged


if __name__ == '__main__':
    out_path = Path('../data/processed/df_merged.csv')
    if out_path.exists():
        print(f"{out_path} already exists, skipping. Delete it to regenerate.")
    else:
        df = create_df()
        print(df.shape)
        print(df['diagnosis'].value_counts())
        df.to_csv(out_path, index=False)