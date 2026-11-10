import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv('../data/processed/df_merged.csv')

    DIAGNOSES_TO_KEEP = ['MEL', 'BKL']

    filtered_df = df[df['diagnosis'].isin(DIAGNOSES_TO_KEEP)].copy()

    # Re-map target to contiguous integers 0, 1, 2, 3
    diag_to_target = {diag: idx for idx, diag in enumerate(DIAGNOSES_TO_KEEP)}
    filtered_df['target'] = filtered_df['diagnosis'].map(diag_to_target)

    filtered_df.to_csv('../data/processed/df_merged_2_classes.csv', index=False)
    print(f"Original shape: {df.shape} | Filtered shape: {filtered_df.shape}")
    print(filtered_df['target'].value_counts())