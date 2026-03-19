import pandas as pd
import os

def merge_csvs(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    return merged_df

if __name__ == "__main__":
    df = merge_csvs("data/raw/")
    df.to_csv("data/processed/merged.csv", index=False)
    print("Merged dataset saved!")