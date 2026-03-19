import pandas as pd

def build_player_stats(df):
    # Rename columns
    df = df.rename(columns={
        'ID': 'match_id',
        'Batter': 'batter',
        'BatsmanRun': 'batsman_runs',
        'BallNumber': 'ball'
    })

    # Keep only required columns
    df = df[['match_id', 'batter', 'batsman_runs', 'ball']]

    # Remove invalid rows (if any)
    df = df.dropna()

    # Aggregate to player level
    player_stats = df.groupby(['match_id', 'batter']).agg(
        runs=('batsman_runs', 'sum'),
        balls_faced=('ball', 'count')
    ).reset_index()

    return player_stats


if __name__ == "__main__":
    df = pd.read_csv("data/processed/merged.csv", low_memory=False)

    print("Original Columns:", df.columns.tolist())

    player_df = build_player_stats(df)

    print(player_df.head())

    player_df.to_csv("data/processed/player_stats.csv", index=False)

    print("✅ player_stats.csv created successfully!")