import pandas as pd

def create_features():
    df = pd.read_csv("data/processed/merged.csv", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'id': 'match_id',
        'batter': 'batter',
        'batsmanrun': 'batsman_runs',
    })

    df = df[['match_id', 'batter', 'batsman_runs']].dropna()

    df['batsman_runs'] = pd.to_numeric(df['batsman_runs'], errors='coerce')
    df = df.dropna()

    # Aggregate
    runs_df = df.groupby(['match_id', 'batter'])['batsman_runs'].sum().reset_index()
    balls_df = df.groupby(['match_id', 'batter'])['batsman_runs'].count().reset_index()

    runs_df.rename(columns={'batsman_runs': 'runs'}, inplace=True)
    balls_df.rename(columns={'batsman_runs': 'balls_faced'}, inplace=True)

    player_df = pd.merge(runs_df, balls_df, on=['match_id', 'batter'])

    # Create fake timeline
    player_df['match_date'] = pd.to_datetime(player_df['match_id'], errors='coerce')
    player_df = player_df.sort_values(['batter', 'match_date'])

    # Features
    player_df['avg_last_5'] = player_df.groupby('batter')['runs'].transform(lambda x: x.rolling(5).mean())
    player_df['avg_last_10'] = player_df.groupby('batter')['runs'].transform(lambda x: x.rolling(10).mean())

    player_df['strike_rate'] = (player_df['runs'] / player_df['balls_faced']) * 100

    player_df['match_number'] = player_df.groupby('batter').cumcount()

    player_df['consistency'] = player_df.groupby('batter')['runs'].transform(lambda x: x.rolling(5).std())

    player_df['form_trend'] = player_df.groupby('batter')['runs'].transform(lambda x: x.diff().rolling(3).mean())

    player_df.fillna(0, inplace=True)

    return player_df


if __name__ == "__main__":
    df = create_features()
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("✅ final_dataset.csv created")