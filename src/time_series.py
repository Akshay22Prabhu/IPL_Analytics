import pandas as pd
from prophet import Prophet

def forecast_player(player_name):
    df = pd.read_csv("data/processed/final_dataset.csv")

    df = df[df['batter'] == player_name].copy()
    df = df.sort_values('match_id')

    df['ds'] = pd.date_range(start='2020-01-01', periods=len(df))
    df['y'] = df['runs']

    model = Prophet()
    model.fit(df[['ds', 'y']])

    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat']].tail(10)