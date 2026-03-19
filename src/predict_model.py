import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_runs(batter, avg_last_5, avg_last_10, strike_rate):
    input_df = pd.DataFrame([{
        "batter": batter,
        "avg_last_5": avg_last_5,
        "avg_last_10": avg_last_10,
        "strike_rate": strike_rate
    }])

    prediction = model.predict(input_df)

    return prediction[0]