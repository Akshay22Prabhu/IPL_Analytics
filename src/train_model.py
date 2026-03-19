import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


def train():
    df = pd.read_csv("data/processed/final_dataset.csv")

    X = df[
        [
            'batter',
            'avg_last_5',
            'avg_last_10',
            'strike_rate',
            'match_number',
            'consistency',
            'form_trend'
        ]
    ]

    y = df['runs']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['batter']),
            ('num', 'passthrough', [
                'avg_last_5',
                'avg_last_10',
                'strike_rate',
                'match_number',
                'consistency',
                'form_trend'
            ])
        ]
    )

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"🔥 MAE: {mae:.2f}")

    joblib.dump(pipeline, "models/model.pkl")
    print("✅ Model saved")


if __name__ == "__main__":
    train()