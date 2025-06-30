# predictor.py
from joblib import load
import pandas as pd

class Predictor:
    def __init__(self, model_path: str):
        self.model = load(model_path)

    def prepare_input(self, date, store, item):
        date = pd.to_datetime(date)
        data = {
            "store": [store],
            "item": [item],
            "year": [date.year],
            "month": [date.month],
            "day": [date.day],
            "dayofweek": [date.dayofweek],
        }
        return pd.DataFrame(data)

    def predict(self, input_df: pd.DataFrame) -> float:
        return self.model.predict(input_df)[0]
