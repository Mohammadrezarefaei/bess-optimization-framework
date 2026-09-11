import pandas as pd
from xgboost import XGBRegressor

class BESSPricePredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

    def _engineer_features(self, df: pd.DataFrame, price_col: str = 'price_eur_mwh') -> pd.DataFrame:
        data = df.copy()
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['rolling_mean_4h'] = data[price_col].shift(1).rolling(window=4).mean()
        data['lag_1h'] = data[price_col].shift(4)
        data.dropna(inplace=True)
        return data

    def forecast(self, df_history: pd.DataFrame, price_col: str = 'price_eur_mwh') -> pd.DataFrame:
        processed_data = self._engineer_features(df_history, price_col)
        features = ['hour', 'dayofweek', 'rolling_mean_4h', 'lag_1h']
        X = processed_data[features]
        y = processed_data[price_col]
        self.model.fit(X, y)
        processed_data['forecasted_price'] = self.model.predict(X)
        return processed_data[['price_eur_mwh', 'forecasted_price']]
