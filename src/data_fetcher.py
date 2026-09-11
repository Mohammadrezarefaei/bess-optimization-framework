import pandas as pd
from entsoe import EntsoePandasClient

class ENTSOEDataFetcher:
    def __init__(self, api_key: str):
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_day_ahead_prices(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.Timestamp(start_date, tz='Europe/Brussels')
        end = pd.Timestamp(end_date, tz='Europe/Brussels')
        prices = self.client.query_day_ahead_prices(country_code, start=start, end=end)
        df = prices.reset_index()
        df.columns = ['timestamp', 'price_eur_mwh']
        df.set_index('timestamp', inplace=True)
        return df
