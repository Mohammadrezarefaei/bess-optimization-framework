import numpy as np
import pandas as pd

class BESSParametricOptimizer:
    def __init__(self, capacity_mwh: float, max_power_mw: float, 
                 min_soc_limit: float = 0.1, max_soc_limit: float = 0.9, 
                 roundtrip_efficiency: float = 0.85, rolling_window: int = 96):
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.min_soc = min_soc_limit
        self.max_soc = max_soc_limit
        self.eta = np.sqrt(roundtrip_efficiency)
        self.rolling_window = rolling_window

    def optimize_dispatch(self, df_prices: pd.DataFrame, price_col: str = 'forecasted_price') -> pd.DataFrame:
        df = df_prices.copy()
        
        # Calculate dynamic rolling thresholds
        df['p50'] = df[price_col].rolling(window=self.rolling_window, min_periods=4).quantile(0.50)
        df['p90'] = df[price_col].rolling(window=self.rolling_window, min_periods=4).quantile(0.90)
        df['p50'] = df['p50'].bfill().ffill()
        df['p90'] = df['p90'].bfill().ffill()
        
        soc = self.min_soc * self.capacity
        soc_history = []
        actions = []
        revenue_cost = []

        for idx, row in df.iterrows():
            price = row[price_col]
            p50 = row['p50']
            p90 = row['p90']
            
            action = 0
            power_flow = 0.0

            if price <= p50 and soc < (self.max_soc * self.capacity):
                action = 1
                power_flow = self.max_power
                energy_added = power_flow * 0.25 * self.eta
                if soc + energy_added <= self.max_soc * self.capacity:
                    soc += energy_added
                else:
                    power_flow = (self.max_soc * self.capacity - soc) / (0.25 * self.eta)
                    soc = self.max_soc * self.capacity

            elif price >= p90 and soc > (self.min_soc * self.capacity):
                action = -1
                power_flow = -self.max_power
                energy_removed = (abs(power_flow) * 0.25) / self.eta
                if soc - energy_removed >= self.min_soc * self.capacity:
                    soc -= energy_removed
                else:
                    power_flow = -(soc - self.min_soc * self.capacity) * self.eta / 0.25
                    soc = self.min_soc * self.capacity

            actions.append(action)
            revenue_cost.append(-power_flow * price * 0.25)
            soc_history.append(soc / self.capacity)

        df['bess_action'] = actions
        df['soc_level'] = soc_history
        df['cash_flow_eur'] = revenue_cost
        return df
