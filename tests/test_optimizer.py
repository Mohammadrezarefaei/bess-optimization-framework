import pandas as pd
import numpy as np
from src.optimizer import BESSParametricOptimizer

def test_optimizer_bounds():
    time_index = pd.date_range(start="2026-09-10 00:00:00", periods=96, freq="15min")
    prices = np.full(96, 50.0)
    prices[40:50] = 600.0  # Introduce a sharp price spike
    df = pd.DataFrame({'forecasted_price': prices}, index=time_index)
    
    optimizer = BESSParametricOptimizer(capacity_mwh=4.0, max_power_mw=2.0)
    res = optimizer.optimize_dispatch(df)
    
    assert res['soc_level'].min() >= 0.1
    assert res['soc_level'].max() <= 0.9
    assert 'bess_action' in res.columns
    assert 'cash_flow_eur' in res.columns
