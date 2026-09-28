import numpy as np
import pandas as pd

class OptionsCalculator:
    @staticmethod
    def calc_call_payoff(spot_at_expiry, strike, premium):
        \"\"\"Max(0, S - K) - Premium\"\"\"
        return max(0, spot_at_expiry - strike) - premium

    @staticmethod
    def calc_put_payoff(spot_at_expiry, strike, premium):
        \"\"\"Max(0, K - S) - Premium\"\"\"
        return max(0, strike - spot_at_expiry) - premium

    @staticmethod
    def calc_covered_call_payoff(spot_at_expiry, strike, premium, entry_price):
        \"\"\"Long Stock + Short Call: (S - entry) + CallPayoff(premium - Max(0, S - K))\"\"\"
        stock_pnl = spot_at_expiry - entry_price
        call_pnl = premium - max(0, spot_at_expiry - strike)
        return stock_pnl + call_pnl

    def generate_payoff_table(self, strategy_type, spot_range, **kwargs):
        \"\"\"Generates a table of spot prices vs P&L for a strategy\"\"\"
        results = []
        for spot in spot_range:
            if strategy_type == "call":
                pnl = self.calc_call_payoff(spot, kwargs['strike'], kwargs['premium'])
            elif strategy_type == "put":
                pnl = self.calc_put_payoff(spot, kwargs['strike'], kwargs['premium'])
            elif strategy_type == "covered_call":
                pnl = self.calc_covered_call_payoff(spot, kwargs['strike'], kwargs['premium'], kwargs['entry_price'])
            else:
                pnl = 0
            results.append({"spot": spot, "pnl": pnl})
        
        return pd.DataFrame(results)
