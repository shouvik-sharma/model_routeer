from backtesting import Backtest, Strategy
import pandas as pd
from tools.market import MarketData

class SMACross(Strategy):
    n1 = 20
    n2 = 50

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), close)

    def next(self):
        if self.sma1[-1] > self.sma2[-1] and self.sma1[-2] <= self.sma2[-2]:
            self.buy()
        elif self.sma1[-1] < self.sma2[-1] and self.sma1[-2] >= self.sma2[-2]:
            self.position.close()

class RSIReversion(Strategy):
    def init(self):
        close = self.data.Close
        self.rsi = self.I(self.calculate_rsi, close)

    def calculate_rsi(self, prices):
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def next(self):
        if self.rsi[-1] < 30:
            self.buy()
        elif self.rsi[-1] > 70:
            self.position.close()

class BacktestEngine:
    def __init__(self):
        self.market = MarketData()

    def run_equity_backtest(self, symbol: str, strategy_name: str):
        data = self.market.get_daily_data(symbol)
        if data.empty:
            return {"error": "No data available"}

        strategy_map = {
            "SMA_CROSS": SMACross,
            "RSI_REVERSION": RSIReversion
        }

        strategy = strategy_map.get(strategy_name)
        if not strategy:
            return {"error": f"Strategy {strategy_name} not implemented"}

        bt = Backtest(data, strategy, cash=10000, commission=.002)
        stats = bt.run()
        
        return {
            "Return [%]": stats['Return [%]'],
            "Buy & Hold Return [%]": stats['Buy & Hold Return [%]'],
            "Max. Drawdown [%]": stats['Max. Drawdown [%]'],
            "# Trades": stats['# Trades'],
            "Win Rate [%]": stats['Win Rate [%]'],
            "Sharpe Ratio": stats['Sharpe Ratio']
        }
