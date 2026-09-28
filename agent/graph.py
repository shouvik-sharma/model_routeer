import yaml
import sqlite3
from datetime import datetime
from tools.equity_backtest import BacktestEngine
from tools.options import OptionsCalculator

class QuantMetronAgent:
    def __init__(self, db_path="quantpath.db"):
        self.db_path = db_path
        self._init_db()
        self.bt_engine = BacktestEngine()
        self.opt_calc = OptionsCalculator()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS users 
                (user_id TEXT PRIMARY KEY, path_len TEXT, quant_level TEXT, options_level TEXT, current_day INTEGER)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS feedback 
                (user_id TEXT, day INTEGER, confidence INTEGER, note TEXT, pnl REAL)''')

    def onboard_user(self, user_id, path_len, quant_level, options_level):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?)", 
                         (user_id, path_len, quant_level, options_level, 1))
        return f"Welcome to QuantMetron! Your {path_len} path is set. Starting Day 1."

    def get_daily_lesson(self, user_id):
        # 1. Recall User State
        with sqlite3.connect(self.db_path) as conn:
            user = conn.execute("SELECT path_len, current_day FROM users WHERE user_id=?", (user_id,)).fetchone()
        
        if not user:
            return {"error": "User not found. Please onboard first."}
        
        path_len, current_day = user
        
        # 2. Load Curriculum
        with open(f"curriculum/{path_len}.yaml", "r") as f:
            curriculum = yaml.safe_load(f)
        
        # 3. Pick Topic
        day_data = next((d for d in curriculum['curriculum']['days'] if d['day'] == current_day), None)
        if not day_data:
            return {"message": "Curriculum completed! Congratulations."}

        # 4. Tool Execution based on lesson
        result = {"topic": day_data['topic'], "explanation": day_data['goal']}
        
        if day_data.get('backtest_required'):
            # For MVP, we use SPY as default test asset
            bt_results = self.bt_engine.run_equity_backtest("SPY", day_data['strategy'])
            result['backtest'] = bt_results

        return result

    def record_feedback(self, user_id, day, confidence, note, pnl):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)", (user_id, day, confidence, note, pnl))
            conn.execute("UPDATE users SET current_day = current_day + 1 WHERE user_id=?", (user_id,))
        return "Feedback recorded. See you tomorrow for Day " + str(day + 1)
