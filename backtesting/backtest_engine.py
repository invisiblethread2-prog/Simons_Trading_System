
# backtesting/backtest_engine.py
# ============================================================
# PROOF MACHINE — COMPLETE CLEAN VERSION
# Walk-forward backtesting engine
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Complete backtest performance report"""
    strategy_name:  str
    total_trades:   int
    win_rate:       float
    profit_factor:  float
    sharpe_ratio:   float
    sortino_ratio:  float
    max_drawdown:   float
    total_return:   float
    annual_return:  float
    calmar_ratio:   float
    max_dd_duration: int
    avg_win_r:      float
    avg_loss_r:     float
    expectancy:     float
    equity_curve:   np.ndarray
    daily_returns:  np.ndarray

    @property
    def is_deployable(self) -> bool:
        return (
            self.sharpe_ratio  >= 1.5  and
            self.win_rate      >= 0.52 and
            self.profit_factor >= 1.3  and
            self.max_drawdown  <= 0.25 and
            self.total_trades  >= 30
        )

    def print_report(self):
        status = "✅ DEPLOYABLE" if self.is_deployable else "❌ NOT READY"
        print(f"""
╔══════════════════════════════════════════════╗
║  BACKTEST: {self.strategy_name:<34} ║
╠══════════════════════════════════════════════╣
║  Trades:        {self.total_trades:<28} ║
║  Win Rate:      {self.win_rate:<27.1%} ║
║  Profit Factor: {self.profit_factor:<27.2f} ║
║  Sharpe:        {self.sharpe_ratio:<27.2f} ║
║  Max Drawdown:  {self.max_drawdown:<27.2%} ║
║  Total Return:  {self.total_return:<27.2%} ║
║  Avg Win R:     {self.avg_win_r:<27.3f} ║
║  Avg Loss R:    {self.avg_loss_r:<27.3f} ║
║  Expectancy:    {self.expectancy:<27.3f} ║
╠══════════════════════════════════════════════╣
║  {status:<44} ║
╚══════════════════════════════════════════════╝""")


class BacktestEngine:
    """
    Walk-Forward Backtesting Engine.

    Train on 70% → Test on 30% (unseen data).
    No look-ahead bias. No curve fitting.
    """

    def __init__(self, starting_capital: float = 100.0):
        self.starting_capital = starting_capital

    def run(
        self,
        closes:      np.ndarray,
        highs:       np.ndarray,
        lows:        np.ndarray,
        volumes:     np.ndarray,
        strategy_fn,
        name:        str = "Strategy"
    ) -> BacktestResult:
        """
        Main backtest runner.
        Train on 70%, test on unseen 30%.
        """
        n         = len(closes)
        train_end = int(n * 0.70)

        # Train
        strategy_fn.fit(
            closes[:train_end],
            highs[:train_end],
            lows[:train_end],
            volumes[:train_end]
        )

        # Test on UNSEEN data only
        return self._simulate(
            closes[train_end:],
            highs[train_end:],
            lows[train_end:],
            volumes[train_end:],
            strategy_fn,
            name
        )

    def _simulate(
        self,
        closes:      np.ndarray,
        highs:       np.ndarray,
        lows:        np.ndarray,
        volumes:     np.ndarray,
        strategy_fn,
        name:        str
    ) -> BacktestResult:
        """Simulate trading on test data"""

        capital    = self.starting_capital
        peak       = capital
        max_dd     = 0.0
        equity     = [capital]
        trades     = []
        daily_rets = []
        i          = 50  # Warmup

        while i < len(closes) - 1:

            # Get signal
            sig = strategy_fn.predict(
                closes[:i+1],
                highs[:i+1],
                lows[:i+1],
                volumes[:i+1]
            )

            if sig is None or sig.get('direction', 'FLAT') == 'FLAT':
                equity.append(capital)
                i += 1
                continue

            # Trade parameters
            entry  = closes[i]
            atr    = sig.get('atr', entry * 0.015)
            stop_d = atr * 1.5
            risk   = capital * 0.02
            direct = sig['direction']

            if atr == 0 or risk == 0:
                i += 1
                continue

            if direct == 'LONG':
                stop = entry - stop_d
                tp1  = entry + stop_d
                tp2  = entry + stop_d * 2.0
            else:
                stop = entry + stop_d
                tp1  = entry - stop_d
                tp2  = entry - stop_d * 2.0

            # Simulate forward bars
            pnl       = 0.0
            remaining = 1.0
            bars_held = 0
            exited    = False

            for j in range(i + 1, min(i + 25, len(closes))):
                h = highs[j]
                l = lows[j]
                bars_held += 1

                if direct == 'LONG':
                    if l <= stop:
                        pnl    = -risk * remaining
                        exited = True
                        break
                    if h >= tp1 and remaining == 1.0:
                        pnl      += risk * 1.0 * 0.5
                        remaining = 0.5
                    if h >= tp2 and remaining > 0:
                        pnl      += risk * 2.0 * remaining
                        remaining = 0.0
                        exited    = True
                        break
                else:
                    if h >= stop:
                        pnl    = -risk * remaining
                        exited = True
                        break
                    if l <= tp1 and remaining == 1.0:
                        pnl      += risk * 1.0 * 0.5
                        remaining = 0.5
                    if l <= tp2 and remaining > 0:
                        pnl      += risk * 2.0 * remaining
                        remaining = 0.0
                        exited    = True
                        break

            # Timeout exit
            if not exited and remaining > 0:
                pnl += 0.0  # Exit at entry (no profit/loss)

            r = pnl / risk if risk > 0 else 0.0
            trades.append({'pnl': pnl, 'r': r})
            capital += pnl

            # Drawdown
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

            equity.append(capital)
            if len(equity) >= 2 and equity[-2] > 0:
                daily_rets.append(equity[-1] / equity[-2] - 1)

            i += max(bars_held, 1)

        return self._calculate_metrics(
            trades        = trades,
            equity_curve  = np.array(equity),
            daily_returns = np.array(daily_rets) if daily_rets else np.array([0.0]),
            name          = name
        )

    def _calculate_metrics(
        self,
        trades:       List[Dict],
        equity_curve: np.ndarray,
        daily_returns: np.ndarray,
        name:         str
    ) -> BacktestResult:
        """Calculate all performance metrics"""

        if not trades:
            return self._empty_result(name)

        wins   = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total    = len(trades)
        win_rate = len(wins) / total if total > 0 else 0.0

        avg_win_r  = float(np.mean([t['r'] for t in wins])) if wins else 0.0
        avg_loss_r = float(abs(np.mean([t['r'] for t in losses]))) if losses else 1.0

        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss   = abs(sum(t['pnl'] for t in losses))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float('inf')
        )

        expectancy = (
            win_rate * avg_win_r -
            (1 - win_rate) * avg_loss_r
        )

        # Sharpe
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(
                np.mean(daily_returns) /
                np.std(daily_returns) *
                np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Sortino
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(
                np.mean(daily_returns) /
                np.std(downside) *
                np.sqrt(252)
            )
        else:
            sortino = 0.0

        # Drawdown duration
        peak_i  = 0
        max_dur = 0
        dd_dur  = 0
        peak_v  = equity_curve[0]

        for val in equity_curve:
            if val >= peak_v:
                peak_v = val
                dd_dur = 0
            else:
                dd_dur += 1
            max_dur = max(max_dur, dd_dur)

        # Returns
        total_return = (
            float(equity_curve[-1] / equity_curve[0] - 1)
            if equity_curve[0] > 0 else 0.0
        )
        n_days = max(len(daily_returns), 1)
        annual_return = float(
            (1 + total_return) ** (252 / n_days) - 1
        )

        max_dd_val = float(np.max(
            1 - equity_curve / np.maximum.accumulate(equity_curve)
        )) if len(equity_curve) > 1 else 0.0

        calmar = (
            annual_return / max_dd_val
            if max_dd_val > 0 else 0.0
        )

        return BacktestResult(
            strategy_name   = name,
            total_trades    = total,
            win_rate        = win_rate,
            profit_factor   = profit_factor,
            sharpe_ratio    = sharpe,
            sortino_ratio   = sortino,
            max_drawdown    = max_dd_val,
            total_return    = total_return,
            annual_return   = annual_return,
            calmar_ratio    = calmar,
            max_dd_duration = max_dur,
            avg_win_r       = avg_win_r,
            avg_loss_r      = avg_loss_r,
            expectancy      = expectancy,
            equity_curve    = equity_curve,
            daily_returns   = daily_returns
        )

    def _empty_result(self, name: str) -> BacktestResult:
        """Empty result when no trades"""
        return BacktestResult(
            strategy_name   = name,
            total_trades    = 0,
            win_rate        = 0.0,
            profit_factor   = 0.0,
            sharpe_ratio    = 0.0,
            sortino_ratio   = 0.0,
            max_drawdown    = 0.0,
            total_return    = 0.0,
            annual_return   = 0.0,
            calmar_ratio    = 0.0,
            max_dd_duration = 0,
            avg_win_r       = 0.0,
            avg_loss_r      = 0.0,
            expectancy      = 0.0,
            equity_curve    = np.array([self.starting_capital]),
            daily_returns   = np.array([0.0])
        )

    def run_walkforward(
        self,
        data:            pd.DataFrame,
        signal_generator,
        train_size:      int = 180,
        test_size:       int = 30,
        step_size:       int = 30
    ) -> List[BacktestResult]:
        """
        Walk-Forward Analysis.
        Train 6 months → Test 1 month → Move forward.
        """
        results = []
        n       = len(data)
        start   = 0

        while start + train_size + test_size <= n:
            train_end  = start + train_size
            test_end   = train_end + test_size
            train_data = data.iloc[start:train_end]
            test_data  = data.iloc[train_end:test_end]

            closes  = test_data['close'].values
            highs   = test_data['high'].values
            lows    = test_data['low'].values
            volumes = test_data['volume'].values

            signal_generator.fit(
                train_data['close'].values,
                train_data['high'].values,
                train_data['low'].values,
                train_data['volume'].values
            )

            result = self._simulate(
                closes, highs, lows, volumes,
                signal_generator,
                f"WF_{start}_{test_end}"
            )
            results.append(result)
            start += step_size

        return results
