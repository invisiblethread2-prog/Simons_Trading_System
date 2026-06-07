# analytics/performance_engine.py
# ============================================================
# THE SCIENTIFIC REVIEW SYSTEM
# Measures everything. Questions everything.
# Improves everything.
# ============================================================

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from config.settings import config
from models.data_models import (
    Trade, PerformanceMetrics, RegimeState
)

logger = logging.getLogger(__name__)


class PerformanceEngine:
    """
    Comprehensive performance analytics.

    Calculates:
    - Core trade metrics (win rate, PF, Sharpe)
    - R-multiple analysis
    - Signal Information Coefficient (IC)
    - Drawdown analysis
    - Monte Carlo simulation
    - Compounding projections
    - Weekly performance reports
    """

    def __init__(self):
        self.all_trades: List[Trade] = []
        self._ic_history: Dict[str, List[float]] = {}

    # ─── CORE METRICS ─────────────────────────────────────

    def calculate_metrics(
        self,
        trades: Optional[List[Trade]] = None,
        period_days: int = 30
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        Uses last N days if no trades provided.
        """
        if trades is None:
            cutoff = datetime.now() - timedelta(days=period_days)
            trades = [
                t for t in self.all_trades
                if t.entry_time >= cutoff
                and t.status.value == 'CLOSED'
            ]

        closed_trades = [t for t in trades if t.status.value == 'CLOSED']

        if not closed_trades:
            logger.warning("No closed trades to analyze")
            return self._empty_metrics()

        # ─── BASIC COUNTS ─────────────────────────────────
        total = len(closed_trades)
        wins = [t for t in closed_trades if t.realized_pnl > 0]
        losses = [t for t in closed_trades if t.realized_pnl <= 0]

        win_rate = len(wins) / total if total > 0 else 0

        # ─── P&L METRICS ──────────────────────────────────
        total_pnl = sum(t.realized_pnl for t in closed_trades)

        avg_win = (
            float(np.mean([t.realized_pnl for t in wins]))
            if wins else 0.0
        )
        avg_loss = (
            float(np.mean([abs(t.realized_pnl) for t in losses]))
            if losses else 0.0
        )

        gross_profit = sum(t.realized_pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.realized_pnl for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss

        # ─── R-MULTIPLE STATISTICS ────────────────────────
        r_multiples = [t.r_multiple for t in closed_trades if t.r_multiple != 0]
        avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0

        expectancy = (
            win_rate * avg_win - (1 - win_rate) * avg_loss
        ) / avg_loss if avg_loss > 0 else 0

        # ─── RISK METRICS ─────────────────────────────────
        # Daily returns for Sharpe calculation
        daily_returns = self._calculate_daily_returns(closed_trades)

        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)

        max_dd, max_dd_duration = self._calculate_max_drawdown(closed_trades)

        calmar = (
            (total_pnl / len(closed_trades) * 252) / abs(max_dd)
            if max_dd != 0 else 0
        )

        # ─── SIGNAL IC ────────────────────────────────────
        signal_ic = self._calculate_signal_ic(closed_trades)

        # ─── PERFORMANCE BY REGIME ────────────────────────
        perf_by_regime = self._performance_by_regime(closed_trades)

        # ─── INITIAL CAPITAL FOR PCT CALC ─────────────────
        initial_capital = (
            closed_trades[0].entry_value
            if closed_trades else config.trading.STARTING_CAPITAL
        )

        return PerformanceMetrics(
            period_start=closed_trades[0].entry_time,
            period_end=closed_trades[-1].exit_time or datetime.now(),
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / initial_capital,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_r_multiple=avg_r,
            expectancy=expectancy,
            signal_ic=signal_ic,
            performance_by_regime=perf_by_regime
        )

    # ─── SHARPE & SORTINO ─────────────────────────────────

    def _calculate_sharpe(
        self,
        daily_returns: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Sharpe Ratio = (Rp - Rf) / σp
        Annualized using √252 factor.
        """
        if len(daily_returns) < 2:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        )
        return float(sharpe)

    def _calculate_sortino(
        self,
        daily_returns: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Sortino Ratio = (Rp - Rf) / σ_downside
        Only penalizes downside volatility.
        Better metric than Sharpe for non-normal returns.
        """
        if len(daily_returns) < 2:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf

        downside = daily_returns[daily_returns < 0]

        if len(downside) == 0 or np.std(downside) == 0:
            return float('inf')

        sortino = (
            np.mean(excess_returns) / np.std(downside) * np.sqrt(252)
        )
        return float(sortino)

    # ─── DRAWDOWN ANALYSIS ────────────────────────────────

    def _calculate_max_drawdown(
        self,
        trades: List[Trade]
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and its duration.

        Drawdown = (Peak - Trough) / Peak
        Duration = number of trades to recover
        """
        if not trades:
            return 0.0, 0

        cumulative_pnl = np.cumsum([t.realized_pnl for t in trades])

        peak = cumulative_pnl[0]
        max_dd = 0.0
        max_dd_duration = 0
        dd_start = 0

        for i, val in enumerate(cumulative_pnl):
            if val > peak:
                peak = val
                dd_start = i

            dd = (peak - val) / (abs(peak) + 1e-10)

            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - dd_start

        return float(max_dd), max_dd_duration

    # ─── SIGNAL IC CALCULATION ────────────────────────────

    def _calculate_signal_ic(
        self,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """
        Information Coefficient for each signal.
        IC = Correlation(Signal Score, Actual Return)

        IC > 0.10: Strong signal
        IC 0.05-0.10: Weak but valid
        IC < 0.05: Decaying — reduce weight
        """
        signal_scores: Dict[str, List[float]] = {}
        actual_returns: Dict[str, List[float]] = []

        actual_returns_list = []

        for trade in trades:
            if not trade.signals_at_entry:
                continue

            # Normalize return to R-multiple
            r = trade.r_multiple
            actual_returns_list.append(r)

            for sig_name, sig_data in trade.signals_at_entry.items():
                if sig_name not in signal_scores:
                    signal_scores[sig_name] = []
                signal_scores[sig_name].append(
                    sig_data.get('score', 0.0)
                )

        ic_dict = {}

        for sig_name, scores in signal_scores.items():
            n = min(len(scores), len(actual_returns_list))

            if n < 10:
                ic_dict[sig_name] = 0.0
                continue

            correlation, p_value = stats.pearsonr(
                scores[:n],
                actual_returns_list[:n]
            )

            # Only count IC if statistically significant
            ic = correlation if p_value < 0.1 else 0.0
            ic_dict[sig_name] = float(ic)

            # Alert on decaying signals
            if abs(ic) < config.trading.MIN_IC_THRESHOLD and n >= 30:
                logger.warning(
                    f"⚠️ Signal IC DECAYING: {sig_name} | "
                    f"IC: {ic:.4f} | "
                    f"Consider reducing weight"
                )

        return ic_dict

    # ─── MONTE CARLO SIMULATION ───────────────────────────

    def run_monte_carlo(
        self,
        n_simulations: int = 10000,
        n_trades: int = 100,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        starting_capital: float = 100.0,
        risk_per_trade: float = 0.02
    ) -> Dict:
        """
        Monte Carlo simulation of trading outcomes.

        Runs 10,000 simulations of N trades.
        Reports distribution of outcomes.

        MUST be run before deploying any strategy.
        """
        if win_rate is None:
            metrics = self.calculate_metrics()
            win_rate = metrics.win_rate or 0.55
            avg_win = metrics.avg_win or 1.5
            avg_loss = metrics.avg_loss or 1.0

        results = []
        ruin_count = 0
        drawdown_20_count = 0

        for sim in range(n_simulations):
            capital = starting_capital
            peak_capital = starting_capital
            sim_max_dd = 0.0

            for trade_num in range(n_trades):
                if capital < starting_capital * 0.01:
                    ruin_count += 1
                    break

                # Simulate trade outcome
                if np.random.random() < win_rate:
                    # Win: Random R between 0.5× and 2× avg_win
                    r = avg_win * np.random.uniform(0.5, 2.0)
                    pnl = capital * risk_per_trade * r
                else:
                    # Loss: Random R between 0.5× and 1.5× avg_loss
                    r = avg_loss * np.random.uniform(0.5, 1.5)
                    pnl = -capital * risk_per_trade * r

                capital += pnl

                # Track drawdown
                if capital > peak_capital:
                    peak_capital = capital

                dd = (peak_capital - capital) / peak_capital
                if dd > sim_max_dd:
                    sim_max_dd = dd

            results.append(capital)

            if sim_max_dd >= 0.20:
                drawdown_20_count += 1

        results = np.array(results)

        # Analysis
        mc_results = {
            'n_simulations': n_simulations,
            'n_trades': n_trades,
            'starting_capital': starting_capital,
            'parameters': {
                'win_rate': win_rate,
                'avg_win_r': avg_win,
                'avg_loss_r': avg_loss,
                'risk_per_trade': risk_per_trade
            },
            'outcomes': {
                'median': float(np.median(results)),
                'mean': float(np.mean(results)),
                'percentile_5': float(np.percentile(results, 5)),
                'percentile_1': float(np.percentile(results, 1)),
                'percentile_25': float(np.percentile(results, 25)),
                'percentile_75': float(np.percentile(results, 75)),
                'percentile_95': float(np.percentile(results, 95)),
                'worst_case': float(np.min(results)),
                'best_case': float(np.max(results)),
            },
            'risk_metrics': {
                'probability_of_ruin': ruin_count / n_simulations,
                'probability_of_20pct_drawdown': drawdown_20_count / n_simulations,
                'probability_of_profit': float(np.mean(results > starting_capital)),
                'probability_of_doubling': float(np.mean(results > starting_capital * 2)),
            },
            'verdict': self._monte_carlo_verdict(
                ruin_count / n_simulations,
                drawdown_20_count / n_simulations,
                float(np.percentile(results, 5)),
                starting_capital
            )
        }

        return mc_results

    def _monte_carlo_verdict(
        self,
        ruin_prob: float,
        dd20_prob: float,
        p5_outcome: float,
        starting_capital: float
    ) -> str:
        """
        Assess if strategy passes Monte Carlo thresholds.
        ALL must pass before deployment.
        """
        issues = []

        if ruin_prob > 0.001:
            issues.append(
                f"Risk of ruin too high: {ruin_prob:.2%} > 0.1%"
            )

        if dd20_prob > 0.05:
            issues.append(
                f"P(20% DD) too high: {dd20_prob:.2%} > 5%"
            )

        if p5_outcome < starting_capital:
            issues.append(
                f"5th percentile is negative: "
                f"${p5_outcome:.2f} < ${starting_capital:.2f}"
            )

        if not issues:
            return "✅ STRATEGY APPROVED FOR DEPLOYMENT"
        else:
            return f"❌ STRATEGY NEEDS ADJUSTMENT: {'; '.join(issues)}"

    # ─── COMPOUND PROJECTIONS ─────────────────────────────

    def project_compounding(
        self,
        starting_capital: float = 100.0,
        daily_return: float = 0.03,
        daily_std: float = 0.02,
        n_days: int = 365,
        n_simulations: int = 1000
    ) -> Dict:
        """
        Project capital growth with compounding.
        Uses geometric Brownian motion.
        Shows realistic range of outcomes.
        """
        all_paths = []

        for _ in range(n_simulations):
            path = [starting_capital]
            capital = starting_capital

            for day in range(n_days):
                daily_r = np.random.normal(daily_return, daily_std)
                daily_r = max(daily_r, -0.15)  # Max daily loss cap
                capital = capital * (1 + daily_r)
                path.append(capital)

            all_paths.append(path)

        all_paths = np.array(all_paths)

        # Key milestones
        milestones = {}
        targets = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

        for target in targets:
            if target > starting_capital:
                days_to_target = []
                for path in all_paths:
                    reached = np.where(path >= target)[0]
                    if len(reached) > 0:
                        days_to_target.append(reached[0])

                if days_to_target:
                    milestones[f"${target:,}"] = {
                        'median_days': int(np.median(days_to_target)),
                        'probability': len(days_to_target) / n_simulations,
                        'median_months': round(np.median(days_to_target) / 30, 1)
                    }

        return {
            'starting_capital': starting_capital,
            'parameters': {
                'daily_return_target': daily_return,
                'daily_std': daily_std,
                'trading_days': n_days
            },
            'projections': {
                'day_30': {
                    'median': float(np.median(all_paths[:, 30])),
                    'p10': float(np.percentile(all_paths[:, 30], 10)),
                    'p90': float(np.percentile(all_paths[:, 30], 90))
                },
                'day_90': {
                    'median': float(np.median(all_paths[:, 90])),
                    'p10': float(np.percentile(all_paths[:, 90], 10)),
                    'p90': float(np.percentile(all_paths[:, 90], 90))
                },
                'day_180': {
                    'median': float(np.median(all_paths[:, 180])),
                    'p10': float(np.percentile(all_paths[:, 180], 10)),
                    'p90': float(np.percentile(all_paths[:, 180], 90))
                },
                'day_365': {
                    'median': float(np.median(all_paths[:, 365])),
                    'p10': float(np.percentile(all_paths[:, 365], 10)),
                    'p90': float(np.percentile(all_paths[:, 365], 90))
                }
            },
            'milestones': milestones
        }

    # ─── WEEKLY REPORT ────────────────────────────────────

    def generate_weekly_report(
        self,
        trades: List[Trade],
        current_capital: float
    ) -> str:
        """
        Generate the weekly performance report.
        The Monday morning meeting with yourself.
        """
        metrics = self.calculate_metrics(trades, period_days=7)
        mc = self.run_monte_carlo(
            starting_capital=current_capital,
            win_rate=metrics.win_rate,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss
        )

        report = f"""
╔══════════════════════════════════════════════════════╗
║         WEEKLY PERFORMANCE REPORT                    ║
║         {datetime.now().strftime('%Y-%m-%d %H:%M')}                       ║
╠══════════════════════════════════════════════════════╣
║  CAPITAL STATUS                                      ║
║  Current Capital:    ${current_capital:>12,.2f}              ║
║  Week P&L:          ${metrics.total_pnl:>+12,.2f}              ║
║  Week Return:       {metrics.total_pnl_pct:>+11.2%}               ║
╠══════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                    ║
║  Total Trades:      {metrics.total_trades:>12}               ║
║  Win Rate:          {metrics.win_rate:>+11.1%}               ║
║  Profit Factor:     {metrics.profit_factor:>12.2f}              ║
║  Avg Win:           ${metrics.avg_win:>12,.2f}              ║
║  Avg Loss:          ${metrics.avg_loss:>12,.2f}              ║
║  Expectancy:        {metrics.expectancy:>+12.2f}R             ║
╠══════════════════════════════════════════════════════╣
║  RISK METRICS                                        ║
║  Max Drawdown:      {metrics.max_drawdown:>+11.2%}               ║
║  Sharpe Ratio:      {metrics.sharpe_ratio:>12.2f}              ║
║  Sortino Ratio:     {metrics.sortino_ratio:>12.2f}              ║
╠══════════════════════════════════════════════════════╣
║  SIGNAL HEALTH (IC Scores)                           ║"""

        for sig, ic in metrics.signal_ic.items():
            status = "✅" if abs(ic) >= 0.05 else "⚠️"
            report += f"\n║  {status} {sig:<20} IC: {ic:>+.4f}              ║"

        report += f"""
╠══════════════════════════════════════════════════════╣
║  PERFORMANCE CHECK                                   ║
║  Win Rate OK:       {'✅' if metrics.win_rate >= 0.55 else '❌'} (target ≥ 55%)                   ║
║  PF OK:             {'✅' if metrics.profit_factor >= 1.5 else '❌'} (target ≥ 1.5)                  ║
║  Sharpe OK:         {'✅' if metrics.sharpe_ratio >= 1.5 else '❌'} (target ≥ 1.5)                  ║
╠══════════════════════════════════════════════════════╣
║  MONTE CARLO VERDICT                                 ║
║  {mc['verdict']:<52} ║
╠══════════════════════════════════════════════════════╣
║  COMPOUNDING PHASE: {config.get_phase(current_capital):<32} ║
║  Reinvestment Rate: {config.get_reinvestment_rate(current_capital):<32.0%} ║
╚══════════════════════════════════════════════════════╝
"""
        return report

    # ─── HELPER METHODS ───────────────────────────────────

    def _calculate_daily_returns(
        self,
        trades: List[Trade]
    ) -> np.ndarray:
        """Group trade P&L by day and calculate daily returns"""
        if not trades:
            return np.array([])

        daily_pnl: Dict[str, float] = {}

        for trade in trades:
            day = trade.entry_time.strftime('%Y-%m-%d')
            daily_pnl[day] = daily_pnl.get(day, 0) + trade.realized_pnl

        pnl_values = list(daily_pnl.values())

        if not pnl_values:
            return np.array([])

        # Convert to returns (normalize by approximate capital)
        initial_capital = config.trading.STARTING_CAPITAL
        return np.array(pnl_values) / initial_capital

    def _performance_by_regime(
        self,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate average R-multiple by regime"""
        regime_returns: Dict[str, List[float]] = {}

        for trade in trades:
            if trade.regime_at_entry:
                regime_name = trade.regime_at_entry.name
                if regime_name not in regime_returns:
                    regime_returns[regime_name] = []
                regime_returns[regime_name].append(trade.r_multiple)

        return {
            regime: float(np.mean(returns))
            for regime, returns in regime_returns.items()
            if returns
        }

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades available"""
        return PerformanceMetrics(
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            avg_r_multiple=0.0,
            expectancy=0.0,
            signal_ic={},
            performance_by_regime={}
        )

    def add_trade(self, trade: Trade) -> None:
        """Add completed trade to history"""
        if trade.status.value == 'CLOSED':
            self.all_trades.append(trade)

