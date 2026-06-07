# signals/stat_arb.py
# ============================================================
# STATISTICAL ARBITRAGE ENGINE
# Medallion's core edge — fixed indentation
# ============================================================

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


class StatisticalArbitrageEngine:
    """
    3 core stat arb strategies:
    1. Mean Reversion (OU Process)
    2. Risk-Adjusted Momentum
    3. Pairs Trading (Cointegration)
    """

    def calculate_ou_z_score(
        self,
        prices:   np.ndarray,
        lookback: int = 100
    ) -> Tuple[float, float, float]:
        """
        Ornstein-Uhlenbeck Z-Score.
        Returns: (z_score, half_life, mean_reversion_speed)
        """
        if len(prices) < lookback:
            return 0.0, float('inf'), 0.0

        series = prices[-lookback:]
        x_lag  = series[:-1]
        x_cur  = series[1:]

        X_mat = np.column_stack([np.ones(len(x_lag)), x_lag])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                X_mat, x_cur, rcond=None
            )
            alpha, beta = coeffs

            if 0 < beta < 1:
                theta     = -np.log(beta)
                mu        = alpha / (1 - beta)
                half_life = np.log(2) / theta
            else:
                return 0.0, float('inf'), 0.0

            residuals = x_cur - (alpha + beta * x_lag)
            sigma     = np.std(residuals)

            if sigma > 0:
                z = (prices[-1] - mu) / sigma
            else:
                return 0.0, float('inf'), 0.0

            return float(z), float(half_life), float(theta)

        except np.linalg.LinAlgError:
            return 0.0, float('inf'), 0.0

    def calculate_risk_adjusted_momentum(
        self,
        prices:  np.ndarray,
        periods: List[int] = [5, 10, 20, 60]
    ) -> float:
        """
        Risk-Adjusted Momentum.
        Return / Volatility — more predictive than raw momentum.
        Fixed: array length mismatch resolved.
        """
        if len(prices) < max(periods) + 3:
            return 0.0

        scores = []

        for period in periods:
            if len(prices) <= period + 3:
                continue

            try:
                # Return (skip last bar)
                ret = (
                    prices[-(period + 1)] /
                    prices[-(period + 2)] - 1
                )

                # Volatility of window
                window = prices[-(period + 2):-1]

                if len(window) < 2:
                    continue

                diffs   = np.diff(window)
                bases   = window[:-1]
                min_len = min(len(diffs), len(bases))

                if min_len == 0:
                    continue

                period_returns = diffs[:min_len] / bases[:min_len]
                vol = np.std(period_returns)

                if vol > 0:
                    scores.append(ret / vol)

            except Exception:
                continue

        if not scores:
            return 0.0

        # Inverse period weighting
        valid_periods = periods[:len(scores)]
        weights       = [1.0 / p for p in valid_periods]
        total_w       = sum(weights)
        weights       = [w / total_w for w in weights]

        raw = sum(s * w for s, w in zip(scores, weights))
        return float(np.tanh(raw))

    def calculate_cointegration_strength(
        self,
        series1: np.ndarray,
        series2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Cointegration spread z-score.
        Returns: (spread_z, hedge_ratio, half_life)
        """
        n = min(len(series1), len(series2))
        if n < 30:
            return 0.0, 1.0, float('inf')

        s1 = series1[-n:]
        s2 = series2[-n:]

        X_mat = np.column_stack([np.ones(n), s2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                X_mat, s1, rcond=None
            )
            alpha, beta = coeffs
        except np.linalg.LinAlgError:
            return 0.0, 1.0, float('inf')

        spread      = s1 - (alpha + beta * s2)
        spread_mean = np.mean(spread)
        spread_std  = np.std(spread)

        if spread_std == 0:
            return 0.0, float(beta), float('inf')

        z = (spread[-1] - spread_mean) / spread_std

        _, half_life, _ = self.calculate_ou_z_score(
            spread, lookback=n
        )

        return float(z), float(beta), float(half_life)

    def cross_sectional_rank(
        self,
        returns_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Cross-sectional momentum ranking.
        Best performers → continue outperforming.
        Returns: {symbol: score} normalized to [-1, +1]
        """
        if len(returns_dict) < 2:
            return {k: 0.0 for k in returns_dict}

        symbols = list(returns_dict.keys())
        returns = np.array([returns_dict[s] for s in symbols])

        ranks  = stats.rankdata(returns)
        n      = len(ranks)
        scores = (2 * ranks - n - 1) / (n - 1)

        return {
            s: float(sc)
            for s, sc in zip(symbols, scores)
        }

    def detect_volatility_regime(
        self,
        returns:      np.ndarray,
        short_window: int = 5,
        long_window:  int = 60
    ) -> Tuple[str, float]:
        """
        Volatility regime detection.

        HIGH_EXPANDING:  Reduce position size
        LOW_COMPRESSING: Breakout coming — momentum
        NORMAL:          Standard sizing

        Returns: (regime_name, vol_ratio)
        """
        if len(returns) < long_window:
            return 'UNKNOWN', 1.0

        short_vol = np.std(returns[-short_window:])
        long_vol  = np.std(returns[-long_window:])

        if long_vol == 0:
            return 'UNKNOWN', 1.0

        vol_ratio = short_vol / long_vol

        if vol_ratio > 1.5:
            regime = 'HIGH_EXPANDING'
        elif vol_ratio > 1.2:
            regime = 'EXPANDING'
        elif vol_ratio < 0.7:
            regime = 'LOW_COMPRESSING'
        elif vol_ratio < 0.8:
            regime = 'COMPRESSING'
        else:
            regime = 'NORMAL'

        return regime, float(vol_ratio)
