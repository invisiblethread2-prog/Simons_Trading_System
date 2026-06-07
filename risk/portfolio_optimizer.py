# risk/portfolio_optimizer.py
# ============================================================
# MODERN PORTFOLIO THEORY
# Maximize return for given risk level
# ============================================================

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Markowitz Portfolio Optimization.

    Medallion's insight:
    Don't optimize each trade individually.
    Optimize the PORTFOLIO as a whole.

    Goal: Maximum Sharpe Ratio portfolio
    Constraint: Sum of weights = 1, no short
    """

    def optimize_kelly_portfolio(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        symbols:           List[str],
        risk_aversion:     float = 2.0
    ) -> Dict[str, float]:
        """
        Kelly-Optimal Portfolio Weights.

        Maximizes: E[R] - (risk_aversion/2) * Var[R]
        This is the portfolio-level Kelly criterion.

        risk_aversion = 2.0 means quarter Kelly
        (same conservative approach as single-trade Kelly)
        """
        n  = len(symbols)
        mu = np.array([expected_returns[s] for s in symbols])

        def neg_objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_var    = np.dot(
                weights.T,
                np.dot(covariance_matrix, weights)
            )
            # Maximize: return - (risk_aversion/2) * variance
            return -(
                portfolio_return -
                (risk_aversion / 2) * portfolio_var
            )

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 0.4) for _ in range(n)]  # Max 40% per asset

        # Initial guess: equal weights
        w0 = np.ones(n) / n

        try:
            result = minimize(
                neg_objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )

            if result.success:
                weights = result.x
                weights = np.clip(weights, 0, None)
                weights /= weights.sum()

                optimal = {
                    s: float(w)
                    for s, w in zip(symbols, weights)
                }
                logger.info(
                    f"✅ Portfolio optimized: {optimal}"
                )
                return optimal

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")

        # Fallback: equal weights
        return {s: 1.0/n for s in symbols}

    def calculate_rolling_covariance(
        self,
        returns_dict: Dict[str, np.ndarray],
        lookback:     int = 60
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate rolling covariance matrix.

        CRITICAL: Correlations change over time.
        Use rolling window, not full history.
        During crises: All correlations → 1.0
        System must adapt to this.
        """
        symbols = list(returns_dict.keys())
        n       = len(symbols)

        # Build returns matrix
        min_len = min(
            len(r[-lookback:]) for r in returns_dict.values()
        )
        R = np.array([
            returns_dict[s][-min_len:]
            for s in symbols
        ])

        # Ledoit-Wolf shrinkage estimator
        # More stable than sample covariance with few observations
        cov = np.cov(R)

        # Add small diagonal for numerical stability
        cov += np.eye(n) * 1e-8

        return cov, symbols

    def portfolio_heat_check(
        self,
        open_positions: Dict[str, float],
        covariance_matrix: np.ndarray,
        symbols:           List[str],
        max_heat:          float = 0.10
    ) -> Tuple[bool, float]:
        """
        Portfolio Heat = Total correlated risk exposure.

        Simple position count misses correlation.
        BTC + ETH = ONE concentrated position (85% corr).
        BTC + Gold = TWO separate positions (10% corr).

        Returns: (is_safe, portfolio_variance)
        """
        n        = len(symbols)
        weights  = np.zeros(n)

        for i, symbol in enumerate(symbols):
            weights[i] = open_positions.get(symbol, 0.0)

        portfolio_variance = float(np.dot(
            weights.T,
            np.dot(covariance_matrix, weights)
        ))

        is_safe = portfolio_variance <= max_heat

        return is_safe, portfolio_variance
