# ml/signal_optimizer.py
# ============================================================
# THE ADAPTIVE BRAIN
# Learns which signals work in which conditions
# ============================================================

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveSignalOptimizer:
    """
    Learns optimal signal weights using Ridge Regression.

    Medallion's core ML insight:
    Signal weights should CHANGE based on:
    1. Current market regime
    2. Recent signal performance
    3. Correlation between signals

    Static weights = leaving money on the table.
    Adaptive weights = extracting maximum edge.
    """

    def __init__(self, n_signals: int = 7):
        self.n_signals  = n_signals
        self.model      = Ridge(alpha=0.1)
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.ic_history: Dict[str, List[float]] = {}

        # Performance tracking per regime
        self.regime_weights: Dict[str, np.ndarray] = {}

    def fit(
        self,
        signal_history:   np.ndarray,  # Shape: (n_trades, n_signals)
        return_history:   np.ndarray,  # Shape: (n_trades,)
        regime_history:   List[str]    # Regime for each trade
    ) -> None:
        """
        Train signal weight optimizer.

        Uses ONLY past data to predict future returns.
        No look-ahead bias.
        Cross-validated to prevent overfitting.
        """
        if len(signal_history) < 20:
            logger.warning(
                "Insufficient data for ML training "
                f"(need 20+, have {len(signal_history)})"
            )
            return

        try:
            # Scale features
            X = self.scaler.fit_transform(signal_history)
            y = return_history

            # Cross-validate to find best regularization
            best_alpha  = 0.1
            best_score  = -float('inf')

            for alpha in [0.01, 0.1, 1.0, 10.0]:
                model = Ridge(alpha=alpha)
                scores = cross_val_score(
                    model, X, y,
                    cv=min(5, len(y)//4),
                    scoring='neg_mean_squared_error'
                )
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_alpha = alpha

            # Train final model
            self.model = Ridge(alpha=best_alpha)
            self.model.fit(X, y)
            self.is_trained = True

            # Train per-regime models
            for regime in set(regime_history):
                mask = np.array(regime_history) == regime
                if mask.sum() >= 10:
                    regime_model = Ridge(alpha=best_alpha)
                    regime_model.fit(X[mask], y[mask])
                    self.regime_weights[regime] = (
                        regime_model.coef_
                    )

            logger.info(
                f"✅ ML signal optimizer trained | "
                f"Alpha: {best_alpha} | "
                f"Samples: {len(y)}"
            )

        except Exception as e:
            logger.error(f"ML training error: {e}")

    def get_optimal_weights(
        self,
        current_regime: str
    ) -> Dict[str, float]:
        """
        Get current optimal signal weights.
        Uses regime-specific weights if available.
        Falls back to global weights.
        """
        signal_names = [
            'momentum', 'zscore_reversion',
            'order_flow', 'volume_confirmation',
            'trend_alignment', 'order_book_imbalance',
            'cointegration'
        ]

        if not self.is_trained:
            # Default equal weights
            n = len(signal_names)
            return {name: 1.0/n for name in signal_names}

        # Use regime-specific weights if available
        if current_regime in self.regime_weights:
            raw_weights = self.regime_weights[current_regime]
        else:
            raw_weights = self.model.coef_

        # Normalize to positive weights summing to 1
        weights = np.abs(raw_weights)
        total   = weights.sum()

        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(len(signal_names)) / len(signal_names)

        return {
            name: float(w)
            for name, w in zip(signal_names, weights)
        }

    def calculate_information_coefficient(
        self,
        signal_scores: np.ndarray,
        actual_returns: np.ndarray
    ) -> float:
        """
        Information Coefficient = Correlation(Prediction, Reality)

        IC > 0.05: Signal has edge
        IC > 0.10: Signal has strong edge
        IC < 0.02: Signal is noise — remove it

        Medallion targeted IC > 0.15 for each signal.
        """
        if len(signal_scores) < 10:
            return 0.0

        from scipy.stats import spearmanr
        # Use Spearman (rank correlation) — robust to outliers
        ic, p_value = spearmanr(signal_scores, actual_returns)

        # Only count if statistically significant
        if p_value > 0.1:
            return 0.0

        return float(ic)

    def detect_signal_decay(
        self,
        signal_name: str,
        recent_ic:   float,
        threshold:   float = 0.03
    ) -> bool:
        """
        Detect if a signal's edge is decaying.

        All edges decay eventually.
        Medallion's edge was FINDING NEW ONES
        before old ones died.

        Returns True if signal needs replacement.
        """
        if signal_name not in self.ic_history:
            self.ic_history[signal_name] = []

        self.ic_history[signal_name].append(recent_ic)

        # Keep last 20 readings
        self.ic_history[signal_name] = (
            self.ic_history[signal_name][-20:]
        )

        history = self.ic_history[signal_name]

        if len(history) < 5:
            return False

        # Check trend: Is IC declining?
        recent_avg = np.mean(history[-5:])
        old_avg    = np.mean(history[:5]) if len(history) >= 10 else recent_avg

        is_decaying = (
            recent_avg < threshold or
            (old_avg > threshold and recent_avg < old_avg * 0.5)
        )

        if is_decaying:
            logger.warning(
                f"⚠️ Signal decay detected: {signal_name} | "
                f"IC: {recent_avg:.4f} | "
                f"Consider replacing"
            )

        return is_decaying
