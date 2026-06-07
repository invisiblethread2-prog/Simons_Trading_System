# signals/indicators.py
# ============================================================
# PURE MATHEMATICAL INDICATORS
# No opinions. Only mathematics.
# ============================================================

import numpy as np
from scipy import stats
from typing import Tuple, Optional


class Indicators:
    """
    Mathematical indicator calculations.
    All methods are static - pure functions.
    Input: numpy arrays. Output: numpy arrays or scalars.
    """

    # ─── TREND INDICATORS ─────────────────────────────────

    @staticmethod
    def ema(values: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average.
        Weights recent data more heavily than SMA.
        """
        if len(values) < period:
            return np.full_like(values, np.nan)

        alpha = 2.0 / (period + 1)
        result = np.empty_like(values, dtype=float)
        result[:period] = np.nan
        result[period - 1] = np.mean(values[:period])

        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]

        return result

    @staticmethod
    def sma(values: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        for i in range(period - 1, len(values)):
            result[i] = np.mean(values[i - period + 1:i + 1])
        return result

    @staticmethod
    def atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Average True Range.
        Measures market volatility.
        ATR = Average of True Ranges over period
        True Range = max(H-L, |H-PrevC|, |L-PrevC|)
        """
        n = len(closes)
        true_ranges = np.empty(n)
        true_ranges[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i-1])
            lpc = abs(lows[i] - closes[i-1])
            true_ranges[i] = max(hl, hpc, lpc)

        # RMA (Wilder's smoothing)
        atr_values = np.full(n, np.nan)
        atr_values[period - 1] = np.mean(true_ranges[:period])

        for i in range(period, n):
            atr_values[i] = (
                atr_values[i-1] * (period - 1) + true_ranges[i]
            ) / period

        return atr_values

    @staticmethod
    def adx(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average Directional Index.
        Measures TREND STRENGTH (not direction).

        ADX > 25: Trending market
        ADX < 20: Ranging market

        Returns: (ADX, +DI, -DI)
        """
        n = len(closes)

        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with Wilder's method
        atr_vals = Indicators.atr(highs, lows, closes, period)

        smooth_plus_dm = np.full(n, np.nan)
        smooth_minus_dm = np.full(n, np.nan)

        smooth_plus_dm[period] = np.sum(plus_dm[1:period+1])
        smooth_minus_dm[period] = np.sum(minus_dm[1:period+1])

        for i in range(period + 1, n):
            smooth_plus_dm[i] = (
                smooth_plus_dm[i-1] - smooth_plus_dm[i-1]/period
                + plus_dm[i]
            )
            smooth_minus_dm[i] = (
                smooth_minus_dm[i-1] - smooth_minus_dm[i-1]/period
                + minus_dm[i]
            )

        # Calculate +DI and -DI
        plus_di = 100 * smooth_plus_dm / atr_vals
        minus_di = 100 * smooth_minus_dm / atr_vals

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        adx_vals = np.full(n, np.nan)
        valid_start = 2 * period

        if valid_start < n:
            adx_vals[valid_start] = np.nanmean(dx[period:valid_start+1])
            for i in range(valid_start + 1, n):
                adx_vals[i] = (
                    adx_vals[i-1] * (period - 1) + dx[i]
                ) / period

        return adx_vals, plus_di, minus_di

    # ─── MOMENTUM INDICATORS ──────────────────────────────

    @staticmethod
    def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index.
        Measures momentum and overbought/oversold conditions.
        RSI > 70: Overbought
        RSI < 30: Oversold
        """
        deltas = np.diff(closes)
        n = len(closes)
        rsi_values = np.full(n, np.nan)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi_values

    @staticmethod
    def macd(
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD - Moving Average Convergence Divergence.
        Returns: (MACD line, Signal line, Histogram)
        """
        ema_fast = Indicators.ema(closes, fast)
        ema_slow = Indicators.ema(closes, slow)

        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    # ─── VOLATILITY INDICATORS ────────────────────────────

    @staticmethod
    def bollinger_bands(
        closes: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands.
        Returns: (Upper, Middle, Lower)
        """
        middle = Indicators.sma(closes, period)

        rolling_std = np.full_like(closes, np.nan, dtype=float)
        for i in range(period - 1, len(closes)):
            rolling_std[i] = np.std(closes[i - period + 1:i + 1])

        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std

        return upper, middle, lower

    @staticmethod
    def bollinger_bandwidth(
        closes: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> np.ndarray:
        """
        BB Width = (Upper - Lower) / Middle
        Squeeze (low bandwidth) = Breakout coming
        """
        upper, middle, lower = Indicators.bollinger_bands(
            closes, period, std_dev
        )

        bandwidth = np.where(
            middle > 0,
            (upper - lower) / middle,
            np.nan
        )
        return bandwidth

    # ─── STATISTICAL INDICATORS ───────────────────────────

    @staticmethod
    def hurst_exponent(
        prices: np.ndarray,
        min_lag: int = 2,
        max_lag: int = 50
    ) -> float:
        """
        Hurst Exponent — The Most Important Single Indicator.

        Measures long-range dependence/memory in time series.

        H > 0.55: Trending (persistent) — Use momentum
        H = 0.50: Random walk — No edge
        H < 0.45: Mean-reverting (anti-persistent) — Use reversion

        Uses R/S (Rescaled Range) analysis.
        """
        if len(prices) < max_lag * 2:
            return 0.5  # Default to random walk

        # Use log returns for stationarity
        log_returns = np.log(prices[1:] / prices[:-1])

        lags = range(min_lag, min(max_lag, len(log_returns) // 2))
        rs_values = []
        lag_values = []

        for lag in lags:
            # Split series into non-overlapping windows
            n_windows = len(log_returns) // lag
            rs_per_window = []

            for w in range(n_windows):
                window = log_returns[w*lag:(w+1)*lag]

                if len(window) < 2:
                    continue

                # Mean-adjust
                mean = np.mean(window)
                deviation = np.cumsum(window - mean)

                # Range
                R = np.max(deviation) - np.min(deviation)

                # Standard deviation
                S = np.std(window, ddof=1)

                if S > 0:
                    rs_per_window.append(R / S)

            if rs_per_window:
                rs_values.append(np.mean(rs_per_window))
                lag_values.append(lag)

        if len(rs_values) < 2:
            return 0.5

        # Log-log linear regression
        log_lags = np.log(lag_values)
        log_rs = np.log(rs_values)

        slope, _, r_value, _, _ = stats.linregress(log_lags, log_rs)

        # Clamp to reasonable range
        hurst = np.clip(slope, 0.0, 1.0)
        return float(hurst)

    @staticmethod
    def zscore(
        values: np.ndarray,
        lookback: int = 100
    ) -> np.ndarray:
        """
        Rolling Z-Score.
        Z = (x - mean) / std

        |Z| > 2: Significant deviation
        |Z| > 3: Extreme deviation
        """
        result = np.full_like(values, np.nan, dtype=float)

        for i in range(lookback - 1, len(values)):
            window = values[i - lookback + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                result[i] = (values[i] - mean) / std

        return result

    @staticmethod
    def ou_parameters(
        prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimate Ornstein-Uhlenbeck parameters.
        dX = θ(μ - X)dt + σdW

        Returns:
            θ (mean reversion speed)
            μ (long-term mean)
            σ (volatility)

        Half-life of mean reversion: τ = ln(2) / θ
        """
        if len(prices) < 10:
            return 0.0, float(np.mean(prices)), float(np.std(prices))

        # OLS regression: X(t) = a + b*X(t-1) + ε
        x = prices[:-1]
        y = prices[1:]

        # Add constant for regression
        X_mat = np.column_stack([np.ones_like(x), x])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            a, b = coeffs

            # Convert to OU parameters
            # b = exp(-θΔt), for Δt=1: θ = -ln(b)
            if b <= 0 or b >= 1:
                theta = 0.0
            else:
                theta = -np.log(b)

            mu = a / (1 - b)

            # Residual std
            y_pred = a + b * x
            residuals = y - y_pred
            sigma = float(np.std(residuals))

            return float(theta), float(mu), sigma

        except np.linalg.LinAlgError:
            return 0.0, float(np.mean(prices)), float(np.std(prices))

    @staticmethod
    def mean_reversion_halflife(theta: float) -> float:
        """
        Calculate half-life of mean reversion.
        τ = ln(2) / θ

        τ < 10 bars: Fast reversion (use short TF)
        τ > 10 bars: Slow reversion (use longer TF)
        """
        if theta <= 0:
            return float('inf')
        return float(np.log(2) / theta)

    # ─── VOLUME INDICATORS ────────────────────────────────

    @staticmethod
    def vwap(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """
        Volume Weighted Average Price.
        VWAP = Σ(typical_price × volume) / Σ(volume)

        Price above VWAP: Bullish institutional bias
        Price below VWAP: Bearish institutional bias
        """
        typical_price = (highs + lows + closes) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volumes)
        cumulative_vol = np.cumsum(volumes)

        return np.where(
            cumulative_vol > 0,
            cumulative_tp_vol / cumulative_vol,
            np.nan
        )

    @staticmethod
    def volume_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Current volume vs average volume.
        Ratio > 1.5: Significant volume event
        Ratio > 2.0: Major volume event
        """
        ma_vol = Indicators.sma(volumes, period)
        return np.where(ma_vol > 0, volumes / ma_vol, np.nan)

    # ─── COMPOSITE CALCULATIONS ───────────────────────────

    @staticmethod
    def calculate_momentum_score(
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Adaptive Momentum Score (AMS).

        Multi-period momentum weighted by inverse volatility.
        Returns score from -1 to +1.
        """
        if len(closes) < 60:
            return 0.0

        # Multi-period returns
        periods = [5, 15, 30, 60]
        returns = []

        for p in periods:
            if len(closes) > p:
                r = (closes[-1] - closes[-p]) / closes[-p]
                returns.append(r)
            else:
                returns.append(0.0)

        # Volatility for each period (inverse weighting)
        vols = []
        for p in periods:
            period_returns = np.diff(closes[-p:]) / closes[-p:-1]
            vols.append(np.std(period_returns) if len(period_returns) > 0 else 1.0)

        # Weights = inverse volatility, normalized
        inv_vols = [1/v if v > 0 else 0 for v in vols]
        total_inv_vol = sum(inv_vols)

        if total_inv_vol == 0:
            return 0.0

        weights = [iv / total_inv_vol for iv in inv_vols]

        # Current ATR for normalization
        atr_current = np.mean(np.abs(np.diff(closes[-20:])))

        if atr_current == 0:
            return 0.0

        # Weighted momentum normalized by ATR
        raw_momentum = sum(w * r for w, r in zip(weights, returns))
        normalized = raw_momentum / (atr_current / closes[-1])

        # Convert to [-1, +1] via tanh
        return float(np.tanh(normalized * 10))

    @staticmethod
    def calculate_price_acceleration(closes: np.ndarray) -> float:
        """
        Second derivative of price (acceleration).

        Positive and increasing: Building momentum
        Positive but decreasing: Momentum fading
        Negative: Reversal in progress

        Key insight: Ball thrown upward — still going up
        but decelerating → WILL reverse
        """
        if len(closes) < 5:
            return 0.0

        # First derivative (velocity)
        velocity = np.diff(closes[-5:])

        # Second derivative (acceleration)
        acceleration = np.diff(velocity)

        if len(acceleration) == 0:
            return 0.0

        return float(acceleration[-1])

    @staticmethod
    def calculate_garch_volatility(
        returns: np.ndarray,
        omega: float = 0.01,
        alpha: float = 0.10,
        beta: float = 0.85
    ) -> float:
        """
        GARCH(1,1) Volatility Estimate.
        σ²(t) = ω + α×ε²(t-1) + β×σ²(t-1)

        Key insight: Volatility CLUSTERS.
        This predicts tomorrow's volatility from today's.
        Use for position sizing.

        Returns: Predicted next-period volatility
        """
        if len(returns) < 10:
            return float(np.std(returns)) if len(returns) > 1 else 0.01

        # Initialize
        variance = np.var(returns)

        # Iterate GARCH recursion
        for r in returns:
            variance = omega + alpha * r**2 + beta * variance

        return float(np.sqrt(variance))

