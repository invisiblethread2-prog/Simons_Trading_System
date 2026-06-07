# signals/regime_detector.py
# ============================================================
# THE MARKET STATE READER
# ============================================================

import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from scipy import stats

from config.settings import config
from models.data_models import MarketData, RegimeState, SignalResult
from signals.indicators import Indicators

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects current market regime.

    Methods:
    1. ADX (trend strength)
    2. Hurst Exponent (trending vs mean-reverting)
    3. ATR ratio (volatility level)
    4. Bollinger Band Width (compression)
    5. Price vs EMA200 (macro trend)
    """

    def __init__(self):
        self.history:          list       = []
        self.current_regime:   RegimeState = RegimeState.TRANSITION
        self.regime_confidence: float     = 0.0
        self.regime_duration:  int        = 0
        self._last_regime:     RegimeState = RegimeState.TRANSITION

    def detect(
        self,
        data: MarketData
    ) -> Tuple[RegimeState, float, Dict]:
        """
        Main regime detection.
        Returns: (RegimeState, confidence, details_dict)
        """
        closes  = data.closes
        highs   = data.highs
        lows    = data.lows
        volumes = data.volumes

        if len(closes) < config.trading.REGIME_LOOKBACK:
            logger.warning("Insufficient data for regime detection")
            return RegimeState.TRANSITION, 0.0, {}

        # ─── INDICATORS ───────────────────────────────────

        # ADX
        adx_vals, plus_di, minus_di = Indicators.adx(
            highs, lows, closes, config.trading.ADX_PERIOD
        )
        current_adx = float(adx_vals[-1]) if not np.isnan(adx_vals[-1]) else 0

        # Hurst
        hurst = Indicators.hurst_exponent(
            closes[-config.trading.HURST_LOOKBACK:]
        )

        # ATR ratio
        atr_vals       = Indicators.atr(highs, lows, closes, config.trading.ATR_PERIOD)
        current_atr    = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else 0
        historical_atr = float(np.nanmean(atr_vals[-50:]))
        atr_ratio      = current_atr / historical_atr if historical_atr > 0 else 1.0

        # Bollinger Band Width
        bb_width = Indicators.bollinger_bandwidth(
            closes, config.trading.BB_PERIOD, config.trading.BB_STD
        )
        valid_bb = bb_width[~np.isnan(bb_width)]
        if len(valid_bb) > 0:
            bb_percentile = float(
                stats.percentileofscore(valid_bb, float(valid_bb[-1]))
            )
        else:
            bb_percentile = 50.0

        # EMA200
        ema200 = Indicators.ema(closes, 200)
        price_vs_ema200 = (
            closes[-1] / ema200[-1] - 1
            if not np.isnan(ema200[-1]) and ema200[-1] > 0
            else 0
        )

        # Volume trend
        vol_ma = Indicators.sma(volumes, 20)
        volume_trending_up = (
            volumes[-1] > vol_ma[-1] * 1.2
            if not np.isnan(vol_ma[-1])
            else False
        )

        # Directional bias
        directional_bias = (
            float(plus_di[-1] - minus_di[-1])
            if not np.isnan(plus_di[-1]) else 0
        )

        # ─── REGIME SCORING ───────────────────────────────

        regime_scores = {
            RegimeState.BULL_TREND:  0.0,
            RegimeState.BEAR_TREND:  0.0,
            RegimeState.RANGING:     0.0,
            RegimeState.TRANSITION:  0.0,
            RegimeState.CHAOS:       0.0,
        }

        # CHAOS
        if atr_ratio > config.trading.ATR_CHAOS_MULTIPLIER:
            regime_scores[RegimeState.CHAOS] += 0.6
        if atr_ratio > 2.0:
            regime_scores[RegimeState.CHAOS] += 0.3
        if current_adx > 60:
            regime_scores[RegimeState.CHAOS] += 0.1

        # TRENDING
        if current_adx > config.trading.ADX_TREND_THRESHOLD:
            if hurst > config.trading.HURST_TREND_THRESHOLD:
                trend_score = min(
                    (current_adx - 25) / 25 + (hurst - 0.55) / 0.45,
                    1.0
                )
                if directional_bias > 0 and price_vs_ema200 > 0:
                    regime_scores[RegimeState.BULL_TREND] += trend_score
                elif directional_bias < 0 and price_vs_ema200 < 0:
                    regime_scores[RegimeState.BEAR_TREND] += trend_score
                else:
                    regime_scores[RegimeState.BULL_TREND] += trend_score * 0.5
                    regime_scores[RegimeState.BEAR_TREND] += trend_score * 0.5

        # RANGING
        if current_adx < config.trading.ADX_RANGE_THRESHOLD:
            if hurst < config.trading.HURST_RANGE_THRESHOLD:
                range_score = min(
                    (20 - current_adx) / 20 + (0.45 - hurst) / 0.45,
                    1.0
                )
                regime_scores[RegimeState.RANGING] += range_score

        if bb_percentile < 25:
            regime_scores[RegimeState.RANGING] += 0.3

        # TRANSITION
        if (config.trading.ADX_RANGE_THRESHOLD
                <= current_adx
                <= config.trading.ADX_TREND_THRESHOLD):
            regime_scores[RegimeState.TRANSITION] += 0.5

        if (config.trading.HURST_RANGE_THRESHOLD
                <= hurst
                <= config.trading.HURST_TREND_THRESHOLD):
            regime_scores[RegimeState.TRANSITION] += 0.3

        # ─── DETERMINE WINNER ─────────────────────────────

        total_score = sum(regime_scores.values())
        if total_score > 0:
            for state in regime_scores:
                regime_scores[state] /= total_score

        if regime_scores[RegimeState.CHAOS] > 0.4:
            detected_regime = RegimeState.CHAOS
            confidence      = regime_scores[RegimeState.CHAOS]
        else:
            del regime_scores[RegimeState.CHAOS]
            detected_regime = max(regime_scores, key=regime_scores.get)
            confidence      = regime_scores[detected_regime]

        # ─── PERSISTENCE FILTER ───────────────────────────

        if detected_regime != self._last_regime:
            self.regime_duration = 0
        else:
            self.regime_duration += 1

        if (detected_regime != self.current_regime
                and self.regime_duration < 3
                and confidence < 0.7):
            detected_regime = RegimeState.TRANSITION
            confidence      = 0.3

        self._last_regime      = detected_regime
        self.current_regime    = detected_regime
        self.regime_confidence = confidence

        details = {
            'adx':               current_adx,
            'hurst':             hurst,
            'atr_ratio':         atr_ratio,
            'bb_percentile':     bb_percentile,
            'price_vs_ema200_pct': price_vs_ema200 * 100,
            'directional_bias':  directional_bias,
            'volume_trending':   volume_trending_up,
            'regime_scores':     {
                k.name: v for k, v in regime_scores.items()
            },
            'regime_duration':   self.regime_duration,
            'atr_current':       current_atr,
        }

        logger.debug(
            f"Regime: {detected_regime.name} "
            f"(conf: {confidence:.2f}) | "
            f"ADX: {current_adx:.1f} | "
            f"Hurst: {hurst:.3f}"
        )

        return detected_regime, confidence, details

    def get_position_multiplier(
        self,
        regime: RegimeState
    ) -> float:
        """Position size multiplier per regime"""
        return config.regime.REGIME_MULTIPLIERS.get(
            regime.value, 0.5
        )

    def is_tradeable(
        self,
        regime:     RegimeState,
        confidence: float
    ) -> bool:
        """
        Can we trade in this regime?

        CHAOS:                  Never trade
        TRANSITION + low conf:  Skip
        Everything else:        Trade (size via multiplier)
        """
        if regime == RegimeState.CHAOS:
            logger.warning("🚫 CHAOS regime — No trading")
            return False

        if regime == RegimeState.TRANSITION and confidence < 0.25:
            logger.info("⚠ Very low confidence TRANSITION — Skipping")
            return False

        return True
