# signals/signal_engine.py
# ============================================================
# THE SIGNAL FACTORY
# Generates all trading signals and combines them
# into a single actionable composite score
# ============================================================

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from config.settings import config
from models.data_models import (
    MarketData, OrderBookData, CompositeSignal,
    SignalResult, RegimeState, Side, SignalStrength
)
from signals.indicators import Indicators
from signals.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Generates all 7 trading signals and combines them.

    Signal Hierarchy:
    Tier 1: Regime (filter)
    Tier 2: Direction (AMS, SDS, OFI, Cointegration)
    Tier 3: Timing (OBI, Volume)
    Tier 4: Sizing (handled by Kelly)
    """

    # Signal weights (must sum to 1.0)
    SIGNAL_WEIGHTS = {
        'momentum': 0.20,
        'zscore_reversion': 0.15,
        'order_flow': 0.15,
        'volume_confirmation': 0.10,
        'trend_alignment': 0.20,
        'order_book_imbalance': 0.10,
        'cointegration': 0.10,
    }

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self._signal_history: Dict[str, List[SignalResult]] = {
            name: [] for name in self.SIGNAL_WEIGHTS.keys()
        }
        self._ic_tracker: Dict[str, List[float]] = {
            name: [] for name in self.SIGNAL_WEIGHTS.keys()
        }

        # Adaptive weights (start equal, update via ML)
        self._current_weights = self.SIGNAL_WEIGHTS.copy()

    # ─── INDIVIDUAL SIGNALS ───────────────────────────────

    def _signal_momentum(self, data: MarketData) -> SignalResult:
        """
        Signal 1: Adaptive Momentum Score (AMS)

        Multi-period momentum weighted by inverse volatility.
        Adapts lookback to current volatility regime.
        """
        score = Indicators.calculate_momentum_score(
            data.closes, data.volumes
        )

        # Acceleration confirmation
        acceleration = Indicators.calculate_price_acceleration(data.closes)

        # RSI confirmation
        rsi = Indicators.rsi(data.closes, config.trading.RSI_PERIOD)
        current_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

        # RSI modifier:
        # RSI 50-70: Confirms bullish momentum
        # RSI 30-50: Confirms bearish momentum
        # RSI >70 or <30: Momentum may be exhausted
        rsi_modifier = 1.0
        if score > 0:
            if 50 <= current_rsi <= 70:
                rsi_modifier = 1.2  # Confirm
            elif current_rsi > 70:
                rsi_modifier = 0.7  # Overbought warning
        elif score < 0:
            if 30 <= current_rsi <= 50:
                rsi_modifier = 1.2  # Confirm
            elif current_rsi < 30:
                rsi_modifier = 0.7  # Oversold warning

        final_score = np.tanh(score * rsi_modifier)

        return SignalResult(
            signal_name='momentum',
            value=score,
            score=float(final_score),
            weight=self._current_weights['momentum'],
            timestamp=data.timestamp,
            metadata={
                'rsi': current_rsi,
                'acceleration': acceleration,
                'rsi_modifier': rsi_modifier
            }
        )

    def _signal_zscore_reversion(
        self,
        data: MarketData,
        regime: RegimeState
    ) -> SignalResult:
        """
        Signal 2: Z-Score Reversion/Confirmation

        REGIME-AWARE interpretation:
        - In ranging market: Z-score signals REVERSION
        - In trending market: Z-score CONFIRMS trend

        This is the key insight most traders miss.
        """
        # Calculate Z-score
        zscores = Indicators.zscore(data.closes, config.trading.ZSCORE_LOOKBACK)
        current_z = float(zscores[-1]) if not np.isnan(zscores[-1]) else 0.0

        # Estimate OU parameters
        theta, mu, sigma = Indicators.ou_parameters(
            data.closes[-config.trading.ZSCORE_LOOKBACK:]
        )
        half_life = Indicators.mean_reversion_halflife(theta)

        if regime in [RegimeState.RANGING]:
            # MEAN REVERSION mode
            # High Z = overbought = SHORT signal
            # Low Z = oversold = LONG signal
            if abs(current_z) > 3.0:
                raw_score = -np.sign(current_z) * 1.0
            elif abs(current_z) > 2.0:
                raw_score = -np.sign(current_z) * 0.8
            elif abs(current_z) > 1.5:
                raw_score = -np.sign(current_z) * 0.4
            else:
                raw_score = 0.0

            mode = "REVERSION"

        elif regime in [RegimeState.BULL_TREND, RegimeState.BEAR_TREND]:
            # MOMENTUM CONFIRMATION mode
            # High positive Z confirms bull trend
            # High negative Z confirms bear trend
            raw_score = np.tanh(current_z / 2.0)
            mode = "CONFIRMATION"

        else:
            # TRANSITION: Reduced signal
            raw_score = -np.sign(current_z) * 0.3 if abs(current_z) > 2.0 else 0.0
            mode = "TRANSITION"

        return SignalResult(
            signal_name='zscore_reversion',
            value=current_z,
            score=float(np.clip(raw_score, -1.0, 1.0)),
            weight=self._current_weights['zscore_reversion'],
            timestamp=data.timestamp,
            metadata={
                'z_score': current_z,
                'theta': theta,
                'mu': mu,
                'half_life': half_life,
                'mode': mode
            }
        )

    def _signal_order_flow(self, trades_df) -> SignalResult:
        """
        Signal 3: Volume-Weighted Order Flow Imbalance

        OFI = (Buy Volume - Sell Volume) / Total Volume

        Measures REAL buying/selling pressure.
        Not price movement — actual aggressor volume.
        """
        if trades_df is None or len(trades_df) == 0:
            return SignalResult(
                signal_name='order_flow',
                value=0.0,
                score=0.0,
                weight=self._current_weights['order_flow'],
                timestamp=datetime.now(),
                metadata={'source': 'unavailable'}
            )

        total_buy = float(trades_df['buy_volume'].sum())
        total_sell = float(trades_df['sell_volume'].sum())
        total_volume = total_buy + total_sell

        if total_volume == 0:
            return SignalResult(
                signal_name='order_flow',
                value=0.0,
                score=0.0,
                weight=self._current_weights['order_flow'],
                timestamp=datetime.now()
            )

        # Basic OFI
        ofi = (total_buy - total_sell) / total_volume

        # Size-weighted OFI (large orders matter more)
        avg_size = float(trades_df['qty'].mean())
        large_trades = trades_df[trades_df['qty'] > 2 * avg_size]

        if len(large_trades) > 0:
            large_buy = float(large_trades['buy_volume'].sum())
            large_sell = float(large_trades['sell_volume'].sum())
            large_total = large_buy + large_sell

            if large_total > 0:
                large_ofi = (large_buy - large_sell) / large_total
                # Weight large OFI more (2:1 ratio)
                final_ofi = (ofi + 2 * large_ofi) / 3
            else:
                final_ofi = ofi
        else:
            final_ofi = ofi

        # Cumulative delta trend
        trades_df = trades_df.copy()
        trades_df['delta'] = trades_df['buy_volume'] - trades_df['sell_volume']
        cum_delta_recent = float(trades_df['delta'].tail(50).sum())
        cum_delta_all = float(trades_df['delta'].sum())

        # If recent delta diverges from overall → Regime change signal
        delta_acceleration = cum_delta_recent / (cum_delta_all + 1e-10)

        score = np.tanh(final_ofi * 3)  # Scale and bound

        return SignalResult(
            signal_name='order_flow',
            value=final_ofi,
            score=float(score),
            weight=self._current_weights['order_flow'],
            timestamp=datetime.now(),
            metadata={
                'ofi': ofi,
                'large_ofi': large_ofi if len(large_trades) > 0 else 0.0,
                'cumulative_delta': cum_delta_all,
                'delta_acceleration': delta_acceleration
            }
        )

    def _signal_volume_confirmation(self, data: MarketData) -> SignalResult:
        """
        Signal 4: Volume Confirmation

        Price movement with HIGH volume = REAL movement
        Price movement with LOW volume = FAKE movement

        Also checks VWAP position for institutional bias.
        """
        # Volume ratio
        vol_ratio = Indicators.volume_ratio(
            data.volumes, config.trading.VOLUME_MA_PERIOD
        )
        current_vol_ratio = float(vol_ratio[-1]) if not np.isnan(vol_ratio[-1]) else 1.0

        # VWAP
        vwap = Indicators.vwap(
            data.highs, data.lows, data.closes, data.volumes
        )
        current_vwap = float(vwap[-1]) if not np.isnan(vwap[-1]) else data.closes[-1]
        current_price = float(data.closes[-1])

        # Price vs VWAP
        vwap_pct = (current_price - current_vwap) / current_vwap

        # Price direction from recent candles
        recent_return = (data.closes[-1] - data.closes[-5]) / data.closes[-5]

        # Volume-confirmed directional score
        if current_vol_ratio > 1.5:
            # High volume: direction is real
            direction_score = np.sign(recent_return) * min(current_vol_ratio / 3, 1.0)
        elif current_vol_ratio < 0.8:
            # Low volume: direction is suspect, reduce score
            direction_score = np.sign(recent_return) * 0.2
        else:
            direction_score = np.sign(recent_return) * 0.5

        # VWAP modifier
        vwap_modifier = np.tanh(vwap_pct * 20)  # Institutional bias

        final_score = 0.7 * direction_score + 0.3 * vwap_modifier

        return SignalResult(
            signal_name='volume_confirmation',
            value=current_vol_ratio,
            score=float(np.clip(final_score, -1.0, 1.0)),
            weight=self._current_weights['volume_confirmation'],
            timestamp=data.timestamp,
            metadata={
                'volume_ratio': current_vol_ratio,
                'vwap': current_vwap,
                'price_vs_vwap_pct': vwap_pct * 100,
                'recent_return': recent_return * 100
            }
        )

    def _signal_trend_alignment(
        self,
        multi_tf_data: Dict[str, MarketData]
    ) -> SignalResult:
        """
        Signal 5: Cross-Timeframe Trend Alignment Score (CTAS)

        Scores each timeframe independently and weights:
        CTAS = 0.1(1m) + 0.2(5m) + 0.3(15m) + 0.4(1h)

        Only trade when multiple timeframes AGREE.
        Disagreement = uncertainty = no statistical edge.
        """
        tf_weights = {
            config.trading.TICK: 0.10,   # 1m
            config.trading.ETF: 0.20,    # 5m
            config.trading.MTF: 0.30,    # 15m
            config.trading.HTF: 0.40,    # 1h
        }

        tf_scores = {}

        for tf, weight in tf_weights.items():
            if tf not in multi_tf_data:
                tf_scores[tf] = 0.0
                continue

            tf_data = multi_tf_data[tf]
            closes = tf_data.closes

            if len(closes) < 50:
                tf_scores[tf] = 0.0
                continue

            # EMA cross for trend direction
            ema_fast = Indicators.ema(closes, config.trading.EMA_FAST)
            ema_slow = Indicators.ema(closes, config.trading.EMA_SLOW)

            if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
                tf_scores[tf] = 0.0
                continue

            # Score: +1 bullish, -1 bearish, magnitude = strength
            ema_diff = (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]
            tf_score = np.tanh(ema_diff * 100)

            # Price momentum on this TF
            if len(closes) >= 10:
                recent_momentum = (closes[-1] - closes[-10]) / closes[-10]
                # Combine EMA cross with momentum
                tf_score = 0.6 * tf_score + 0.4 * np.tanh(recent_momentum * 50)

            tf_scores[tf] = float(tf_score)

        # Weighted composite
        total_weight = 0
        weighted_score = 0

        for tf, weight in tf_weights.items():
            if tf in tf_scores:
                weighted_score += weight * tf_scores[tf]
                total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0

        # Count how many TFs agree with the direction
        if final_score > 0:
            aligned = sum(1 for s in tf_scores.values() if s > 0.2)
        elif final_score < 0:
            aligned = sum(1 for s in tf_scores.values() if s < -0.2)
        else:
            aligned = 0

        return SignalResult(
            signal_name='trend_alignment',
            value=final_score,
            score=float(np.clip(final_score, -1.0, 1.0)),
            weight=self._current_weights['trend_alignment'],
            timestamp=datetime.now(),
            metadata={
                'tf_scores': tf_scores,
                'timeframes_aligned': aligned,
                'total_timeframes': len(tf_weights)
            }
        )

    def _signal_order_book_imbalance(
        self,
        order_book: Optional[OrderBookData]
    ) -> SignalResult:
        """
        Signal 6: Order Book Imbalance (Timing Signal)

        OBI = (Bid Volume - Ask Volume) / (Bid + Ask)

        Predicts short-term (1-5 minute) price direction.
        Used primarily for ENTRY TIMING not direction bias.
        """
        if order_book is None:
            return SignalResult(
                signal_name='order_book_imbalance',
                value=0.0,
                score=0.0,
                weight=self._current_weights['order_book_imbalance'],
                timestamp=datetime.now(),
                metadata={'source': 'unavailable'}
            )

        obi = order_book.order_book_imbalance

        # Spread quality check
        # Wide spread = low liquidity = less reliable signal
        spread_pct = order_book.spread_pct
        spread_penalty = 1.0 if spread_pct < 0.001 else max(0.3, 1 - spread_pct * 100)

        # Large order detection (top level)
        if order_book.bids and order_book.asks:
            total_top_bid = order_book.bids[0][1]
            total_top_ask = order_book.asks[0][1]
            avg_level_size = (
                order_book.bid_volume_5_levels
                + order_book.ask_volume_5_levels
            ) / 10

            # Large bid at top = strong support
            large_bid = total_top_bid > 3 * avg_level_size
            large_ask = total_top_ask > 3 * avg_level_size

            if large_bid and not large_ask:
                obi = min(obi + 0.1, 1.0)  # Boost buy signal
            elif large_ask and not large_bid:
                obi = max(obi - 0.1, -1.0)  # Boost sell signal

        score = obi * spread_penalty

        return SignalResult(
            signal_name='order_book_imbalance',
            value=obi,
            score=float(np.clip(score, -1.0, 1.0)),
            weight=self._current_weights['order_book_imbalance'],
            timestamp=datetime.now(),
            metadata={
                'obi': obi,
                'spread_pct': spread_pct,
                'spread_penalty': spread_penalty,
                'bid_volume': order_book.bid_volume_5_levels,
                'ask_volume': order_book.ask_volume_5_levels
            }
        )

    def _signal_cointegration(
        self,
        primary_data: MarketData,
        secondary_data: Optional[MarketData]
    ) -> SignalResult:
        """
        Signal 7: Cointegration Spread Signal

        For BTC/ETH pair (most cointegrated crypto pair):
        Spread = BTC - β × ETH

        When spread deviates > 2σ:
        Long underperformer, short outperformer
        """
        if secondary_data is None or len(primary_data.closes) < 50:
            return SignalResult(
                signal_name='cointegration',
                value=0.0,
                score=0.0,
                weight=self._current_weights['cointegration'],
                timestamp=datetime.now(),
                metadata={'source': 'unavailable'}
            )

        n = min(len(primary_data.closes), len(secondary_data.closes), 100)

        primary = primary_data.closes[-n:]
        secondary = secondary_data.closes[-n:]

        # Estimate beta (hedge ratio) via OLS
        x = secondary
        y = primary

        X_mat = np.column_stack([np.ones(len(x)), x])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            alpha_coef, beta = coeffs
        except np.linalg.LinAlgError:
            return SignalResult(
                signal_name='cointegration',
                value=0.0,
                score=0.0,
                weight=self._current_weights['cointegration'],
                timestamp=datetime.now()
            )

        # Calculate spread
        spread = y - (alpha_coef + beta * x)

        # Z-score of spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        if spread_std == 0:
            return SignalResult(
                signal_name='cointegration',
                value=0.0,
                score=0.0,
                weight=self._current_weights['cointegration'],
                timestamp=datetime.now()
            )

        current_z = float((spread[-1] - spread_mean) / spread_std)

        # Signal: Spread too high = primary overvalued vs secondary
        # Short primary (score negative)
        # Spread too low = primary undervalued
        # Long primary (score positive)

        if abs(current_z) < 1.0:
            score = 0.0  # No signal in neutral zone
        elif abs(current_z) > 3.0:
            score = -np.sign(current_z) * 1.0
        else:
            # Linear scaling between 1 and 3 sigma
            score = -np.sign(current_z) * (abs(current_z) - 1.0) / 2.0

        # Mean reversion confirmation
        # Is spread already moving back?
        if len(spread) >= 3:
            spread_velocity = spread[-1] - spread[-3]
            if np.sign(spread_velocity) != np.sign(current_z):
                # Spread moving back already — STRONG signal
                score = score * 1.3

        return SignalResult(
            signal_name='cointegration',
            value=current_z,
            score=float(np.clip(score, -1.0, 1.0)),
            weight=self._current_weights['cointegration'],
            timestamp=datetime.now(),
            metadata={
                'spread_zscore': current_z,
                'beta': float(beta),
                'spread_std': float(spread_std),
                'spread_current': float(spread[-1])
            }
        )

    # ─── COMPOSITE SIGNAL GENERATION ──────────────────────

    def generate_composite_signal(
        self,
        symbol: str,
        primary_data: MarketData,
        multi_tf_data: Dict[str, MarketData],
        order_book: Optional[OrderBookData],
        trades_df,
        secondary_data: Optional[MarketData] = None
    ) -> CompositeSignal:
        """
        Generate the final composite trading signal.

        This is the main brain of the system.
        All signals combined into one clear decision.
        """
        # ─── STEP 1: DETECT REGIME ────────────────────────
        regime, regime_confidence, regime_details = (
            self.regime_detector.detect(primary_data)
        )

        # ─── STEP 2: CHECK IF TRADEABLE ───────────────────
        if not self.regime_detector.is_tradeable(regime, regime_confidence):
            return self._create_no_trade_signal(
                symbol, regime, "Regime not tradeable"
            )

        # ─── STEP 3: GENERATE ALL SIGNALS ─────────────────
        signals = {}

        # Signal 1: Momentum
        signals['momentum'] = self._signal_momentum(primary_data)

        # Signal 2: Z-Score (regime-aware)
        signals['zscore_reversion'] = self._signal_zscore_reversion(
            primary_data, regime
        )

        # Signal 3: Order Flow
        signals['order_flow'] = self._signal_order_flow(trades_df)

        # Signal 4: Volume Confirmation
        signals['volume_confirmation'] = self._signal_volume_confirmation(
            primary_data
        )

        # Signal 5: Trend Alignment
        signals['trend_alignment'] = self._signal_trend_alignment(
            multi_tf_data
        )

        # Signal 6: Order Book Imbalance
        signals['order_book_imbalance'] = self._signal_order_book_imbalance(
            order_book
        )

        # Signal 7: Cointegration
        signals['cointegration'] = self._signal_cointegration(
            primary_data, secondary_data
        )

        # ─── STEP 4: CALCULATE COMPOSITE SCORE ───────────
        composite_score = sum(
            sig.weighted_score
            for sig in signals.values()
        )
        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # ─── STEP 5: COUNT ALIGNED SIGNALS ───────────────
        if composite_score > 0:
            aligned = sum(
                1 for sig in signals.values()
                if sig.score > 0.2
            )
        elif composite_score < 0:
            aligned = sum(
                1 for sig in signals.values()
                if sig.score < -0.2
            )
        else:
            aligned = 0

        # ─── STEP 6: DETERMINE TRADE DECISION ─────────────
        threshold = config.trading.COMPOSITE_SCORE_THRESHOLD
        min_aligned = config.trading.MIN_SIGNALS_ALIGNED

        should_trade = (
            abs(composite_score) >= threshold
            and aligned >= min_aligned
        )

        # Determine direction
        if composite_score > threshold:
            direction = Side.LONG
            if composite_score > 0.8:
                strength = SignalStrength.STRONG_LONG
            else:
                strength = SignalStrength.WEAK_LONG
        elif composite_score < -threshold:
            direction = Side.SHORT
            if composite_score < -0.8:
                strength = SignalStrength.STRONG_SHORT
            else:
                strength = SignalStrength.WEAK_SHORT
        else:
            direction = Side.FLAT
            strength = SignalStrength.NEUTRAL
            should_trade = False

        # Confidence = how far above threshold
        if should_trade:
            confidence = min(
                (abs(composite_score) - threshold) / (1 - threshold),
                1.0
            )
        else:
            confidence = 0.0

        logger.info(
            f"📊 {symbol} | Score: {composite_score:+.3f} | "
            f"Regime: {regime.name} | "
            f"Aligned: {aligned}/7 | "
            f"Trade: {'✅' if should_trade else '❌'} {direction.value}"
        )

        return CompositeSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            individual_signals=signals,
            composite_score=composite_score,
            signals_aligned=aligned,
            regime=regime,
            should_trade=should_trade,
            direction=direction,
            strength=strength,
            confidence=confidence
        )

    def _create_no_trade_signal(
        self,
        symbol: str,
        regime: RegimeState,
        reason: str
    ) -> CompositeSignal:
        """Create a no-trade composite signal"""
        logger.info(f"🚫 No trade signal for {symbol}: {reason}")

        return CompositeSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            individual_signals={},
            composite_score=0.0,
            signals_aligned=0,
            regime=regime,
            should_trade=False,
            direction=Side.FLAT,
            strength=SignalStrength.NEUTRAL,
            confidence=0.0
        )

    def update_signal_weights(
        self,
        performance_by_signal: Dict[str, float]
    ) -> None:
        """
        Adaptively update signal weights based on recent performance.
        Signals with higher IC get higher weights.

        Uses ridge regression for stability.
        """
        if not performance_by_signal:
            return

        # Normalize IC scores to weights
        ic_scores = {
            k: max(v, 0)
            for k, v in performance_by_signal.items()
        }

        total_ic = sum(ic_scores.values())

        if total_ic > 0:
            new_weights = {
                k: v / total_ic
                for k, v in ic_scores.items()
            }

            # Blend: 70% new weights, 30% original
            # (Prevent overreaction to short-term performance)
            for signal_name in self._current_weights:
                if signal_name in new_weights:
                    self._current_weights[signal_name] = (
                        0.7 * new_weights[signal_name]
                        + 0.3 * self.SIGNAL_WEIGHTS[signal_name]
                    )

            logger.info(f"📈 Signal weights updated: {self._current_weights}")


