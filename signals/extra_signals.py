# signals/extra_signals.py
# ============================================================
# ADDITIONAL SIGNALS — FIXED VERSION
# Fear & Greed neutral zone broadened (20-80)
# Stops interfering with trend signals
# ============================================================

import logging
import aiohttp
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional

from models.data_models import SignalResult

logger = logging.getLogger(__name__)


class ExtraSignals:
    """
    Additional free signals:
    1. Crypto Fear & Greed Index (FIXED)
    2. Funding Rate Signal
    3. GARCH Volatility Sizing
    """

    def __init__(self):
        self._fear_greed_cache: Optional[dict] = None
        self._fear_greed_time:  Optional[datetime] = None
        self._cache_minutes:    int = 60

    # ─── SIGNAL 8: FEAR & GREED (FIXED) ──────────────────

    async def get_fear_greed_signal(self) -> SignalResult:
        """
        Fear & Greed Index — FIXED VERSION.

        OLD: 25-44 = Fear = +0.40 score (was fighting bear trend)
        NEW: 20-80 = Neutral = 0.00 (does not interfere)

        Only extreme readings (< 20 or > 80) generate signal.
        This stops the signal from reducing bearish scores.
        """
        try:
            # Check cache (update every 60 minutes)
            if self._fear_greed_cache and self._fear_greed_time:
                age_minutes = (
                    datetime.now() - self._fear_greed_time
                ).seconds / 60
                if age_minutes < self._cache_minutes:
                    return self._fear_greed_to_signal(
                        self._fear_greed_cache
                    )

            # Fetch fresh data
            url = "https://api.alternative.me/fng/?limit=1"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data     = await resp.json()
                        fng_data = data['data'][0]

                        self._fear_greed_cache = fng_data
                        self._fear_greed_time  = datetime.now()

                        signal = self._fear_greed_to_signal(fng_data)

                        logger.info(
                            f"😱 Fear & Greed: {fng_data['value']} "
                            f"({fng_data['value_classification']}) "
                            f"→ Score: {signal.score:+.2f}"
                        )
                        return signal

        except Exception as e:
            logger.debug(f"Fear & Greed unavailable: {e}")

        # Return neutral if unavailable
        return SignalResult(
            signal_name='fear_greed',
            value=50.0,
            score=0.0,
            weight=0.08,
            timestamp=datetime.now(),
            metadata={'source': 'unavailable'}
        )

    def _fear_greed_to_signal(self, fng_data: dict) -> SignalResult:
        """
        FIXED: Neutral zone is now 20-80.

        Only EXTREME readings generate signal.
        Fear & Greed = 28 → Score = 0.0 (was +0.40 before)
        This was reducing bearish signals — now fixed.

        Extreme Fear  (< 20): Strong buy  signal
        Extreme Greed (> 80): Strong sell signal
        Middle zone (20-80):  Neutral — no interference
        """
        value = int(fng_data.get('value', 50))

        # FIXED SCORING — only extreme values matter
        if value <= 10:
            score = +1.0      # Extreme fear = very strong buy
        elif value <= 15:
            score = +0.7      # Extreme fear = strong buy
        elif value <= 20:
            score = +0.4      # High fear = moderate buy
        elif value >= 90:
            score = -1.0      # Extreme greed = very strong sell
        elif value >= 85:
            score = -0.7      # Extreme greed = strong sell
        elif value >= 80:
            score = -0.4      # High greed = moderate sell
        else:
            score = 0.0       # 20-80 range = NEUTRAL
            # Does not fight trend signals in this range

        return SignalResult(
            signal_name='fear_greed',
            value=float(value),
            score=float(score),
            weight=0.08,
            timestamp=datetime.now(),
            metadata={
                'value':          value,
                'classification': fng_data.get(
                    'value_classification', 'Unknown'
                ),
                'neutral_zone':   '20-80 (does not interfere)'
            }
        )

    # ─── SIGNAL 9: FUNDING RATE ───────────────────────────

    async def get_funding_rate_signal(
        self,
        symbol:       str,
        funding_rate: float
    ) -> SignalResult:
        """
        Perpetual Futures Funding Rate Signal.

        Positive funding → longs pay shorts
        → Overleveraged longs = contrarian SELL

        Negative funding → shorts pay longs
        → Overleveraged shorts = contrarian BUY

        Free edge most retail traders ignore.
        """
        try:
            rate = float(funding_rate) if funding_rate else 0.0

            if rate > 0.001:
                score = -1.0      # Very high positive = strong sell
            elif rate > 0.0005:
                score = -0.6      # High positive = sell
            elif rate > 0.0001:
                score = -0.3      # Moderate positive = weak sell
            elif rate < -0.001:
                score = +1.0      # Very negative = strong buy
            elif rate < -0.0005:
                score = +0.6      # High negative = buy
            elif rate < -0.0001:
                score = +0.3      # Moderate negative = weak buy
            else:
                score = 0.0       # Neutral funding

            logger.debug(
                f"💰 Funding {symbol}: "
                f"{rate:.6f} → Score: {score:+.2f}"
            )

            return SignalResult(
                signal_name='funding_rate',
                value=float(rate),
                score=float(score),
                weight=0.07,
                timestamp=datetime.now(),
                metadata={
                    'rate':           rate,
                    'annualized':     rate * 3 * 365,
                    'interpretation': (
                        'overleveraged_longs'
                        if rate > 0.0005
                        else 'overleveraged_shorts'
                        if rate < -0.0005
                        else 'balanced'
                    )
                }
            )

        except Exception as e:
            logger.debug(f"Funding rate signal error: {e}")

        return SignalResult(
            signal_name='funding_rate',
            value=0.0,
            score=0.0,
            weight=0.07,
            timestamp=datetime.now(),
            metadata={'source': 'unavailable'}
        )

    # ─── SIGNAL 10: GARCH VOLATILITY ──────────────────────

    def get_garch_sizing_multiplier(
        self,
        returns,
        target_vol: float = 0.02
    ) -> float:
        """
        GARCH(1,1) Volatility-Based Position Multiplier.

        σ²(t) = ω + α×ε²(t-1) + β×σ²(t-1)

        Predicts NEXT period volatility.
        Sizes position INVERSELY to predicted volatility.

        High vol → smaller position (more risk)
        Low vol  → larger position  (less risk)

        Returns: multiplier between 0.3 and 2.0
        """
        try:
            if returns is None or len(returns) < 20:
                return 1.0

            omega = 0.000001
            alpha = 0.10
            beta  = 0.85

            variance = float(np.var(returns))

            if variance <= 0:
                return 1.0

            # GARCH recursion
            for r in returns[-50:]:
                variance = (
                    omega
                    + alpha * float(r) ** 2
                    + beta  * variance
                )

            predicted_vol = float(np.sqrt(max(variance, 1e-10)))

            if predicted_vol <= 0:
                return 1.0

            multiplier = target_vol / predicted_vol
            multiplier = float(np.clip(multiplier, 0.3, 2.0))

            logger.debug(
                f"📊 GARCH vol: {predicted_vol:.4f} "
                f"→ Multiplier: {multiplier:.2f}x"
            )

            return multiplier

        except Exception as e:
            logger.debug(f"GARCH error: {e}")
            return 1.0
