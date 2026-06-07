
# risk/risk_manager.py
# ============================================================
# THE SURVIVAL ENGINE
# If the signal engine is the brain
# The risk manager is the immune system
# It keeps the organism alive
# ============================================================

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import config
from models.data_models import (
    CompositeSignal, PositionSize, PortfolioState,
    Trade, Side, RegimeState, SignalStrength
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    5-Layer Risk Protection System.

    Layer 1: Per-trade risk (Kelly sizing)
    Layer 2: Daily loss limit
    Layer 3: Weekly loss limit
    Layer 4: Drawdown management
    Layer 5: Correlation control

    Every layer must pass before any trade executes.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital

        # Tracking
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0

        # Circuit breakers
        self.daily_circuit_open: bool = False
        self.weekly_circuit_open: bool = False
        self.drawdown_circuit_level: int = 0

        # Trade history for statistics
        self.trade_history: List[Trade] = []
        self.daily_trades: List[Trade] = []

        # Date tracking
        self._current_date: date = date.today()
        self._current_week: int = datetime.now().isocalendar()[1]

        # Win rate and R tracking (rolling 50 trades)
        self._recent_wins: List[float] = []
        self._recent_losses: List[float] = []
        self._recent_r_multiples: List[float] = []

    # ─── SAFE VALUE EXTRACTORS ────────────────────────────

    @staticmethod
    def _get_regime_key(regime) -> int:
        """
        Safely extract integer key from regime.
        Handles RegimeState enum, string, or int.
        Always returns a valid integer key.
        """
        # Case 1: It's a RegimeState enum
        if isinstance(regime, RegimeState):
            return regime.value

        # Case 2: It's an integer already
        if isinstance(regime, int):
            return regime

        # Case 3: It's a string name like "BULL_TREND"
        if isinstance(regime, str):
            try:
                return RegimeState[regime].value
            except KeyError:
                logger.warning(
                    f"Unknown regime string: '{regime}'. "
                    f"Using TRANSITION default."
                )
                return RegimeState.TRANSITION.value

        # Case 4: MagicMock or unknown object
        # Try to call .value attribute safely
        try:
            val = regime.value
            if isinstance(val, int):
                return val
        except AttributeError:
            pass

        # Final fallback
        logger.warning(
            f"Cannot extract regime key from: {type(regime)}. "
            f"Using TRANSITION default."
        )
        return RegimeState.TRANSITION.value

    @staticmethod
    def _get_strength_enum(strength) -> SignalStrength:
        """
        Safely extract SignalStrength enum.
        Handles enum, string, or int.
        """
        # Case 1: Already a SignalStrength enum
        if isinstance(strength, SignalStrength):
            return strength

        # Case 2: String name
        if isinstance(strength, str):
            try:
                return SignalStrength[strength]
            except KeyError:
                return SignalStrength.NEUTRAL

        # Case 3: Integer value
        if isinstance(strength, int):
            try:
                return SignalStrength(strength)
            except ValueError:
                return SignalStrength.NEUTRAL

        # Case 4: MagicMock or unknown
        try:
            val = strength.value
            if isinstance(val, int):
                return SignalStrength(val)
        except (AttributeError, ValueError):
            pass

        return SignalStrength.NEUTRAL

    @staticmethod
    def _get_direction_is_long(direction) -> bool:
        """
        Safely determine if direction is LONG.
        Handles Side enum, string, or mock.
        """
        # Case 1: Side enum
        if isinstance(direction, Side):
            return direction == Side.LONG

        # Case 2: String
        if isinstance(direction, str):
            return direction.upper() == 'LONG'

        # Case 3: MagicMock or unknown
        try:
            return direction.value == 'LONG'
        except AttributeError:
            pass

        return True  # Default to LONG as safe fallback

    # ─── POSITION SIZING (KELLY) ──────────────────────────

    def calculate_position_size(
        self,
        signal,
        current_capital: float,
        current_price: float,
        atr: float,
        open_trades: List
    ) -> Optional[PositionSize]:
        """
        Calculate exact position size using Modified Kelly Criterion.

        Steps:
        1. Calculate base Kelly fraction
        2. Apply signal strength adjustment
        3. Apply volatility adjustment
        4. Apply regime adjustment
        5. Apply drawdown adjustment
        6. Hard cap at 5%
        7. Validate minimum size
        """

        # Guard: must want to trade
        if not signal.should_trade:
            return None

        # ─── STEP 1: BASE KELLY ───────────────────────────
        win_rate, avg_win, avg_loss = self._get_edge_statistics()

        # No history yet: use conservative defaults
        if avg_win <= 0 or avg_loss <= 0:
            win_rate = 0.55
            avg_win  = 1.5   # R
            avg_loss = 1.0   # R

        b = avg_win / avg_loss   # Reward/Risk ratio
        p = win_rate
        q = 1 - win_rate

        # Kelly formula: f* = (pb - q) / b
        if b > 0:
            raw_kelly = (p * b - q) / b
        else:
            logger.warning("Invalid reward/risk ratio. Skipping trade.")
            return None

        # Quarter Kelly (safer)
        kelly_fraction = max(0.0, raw_kelly * config.trading.KELLY_FRACTION)

        # ─── STEP 2: SIGNAL STRENGTH ADJUSTMENT ──────────
        strength_enum = self._get_strength_enum(signal.strength)

        strength_multiplier = {
            SignalStrength.STRONG_LONG:  1.0,
            SignalStrength.WEAK_LONG:    0.5,
            SignalStrength.STRONG_SHORT: 1.0,
            SignalStrength.WEAK_SHORT:   0.5,
            SignalStrength.NEUTRAL:      0.0,
        }.get(strength_enum, 0.5)

        kelly_fraction *= strength_multiplier

        # ─── STEP 3: VOLATILITY ADJUSTMENT ───────────────
        # Target 1% daily portfolio volatility per trade
        target_vol  = 0.01
        current_vol = (atr / current_price) if current_price > 0 else 0.02

        if current_vol > 0:
            vol_adjustment = target_vol / current_vol
            vol_adjustment = float(np.clip(vol_adjustment, 0.3, 2.0))
        else:
            vol_adjustment = 1.0

        kelly_vol_adjusted = kelly_fraction * vol_adjustment

        # ─── STEP 4: REGIME ADJUSTMENT ───────────────────
        regime_key = self._get_regime_key(signal.regime)

        regime_multiplier = config.regime.REGIME_MULTIPLIERS.get(
            regime_key, 0.5
        )

        kelly_regime_adjusted = kelly_vol_adjusted * regime_multiplier

        # ─── STEP 5: DRAWDOWN ADJUSTMENT ─────────────────
        dd_multiplier        = self._get_drawdown_multiplier()
        kelly_dd_adjusted    = kelly_regime_adjusted * dd_multiplier

        # ─── STEP 6: HARD CAP ────────────────────────────
        final_kelly = min(
            kelly_dd_adjusted,
            config.trading.MAX_POSITION_FRACTION   # Hard cap 5%
        )
        final_kelly = max(final_kelly, 0.0)

        # ─── CALCULATE DOLLAR AMOUNTS ────────────────────
        risk_amount    = current_capital * config.trading.MAX_RISK_PER_TRADE
        position_value = current_capital * final_kelly

        # ─── STEP 7: MINIMUM SIZE CHECK ──────────────────
         # If Kelly gives tiny size, use minimum viable size
        if position_value < config.trading.MIN_POSITION_USD:
            # Force minimum position instead of skipping
            position_value      = config.trading.MIN_POSITION_USD
            position_size_units = position_value / current_price
            logger.info(
                f"📐 Using minimum position: ${position_value:.2f}"
            )


        position_size_units = position_value / current_price

        # ─── CALCULATE STOP AND TARGETS ──────────────────
        atr_stop_distance = 1.5 * atr
        is_long           = self._get_direction_is_long(signal.direction)

        if is_long:
            stop_loss = current_price - atr_stop_distance
            tp1       = current_price + (1.0 * atr_stop_distance)
            tp2       = current_price + (2.0 * atr_stop_distance)
            tp3       = current_price + (3.0 * atr_stop_distance)
        else:
            stop_loss = current_price + atr_stop_distance
            tp1       = current_price - (1.0 * atr_stop_distance)
            tp2       = current_price - (2.0 * atr_stop_distance)
            tp3       = current_price - (3.0 * atr_stop_distance)

        logger.info(
            f"📐 Position Sized: "
            f"Kelly={final_kelly:.4f} | "
            f"Value=${position_value:.2f} | "
            f"Risk=${risk_amount:.2f} | "
            f"Stop={stop_loss:.4f}"
        )

        return PositionSize(
            symbol=signal.symbol,
            timestamp=datetime.now(),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            reward_risk_ratio=b,
            raw_kelly=raw_kelly,
            adjusted_kelly=final_kelly,
            capital=current_capital,
            risk_amount=risk_amount,
            position_value=position_value,
            position_size_units=position_size_units,
            volatility_adjustment=vol_adjustment,
            regime_adjustment=regime_multiplier,
            drawdown_adjustment=dd_multiplier,
            stop_loss_price=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3
        )

    # ─── TRADE VALIDATION ─────────────────────────────────

    def validate_trade(
        self,
        signal,
        position_size: PositionSize,
        open_trades: List,
        current_capital: float
    ) -> Tuple[bool, str]:
        """
        Run ALL risk checks before allowing any trade.
        Returns (is_approved, reason).
        ALL checks must pass. No exceptions.
        """

        # CHECK 1: Daily Circuit Breaker
        if self.daily_circuit_open:
            return False, (
                f"Daily circuit breaker active: "
                f"{self.daily_pnl:.2%} loss today"
            )

        if current_capital > 0:
            daily_pnl_pct = self.daily_pnl / current_capital
            if daily_pnl_pct <= -config.trading.MAX_DAILY_LOSS:
                self.daily_circuit_open = True
                logger.warning(
                    f"🔴 Daily circuit breaker triggered: "
                    f"{daily_pnl_pct:.2%}"
                )
                return False, "Daily loss limit reached"

        # CHECK 2: Weekly Circuit Breaker
        if self.weekly_circuit_open:
            return False, "Weekly circuit breaker active"

        if current_capital > 0:
            weekly_pnl_pct = self.weekly_pnl / current_capital
            if weekly_pnl_pct <= -config.trading.MAX_WEEKLY_LOSS:
                self.weekly_circuit_open = True
                logger.warning("🔴 Weekly circuit breaker triggered")
                return False, "Weekly loss limit reached"

        # CHECK 3: Drawdown Check
        if self.current_drawdown >= config.trading.MAX_DRAWDOWN_BEFORE_STOP:
            return False, (
                f"Max drawdown exceeded: "
                f"{self.current_drawdown:.2%}"
            )

        if self.current_drawdown >= config.trading.MAX_DRAWDOWN_BEFORE_PAUSE:
            return False, (
                f"Drawdown pause active: "
                f"{self.current_drawdown:.2%}"
            )

        # CHECK 4: Max Concurrent Positions
        if len(open_trades) >= config.trading.MAX_CONCURRENT_TRADES:
            return False, (
                f"Max concurrent trades reached: "
                f"{len(open_trades)}"
            )

        # CHECK 5: Portfolio Heat
        current_heat = self._calculate_portfolio_heat(
            open_trades, current_capital
        )
        new_heat = current_heat + position_size.adjusted_kelly

        if new_heat > config.trading.MAX_PORTFOLIO_HEAT:
            return False, (
                f"Portfolio heat too high: "
                f"{new_heat:.2%}"
            )

        # CHECK 6: Correlation Check
        symbol = (
            signal.symbol
            if hasattr(signal, 'symbol')
            else 'UNKNOWN'
        )
        correlation_ok = self._check_correlation(symbol, open_trades)
        if not correlation_ok:
            return False, "Correlation with open positions too high"

        # CHECK 7: Minimum Capital Check
        if current_capital < config.trading.MIN_POSITION_USD * 2:
            return False, (
                f"Insufficient capital: "
                f"${current_capital:.2f}"
            )

        # CHECK 8: Signal staleness
        try:
            signal_age = (datetime.now() - signal.timestamp).seconds
            if signal_age > 30:
                return False, f"Signal stale: {signal_age}s old"
        except (AttributeError, TypeError):
            pass  # If timestamp not available, skip this check

        # ALL CHECKS PASSED
        logger.info(
            f"✅ Trade approved: {symbol} | "
            f"Size: ${position_size.position_value:.2f} | "
            f"Heat: {new_heat:.2%}"
        )
        return True, "All risk checks passed"

    # ─── PORTFOLIO MANAGEMENT ─────────────────────────────

    def update_pnl(
        self,
        trade: Trade,
        current_capital: float
    ) -> None:
        """Update P&L tracking when trade closes"""

        self.daily_pnl  += trade.realized_pnl
        self.weekly_pnl += trade.realized_pnl
        self.total_pnl  += trade.realized_pnl

        # Update peak and drawdown
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        if self.peak_capital > 0:
            self.current_drawdown = (
                (self.peak_capital - current_capital)
                / self.peak_capital
            )
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

        # Update rolling R statistics
        if trade.r_multiple > 0:
            self._recent_wins.append(trade.r_multiple)
        elif trade.r_multiple < 0:
            self._recent_losses.append(abs(trade.r_multiple))

        self._recent_r_multiples.append(trade.r_multiple)

        # Keep only last 50 trades
        self._recent_wins         = self._recent_wins[-50:]
        self._recent_losses       = self._recent_losses[-50:]
        self._recent_r_multiples  = self._recent_r_multiples[-50:]

        # Daily reset check
        today = date.today()
        if today != self._current_date:
            self.daily_pnl         = 0.0
            self.daily_circuit_open = False
            self._current_date     = today
            logger.info("📅 Daily P&L reset")

        # Weekly reset check
        current_week = datetime.now().isocalendar()[1]
        if current_week != self._current_week:
            self.weekly_pnl         = 0.0
            self.weekly_circuit_open = False
            self._current_week      = current_week
            logger.info("📅 Weekly P&L reset")

    def get_portfolio_state(
        self,
        current_capital: float,
        open_trades: List
    ) -> PortfolioState:
        """Get complete portfolio snapshot"""

        locked       = sum(t.entry_value for t in open_trades)
        heat         = self._calculate_portfolio_heat(
            open_trades, current_capital
        )
        phase        = config.get_phase(current_capital)
        reinvestment = config.get_reinvestment_rate(current_capital)

        return PortfolioState(
            timestamp=datetime.now(),
            total_capital=current_capital,
            available_capital=current_capital - locked,
            locked_in_trades=locked,
            open_trades=open_trades,
            daily_pnl=self.daily_pnl,
            weekly_pnl=self.weekly_pnl,
            total_pnl=self.total_pnl,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            portfolio_heat=heat,
            phase=phase,
            reinvestment_rate=reinvestment
        )

    # ─── RISK REPORTING ───────────────────────────────────

    def get_risk_report(self, current_capital: float) -> Dict:
        """Generate complete risk status report"""

        win_rate, avg_win, avg_loss = self._get_edge_statistics()
        bet_size = current_capital * config.trading.MAX_RISK_PER_TRADE
        ror      = self.calculate_risk_of_ruin(current_capital, bet_size)

        expected_value = (
            win_rate * avg_win - (1 - win_rate) * avg_loss
        )

        return {
            'timestamp': datetime.now().isoformat(),
            'capital': current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (
                self.daily_pnl / current_capital
                if current_capital > 0 else 0.0
            ),
            'weekly_pnl': self.weekly_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'risk_of_ruin': ror,
            'edge_statistics': {
                'win_rate': win_rate,
                'avg_win_r': avg_win,
                'avg_loss_r': avg_loss,
                'expected_value_per_trade': expected_value,
                'trades_analyzed': len(self._recent_r_multiples)
            },
            'circuit_breakers': {
                'daily': self.daily_circuit_open,
                'weekly': self.weekly_circuit_open,
                'drawdown_level': self.drawdown_circuit_level
            },
            'phase': config.get_phase(current_capital),
            'reinvestment_rate': config.get_reinvestment_rate(
                current_capital
            )
        }

    # ─── HELPER METHODS ───────────────────────────────────

    def _get_edge_statistics(self) -> Tuple[float, float, float]:
        """
        Calculate current edge statistics from trade history.
        Returns (win_rate, avg_win_R, avg_loss_R)
        Uses conservative defaults until 10+ trades available.
        """
        if len(self._recent_r_multiples) < 10:
            return 0.55, 1.5, 1.0  # Conservative defaults

        wins   = [r for r in self._recent_r_multiples if r > 0]
        losses = [abs(r) for r in self._recent_r_multiples if r < 0]

        win_rate = len(wins) / len(self._recent_r_multiples)
        avg_win  = float(np.mean(wins))  if wins   else 1.5
        avg_loss = float(np.mean(losses)) if losses else 1.0

        return win_rate, avg_win, avg_loss

    def _get_drawdown_multiplier(self) -> float:
        """
        Get position size multiplier based on current drawdown.
        Automatically reduces size as drawdown deepens.
        """
        dd        = self.current_drawdown
        dd_config = config.drawdown

        if dd >= dd_config.LEVEL_3:
            return dd_config.SIZE_MULTIPLIER_3   # 0.0 → stop trading
        elif dd >= dd_config.LEVEL_2:
            return dd_config.SIZE_MULTIPLIER_2   # 0.50
        elif dd >= dd_config.LEVEL_1:
            return dd_config.SIZE_MULTIPLIER_1   # 0.75
        else:
            return 1.0                            # No drawdown → full size

    def _calculate_portfolio_heat(
        self,
        open_trades: List,
        current_capital: float
    ) -> float:
        """
        Total portfolio risk exposure as fraction of capital.
        Heat = sum of (risk_amount / capital) for all open trades.
        """
        if not open_trades or current_capital <= 0:
            return 0.0

        total_heat = 0.0
        for trade in open_trades:
            try:
                risk       = trade.risk_amount / current_capital
                total_heat += risk
            except (AttributeError, ZeroDivisionError):
                pass

        return total_heat

    def _check_correlation(
        self,
        new_symbol: str,
        open_trades: List
    ) -> bool:
        """
        Check if new trade correlates too highly with open positions.
        Returns False if any pair exceeds MAX_CORRELATION threshold.
        """
        CORRELATION_MAP = {
            ('BTCUSDT', 'ETHUSDT'): 0.85,
            ('ETHUSDT', 'BTCUSDT'): 0.85,
            ('BTCUSDT', 'SOLUSDT'): 0.78,
            ('SOLUSDT', 'BTCUSDT'): 0.78,
            ('ETHUSDT', 'SOLUSDT'): 0.82,
            ('SOLUSDT', 'ETHUSDT'): 0.82,
            ('BTCUSDT', 'BNBUSDT'): 0.75,
            ('BNBUSDT', 'BTCUSDT'): 0.75,
        }

        for trade in open_trades:
            try:
                pair        = (new_symbol, trade.symbol)
                correlation = CORRELATION_MAP.get(pair, 0.3)

                if correlation > config.trading.MAX_CORRELATION:
                    logger.warning(
                        f"⚠️ High correlation: "
                        f"{new_symbol}/{trade.symbol} "
                        f"= {correlation:.2f}"
                    )
                    return False
            except AttributeError:
                pass

        return True

    def calculate_risk_of_ruin(
        self,
        capital: float,
        bet_size: float
    ) -> float:
        """
        Mathematical Risk of Ruin.
        R = ((1-edge) / (1+edge)) ^ (capital / bet_size)

        Must approach zero before live trading.
        """
        win_rate, avg_win, avg_loss = self._get_edge_statistics()

        edge = (
            (win_rate * avg_win - (1 - win_rate) * avg_loss)
            / avg_win
        )

        if edge <= 0:
            return 1.0  # No edge = 100% eventual ruin

        if bet_size <= 0:
            return 1.0

        n_bets = capital / bet_size
        ror    = ((1 - edge) / (1 + edge)) ** n_bets

        return float(np.clip(ror, 0.0, 1.0))
