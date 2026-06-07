
# core/trading_engine.py
# ============================================================
# THE MASTER BRAIN — COMPLETE FINAL VERSION
# ✅ GARCH disabled (fixed $0.00 bug)
# ✅ Minimum position size enforced
# ✅ save_open_trade() integrated
# ✅ update_trade_on_close() integrated
# ✅ All syntax correct
# ============================================================

import asyncio
import logging
import signal
import sys
from datetime import datetime, date
from typing import Dict, List, Optional

from binance import AsyncClient

from config.settings import config
from data.data_collector import DataCollector
from data.database import TradeDatabase
from signals.signal_engine import SignalEngine
from signals.indicators import Indicators
from signals.extra_signals import ExtraSignals
from signals.stat_arb import StatisticalArbitrageEngine
from signals.sentiment_signal import SentimentSignal
from risk.risk_manager import RiskManager
from risk.portfolio_optimizer import PortfolioOptimizer
from execution.executor import ExecutionEngine
from analytics.performance_engine import PerformanceEngine
from backtesting.backtest_engine import BacktestEngine
from ml.signal_optimizer import AdaptiveSignalOptimizer
from models.data_models import (
    Side, TradeStatus, RegimeState
)

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Complete algorithmic trading system.

    Flow per cycle:
    1. Collect market data
    2. Generate 7 core signals
    3. Blend extra signals (Fear & Greed, Funding)
    4. StatArb + Sentiment adjustments
    5. Recalculate should_trade after blend
    6. Validate 5-layer risk
    7. Force minimum position size
    8. Execute trade
    9. Monitor + manage open trades
    10. Save everything to database
    11. ML retrain every 20 trades
    """

    def __init__(self):
        self.is_running      = False
        self.current_capital = config.trading.STARTING_CAPITAL
        self.cycle_count     = 0
        self.start_time      = datetime.now()

        # ─── CORE COMPONENTS ──────────────────────────────
        self.data_collector:  Optional[DataCollector]     = None
        self.signal_engine:   Optional[SignalEngine]      = None
        self.risk_manager:    Optional[RiskManager]       = None
        self.executor:        Optional[ExecutionEngine]   = None
        self.performance:     Optional[PerformanceEngine] = None
        self.client:          Optional[AsyncClient]       = None
        self.database:        Optional[TradeDatabase]     = None
        self.extra_signals:   Optional[ExtraSignals]      = None

        # ─── ADVANCED COMPONENTS ──────────────────────────
        self.backtest_engine:     Optional[BacktestEngine]             = None
        self.signal_optimizer:    Optional[AdaptiveSignalOptimizer]    = None
        self.portfolio_optimizer: Optional[PortfolioOptimizer]         = None
        self.stat_arb:            Optional[StatisticalArbitrageEngine] = None
        self.sentiment:           Optional[SentimentSignal]            = None

        # ─── TRACKING ─────────────────────────────────────
        self._trade_count_since_retrain: int   = 0
        self._daily_start_capital:       float = config.trading.STARTING_CAPITAL
        self._last_daily_save:           str   = datetime.now().strftime('%Y-%m-%d')

    # ═══════════════════════════════════════════════════════
    # STARTUP
    # ═══════════════════════════════════════════════════════

    async def start(self) -> None:
        """Initialize all components and start trading."""
        logger.info(
            f"\n{'#'*60}\n"
            f"# THE EMPIRE STARTS HERE\n"
            f"# Mode: {'PAPER' if config.trading.PAPER_TRADING else 'LIVE'}\n"
            f"# Capital: ${self.current_capital:.2f}\n"
            f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'#'*60}"
        )

        # Binance client
        self.client = await AsyncClient.create(
            api_key=config.exchange.BINANCE_API_KEY,
            api_secret=config.exchange.BINANCE_SECRET_KEY,
            testnet=config.exchange.BINANCE_TESTNET
        )

        # Database first
        self.database = TradeDatabase(config.trading.DB_PATH)
        logger.info("✅ Database initialized")

        # Restore saved capital
        saved_capital = self.database.load_state(
            'current_capital', config.trading.STARTING_CAPITAL
        )
        if saved_capital and saved_capital != config.trading.STARTING_CAPITAL:
            self.current_capital      = saved_capital
            self._daily_start_capital = saved_capital
            logger.info(f"✅ Restored capital: ${self.current_capital:.2f}")

        # Core components
        self.data_collector = DataCollector()
        await self.data_collector.initialize()

        self.signal_engine = SignalEngine()
        self.extra_signals = ExtraSignals()
        logger.info("✅ Core signals initialized")

        self.risk_manager = RiskManager(
            initial_capital=self.current_capital
        )

        self.executor = ExecutionEngine(
            client=self.client,
            paper_trading=config.trading.PAPER_TRADING
        )

        self.performance = PerformanceEngine()

        # Advanced components
        self.backtest_engine     = BacktestEngine(self.current_capital)
        self.signal_optimizer    = AdaptiveSignalOptimizer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.stat_arb            = StatisticalArbitrageEngine()
        self.sentiment           = SentimentSignal()
        logger.info("✅ Advanced components initialized")

        # Validate before trading
        if not await self._validate_system():
            logger.error("❌ Validation failed.")
            await self.stop()
            return

        # Shutdown handlers
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Start
        self.is_running = True
        logger.info("✅ System validated. Starting trading loop.")
        await self._main_loop()

    # ═══════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════

    async def _main_loop(self) -> None:
        """Main trading loop — runs every 60 seconds."""
        while self.is_running:
            cycle_start      = datetime.now()
            self.cycle_count += 1

            try:
                # Portfolio state
                portfolio = self.risk_manager.get_portfolio_state(
                    self.current_capital,
                    list(self.executor.open_trades.values())
                )

                # Monitor open trades first
                await self._monitor_open_trades()

                # Circuit breaker check
                if portfolio.is_circuit_breaker_active:
                    logger.warning(
                        f"⛔ Circuit breaker | "
                        f"Daily: {self.risk_manager.daily_pnl / self.current_capital:.2%}"
                    )
                    await self._wait_for_next_cycle(60)
                    continue

                # Analyze each symbol
                for symbol in config.trading.PRIMARY_PAIRS:
                    await self._process_symbol(symbol)
                    await asyncio.sleep(2)

                # Save state
                if self.database:
                    self.database.save_state(
                        'current_capital', self.current_capital
                    )
                    self.database.save_capital(
                        capital=self.current_capital,
                        daily_pnl=self.risk_manager.daily_pnl,
                        total_pnl=self.risk_manager.total_pnl,
                        drawdown=self.risk_manager.current_drawdown,
                        phase=config.get_phase(self.current_capital)
                    )

                # Daily save check
                await self._check_daily_save()

                # Periodic reviews
                if self.cycle_count % 60 == 0:
                    await self._hourly_review()

                if self.cycle_count % (60 * 24 * 7) == 0:
                    await self._weekly_review()

                # Wait for next cycle
                elapsed   = (datetime.now() - cycle_start).seconds
                wait_time = max(60 - elapsed, 5)
                await self._wait_for_next_cycle(wait_time)

            except Exception as e:
                logger.error(f"❌ Main loop error: {e}", exc_info=True)
                await asyncio.sleep(10)

    # ═══════════════════════════════════════════════════════
    # PROCESS ONE SYMBOL
    # ═══════════════════════════════════════════════════════

    async def _process_symbol(self, symbol: str) -> None:
        """
        Complete analysis and trading cycle for one symbol.

        KEY FIXES:
        1. GARCH disabled → no more $0.00 positions
        2. Minimum position size enforced before validation
        3. should_trade recalculated AFTER all blending
        """
        try:
            # ─── DATA COLLECTION ──────────────────────────
            data_package = await self.data_collector.get_complete_data_package(
                symbol
            )

            multi_tf     = data_package['multi_tf']
            order_book   = data_package['order_book']
            trades_df    = data_package['trades']
            funding_rate = data_package.get('funding_rate', 0.0)

            if not multi_tf or config.trading.ETF not in multi_tf:
                logger.warning(f"No data for {symbol}")
                return

            primary_data = multi_tf[config.trading.ETF]

            # Secondary for cointegration
            secondary_symbol = {
                'BTCUSDT': 'ETHUSDT',
                'ETHUSDT': 'BTCUSDT',
                'SOLUSDT': 'ETHUSDT',
            }.get(symbol)

            secondary_data = (
                multi_tf.get(secondary_symbol)
                if secondary_symbol else None
            )

            # ─── CORE SIGNAL GENERATION ───────────────────
            signal = self.signal_engine.generate_composite_signal(
                symbol=symbol,
                primary_data=primary_data,
                multi_tf_data=multi_tf,
                order_book=order_book,
                trades_df=trades_df,
                secondary_data=secondary_data
            )

            # Save raw signal
            try:
                self.database.save_signal(symbol, signal)
            except Exception:
                pass

            # ─── EXTRA SIGNALS ────────────────────────────
            extra_score = 0.0
            try:
                fear_greed_signal = await self.extra_signals.get_fear_greed_signal()
                funding_signal    = await self.extra_signals.get_funding_rate_signal(
                    symbol, funding_rate
                )
                extra_score = (
                    fear_greed_signal.score * 0.05 +
                    funding_signal.score    * 0.05
                )
            except Exception as e:
                logger.debug(f"Extra signals error: {e}")

            # ─── BLEND BASE + EXTRA ────────────────────────
            base_score    = signal.composite_score
            blended_score = float(
                max(-1.0, min(1.0, base_score + extra_score))
            )

            # ─── STAT ARB ADJUSTMENT ──────────────────────
            try:
                ou_z, half_life, _ = self.stat_arb.calculate_ou_z_score(
                    primary_data.closes
                )
                ram = self.stat_arb.calculate_risk_adjusted_momentum(
                    primary_data.closes
                )
                vol_regime, vol_ratio = self.stat_arb.detect_volatility_regime(
                    primary_data.returns
                )

                if vol_regime == 'HIGH_EXPANDING':
                    blended_score *= 0.7
                elif vol_regime == 'LOW_COMPRESSING':
                    blended_score *= 1.1

                blended_score = float(max(-1.0, min(1.0, blended_score)))

            except Exception as e:
                logger.debug(f"StatArb error: {e}")

            # ─── SENTIMENT ADJUSTMENT ─────────────────────
            try:
                sent_score    = await self.sentiment.get_combined_sentiment(symbol)
                blended_score = float(
                    max(-1.0, min(1.0, blended_score + sent_score * 0.03))
                )
            except Exception as e:
                logger.debug(f"Sentiment error: {e}")

            # ─── UPDATE SIGNAL WITH BLENDED VALUES ────────
            # CRITICAL: Update should_trade AFTER all blending
            signal.composite_score = blended_score
            threshold = config.trading.COMPOSITE_SCORE_THRESHOLD
            regime_ok = (signal.regime != RegimeState.CHAOS)

            signal.should_trade = (
                abs(blended_score) >= threshold
                and signal.signals_aligned >= config.trading.MIN_SIGNALS_ALIGNED
                and regime_ok
            )

            # Update direction
            if blended_score >= threshold:
                signal.direction = Side.LONG
            elif blended_score <= -threshold:
                signal.direction = Side.SHORT
            else:
                signal.direction = Side.FLAT

            # Log significant signals
            if abs(blended_score) >= 0.3:
                logger.info(
                    f"📊 {symbol} | "
                    f"Base: {base_score:+.3f} | "
                    f"Extra: {extra_score:+.3f} | "
                    f"Final: {blended_score:+.3f} | "
                    f"Regime: {signal.regime.name} | "
                    f"Aligned: {signal.signals_aligned}/7 | "
                    f"Trade: {'✅' if signal.should_trade else '❌'} "
                    f"{signal.direction.value}"
                )

            # ─── NO TRADE CHECK ───────────────────────────
            if not signal.should_trade:
                return

            # ─── ALREADY IN THIS SYMBOL? ──────────────────
            existing = [
                t for t in self.executor.open_trades.values()
                if t.symbol == symbol
            ]
            if existing:
                logger.debug(f"Already in {symbol}, skipping")
                return

            # ─── ATR FOR STOPS ────────────────────────────
            atr_values = Indicators.atr(
                primary_data.highs,
                primary_data.lows,
                primary_data.closes,
                config.trading.ATR_PERIOD
            )
            current_atr   = float(atr_values[-1])
            current_price = primary_data.current_price

            # ─── KELLY POSITION SIZE ──────────────────────
            position_size = self.risk_manager.calculate_position_size(
                signal=signal,
                current_capital=self.current_capital,
                current_price=current_price,
                atr=current_atr,
                open_trades=list(self.executor.open_trades.values())
            )

            if position_size is None:
                logger.warning(f"⚠️ Position size calculation failed for {symbol}")
                return

            # ─── FORCE MINIMUM POSITION SIZE ──────────────
            # FIX: Kelly with small capital gives ~$0
            # Force minimum to ensure trades actually execute
            min_size = config.trading.MIN_POSITION_USD
            if position_size.position_value < min_size:
                logger.info(
                    f"📐 Forcing minimum position: "
                    f"${position_size.position_value:.2f} → ${min_size:.2f}"
                )
                position_size.position_value      = min_size
                position_size.position_size_units = (
                    min_size / current_price
                    if current_price > 0 else 0
                )

            # ─── GARCH DISABLED ───────────────────────────
            # Re-enable when capital > $1000
            # garch_multiplier = 1.0 (no-op multiply)
            position_size.position_size_units = (
                position_size.position_value / current_price
                if current_price > 0 else 0
            )

            # ─── RISK VALIDATION ──────────────────────────
            is_approved, reason = self.risk_manager.validate_trade(
                signal=signal,
                position_size=position_size,
                open_trades=list(self.executor.open_trades.values()),
                current_capital=self.current_capital
            )

            if not is_approved:
                logger.info(f"🚫 Rejected ({symbol}): {reason}")
                return

            # ─── EXECUTE ──────────────────────────────────
            logger.info(
                f"\n{'='*50}\n"
                f"🎯 SIGNAL APPROVED — EXECUTING TRADE\n"
                f"   Symbol:    {symbol}\n"
                f"   Direction: {signal.direction.value}\n"
                f"   Score:     {blended_score:+.3f}\n"
                f"   Regime:    {signal.regime.name}\n"
                f"   Aligned:   {signal.signals_aligned}/7\n"
                f"   Price:     {current_price:.4f}\n"
                f"   Size:      ${position_size.position_value:.2f}\n"
                f"{'='*50}"
            )

            trade = await self.executor.execute_entry(
                signal=signal,
                position_size=position_size,
                current_price=current_price
            )

            if trade:
                logger.info(
                    f"🚀 NEW TRADE: {trade.trade_id} | "
                    f"{symbol} {signal.direction.value} | "
                    f"Score: {blended_score:+.3f} | "
                    f"Size: ${position_size.position_value:.2f}"
                )

                # Save to database immediately
                if self.database:
                    try:
                        self.database.save_open_trade(
                            trade=trade,
                            capital_before=self.current_capital
                        )
                        self.database.save_capital(
                            capital=self.current_capital,
                            daily_pnl=self.risk_manager.daily_pnl,
                            total_pnl=self.risk_manager.total_pnl,
                            drawdown=self.risk_manager.current_drawdown,
                            phase=config.get_phase(self.current_capital),
                            note=f"Trade opened: {trade.trade_id}"
                        )
                        logger.info(
                            f"💾 Trade {trade.trade_id} saved to database"
                        )
                    except Exception as db_err:
                        logger.error(f"DB save error: {db_err}")
            else:
                logger.error(f"❌ Trade execution failed for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════════
    # MONITOR OPEN TRADES
    # ═══════════════════════════════════════════════════════

    async def _monitor_open_trades(self) -> None:
        """
        Monitor all open trades.
        Handles stops, take profits, trailing stops.
        Saves closed trades to database.
        """
        for trade_id, trade in list(self.executor.open_trades.items()):
            try:
                # Get current price with retry
                data = await self.data_collector.fetch_ohlcv(
                    trade.symbol,
                    config.trading.TICK,
                    limit=20,
                    use_cache=False
                )

                current_price  = data.current_price
                atr_vals       = Indicators.atr(
                    data.highs, data.lows, data.closes
                )
                current_atr    = float(atr_vals[-1])
                capital_before = self.current_capital

                # Monitor trade
                updated_trade = await self.executor.monitor_and_manage_trade(
                    trade=trade,
                    current_price=current_price,
                    current_atr=current_atr
                )

                # Log unrealized PnL for open trades
                if updated_trade.status != TradeStatus.CLOSED:
                    unrealized = updated_trade.update_unrealized_pnl(
                        current_price
                    )
                    logger.debug(
                        f"📊 {trade_id}: {trade.symbol} | "
                        f"Price: {current_price:.2f} | "
                        f"Unrealized: ${unrealized:+.2f}"
                    )

                # ─── TRADE CLOSED ─────────────────────────
                if updated_trade.status == TradeStatus.CLOSED:

                    # Update capital
                    self.current_capital += updated_trade.realized_pnl
                    capital_after         = self.current_capital

                    # Update risk manager
                    self.risk_manager.update_pnl(
                        updated_trade, self.current_capital
                    )

                    # Record in performance engine
                    self.performance.add_trade(updated_trade)

                    # Save to database
                    if self.database:
                        try:
                            self.database.update_trade_on_close(
                                trade=updated_trade,
                                capital_after=capital_after
                            )
                            self.database.save_capital(
                                capital=self.current_capital,
                                daily_pnl=self.risk_manager.daily_pnl,
                                total_pnl=self.risk_manager.total_pnl,
                                drawdown=self.risk_manager.current_drawdown,
                                phase=config.get_phase(self.current_capital),
                                note=f"Trade closed: {trade_id}"
                            )
                            self.database.save_state(
                                'current_capital', self.current_capital
                            )
                        except Exception as db_err:
                            logger.error(f"DB update error: {db_err}")

                    # Update signal weights
                    metrics = self.performance.calculate_metrics(
                        period_days=7
                    )
                    if metrics.signal_ic:
                        self.signal_engine.update_signal_weights(
                            metrics.signal_ic
                        )

                    # ML retrain check
                    self._trade_count_since_retrain += 1
                    if self._trade_count_since_retrain >= 20:
                        await self._retrain_ml_optimizer()
                        self._trade_count_since_retrain = 0

                    # Log result
                    logger.info(
                        f"\n{'─'*50}\n"
                        f"💰 TRADE CLOSED: {trade_id}\n"
                        f"   Symbol: {updated_trade.symbol}\n"
                        f"   Side:   {updated_trade.side.value}\n"
                        f"   PnL:    ${updated_trade.realized_pnl:+.2f}\n"
                        f"   R:      {updated_trade.r_multiple:+.2f}R\n"
                        f"   Capital:${capital_before:.2f} → "
                        f"${self.current_capital:.2f}\n"
                        f"   Phase:  {config.get_phase(self.current_capital)}\n"
                        f"{'─'*50}"
                    )

            except Exception as e:
                logger.error(
                    f"Error monitoring {trade_id}: {e}",
                    exc_info=True
                )

    # ═══════════════════════════════════════════════════════
    # ML RETRAIN
    # ═══════════════════════════════════════════════════════

    async def _retrain_ml_optimizer(self) -> None:
        """Retrain ML signal optimizer every 20 trades."""
        try:
            import numpy as np

            closed_trades = [
                t for t in self.performance.all_trades
                if t.status.value == 'CLOSED' and t.signals_at_entry
            ]

            if len(closed_trades) < 20:
                logger.debug("Not enough trades for ML retrain")
                return

            signal_names = [
                'momentum', 'zscore_reversion', 'order_flow',
                'volume_confirmation', 'trend_alignment',
                'order_book_imbalance', 'cointegration'
            ]

            X_rows, y_rows, regimes = [], [], []

            for trade in closed_trades[-100:]:
                row = []
                for name in signal_names:
                    sig_data = trade.signals_at_entry.get(name, {})
                    score    = (
                        sig_data.get('score', 0.0)
                        if isinstance(sig_data, dict) else 0.0
                    )
                    row.append(score)
                X_rows.append(row)
                y_rows.append(trade.r_multiple)
                regimes.append(
                    trade.regime_at_entry.name
                    if trade.regime_at_entry else 'UNKNOWN'
                )

            X = np.array(X_rows)
            y = np.array(y_rows)

            self.signal_optimizer.fit(X, y, regimes)
            new_weights = self.signal_optimizer.get_optimal_weights('BULL_TREND')
            self.signal_engine.update_signal_weights(new_weights)

            logger.info(
                f"🤖 ML retrained | "
                f"Trades: {len(closed_trades)} | "
                f"Weights updated"
            )

        except Exception as e:
            logger.error(f"ML retrain error: {e}")

    # ═══════════════════════════════════════════════════════
    # DAILY SAVE
    # ═══════════════════════════════════════════════════════

    async def _check_daily_save(self) -> None:
        """Save daily performance at end of each day."""
        today = datetime.now().strftime('%Y-%m-%d')

        if today != self._last_daily_save:
            try:
                self.database.save_daily_performance(
                    date=self._last_daily_save,
                    starting_capital=self._daily_start_capital,
                    ending_capital=self.current_capital,
                    trades=self.performance.all_trades
                )
                daily_pnl = self.current_capital - self._daily_start_capital
                logger.info(
                    f"📅 Daily saved: {self._last_daily_save} | "
                    f"${self._daily_start_capital:.2f} → "
                    f"${self.current_capital:.2f} | "
                    f"PnL: ${daily_pnl:+.2f}"
                )
            except Exception as e:
                logger.error(f"Daily save error: {e}")

            self._daily_start_capital = self.current_capital
            self._last_daily_save     = today

    # ═══════════════════════════════════════════════════════
    # REVIEWS
    # ═══════════════════════════════════════════════════════

    async def _hourly_review(self) -> None:
        """Hourly performance check."""
        try:
            risk   = self.risk_manager.get_risk_report(self.current_capital)
            uptime = datetime.now() - self.start_time
            db_sum = self.database.get_performance_summary()

            logger.info(
                f"\n{'─'*50}\n"
                f"📊 HOURLY REVIEW\n"
                f"   Capital:      ${self.current_capital:,.2f}\n"
                f"   Daily P&L:    ${risk['daily_pnl']:+,.2f} "
                f"({risk['daily_pnl_pct']:+.2%})\n"
                f"   Drawdown:     {risk['current_drawdown']:.2%}\n"
                f"   Win Rate:     {risk['edge_statistics']['win_rate']:.1%}\n"
                f"   Risk of Ruin: {risk['risk_of_ruin']:.6%}\n"
                f"   Total Trades: {db_sum.get('total_trades', 0)}\n"
                f"   Total PnL:    ${db_sum.get('total_pnl', 0):+.2f}\n"
                f"   Open:         {len(self.executor.open_trades)}\n"
                f"   Uptime:       {str(uptime).split('.')[0]}\n"
                f"{'─'*50}"
            )
        except Exception as e:
            logger.error(f"Hourly review error: {e}")

    async def _weekly_review(self) -> None:
        """Weekly performance review."""
        try:
            recent = [
                t for t in self.performance.all_trades
                if (datetime.now() - t.entry_time).days <= 7
            ]
            report = self.performance.generate_weekly_report(
                trades=recent,
                current_capital=self.current_capital
            )
            logger.info(report)

            mc = self.performance.run_monte_carlo(
                starting_capital=self.current_capital
            )
            logger.info(
                f"Monte Carlo: {mc['verdict']}\n"
                f"  Median: ${mc['outcomes']['median']:,.2f}\n"
                f"  Ruin:   {mc['risk_metrics']['probability_of_ruin']:.3%}"
            )

            self.database.print_summary()

        except Exception as e:
            logger.error(f"Weekly review error: {e}")

    # ═══════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════

    async def _validate_system(self) -> bool:
        """Validate system before trading starts."""
        logger.info("🔍 Validating system...")

        # API connection
        try:
            await self.client.get_server_time()
            logger.info("✅ API connection verified")
        except Exception as e:
            logger.error(f"❌ API failed: {e}")
            return False

        # Data collection
        try:
            test_data = await self.data_collector.fetch_ohlcv(
                'BTCUSDT', '5m', limit=100
            )
            is_valid, msg = self.data_collector.validate_market_data(test_data)
            if not is_valid:
                logger.error(f"❌ Data invalid: {msg}")
                return False
            logger.info("✅ Data collection verified")
        except Exception as e:
            logger.error(f"❌ Data failed: {e}")
            return False

        # Fear & Greed (non-critical)
        try:
            fg = await self.extra_signals.get_fear_greed_signal()
            logger.info(
                f"✅ Fear & Greed: {fg.value:.0f} "
                f"(score: {fg.score:+.2f})"
            )
        except Exception as e:
            logger.warning(f"⚠️ F&G unavailable: {e}")

        # Monte Carlo
        try:
            mc = self.performance.run_monte_carlo(
                starting_capital=self.current_capital
            )
            logger.info(f"Monte Carlo: {mc['verdict']}")
        except Exception as e:
            logger.warning(f"Monte Carlo error: {e}")

        logger.info("✅ System validation complete")
        return True

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════

    async def _wait_for_next_cycle(self, seconds: int) -> None:
        """Wait for next cycle."""
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass

    def _handle_shutdown(self, signum, frame) -> None:
        """Graceful shutdown handler."""
        logger.info("⚡ Shutdown signal received...")
        self.is_running = False

    # ═══════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════

    async def stop(self) -> None:
        """Graceful shutdown — saves state, closes connections."""
        logger.info("🔴 Shutting down...")
        self.is_running = False

        # Close live positions if needed
        if not config.trading.PAPER_TRADING and self.executor:
            logger.info("Closing all open positions...")
            for trade in list(self.executor.open_trades.values()):
                try:
                    data = await self.data_collector.fetch_ohlcv(
                        trade.symbol, config.trading.TICK, limit=5
                    )
                    await self.executor.execute_full_exit(
                        trade, data.current_price, "SHUTDOWN"
                    )
                except Exception as e:
                    logger.error(f"Close trade error: {e}")

        # Save final state
        if self.database:
            try:
                self.database.save_state(
                    'current_capital', self.current_capital
                )
                self.database.save_capital(
                    capital=self.current_capital,
                    daily_pnl=self.risk_manager.daily_pnl
                    if self.risk_manager else 0,
                    total_pnl=self.risk_manager.total_pnl
                    if self.risk_manager else 0,
                    drawdown=self.risk_manager.current_drawdown
                    if self.risk_manager else 0,
                    phase=config.get_phase(self.current_capital),
                    note="System shutdown"
                )
                logger.info("✅ Final state saved")
            except Exception as e:
                logger.error(f"Save error: {e}")

        # Final report
        if self.performance:
            try:
                report = self.performance.generate_weekly_report(
                    trades=self.performance.all_trades,
                    current_capital=self.current_capital
                )
                logger.info(report)
                if self.database:
                    self.database.print_summary()
            except Exception as e:
                logger.error(f"Report error: {e}")

        # Close data collector
        if self.data_collector:
            try:
                await self.data_collector.close()
            except Exception as e:
                logger.error(f"Data collector close error: {e}")

        # Close Binance client
        if self.client:
            try:
                await self.client.close_connection()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Client close error: {e}")

        logger.info(
            f"✅ Shutdown complete. "
            f"Final capital: ${self.current_capital:.2f}"
        )



