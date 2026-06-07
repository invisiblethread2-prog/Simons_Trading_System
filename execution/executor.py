# execution/executor.py
# ============================================================
# THE HANDS OF THE SYSTEM
# Takes decisions from brain → converts to market orders
# Manages the full trade lifecycle
# ============================================================

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from config.settings import config
from models.data_models import (
    CompositeSignal, PositionSize, Trade,
    Side, TradeStatus, OrderType
)

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Handles all order execution.

    Responsibilities:
    - Place entry orders (limit preferred)
    - Set stop losses
    - Manage scaled exits (3 targets + runner)
    - Trail runner position
    - Handle partial fills
    - Track all positions
    """

    def __init__(self, client: AsyncClient, paper_trading: bool = True):
        self.client = client
        self.paper_trading = paper_trading
        self.open_trades: Dict[str, Trade] = {}
        self._paper_balance = config.trading.STARTING_CAPITAL
        self._paper_trade_counter = 0

        # Track all placed orders
        self._active_orders: Dict[str, Dict] = {}

        # Execution statistics
        self._total_slippage: float = 0.0
        self._total_commission: float = 0.0
        self._execution_count: int = 0

    # ─── ENTRY EXECUTION ──────────────────────────────────

    async def execute_entry(
        self,
        signal: CompositeSignal,
        position_size: PositionSize,
        current_price: float
    ) -> Optional[Trade]:
        """
        Execute trade entry.

        Order priority:
        1. Limit order (saves 50% of spread cost)
        2. Market order fallback (if momentum too strong)

        Returns Trade object if successful, None if failed.
        """
        trade_id = f"T{str(uuid.uuid4())[:8].upper()}"

        logger.info(
            f"\n{'='*50}\n"
            f"🎯 EXECUTING ENTRY\n"
            f"   ID:        {trade_id}\n"
            f"   Symbol:    {signal.symbol}\n"
            f"   Direction: {signal.direction.value}\n"
            f"   Size:      ${position_size.position_value:.2f}\n"
            f"   Price:     {current_price:.4f}\n"
            f"   Stop:      {position_size.stop_loss_price:.4f}\n"
            f"   TP1/2/3:   {position_size.take_profit_1:.4f} / "
            f"{position_size.take_profit_2:.4f} / "
            f"{position_size.take_profit_3:.4f}\n"
            f"   Kelly:     {position_size.adjusted_kelly:.3f}\n"
            f"   Risk:      ${position_size.risk_amount:.2f}\n"
            f"{'='*50}"
        )

        if self.paper_trading:
            trade = await self._paper_execute_entry(
                trade_id, signal, position_size, current_price
            )
        else:
            trade = await self._live_execute_entry(
                trade_id, signal, position_size, current_price
            )

        if trade:
            self.open_trades[trade_id] = trade
            self._execution_count += 1
            logger.info(f"✅ Entry executed: {trade_id}")

        return trade

    async def _paper_execute_entry(
        self,
        trade_id: str,
        signal: CompositeSignal,
        position_size: PositionSize,
        current_price: float
    ) -> Optional[Trade]:
        """
        Paper trading entry — simulates real execution
        including slippage and commission.
        """
        # Simulate realistic slippage
        # Limit order: ~0.05% slippage
        # Market order: ~0.1% slippage
        slippage_pct = 0.0005  # Using limit order assumption

        if signal.direction == Side.LONG:
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)

        # Binance maker fee (limit order)
        commission_pct = 0.00075  # 0.075%
        commission = position_size.position_value * commission_pct

        # Check sufficient paper balance
        total_cost = position_size.position_value + commission
        if self._paper_balance < total_cost:
            logger.warning(
                f"❌ Insufficient paper balance: "
                f"${self._paper_balance:.2f} < ${total_cost:.2f}"
            )
            return None

        # Deduct from paper balance
        self._paper_balance -= total_cost
        self._total_slippage += abs(fill_price - current_price) * position_size.position_size_units
        self._total_commission += commission

        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            side=signal.direction,
            status=TradeStatus.ACTIVE,
            entry_time=datetime.now(),
            entry_price=fill_price,
            entry_size=position_size.position_size_units,
            entry_value=position_size.position_value,
            stop_loss=position_size.stop_loss_price,
            take_profit_1=position_size.take_profit_1,
            take_profit_2=position_size.take_profit_2,
            take_profit_3=position_size.take_profit_3,
            risk_amount=position_size.risk_amount,
            remaining_size=position_size.position_size_units,
            regime_at_entry=signal.regime,
            composite_score_at_entry=signal.composite_score,
            signals_at_entry={
                k: {
                    'score': v.score,
                    'value': v.value
                }
                for k, v in signal.individual_signals.items()
            },
            commission_paid=commission,
            slippage=abs(fill_price - current_price) * position_size.position_size_units,
            notes=f"Paper trade | Score: {signal.composite_score:+.3f}"
        )

        self._paper_trade_counter += 1
        # Log trade immediately
        logger.info(
            f"📝 Paper trade created: {trade_id} | "
            f"{signal.symbol} {signal.direction.value} | "
            f"Entry: {fill_price:.4f} | "
            f"Size: ${position_size.position_value:.2f}"
        )
        return trade

    async def _live_execute_entry(
        self,
        trade_id: str,
        signal: CompositeSignal,
        position_size: PositionSize,
        current_price: float
    ) -> Optional[Trade]:
        """
        Live trading entry via Binance API.
        Uses limit orders with market order fallback.
        """
        symbol = signal.symbol
        side = "BUY" if signal.direction == Side.LONG else "SELL"
        quantity = self._round_quantity(
            position_size.position_size_units, symbol
        )

        # Calculate limit price (slightly inside spread)
        limit_offset = 0.0002  # 0.02% inside market

        if signal.direction == Side.LONG:
            limit_price = current_price * (1 + limit_offset)
        else:
            limit_price = current_price * (1 - limit_offset)

        limit_price = self._round_price(limit_price, symbol)

        try:
            # Place limit order
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=limit_price
            )

            logger.info(f"📋 Limit order placed: {order['orderId']}")

            # Wait for fill (up to 30 seconds)
            filled_order = await self._wait_for_fill(
                symbol, order['orderId'], timeout=30
            )

            if filled_order and filled_order['status'] == 'FILLED':
                fill_price = float(filled_order['price'])
                fill_qty = float(filled_order['executedQty'])
                commission = float(
                    filled_order['fills'][0]['commission']
                    if filled_order.get('fills') else 0
                )

                trade = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=signal.direction,
                    status=TradeStatus.ACTIVE,
                    entry_time=datetime.now(),
                    entry_price=fill_price,
                    entry_size=fill_qty,
                    entry_value=fill_price * fill_qty,
                    stop_loss=position_size.stop_loss_price,
                    take_profit_1=position_size.take_profit_1,
                    take_profit_2=position_size.take_profit_2,
                    take_profit_3=position_size.take_profit_3,
                    risk_amount=position_size.risk_amount,
                    remaining_size=fill_qty,
                    regime_at_entry=signal.regime,
                    composite_score_at_entry=signal.composite_score,
                    signals_at_entry={
                        k: {'score': v.score, 'value': v.value}
                        for k, v in signal.individual_signals.items()
                    },
                    commission_paid=commission,
                    slippage=abs(fill_price - current_price) * fill_qty
                )

                # Immediately place stop loss order
                await self._place_stop_loss(trade)

                return trade

            else:
                # Order not filled → Cancel and try market
                logger.warning(
                    f"Limit order not filled within timeout. "
                    f"Attempting market order."
                )
                await self.client.cancel_order(
                    symbol=symbol,
                    orderId=order['orderId']
                )

                # Market order fallback
                return await self._market_order_entry(
                    trade_id, signal, position_size, current_price
                )

        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error on entry: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error on entry: {e}")
            return None

    async def _market_order_entry(
        self,
        trade_id: str,
        signal: CompositeSignal,
        position_size: PositionSize,
        current_price: float
    ) -> Optional[Trade]:
        """Market order fallback for urgent entries"""
        symbol = signal.symbol
        side = "BUY" if signal.direction == Side.LONG else "SELL"
        quantity = self._round_quantity(
            position_size.position_size_units, symbol
        )

        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            if order['status'] == 'FILLED':
                fill_price = float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                fill_qty = float(order['executedQty'])

                trade = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=signal.direction,
                    status=TradeStatus.ACTIVE,
                    entry_time=datetime.now(),
                    entry_price=fill_price,
                    entry_size=fill_qty,
                    entry_value=fill_price * fill_qty,
                    stop_loss=position_size.stop_loss_price,
                    take_profit_1=position_size.take_profit_1,
                    take_profit_2=position_size.take_profit_2,
                    take_profit_3=position_size.take_profit_3,
                    risk_amount=position_size.risk_amount,
                    remaining_size=fill_qty,
                    regime_at_entry=signal.regime,
                    composite_score_at_entry=signal.composite_score,
                    signals_at_entry={},
                    notes="Market order fallback"
                )

                await self._place_stop_loss(trade)
                return trade

        except BinanceAPIException as e:
            logger.error(f"❌ Market order failed: {e}")
            return None

    # ─── STOP LOSS MANAGEMENT ─────────────────────────────

    async def _place_stop_loss(self, trade: Trade) -> None:
        """
        Place stop loss order immediately after entry.
        This is NON-NEGOTIABLE.
        Stop is ALWAYS placed before moving on.
        """
        if self.paper_trading:
            return  # Paper trading handles stops in monitoring

        symbol = trade.symbol

        # Opposite side for stop
        stop_side = "SELL" if trade.side == Side.LONG else "BUY"

        quantity = self._round_quantity(trade.remaining_size, symbol)
        stop_price = self._round_price(trade.stop_loss, symbol)

        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=stop_side,
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=stop_price,
                price=self._round_price(
                    stop_price * 0.999 if stop_side == "SELL"
                    else stop_price * 1.001,
                    symbol
                )
            )

            self._active_orders[f"{trade.trade_id}_SL"] = {
                'order_id': order['orderId'],
                'type': 'STOP_LOSS',
                'trade_id': trade.trade_id
            }

            logger.info(
                f"🛡️ Stop loss placed at {stop_price:.4f} "
                f"for trade {trade.trade_id}"
            )

        except BinanceAPIException as e:
            logger.error(f"❌ CRITICAL: Failed to place stop loss: {e}")
            # If we can't place stop loss → close the trade immediately
            await self.execute_full_exit(trade, trade.entry_price, "STOP_FAILED")

    # ─── EXIT MANAGEMENT ──────────────────────────────────

    async def monitor_and_manage_trade(
        self,
        trade: Trade,
        current_price: float,
        current_atr: float
    ) -> Optional[Trade]:
        """
        Monitor open trade and execute exits based on:
        1. Stop loss hit
        2. Take profit levels hit
        3. Runner trailing stop

        Called on every price update for open trades.
        Returns updated trade (closed if exited).
        """
        if trade.status != TradeStatus.ACTIVE:
            return trade

        trade.update_unrealized_pnl(current_price)

        # ─── CHECK STOP LOSS ──────────────────────────────
        if self._is_stop_hit(trade, current_price):
            logger.info(
                f"🛑 STOP LOSS HIT: {trade.trade_id} | "
                f"Price: {current_price:.4f} | "
                f"Stop: {trade.stop_loss:.4f}"
            )
            return await self.execute_full_exit(
                trade, trade.stop_loss, "STOP_LOSS"
            )

        # ─── CHECK TAKE PROFIT LEVELS ─────────────────────

        # TP1: 40% at 1R
        if (not self._tp1_hit(trade)
                and self._is_tp1_hit(trade, current_price)):

            exit_size = trade.entry_size * config.trading.TP1_CLOSE_FRACTION
            trade = await self._execute_partial_exit(
                trade, current_price, exit_size, "TP1"
            )

            # Move stop to breakeven after TP1
            trade.stop_loss = trade.entry_price
            logger.info(
                f"🎯 TP1 HIT: {trade.trade_id} | "
                f"Stop moved to breakeven: {trade.entry_price:.4f}"
            )

        # TP2: 30% at 2R
        elif (self._tp1_hit(trade)
              and not self._tp2_hit(trade)
              and self._is_tp2_hit(trade, current_price)):

            exit_size = trade.entry_size * config.trading.TP2_CLOSE_FRACTION
            trade = await self._execute_partial_exit(
                trade, current_price, exit_size, "TP2"
            )

            # Move stop to lock partial profit
            if trade.side == Side.LONG:
                trade.stop_loss = trade.entry_price + (
                    (current_price - trade.entry_price) * 0.3
                )
            else:
                trade.stop_loss = trade.entry_price - (
                    (trade.entry_price - current_price) * 0.3
                )

            logger.info(
                f"🎯 TP2 HIT: {trade.trade_id} | "
                f"Stop moved to: {trade.stop_loss:.4f}"
            )

        # TP3: 20% at 3R
        elif (self._tp2_hit(trade)
              and not self._tp3_hit(trade)
              and self._is_tp3_hit(trade, current_price)):

            exit_size = trade.entry_size * config.trading.TP3_CLOSE_FRACTION
            trade = await self._execute_partial_exit(
                trade, current_price, exit_size, "TP3"
            )
            logger.info(f"🎯 TP3 HIT: {trade.trade_id}")

        # ─── RUNNER MANAGEMENT ────────────────────────────
        # Trail stop on remaining 10% position

        elif (self._tp3_hit(trade)
              and trade.remaining_size > 0):

            trade = self._update_trailing_stop(
                trade, current_price, current_atr
            )

            # Check if trailing stop hit
            if self._is_stop_hit(trade, current_price):
                logger.info(
                    f"🏃 RUNNER STOPPED: {trade.trade_id} | "
                    f"Price: {current_price:.4f}"
                )
                return await self.execute_full_exit(
                    trade, current_price, "TRAIL_STOP"
                )

        return trade

    async def _execute_partial_exit(
        self,
        trade: Trade,
        exit_price: float,
        exit_size: float,
        reason: str
    ) -> Trade:
        """Execute a partial position exit"""

        if self.paper_trading:
            fill_price = exit_price
        else:
            # Place market order for the partial exit
            fill_price = await self._place_exit_order(
                trade, exit_size, exit_price
            )

        if fill_price is None:
            fill_price = exit_price  # Fallback

        # Calculate P&L for this partial exit
        if trade.side == Side.LONG:
            partial_pnl = (fill_price - trade.entry_price) * exit_size
        else:
            partial_pnl = (trade.entry_price - fill_price) * exit_size

        # Deduct commission
        commission = fill_price * exit_size * 0.00075
        partial_pnl -= commission

        # Update trade
        trade.realized_pnl += partial_pnl
        trade.remaining_size -= exit_size
        trade.commission_paid += commission
        trade.status = TradeStatus.PARTIAL

        trade.partial_exits.append({
            'time': datetime.now().isoformat(),
            'reason': reason,
            'price': fill_price,
            'size': exit_size,
            'pnl': partial_pnl
        })

        # Paper trading: add proceeds back to balance
        if self.paper_trading:
            self._paper_balance += (fill_price * exit_size) - commission

        r_multiple = partial_pnl / trade.risk_amount if trade.risk_amount > 0 else 0

        logger.info(
            f"💰 PARTIAL EXIT ({reason}): {trade.trade_id} | "
            f"Size: {exit_size:.6f} | "
            f"Price: {fill_price:.4f} | "
            f"PnL: ${partial_pnl:+.2f} | "
            f"R: {r_multiple:+.2f}"
        )

        return trade

    async def execute_full_exit(
        self,
        trade: Trade,
        exit_price: float,
        reason: str
    ) -> Trade:
        """Execute complete position exit"""

        if trade.remaining_size <= 0:
            trade.status = TradeStatus.CLOSED
            return trade

        if self.paper_trading:
            fill_price = exit_price
        else:
            fill_price = await self._place_exit_order(
                trade, trade.remaining_size, exit_price
            ) or exit_price

        # Calculate remaining P&L
        if trade.side == Side.LONG:
            remaining_pnl = (fill_price - trade.entry_price) * trade.remaining_size
        else:
            remaining_pnl = (trade.entry_price - fill_price) * trade.remaining_size

        commission = fill_price * trade.remaining_size * 0.00075
        remaining_pnl -= commission

        # Update trade to closed
        trade.realized_pnl += remaining_pnl
        trade.commission_paid += commission
        trade.exit_time = datetime.now()
        trade.exit_price = fill_price
        trade.exit_value = fill_price * trade.remaining_size
        trade.remaining_size = 0
        trade.status = TradeStatus.CLOSED
        trade.r_multiple = (
            trade.realized_pnl / trade.risk_amount
            if trade.risk_amount > 0
            else 0
        )

        # Paper trading: return proceeds
        if self.paper_trading:
            self._paper_balance += (fill_price * trade.entry_size *
                                    config.trading.RUNNER_FRACTION) - commission

        # Remove from open trades
        if trade.trade_id in self.open_trades:
            del self.open_trades[trade.trade_id]

        logger.info(
            f"\n{'='*50}\n"
            f"🏁 TRADE CLOSED: {trade.trade_id}\n"
            f"   Reason:    {reason}\n"
            f"   Entry:     {trade.entry_price:.4f}\n"
            f"   Exit:      {fill_price:.4f}\n"
            f"   Total PnL: ${trade.realized_pnl:+.2f}\n"
            f"   R Multiple:{trade.r_multiple:+.2f}R\n"
            f"   Duration:  {(trade.exit_time - trade.entry_time)}\n"
            f"   Commission:${trade.commission_paid:.2f}\n"
            f"{'='*50}"
        )

        return trade

    async def _place_exit_order(
        self,
        trade: Trade,
        size: float,
        target_price: float
    ) -> Optional[float]:
        """Place exit order on live exchange"""
        side = "SELL" if trade.side == Side.LONG else "BUY"
        quantity = self._round_quantity(size, trade.symbol)

        try:
            order = await self.client.create_order(
                symbol=trade.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            if order.get('status') == 'FILLED':
                return float(order['cummulativeQuoteQty']) / float(order['executedQty'])

        except BinanceAPIException as e:
            logger.error(f"❌ Exit order failed: {e}")

        return None

    # ─── TRAILING STOP ────────────────────────────────────

    def _update_trailing_stop(
        self,
        trade: Trade,
        current_price: float,
        current_atr: float
    ) -> Trade:
        """
        Update trailing stop for runner position.
        Trail distance = 1× ATR from current price.
        Only moves stop in FAVORABLE direction.
        Never moves stop backward.
        """
        trail_distance = config.trading.RUNNER_TRAIL_ATR * current_atr

        if trade.side == Side.LONG:
            new_stop = current_price - trail_distance
            # Only move stop UP (never down)
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
                logger.debug(
                    f"📈 Trail stop updated: {trade.trade_id} → "
                    f"{new_stop:.4f}"
                )
        else:
            new_stop = current_price + trail_distance
            # Only move stop DOWN (never up for short)
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
                logger.debug(
                    f"📉 Trail stop updated: {trade.trade_id} → "
                    f"{new_stop:.4f}"
                )

        return trade

    # ─── STOP/TARGET CHECK HELPERS ────────────────────────

    def _is_stop_hit(self, trade: Trade, current_price: float) -> bool:
        if trade.side == Side.LONG:
            return current_price <= trade.stop_loss
        else:
            return current_price >= trade.stop_loss

    def _is_tp1_hit(self, trade: Trade, current_price: float) -> bool:
        if trade.side == Side.LONG:
            return current_price >= trade.take_profit_1
        else:
            return current_price <= trade.take_profit_1

    def _is_tp2_hit(self, trade: Trade, current_price: float) -> bool:
        if trade.side == Side.LONG:
            return current_price >= trade.take_profit_2
        else:
            return current_price <= trade.take_profit_2

    def _is_tp3_hit(self, trade: Trade, current_price: float) -> bool:
        if trade.side == Side.LONG:
            return current_price >= trade.take_profit_3
        else:
            return current_price <= trade.take_profit_3

    def _tp1_hit(self, trade: Trade) -> bool:
        return any(p['reason'] == 'TP1' for p in trade.partial_exits)

    def _tp2_hit(self, trade: Trade) -> bool:
        return any(p['reason'] == 'TP2' for p in trade.partial_exits)

    def _tp3_hit(self, trade: Trade) -> bool:
        return any(p['reason'] == 'TP3' for p in trade.partial_exits)

    # ─── WAIT FOR FILL ────────────────────────────────────

    async def _wait_for_fill(
        self,
        symbol: str,
        order_id: int,
        timeout: int = 30
    ) -> Optional[Dict]:
        """Poll for order fill with timeout"""
        import asyncio

        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < timeout:
            try:
                order = await self.client.get_order(
                    symbol=symbol,
                    orderId=order_id
                )

                if order['status'] in ['FILLED', 'PARTIALLY_FILLED']:
                    return order
                elif order['status'] in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    return None

                await asyncio.sleep(1)

            except BinanceAPIException as e:
                logger.error(f"Error checking order: {e}")
                await asyncio.sleep(2)

        return None

    # ─── UTILITY HELPERS ──────────────────────────────────

    def _round_quantity(self, qty: float, symbol: str) -> float:
        """Round quantity to exchange requirements"""
        # Binance BTC minimum: 0.000001
        # This would need exchange info in production
        precision = {
            'BTCUSDT': 6,
            'ETHUSDT': 5,
            'SOLUSDT': 2,
            'BNBUSDT': 3,
        }
        decimals = precision.get(symbol, 4)
        return round(qty, decimals)

    def _round_price(self, price: float, symbol: str) -> float:
        """Round price to exchange tick size"""
        precision = {
            'BTCUSDT': 2,
            'ETHUSDT': 2,
            'SOLUSDT': 3,
            'BNBUSDT': 3,
        }
        decimals = precision.get(symbol, 2)
        return round(price, decimals)

    def get_execution_stats(self) -> Dict:
        """Return execution quality statistics"""
        return {
            'total_trades_executed': self._execution_count,
            'total_slippage_cost': self._total_slippage,
            'total_commission_cost': self._total_commission,
            'total_friction_cost': self._total_slippage + self._total_commission,
            'paper_balance': self._paper_balance,
            'open_positions': len(self.open_trades)
        }

