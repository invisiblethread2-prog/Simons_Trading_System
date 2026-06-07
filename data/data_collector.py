# data/data_collector.py
# ============================================================
# THE EYES OF THE SYSTEM — FIXED VERSION
# Added: Retry logic on timeout (3 attempts)
# Added: Stale cache fallback
# Added: Better error handling
# ============================================================

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import aiohttp
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from config.settings import config
from models.data_models import (
    OHLCV, MarketData, OrderBookData
)

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Handles all market data collection from Binance.

    FIXED:
    - fetch_ohlcv: 3 retries on timeout
    - Stale cache fallback when all retries fail
    - Better timeout handling for testnet
    """

    def __init__(self):
        self.client:         Optional[AsyncClient]       = None
        self.socket_manager: Optional[BinanceSocketManager] = None

        # Data caches
        self._ohlcv_cache:     Dict[str, Dict]  = {}
        self._orderbook_cache: Dict[str, OrderBookData] = {}
        self._trade_cache:     Dict[str, List]  = {}

        # WebSocket
        self._ws_connections: Dict = {}
        self._callbacks:      Dict[str, List] = {}

        # Rate limiting
        self._last_request_time:    float = 0
        self._min_request_interval: float = 0.1

    async def initialize(self) -> None:
        """Initialize Binance async client"""
        try:
            self.client = await AsyncClient.create(
                api_key=config.exchange.BINANCE_API_KEY,
                api_secret=config.exchange.BINANCE_SECRET_KEY,
                testnet=config.exchange.BINANCE_TESTNET
            )
            self.socket_manager = BinanceSocketManager(self.client)
            logger.info("✅ Data collector initialized successfully")

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize data collector: {e}"
            )
            raise

    async def close(self) -> None:
        """Clean shutdown"""
        if self.client:
            await self.client.close_connection()
        logger.info("Data collector closed")

    # ─── RATE LIMITING ────────────────────────────────────

    async def _rate_limit(self) -> None:
        """Ensure we don't hit API rate limits"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(
                self._min_request_interval - elapsed
            )
        self._last_request_time = time.time()

    # ─── HISTORICAL DATA WITH RETRY ───────────────────────

    async def fetch_ohlcv(
        self,
        symbol:     str,
        timeframe:  str,
        limit:      int = 500,
        start_time: Optional[datetime] = None,
        use_cache:  bool = True
    ) -> MarketData:
        """
        Fetch OHLCV candlestick data.

        FIXED: Now retries 3 times on timeout.
        Falls back to stale cache if all retries fail.
        Testnet can be slow — this prevents crashes.
        """
        cache_key = f"{symbol}_{timeframe}"

        # Check fresh cache first
        if use_cache and cache_key in self._ohlcv_cache:
            cached = self._ohlcv_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < 60:
                return cached['data']

        # ─── RETRY LOOP ───────────────────────────────────
        max_retries = 3
        last_error  = None

        for attempt in range(max_retries):
            try:
                await self._rate_limit()

                klines = await self.client.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit,
                    startTime=int(
                        start_time.timestamp() * 1000
                    ) if start_time else None
                )

                if not klines:
                    raise ValueError(
                        f"No data received for {symbol} {timeframe}"
                    )

                # Parse into DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low',
                    'close', 'volume', 'close_time',
                    'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], unit='ms'
                )
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                market_data = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    opens=df['open'].values,
                    highs=df['high'].values,
                    lows=df['low'].values,
                    closes=df['close'].values,
                    volumes=df['volume'].values
                )

                # Update cache
                self._ohlcv_cache[cache_key] = {
                    'data':      market_data,
                    'timestamp': datetime.now()
                }

                logger.debug(
                    f"📊 Fetched {len(klines)} candles "
                    f"{symbol} {timeframe}"
                )
                return market_data

            # ─── TIMEOUT: RETRY ───────────────────────────
            except (TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                wait_time  = (attempt + 1) * 3  # 3s, 6s, 9s
                logger.warning(
                    f"⏱️ Timeout {symbol} {timeframe} "
                    f"(attempt {attempt+1}/{max_retries}) "
                    f"— retry in {wait_time}s"
                )
                await asyncio.sleep(wait_time)

            # ─── API ERROR: RETRY ONCE ────────────────────
            except BinanceAPIException as e:
                last_error = e
                logger.error(
                    f"Binance API error {symbol}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

            # ─── OTHER ERROR ──────────────────────────────
            except Exception as e:
                last_error = e
                logger.error(
                    f"Error fetching {symbol} {timeframe}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        # ─── ALL RETRIES FAILED ───────────────────────────
        # Use stale cache as last resort
        if cache_key in self._ohlcv_cache:
            logger.warning(
                f"⚠️ All retries failed for {symbol} {timeframe}. "
                f"Using stale cache."
            )
            return self._ohlcv_cache[cache_key]['data']

        # Nothing available — raise error
        raise last_error or Exception(
            f"Failed to fetch {symbol} after {max_retries} attempts"
        )

    # ─── MULTI TIMEFRAME ──────────────────────────────────

    async def fetch_multi_timeframe(
        self,
        symbol:     str,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, MarketData]:
        """
        Fetch data across multiple timeframes simultaneously.
        """
        if timeframes is None:
            timeframes = [
                config.trading.TICK,   # 1m
                config.trading.ETF,    # 5m
                config.trading.MTF,    # 15m
                config.trading.HTF,    # 1h
                "4h"
            ]

        tasks = [
            self.fetch_ohlcv(symbol, tf, limit=300)
            for tf in timeframes
        ]

        results = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        multi_tf_data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to fetch {symbol} {tf}: {result}"
                )
            else:
                multi_tf_data[tf] = result

        return multi_tf_data

    # ─── ORDER BOOK ───────────────────────────────────────

    async def fetch_order_book(
        self,
        symbol: str,
        depth:  int = 20
    ) -> OrderBookData:
        """Fetch current order book state"""
        await self._rate_limit()

        try:
            book = await self.client.get_order_book(
                symbol=symbol,
                limit=depth
            )

            return OrderBookData(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=[
                    (float(p), float(v))
                    for p, v in book['bids']
                ],
                asks=[
                    (float(p), float(v))
                    for p, v in book['asks']
                ]
            )

        except Exception as e:
            logger.error(
                f"Error fetching order book {symbol}: {e}"
            )
            raise

    # ─── RECENT TRADES ────────────────────────────────────

    async def fetch_recent_trades(
        self,
        symbol: str,
        limit:  int = 500
    ) -> pd.DataFrame:
        """
        Fetch recent trade data for OFI calculation.
        """
        await self._rate_limit()

        try:
            trades = await self.client.get_recent_trades(
                symbol=symbol,
                limit=limit
            )

            df = pd.DataFrame(trades)
            df['time']  = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['qty']   = df['qty'].astype(float)

            df['is_buy']      = ~df['isBuyerMaker']
            df['buy_volume']  = df['qty'].where(df['is_buy'], 0)
            df['sell_volume'] = df['qty'].where(~df['is_buy'], 0)

            return df

        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            raise

    # ─── FUNDING RATE ─────────────────────────────────────

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch perpetual futures funding rate"""
        await self._rate_limit()

        try:
            funding = await self.client.futures_funding_rate(
                symbol=symbol,
                limit=1
            )
            if funding:
                return float(funding[0]['fundingRate'])
            return 0.0
        except Exception:
            return 0.0

    # ─── WEBSOCKET FEEDS ──────────────────────────────────

    async def start_realtime_feed(
        self,
        symbols:  List[str],
        callback
    ) -> None:
        """Start real-time data feeds via WebSocket"""
        for symbol in symbols:
            try:
                ts = self.socket_manager.kline_socket(
                    symbol=symbol,
                    interval="1m"
                )
                self._ws_connections[symbol] = ts
                logger.info(
                    f"📡 Real-time feed started for {symbol}"
                )
            except Exception as e:
                logger.error(
                    f"Error starting feed for {symbol}: {e}"
                )

    async def start_orderbook_stream(
        self,
        symbol:   str,
        callback
    ) -> None:
        """Real-time order book depth stream"""
        try:
            ds = self.socket_manager.depth_socket(
                symbol=symbol,
                depth=BinanceSocketManager.WEBSOCKET_DEPTH_20
            )
            self._ws_connections[f"{symbol}_depth"] = ds
        except Exception as e:
            logger.error(
                f"Error starting orderbook stream: {e}"
            )

    # ─── DATA VALIDATION ──────────────────────────────────

    def validate_market_data(
        self,
        data: MarketData
    ) -> Tuple[bool, str]:
        """
        Validate data quality before passing to signals.
        Garbage in = garbage out.
        """
        if len(data.closes) < 50:
            return False, "Insufficient data (< 50 bars)"

        for arr_name, arr in [
            ('closes',  data.closes),
            ('volumes', data.volumes),
            ('highs',   data.highs),
            ('lows',    data.lows)
        ]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False, f"NaN/Inf in {arr_name}"

        if np.any(data.highs < data.lows):
            return False, "High < Low detected"

        if np.any(data.closes <= 0) or np.any(data.opens <= 0):
            return False, "Non-positive prices detected"

        if np.all(data.volumes == 0):
            return False, "All volumes zero (stale data)"

        max_return = np.max(np.abs(data.returns))
        if max_return > 0.20:
            return False, f"Extreme price jump: {max_return:.1%}"

        return True, "Data validated successfully"

    # ─── COMPLETE DATA PACKAGE ────────────────────────────

    async def get_complete_data_package(
        self,
        symbol: str
    ) -> Dict:
        """
        Fetch everything needed for signal generation.
        One call = all data for one symbol.

        FIXED: Non-critical failures (order book, trades)
        no longer crash the whole package.
        """
        # Fetch all concurrently
        results = await asyncio.gather(
            self.fetch_multi_timeframe(symbol),
            self.fetch_order_book(symbol),
            self.fetch_recent_trades(symbol),
            self.fetch_funding_rate(symbol),
            return_exceptions=True
        )

        multi_tf, order_book, trades, funding_rate = results

        # Primary data MUST succeed
        if isinstance(multi_tf, Exception):
            raise ValueError(
                f"Failed to fetch market data: {multi_tf}"
            )

        # Validate primary timeframe
        primary_tf = config.trading.ETF
        if primary_tf in multi_tf:
            is_valid, msg = self.validate_market_data(
                multi_tf[primary_tf]
            )
            if not is_valid:
                raise ValueError(
                    f"Data validation failed: {msg}"
                )

        return {
            'multi_tf': multi_tf,
            'order_book': (
                order_book
                if not isinstance(order_book, Exception)
                else None
            ),
            'trades': (
                trades
                if not isinstance(trades, Exception)
                else None
            ),
            'funding_rate': (
                funding_rate
                if not isinstance(funding_rate, Exception)
                else 0.0
            ),
            'timestamp': datetime.now()
        }
