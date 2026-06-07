# config/settings.py
# ============================================================
# THE BRAIN OF THE SYSTEM - ALL PARAMETERS LIVE HERE
# Change behavior of entire system from this single file
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExchangeConfig:
    """Exchange and API Configuration"""
    # Binance Configuration
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_TESTNET: bool = True  # START ON TESTNET ALWAYS
    BINANCE_BASE_URL: str = "https://testnet.binance.vision"
    BINANCE_WS_URL: str = "wss://testnet.binance.vision/ws"

    # Alpaca Configuration (for stocks later)
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER: bool = True


@dataclass
class TradingConfig:
    """Core Trading Parameters"""

    # ─── CAPITAL MANAGEMENT ───────────────────────────────
    STARTING_CAPITAL: float = 100.0
    BASE_CURRENCY: str = "USDT"

    # ─── TRADING PAIRS ────────────────────────────────────
    # Tier 1: Primary pairs (highest liquidity)
    PRIMARY_PAIRS: List[str] = field(default_factory=lambda: [
        "BTCUSDT",
        "ETHUSDT",
    ])

    # Tier 2: Secondary pairs (added at $1000+)
    SECONDARY_PAIRS: List[str] = field(default_factory=lambda: [
        "SOLUSDT",
        "BNBUSDT",
    ])

    # ─── TIMEFRAMES ───────────────────────────────────────
    HTF: str = "1h"      # Higher timeframe: BIAS
    MTF: str = "15m"     # Middle timeframe: SETUP
    ETF: str = "5m"      # Entry timeframe: ENTRY
    TICK: str = "1m"     # Micro timeframe: TIMING

    # ─── SESSION HOURS (UTC) ──────────────────────────────
    BEST_SESSION_START: int = 8   # London open
    BEST_SESSION_END: int = 17    # NY close

    # ─── RISK PARAMETERS ──────────────────────────────────
    MAX_RISK_PER_TRADE: float = 0.02      # 2% per trade
    MAX_DAILY_LOSS: float = 0.04          # 4% daily stop
    MAX_WEEKLY_LOSS: float = 0.10         # 10% weekly stop
    MAX_DRAWDOWN_BEFORE_PAUSE: float = 0.15  # 15% pause
    MAX_DRAWDOWN_BEFORE_STOP: float = 0.20   # 20% full stop
    MAX_PORTFOLIO_HEAT: float = 0.08      # 8% total exposure
    MAX_CORRELATION: float = 0.70         # Max between trades
    MAX_CONCURRENT_TRADES: int = 3        # Max open positions

    # ─── KELLY CRITERION PARAMETERS ───────────────────────
    KELLY_FRACTION: float = 0.25          # Quarter Kelly (safe)
    MAX_POSITION_FRACTION: float = 0.05   # Hard cap 5%
    MIN_POSITION_USD: float = 3.0        # Min trade size

    # ─── SIGNAL THRESHOLDS ────────────────────────────────
    COMPOSITE_SCORE_THRESHOLD: float = 0.42  # Min for trade
    MIN_SIGNALS_ALIGNED: int = 3             # Min confirmations

    # ─── REGIME THRESHOLDS ────────────────────────────────
    ADX_TREND_THRESHOLD: float = 25.0
    ADX_RANGE_THRESHOLD: float = 20.0
    HURST_TREND_THRESHOLD: float = 0.55
    HURST_RANGE_THRESHOLD: float = 0.45
    ATR_CHAOS_MULTIPLIER: float = 2.5        # ATR × this = chaos

    # ─── EXIT PARAMETERS ──────────────────────────────────
    TAKE_PROFIT_1: float = 1.0    # 1R → Take 40%
    TAKE_PROFIT_2: float = 2.0    # 2R → Take 30%
    TAKE_PROFIT_3: float = 3.0    # 3R → Take 20%
    RUNNER_FRACTION: float = 0.10  # 10% let run

    TP1_CLOSE_FRACTION: float = 0.40
    TP2_CLOSE_FRACTION: float = 0.30
    TP3_CLOSE_FRACTION: float = 0.20

    # Trailing stop for runner (in ATR multiples)
    RUNNER_TRAIL_ATR: float = 1.0

    # ─── PERFORMANCE THRESHOLDS ───────────────────────────
    MIN_WIN_RATE: float = 0.55          # Below this: review
    MIN_PROFIT_FACTOR: float = 1.5      # Below this: review
    MIN_SHARPE_RATIO: float = 1.5       # Below this: review
    MIN_IC_THRESHOLD: float = 0.05      # Below this: signal decaying

    # ─── COMPOUNDING PHASES ───────────────────────────────
    PHASE_1_TARGET: float = 1_000.0     # 100% reinvestment
    PHASE_2_TARGET: float = 10_000.0    # 98% reinvestment
    PHASE_3_TARGET: float = 100_000.0   # 90% reinvestment
    PHASE_4_TARGET: float = 1_000_000.0 # 80% reinvestment

    # ─── LOOKBACK PERIODS ─────────────────────────────────
    HURST_LOOKBACK: int = 100
    ATR_PERIOD: int = 14
    ADX_PERIOD: int = 14
    EMA_FAST: int = 8
    EMA_SLOW: int = 21
    EMA_TREND: int = 200
    RSI_PERIOD: int = 14
    VOLUME_MA_PERIOD: int = 20
    ZSCORE_LOOKBACK: int = 100
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    OFI_LOOKBACK: int = 20
    REGIME_LOOKBACK: int = 50

    # ─── STATISTICAL VALIDATION ───────────────────────────
    MIN_SAMPLE_SIZE: int = 200
    MAX_PVALUE: float = 0.05
    MIN_EFFECT_SIZE: float = 0.3
    MIN_OOS_RATIO: float = 0.70  # OOS must be 70% of IS perf

    # ─── ML PARAMETERS ────────────────────────────────────
    ML_RETRAIN_FREQUENCY: int = 50   # Retrain every N trades
    ML_SEQUENCE_LENGTH: int = 10     # LSTM input length
    ML_FEATURES: int = 10            # Input features

    # ─── SYSTEM PARAMETERS ────────────────────────────────
    LOG_LEVEL: str = "INFO"
    PAPER_TRADING: bool = True  # ALWAYS START PAPER
    DB_PATH: str = "data/trading.db"
    LOG_PATH: str = "logs/"
    MODEL_PATH: str = "models/"


@dataclass
class MarketRegime:
    """Market Regime State Definitions"""
    BULL_TREND: int = 1
    BEAR_TREND: int = 2
    RANGING: int = 3
    TRANSITION: int = 4
    CHAOS: int = 5

    REGIME_NAMES: Dict[int, str] = field(default_factory=lambda: {
        1: "BULL_TREND",
        2: "BEAR_TREND",
        3: "RANGING",
        4: "TRANSITION",
        5: "CHAOS"
    })

    # Position size multipliers per regime
    REGIME_MULTIPLIERS: Dict[int, float] = field(default_factory=lambda: {
        1: 1.0,   # Bull trend: normal size
        2: 1.0,   # Bear trend: normal size
        3: 1.2,   # Range: slightly larger (higher win rate)
        4: 0.5,   # Transition: reduce size
        5: 0.0    # Chaos: no trading
    })


@dataclass
class DrawdownConfig:
    """Drawdown Response Configuration"""
    LEVEL_1: float = 0.05   # 5%  → reduce 25%
    LEVEL_2: float = 0.10   # 10% → reduce 50%
    LEVEL_3: float = 0.15   # 15% → stop and review
    LEVEL_4: float = 0.20   # 20% → full system audit

    SIZE_MULTIPLIER_1: float = 0.75
    SIZE_MULTIPLIER_2: float = 0.50
    SIZE_MULTIPLIER_3: float = 0.00
    SIZE_MULTIPLIER_4: float = 0.00


# ─── MASTER CONFIG INSTANCE ───────────────────────────────
class Config:
    exchange = ExchangeConfig()
    trading = TradingConfig()
    regime = MarketRegime()
    drawdown = DrawdownConfig()

    @classmethod
    def is_paper_trading(cls) -> bool:
        return cls.trading.PAPER_TRADING

    @classmethod
    def get_phase(cls, capital: float) -> int:
        """Returns current compounding phase"""
        if capital < cls.trading.PHASE_1_TARGET:
            return 1
        elif capital < cls.trading.PHASE_2_TARGET:
            return 2
        elif capital < cls.trading.PHASE_3_TARGET:
            return 3
        elif capital < cls.trading.PHASE_4_TARGET:
            return 4
        return 5

    @classmethod
    def get_reinvestment_rate(cls, capital: float) -> float:
        """Returns reinvestment rate based on phase"""
        phase = cls.get_phase(capital)
        rates = {1: 1.00, 2: 0.98, 3: 0.90, 4: 0.80, 5: 0.70}
        return rates[phase]


config = Config()

