# models/data_models.py
# ============================================================
# PURE DATA STRUCTURES — THE LANGUAGE OF THE SYSTEM
# ============================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class TradeStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class RegimeState(Enum):
    BULL_TREND = 1
    BEAR_TREND = 2
    RANGING = 3
    TRANSITION = 4
    CHAOS = 5


class SignalStrength(Enum):
    STRONG_LONG = 2
    WEAK_LONG = 1
    NEUTRAL = 0
    WEAK_SHORT = -1
    STRONG_SHORT = -2


@dataclass
class OHLCV:
    """Single candlestick data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class MarketData:
    """Complete market data package for analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime

    # Price arrays
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray

    # Derived
    returns: np.ndarray = field(init=False)
    log_returns: np.ndarray = field(init=False)

    def __post_init__(self):
        self.returns = np.diff(self.closes) / self.closes[:-1]
        self.log_returns = np.log(self.closes[1:] / self.closes[:-1])

    @property
    def current_price(self) -> float:
        return float(self.closes[-1])

    @property
    def current_volume(self) -> float:
        return float(self.volumes[-1])

    @property
    def avg_volume(self) -> float:
        return float(np.mean(self.volumes[-20:]))


@dataclass
class OrderBookData:
    """Level 2 order book data"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price if self.mid_price > 0 else 0

    @property
    def bid_volume_5_levels(self) -> float:
        return sum(size for _, size in self.bids[:5])

    @property
    def ask_volume_5_levels(self) -> float:
        return sum(size for _, size in self.asks[:5])

    @property
    def order_book_imbalance(self) -> float:
        """
        OBI = (BidVol - AskVol) / (BidVol + AskVol)
        Range: [-1, +1]
        > 0: More buy pressure
        < 0: More sell pressure
        """
        bv = self.bid_volume_5_levels
        av = self.ask_volume_5_levels
        total = bv + av
        if total == 0:
            return 0.0
        return (bv - av) / total


@dataclass
class SignalResult:
    """Output from any signal calculation"""
    signal_name: str
    value: float          # Raw signal value
    score: float          # Normalized [-1, +1]
    weight: float         # Signal weight in composite
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    @property
    def is_long(self) -> bool:
        return self.score > 0.3

    @property
    def is_short(self) -> bool:
        return self.score < -0.3

    @property
    def is_neutral(self) -> bool:
        return abs(self.score) <= 0.3


@dataclass
class CompositeSignal:
    """Aggregated signal from all individual signals"""
    timestamp: datetime
    symbol: str

    individual_signals: Dict[str, SignalResult]
    composite_score: float        # Weighted sum [-1, +1]
    signals_aligned: int          # Count of aligned signals
    regime: RegimeState

    # Trading decision
    should_trade: bool
    direction: Side
    strength: SignalStrength
    confidence: float             # 0 to 1

    @property
    def is_strong_long(self) -> bool:
        return self.composite_score > 0.65

    @property
    def is_strong_short(self) -> bool:
        return self.composite_score < -0.65

    @property
    def is_weak_long(self) -> bool:
        return 0.45 < self.composite_score <= 0.65

    @property
    def is_weak_short(self) -> bool:
        return -0.65 <= self.composite_score < -0.45


@dataclass
class PositionSize:
    """Kelly-calculated position sizing result"""
    symbol: str
    timestamp: datetime

    # Kelly components
    win_rate: float
    avg_win: float
    avg_loss: float
    reward_risk_ratio: float

    # Kelly calculations
    raw_kelly: float
    adjusted_kelly: float         # After all adjustments

    # Final sizing
    capital: float
    risk_amount: float            # Dollar risk
    position_value: float         # Total position value
    position_size_units: float    # In base currency

    # Adjustments applied
    volatility_adjustment: float
    regime_adjustment: float
    drawdown_adjustment: float

    stop_loss_price: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float


@dataclass
class Trade:
    """Complete trade lifecycle"""
    trade_id: str
    symbol: str
    side: Side
    status: TradeStatus

    # Entry
    entry_time: datetime
    entry_price: float
    entry_size: float           # Base currency amount
    entry_value: float          # USDT value

    # Risk levels
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_amount: float          # Dollar risk on this trade

    # Exit tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_value: Optional[float] = None

    # Partial exits
    partial_exits: List[Dict] = field(default_factory=list)
    remaining_size: float = 0.0

    # Performance
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    r_multiple: float = 0.0      # Profit in R units

    # Context
    regime_at_entry: Optional[RegimeState] = None
    composite_score_at_entry: float = 0.0
    signals_at_entry: Dict = field(default_factory=dict)

    # Metadata
    commission_paid: float = 0.0
    slippage: float = 0.0
    notes: str = ""

    def calculate_r_multiple(self) -> float:
        """Calculate R-multiple (profit/initial risk)"""
        if self.risk_amount == 0:
            return 0.0
        return self.realized_pnl / self.risk_amount

    def update_unrealized_pnl(self, current_price: float) -> float:
        if self.side == Side.LONG:
            self.unrealized_pnl = (
                (current_price - self.entry_price)
                * self.remaining_size
            )
        else:
            self.unrealized_pnl = (
                (self.entry_price - current_price)
                * self.remaining_size
            )
        return self.unrealized_pnl


@dataclass
class PortfolioState:
    """Real-time portfolio snapshot"""
    timestamp: datetime

    total_capital: float
    available_capital: float
    locked_in_trades: float

    open_trades: List[Trade]
    daily_pnl: float
    weekly_pnl: float
    total_pnl: float

    current_drawdown: float
    max_drawdown: float
    portfolio_heat: float         # Total risk exposure

    # Phase info
    phase: int
    reinvestment_rate: float

    @property
    def daily_return_pct(self) -> float:
        return self.daily_pnl / self.total_capital

    @property
    def is_circuit_breaker_active(self) -> bool:
        from config.settings import config
        return (
            self.daily_return_pct <= -config.trading.MAX_DAILY_LOSS
            or self.current_drawdown >= config.trading.MAX_DRAWDOWN_BEFORE_PAUSE
        )


@dataclass
class PerformanceMetrics:
    """Comprehensive performance analytics"""
    period_start: datetime
    period_end: datetime

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int    # In bars
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # R-multiple statistics
    avg_r_multiple: float
    expectancy: float             # Expected R per trade

    # Signal performance
    signal_ic: Dict[str, float]   # IC for each signal

    # Regime performance
    performance_by_regime: Dict[str, float]

    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate

    @property
    def is_performing(self) -> bool:
        from config.settings import config
        return (
            self.win_rate >= config.trading.MIN_WIN_RATE
            and self.profit_factor >= config.trading.MIN_PROFIT_FACTOR
            and self.sharpe_ratio >= config.trading.MIN_SHARPE_RATIO
        )

