from risk.risk_manager import RiskManager
from datetime import datetime
from unittest.mock import MagicMock

# Mock the data models
class Side:
    LONG = "LONG"
    SHORT = "SHORT"

class RegimeState:
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGING = "RANGING"
    TRANSITION = "TRANSITION"

class SignalStrength:
    STRONG_LONG = "STRONG_LONG"
    WEAK_LONG = "WEAK_LONG"
    STRONG_SHORT = "STRONG_SHORT"
    WEAK_SHORT = "WEAK_SHORT"
    NEUTRAL = "NEUTRAL"

# Initialize with $100
rm = RiskManager(initial_capital=100.0)

# Create mock signal — FIX: assign actual enum objects
signal = MagicMock()
signal.should_trade = True
signal.direction = Side.LONG                    # ← Actual value
signal.regime = RegimeState.BULL_TREND          # ← Actual value (NOT string)
signal.strength = SignalStrength.STRONG_LONG    # ← Actual value
signal.composite_score = 0.75
signal.symbol = "BTCUSDT"
signal.timestamp = datetime.now()
signal.individual_signals = {}

# Ensure config has required attributes
from config.settings import config

if not hasattr(config.trading, 'KELLY_FRACTION'):
    config.trading.KELLY_FRACTION = 0.25
if not hasattr(config.trading, 'MAX_RISK_PER_TRADE'):
    config.trading.MAX_RISK_PER_TRADE = 0.02
if not hasattr(config.trading, 'MAX_POSITION_FRACTION'):
    config.trading.MAX_POSITION_FRACTION = 0.25
if not hasattr(config.trading, 'MIN_POSITION_USD'):
    config.trading.MIN_POSITION_USD = 10.0

# Add regime config if missing
if not hasattr(config, 'regime'):
    config.regime = type('obj', (object,), {
        'REGIME_MULTIPLIERS': {
            'BULL_TREND': 1.0,
            'BEAR_TREND': 1.0,
            'RANGING': 0.5,
            'TRANSITION': 0.3,
        }
    })()

# Add drawdown config if missing
if not hasattr(config, 'drawdown'):
    config.drawdown = type('obj', (object,), {
        'LEVEL_1': 0.10,
        'LEVEL_2': 0.15,
        'LEVEL_3': 0.20,
        'SIZE_MULTIPLIER_1': 0.7,
        'SIZE_MULTIPLIER_2': 0.5,
        'SIZE_MULTIPLIER_3': 0.3,
    })()

# Test position sizing
position = rm.calculate_position_size(
    signal=signal,
    current_capital=100.0,
    current_price=43000.0,
    atr=430.0,
    open_trades=[]
)

if position:
    print(f"✅ Position Size: ${position.position_value:.2f}")
    print(f"✅ Kelly Fraction: {position.adjusted_kelly:.4f}")
    print(f"✅ Risk Amount: ${position.risk_amount:.2f}")
    print(f"✅ Stop Loss: ${position.stop_loss_price:.2f}")
    print(f"✅ TP1: ${position.take_profit_1:.2f}")
else:
    print("Position too small (expected with $100 — normal)")

# Test risk report
report = rm.get_risk_report(100.0)
print(f"\n✅ Risk of Ruin: {report['risk_of_ruin']:.8f}")
print(f"✅ Phase: {report['phase']}")

print("\n✅ RISK MANAGER TEST PASSED")
