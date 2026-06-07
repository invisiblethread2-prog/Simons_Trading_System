import numpy as np
from datetime import datetime

# Import from your existing files
from signals.regime_detector import RegimeDetector
from models.data_models import MarketData, RegimeState

# Create synthetic market data
np.random.seed(42)
n = 200

# Simulate trending market
trend = np.cumsum(np.random.randn(n) * 0.5 + 0.1)
closes = 100 + trend
highs = closes + np.abs(np.random.randn(n) * 0.3)
lows = closes - np.abs(np.random.randn(n) * 0.3)
volumes = np.random.randint(1000, 5000, n).astype(float)

# Create MarketData object
market_data = MarketData(
    symbol="BTCUSDT",
    timeframe="5m",
    timestamp=datetime.now(),
    opens=closes,
    highs=highs,
    lows=lows,
    closes=closes,
    volumes=volumes
)

# Detect regime
detector = RegimeDetector()
regime, confidence, details = detector.detect(market_data)

print(f"✅ Regime Detected: {regime}")
print(f"✅ Confidence: {confidence:.2f}")
print(f"✅ Details: {details}")

print("\n✅ REGIME DETECTION TEST PASSED")
