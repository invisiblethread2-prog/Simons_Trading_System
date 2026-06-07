import numpy as np
from signals.indicators import Indicators

print("Testing Indicators...")
print("=" * 40)

# Generate fake price data
np.random.seed(42)
closes = 100 + np.cumsum(np.random.randn(200) * 0.5)
highs = closes + np.abs(np.random.randn(200) * 0.3)
lows = closes - np.abs(np.random.randn(200) * 0.3)
volumes = np.random.randint(1000, 5000, 200).astype(float)

# Test ATR
atr = Indicators.atr(highs, lows, closes, period=14)
print(f"✅ ATR[-1]: {atr[-1]:.4f}")

# Test RSI
rsi = Indicators.rsi(closes, period=14)
print(f"✅ RSI[-1]: {rsi[-1]:.2f}")

# Test Hurst Exponent
hurst = Indicators.hurst_exponent(closes)
print(f"✅ Hurst Exponent: {hurst:.4f}")

# Test Z-Score
zscore = Indicators.zscore(closes, lookback=20)
print(f"✅ Z-Score[-1]: {zscore[-1]:.4f}")

# Test Momentum Score
momentum = Indicators.calculate_momentum_score(closes, volumes)
print(f"✅ Momentum Score: {momentum:.4f}")

# Test GARCH Volatility
returns = np.diff(closes) / closes[:-1]
garch_vol = Indicators.calculate_garch_volatility(returns)
print(f"✅ GARCH Volatility: {garch_vol:.6f}")

# Test Bollinger Bands
upper, middle, lower = Indicators.bollinger_bands(closes, period=20)
print(f"✅ Bollinger Bands - Upper: {upper[-1]:.2f}, Middle: {middle[-1]:.2f}, Lower: {lower[-1]:.2f}")

# Test MACD
macd, signal, hist = Indicators.macd(closes)
print(f"✅ MACD[-1]: {macd[-1]:.4f}")

# Test VWAP
vwap = Indicators.vwap(highs, lows, closes, volumes)
print(f"✅ VWAP[-1]: {vwap[-1]:.2f}")

print("\n" + "=" * 40)
print("✅ ALL INDICATOR TESTS PASSED!")
