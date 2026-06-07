# test_upgrades.py
# Test all new upgrade files

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_all():
    print("\n" + "="*50)
    print("TESTING ALL UPGRADE FILES")
    print("="*50 + "\n")

    # Test 1: Stat Arb
    print("1. Testing StatisticalArbitrageEngine...")
    from signals.stat_arb import StatisticalArbitrageEngine
    sae    = StatisticalArbitrageEngine()
    prices = np.cumsum(np.random.randn(200)) + 100

    z, hl, theta = sae.calculate_ou_z_score(prices)
    ram          = sae.calculate_risk_adjusted_momentum(prices)
    regime, vr   = sae.detect_volatility_regime(
        np.diff(prices) / prices[:-1]
    )

    print(f"   OU Z-Score:   {z:.3f}")
    print(f"   Half Life:    {hl:.1f} bars")
    print(f"   Risk-Adj Mom: {ram:.3f}")
    print(f"   Vol Regime:   {regime} ({vr:.2f}x)")
    print(f"   ✅ StatArb OK\n")

    # Test 2: ML Optimizer
    print("2. Testing AdaptiveSignalOptimizer...")
    from ml.signal_optimizer import AdaptiveSignalOptimizer
    opt = AdaptiveSignalOptimizer()

    # Fake training data
    X = np.random.randn(50, 7)
    y = np.random.randn(50)
    r = ['BULL_TREND'] * 25 + ['BEAR_TREND'] * 25

    opt.fit(X, y, r)
    weights = opt.get_optimal_weights('BULL_TREND')
    print(f"   Trained: {opt.is_trained}")
    print(f"   Weights: {weights}")
    print(f"   ✅ ML Optimizer OK\n")

    # Test 3: Portfolio Optimizer
    print("3. Testing PortfolioOptimizer...")
    from risk.portfolio_optimizer import PortfolioOptimizer
    po = PortfolioOptimizer()

    expected_r = {'BTCUSDT': 0.03, 'ETHUSDT': 0.025}
    cov        = np.array([[0.04, 0.03], [0.03, 0.05]])
    symbols    = ['BTCUSDT', 'ETHUSDT']

    weights = po.optimize_kelly_portfolio(
        expected_r, cov, symbols
    )
    print(f"   Optimal weights: {weights}")
    is_safe, heat = po.portfolio_heat_check(
        {'BTCUSDT': 0.05}, cov, symbols
    )
    print(f"   Portfolio heat: {heat:.4f} | Safe: {is_safe}")
    print(f"   ✅ Portfolio Optimizer OK\n")

    # Test 4: Backtest Engine
    print("4. Testing BacktestEngine...")
    from backtesting.backtest_engine import BacktestEngine

    class SimpleStrategy:
        def fit(self, closes, highs, lows, volumes):
            self.mean = np.mean(closes)

        def predict(self, closes, highs, lows, volumes):
            if len(closes) < 20:
                return None
            atr = np.mean(np.abs(np.diff(closes[-20:])))
            if closes[-1] > np.mean(closes[-20:]):
                return {'direction': 'LONG', 'atr': atr}
            elif closes[-1] < np.mean(closes[-20:]):
                return {'direction': 'SHORT', 'atr': atr}
            return None

    bt     = BacktestEngine(100.0)
    closes = 100 + np.cumsum(np.random.randn(500) * 0.5)
    highs  = closes + np.abs(np.random.randn(500) * 0.3)
    lows   = closes - np.abs(np.random.randn(500) * 0.3)
    vols   = np.random.randint(1000, 5000, 500).astype(float)

    result = bt.run(
        closes, highs, lows, vols,
        SimpleStrategy(), "Test"
    )

    print(f"   Trades:     {result.total_trades}")
    print(f"   Win Rate:   {result.win_rate:.1%}")
    print(f"   Sharpe:     {result.sharpe_ratio:.2f}")
    print(f"   Deployable: {result.is_deployable}")
    print(f"   ✅ Backtest Engine OK\n")

    # Test 5: Sentiment (sync only)
    print("5. Testing SentimentSignal (import only)...")
    from signals.sentiment_signal import SentimentSignal
    ss = SentimentSignal()
    print(f"   ✅ SentimentSignal import OK\n")

    print("="*50)
    print("✅ ALL UPGRADE FILES WORKING")
    print("="*50 + "\n")


if __name__ == "__main__":
    test_all()
