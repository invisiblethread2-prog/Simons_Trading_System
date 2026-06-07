# monitor.py
# ============================================================
# REAL TIME DASHBOARD
# Run in separate terminal while system runs
# Shows everything happening live
# ============================================================

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

from data.data_collector import DataCollector
from signals.signal_engine import SignalEngine
from data.database import TradeDatabase
from config.settings import config


def clear():
    os.system('clear' if os.name != 'nt' else 'cls')


def score_bar(score: float, width: int = 20) -> str:
    """Visual bar for score"""
    filled = int(abs(score) * width)
    bar    = '█' * filled + '░' * (width - filled)
    return bar


def score_status(score: float) -> str:
    threshold = config.trading.COMPOSITE_SCORE_THRESHOLD
    if abs(score) >= threshold:
        return '🚀 TRADING NOW'
    elif abs(score) >= threshold * 0.85:
        return '🔥 VERY CLOSE '
    elif abs(score) >= threshold * 0.70:
        return '👀 WATCH      '
    else:
        return '😴 QUIET      '


async def run_dashboard():
    collector = DataCollector()
    await collector.initialize()

    signal_engine = SignalEngine()
    db = TradeDatabase()

    print("📡 Dashboard starting...")
    await asyncio.sleep(2)

    cycle = 0

    while True:
        cycle += 1
        clear()

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary = db.get_performance_summary()
        capital_rows = db.get_capital_history(days=1)
        current_capital = 100.0

        if capital_rows:
            current_capital = capital_rows[-1]['capital']

        # ─── HEADER ───────────────────────────────────────
        print(f"""
╔══════════════════════════════════════════════════════╗
║   🤖 SIMONS TRADING SYSTEM — LIVE DASHBOARD          ║
║   {now}                              ║
╠══════════════════════════════════════════════════════╣
║   💰 CAPITAL:  ${current_capital:>10,.2f}                        ║
║   📈 TRADES:   {summary.get('total_trades', 0):>4} total | Win Rate: {summary.get('win_rate', 0):>5.1%}        ║
║   💵 TOTAL PnL: ${summary.get('total_pnl', 0):>+9.2f}                       ║
╠══════════════════════════════════════════════════════╣
║   LIVE SIGNAL SCORES                                 ║""")

        # ─── SIGNAL SCORES ────────────────────────────────
        for symbol in config.trading.PRIMARY_PAIRS:
            try:
                pkg      = await collector.get_complete_data_package(symbol)
                multi_tf = pkg['multi_tf']
                ob       = pkg['order_book']
                trades   = pkg['trades']

                if not multi_tf or config.trading.ETF not in multi_tf:
                    print(f"║   {symbol}: No data available                        ║")
                    continue

                primary = multi_tf[config.trading.ETF]
                sec_map = {
                    'BTCUSDT': 'ETHUSDT',
                    'ETHUSDT': 'BTCUSDT'
                }
                sec = multi_tf.get(sec_map.get(symbol))

                sig = signal_engine.generate_composite_signal(
                    symbol=symbol,
                    primary_data=primary,
                    multi_tf_data=multi_tf,
                    order_book=ob,
                    trades_df=trades,
                    secondary_data=sec
                )

                score     = sig.composite_score
                regime    = sig.regime.name
                aligned   = sig.signals_aligned
                threshold = config.trading.COMPOSITE_SCORE_THRESHOLD
                gap       = threshold - abs(score)
                direction = '▲ LONG ' if score > 0 else '▼ SHORT'
                bar       = score_bar(score)
                status    = score_status(score)

                print(f"║                                                      ║")
                print(f"║   {symbol}                                           ║")
                print(f"║   Score:  {score:>+7.3f}  [{bar}]         ║")
                print(f"║   Regime: {regime:<12} Aligned: {aligned}/7            ║")
                print(f"║   Status: {status}  {direction}               ║")

                if gap > 0:
                    print(f"║   Gap to trade: {gap:>+.3f} more needed              ║")
                else:
                    print(f"║   ✅ THRESHOLD MET — TRADE SHOULD FIRE!              ║")

                # Save signal to database
                db.save_signal(symbol, sig)

            except Exception as e:
                print(f"║   {symbol}: Error — {str(e)[:30]}               ║")

        # ─── RECENT TRADES ────────────────────────────────
        recent = db.get_recent_trades(limit=5)
        print(f"""╠══════════════════════════════════════════════════════╣
║   RECENT TRADES                                      ║""")

        if not recent:
            print("║   No trades yet — system watching markets...        ║")
        else:
            for t in recent:
                pnl    = t.get('realized_pnl', 0) or 0
                symbol = t.get('symbol', '?')
                side   = t.get('side', '?')
                icon   = '✅' if pnl > 0 else '❌'
                print(
                    f"║   {icon} {symbol:<8} {side:<5} "
                    f"PnL: ${pnl:>+7.2f}                    ║"
                )

        # ─── SYSTEM SIGNALS BREAKDOWN ─────────────────────
        print(f"""╠══════════════════════════════════════════════════════╣
║   THRESHOLDS & SETTINGS                              ║
║   Trade threshold:  {config.trading.COMPOSITE_SCORE_THRESHOLD:.2f}                             ║
║   Min signals:      {config.trading.MIN_SIGNALS_ALIGNED}/7                              ║
║   Max position:     {config.trading.MAX_POSITION_FRACTION:.0%} of capital               ║
║   Min position:     ${config.trading.MIN_POSITION_USD:.2f}                            ║
║   Mode:             {'PAPER TRADING ✅' if config.trading.PAPER_TRADING else 'LIVE TRADING ⚠️ '}                  ║
╠══════════════════════════════════════════════════════╣
║   Cycle: {cycle:<6}  |  Updates every 30 seconds           ║
║   Ctrl+C to exit                                     ║
╚══════════════════════════════════════════════════════╝""")

        await asyncio.sleep(30)


if __name__ == "__main__":
    print("🚀 Starting dashboard...")
    print("Run test_full_system.py in another terminal\n")
    asyncio.run(run_dashboard())
