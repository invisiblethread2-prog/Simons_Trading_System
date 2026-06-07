# score_monitor.py
# ============================================================
# SCORE MONITOR
# Run in separate terminal while system runs
# Shows live signal scores every 60 seconds
# ============================================================

import asyncio
import logging
import sys
import os
from datetime import datetime

# Silence most logging
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def monitor():
    """
    Lightweight monitor showing only score progression.
    Run this in a second terminal while test_full_system.py runs.
    """
    from data.data_collector import DataCollector
    from signals.signal_engine import SignalEngine
    from config.settings import config

    collector = DataCollector()
    await collector.initialize()
    engine = SignalEngine()

    print("\n" + "=" * 70)
    print("📡 SCORE MONITOR — Updates every 60 seconds")
    print("=" * 70)
    print(
        f"{'Time':<10} "
        f"{'Symbol':<10} "
        f"{'Score':>8} "
        f"{'Regime':<15} "
        f"{'Gap':>10} "
        f"{'Status':<12} "
        f"{'Dir':<8}"
    )
    print("=" * 70)

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now().strftime('%H:%M:%S')

        for symbol in config.trading.PRIMARY_PAIRS:
            try:
                # ─── GET DATA ─────────────────────────────
                data_pkg = await collector.get_complete_data_package(symbol)
                multi_tf = data_pkg['multi_tf']
                ob       = data_pkg['order_book']
                trades   = data_pkg['trades']

                if not multi_tf or config.trading.ETF not in multi_tf:
                    print(f"{now:<10} {symbol:<10} {'NO DATA':>8}")
                    continue

                primary = multi_tf[config.trading.ETF]

                secondary_map = {
                    'BTCUSDT': 'ETHUSDT',
                    'ETHUSDT': 'BTCUSDT',
                }
                sec_symbol = secondary_map.get(symbol)
                sec_data   = multi_tf.get(sec_symbol) if sec_symbol else None

                # ─── GENERATE SIGNAL ──────────────────────
                sig = engine.generate_composite_signal(
                    symbol=symbol,
                    primary_data=primary,
                    multi_tf_data=multi_tf,
                    order_book=ob,
                    trades_df=trades,
                    secondary_data=sec_data
                )

                score     = sig.composite_score
                regime    = sig.regime.name
                aligned   = sig.signals_aligned
                threshold = config.trading.COMPOSITE_SCORE_THRESHOLD
                gap       = threshold - abs(score)

                # ─── STATUS ───────────────────────────────
                if abs(score) >= threshold:
                    status = "🚀 TRADE!"
                elif abs(score) >= threshold * 0.85:
                    status = "🔥 CLOSE "
                elif abs(score) >= 0.3:
                    status = "👀 WATCH "
                else:
                    status = "😴 QUIET "

                direction  = "▲ LONG " if score > 0 else "▼ SHORT"
                gap_text   = "TRADE NOW!" if gap <= 0 else f"need +{gap:.3f}"

                print(
                    f"{now:<10} "
                    f"{symbol:<10} "
                    f"{score:>+8.3f} "
                    f"{regime:<15} "
                    f"{gap_text:>10} "
                    f"{status:<12} "
                    f"{direction:<8} "
                    f"[{aligned}/7]"
                )

            except Exception as e:
                print(
                    f"{now:<10} "
                    f"{symbol:<10} "
                    f"{'ERROR':>8}  "
                    f"{str(e)[:30]}"
                )

        # ─── SEPARATOR BETWEEN CYCLES ─────────────────────
        print("-" * 70)
        await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped.")
