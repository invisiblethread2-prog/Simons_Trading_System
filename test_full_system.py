# test_full_system.py
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Path fix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trading_engine import TradingEngine
from config.settings import config


def setup_logging():
    """File + Terminal dono mein logging"""
    Path("logs").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f"logs/trading_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'
    )

    # Terminal
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setLevel(logging.INFO)
    terminal.setFormatter(fmt)
    root_logger.addHandler(terminal)

    # File
    file_handler = logging.FileHandler(
        log_file, mode='w', encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return log_file


# ─── LOGGING SETUP ────────────────────────────────────────
log_file = setup_logging()

print(f"\n📁 Log file: {os.path.abspath(log_file)}")

# ─── SAFETY CHECKS ────────────────────────────────────────
print("\n🔍 Checking configuration...")
print(f"  PAPER_TRADING: {config.trading.PAPER_TRADING}")
print(f"  TESTNET:       {config.exchange.BINANCE_TESTNET}")

assert config.trading.PAPER_TRADING == True, "MUST be paper trading!"
assert config.exchange.BINANCE_TESTNET == True, "MUST be testnet!"
print("✅ Safety checks passed\n")


async def test_full_system():
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("🚀 Starting full system test...")
    print("Running 24/7 — Ctrl+C to stop")
    print("=" * 60 + "\n")

    logger.info("=" * 50)
    logger.info("TEST STARTED")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 50)

    engine = TradingEngine()

    try:
        await asyncio.wait_for(
            engine.start(),
            timeout=86400  # 24 hours
        )

    except asyncio.TimeoutError:
        logger.info("⏱️ 24 hour timeout reached")
        print("\n⏱️ 24 hour run complete")

    except KeyboardInterrupt:
        logger.info("⌨️ Stopped by user")
        print("\n⌨️ Stopped by user")

    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")

    finally:
        logger.info("Shutting down...")

        try:
            await engine.stop()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        await asyncio.sleep(1)

        # ─── SAFE FINAL RESULTS ───────────────────────────
        # FIX: Check None before accessing attributes
        current_cap  = engine.current_capital
        cycle_count  = engine.cycle_count
        trades_count = 0

        if engine.performance is not None:
            trades_count = len(engine.performance.all_trades)

        logger.info("=" * 50)
        logger.info("FINAL RESULTS")
        logger.info(f"Capital:  ${current_cap:.2f}")
        logger.info(f"Trades:   {trades_count}")
        logger.info(f"Cycles:   {cycle_count}")
        logger.info("=" * 50)

        print("\n" + "─" * 40)
        print("📊 FINAL RESULTS:")
        print("─" * 40)
        print(f"💰 Capital: ${current_cap:.2f}")
        print(f"📈 Trades:  {trades_count}")
        print(f"🔄 Cycles:  {cycle_count}")
        print("─" * 40)
        print(f"\n📁 Log: {os.path.abspath(log_file)}")
        print("✅ TEST COMPLETED\n")


if __name__ == "__main__":
    asyncio.run(test_full_system())
