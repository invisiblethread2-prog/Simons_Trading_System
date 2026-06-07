# main.py
# ============================================================
# START THE EMPIRE
# python main.py
# ============================================================

import asyncio
import logging
import sys
from pathlib import Path

from config.settings import config
from core.trading_engine import TradingEngine


def setup_logging() -> None:
    """Configure comprehensive logging"""
    Path(config.trading.LOG_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.trading.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.trading.LOG_LEVEL),
        format='%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"{config.trading.LOG_PATH}trading_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.log"
            )
        ]
    )


def print_startup_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     JAMES SIMONS ALGORITHMIC TRADING SYSTEM              ║
║     ─────────────────────────────────────────            ║
║     "The market is a code to be broken"                  ║
║                                                          ║
║     Mode:     PAPER TRADING (Safe Mode)                  ║
║     Capital:  $100                                       ║
║     Strategy: Statistical Momentum Reversion Hybrid      ║
║                                                          ║
║     Mathematics + Discipline + Compounding = Empire      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


async def main() -> None:
    setup_logging()
    print_startup_banner()

    engine = TradingEngine()

    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
```
