# data/database.py
# ============================================================
# PERMANENT TRADE STORAGE — COMPLETE FIXED VERSION
# ✅ save_open_trade() added
# ✅ update_trade_on_close() added
# ✅ get_open_trades() added
# ✅ All indentation fixed
# ============================================================

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeDatabase:

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"✅ Database initialized: {db_path}")

    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id        TEXT UNIQUE NOT NULL,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    entry_time      TEXT NOT NULL,
                    entry_price     REAL NOT NULL,
                    entry_size      REAL NOT NULL,
                    entry_value     REAL NOT NULL,
                    exit_time       TEXT,
                    exit_price      REAL,
                    exit_value      REAL,
                    stop_loss       REAL NOT NULL,
                    take_profit_1   REAL NOT NULL,
                    take_profit_2   REAL NOT NULL,
                    take_profit_3   REAL NOT NULL,
                    risk_amount     REAL NOT NULL,
                    realized_pnl    REAL DEFAULT 0,
                    r_multiple      REAL DEFAULT 0,
                    commission_paid REAL DEFAULT 0,
                    slippage        REAL DEFAULT 0,
                    regime_at_entry TEXT,
                    composite_score REAL,
                    signals_aligned INTEGER,
                    signals_data    TEXT,
                    capital_before  REAL,
                    capital_after   REAL,
                    notes           TEXT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT NOT NULL,
                    symbol              TEXT NOT NULL,
                    regime              TEXT NOT NULL,
                    composite_score     REAL NOT NULL,
                    signals_aligned     INTEGER NOT NULL,
                    should_trade        INTEGER NOT NULL,
                    direction           TEXT NOT NULL,
                    momentum_score      REAL,
                    zscore_score        REAL,
                    order_flow_score    REAL,
                    volume_score        REAL,
                    trend_align_score   REAL,
                    obi_score           REAL,
                    cointegration_score REAL,
                    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS capital_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    capital     REAL NOT NULL,
                    daily_pnl   REAL DEFAULT 0,
                    total_pnl   REAL DEFAULT 0,
                    drawdown    REAL DEFAULT 0,
                    phase       INTEGER DEFAULT 1,
                    note        TEXT,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regime_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    regime      TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    adx         REAL,
                    hurst       REAL,
                    atr_ratio   REAL,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    date             TEXT UNIQUE NOT NULL,
                    starting_capital REAL NOT NULL,
                    ending_capital   REAL NOT NULL,
                    daily_pnl        REAL NOT NULL,
                    daily_return     REAL NOT NULL,
                    trades_taken     INTEGER DEFAULT 0,
                    trades_won       INTEGER DEFAULT 0,
                    trades_lost      INTEGER DEFAULT 0,
                    win_rate         REAL DEFAULT 0,
                    max_drawdown     REAL DEFAULT 0,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    key        TEXT UNIQUE NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            logger.info("✅ All database tables created/verified")

    # ─── SAVE OPEN TRADE (NEW) ────────────────────────────

    def save_open_trade(
        self,
        trade,
        capital_before: float
    ) -> None:
        """Save OPEN trade immediately on entry — survives restart"""
        try:
            signals_json = json.dumps(
                trade.signals_at_entry
                if trade.signals_at_entry else {}
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, symbol, side, status,
                        entry_time, entry_price, entry_size, entry_value,
                        stop_loss, take_profit_1, take_profit_2,
                        take_profit_3, risk_amount,
                        realized_pnl, r_multiple,
                        commission_paid, slippage,
                        regime_at_entry, composite_score,
                        signals_data, capital_before, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    trade.trade_id,
                    trade.symbol,
                    trade.side.value,
                    'ACTIVE',
                    trade.entry_time.isoformat(),
                    trade.entry_price,
                    trade.entry_size,
                    trade.entry_value,
                    trade.stop_loss,
                    trade.take_profit_1,
                    trade.take_profit_2,
                    trade.take_profit_3,
                    trade.risk_amount,
                    0.0,
                    0.0,
                    trade.commission_paid,
                    trade.slippage,
                    trade.regime_at_entry.name
                    if trade.regime_at_entry else None,
                    trade.composite_score_at_entry,
                    signals_json,
                    capital_before,
                    'OPEN TRADE'
                ))
                conn.commit()

            logger.info(
                f"💾 Open trade saved: {trade.trade_id} | "
                f"{trade.symbol} {trade.side.value} @ "
                f"${trade.entry_price:.2f}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save open trade: {e}")

    # ─── UPDATE TRADE ON CLOSE (NEW) ─────────────────────

    def update_trade_on_close(
        self,
        trade,
        capital_after: float
    ) -> None:
        """Update trade record when it closes — ACTIVE → CLOSED"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE trades SET
                        status          = ?,
                        exit_time       = ?,
                        exit_price      = ?,
                        exit_value      = ?,
                        stop_loss       = ?,
                        realized_pnl    = ?,
                        r_multiple      = ?,
                        commission_paid = ?,
                        capital_after   = ?
                    WHERE trade_id = ?
                """, (
                    trade.status.value,
                    trade.exit_time.isoformat()
                    if trade.exit_time else None,
                    trade.exit_price,
                    trade.exit_value,
                    trade.stop_loss,
                    trade.realized_pnl,
                    trade.r_multiple,
                    trade.commission_paid,
                    capital_after,
                    trade.trade_id
                ))
                conn.commit()

            logger.info(
                f"✅ Trade closed in DB: {trade.trade_id} | "
                f"PnL: ${trade.realized_pnl:+.2f}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to update trade on close: {e}")

    # ─── GET OPEN TRADES (NEW) ────────────────────────────

    def get_open_trades(self) -> List[Dict]:
        """Get all currently open/active trades"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM trades "
                    "WHERE status IN ('ACTIVE', 'PARTIAL') "
                    "ORDER BY entry_time DESC"
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get open trades: {e}")
            return []

    # ─── SAVE COMPLETED TRADE (ORIGINAL) ─────────────────

    def save_trade(
        self,
        trade,
        capital_before: float,
        capital_after: float
    ) -> None:
        """Save completed trade (fallback — use update_trade_on_close instead)"""
        try:
            signals_json = json.dumps(
                trade.signals_at_entry
                if trade.signals_at_entry else {}
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, symbol, side, status,
                        entry_time, entry_price, entry_size, entry_value,
                        exit_time, exit_price, exit_value,
                        stop_loss, take_profit_1, take_profit_2,
                        take_profit_3, risk_amount,
                        realized_pnl, r_multiple,
                        commission_paid, slippage,
                        regime_at_entry, composite_score,
                        signals_data, capital_before, capital_after,
                        notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    trade.trade_id,
                    trade.symbol,
                    trade.side.value,
                    trade.status.value,
                    trade.entry_time.isoformat(),
                    trade.entry_price,
                    trade.entry_size,
                    trade.entry_value,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.exit_price,
                    trade.exit_value,
                    trade.stop_loss,
                    trade.take_profit_1,
                    trade.take_profit_2,
                    trade.take_profit_3,
                    trade.risk_amount,
                    trade.realized_pnl,
                    trade.r_multiple,
                    trade.commission_paid,
                    trade.slippage,
                    trade.regime_at_entry.name
                    if trade.regime_at_entry else None,
                    trade.composite_score_at_entry,
                    signals_json,
                    capital_before,
                    capital_after,
                    trade.notes
                ))
                conn.commit()

            logger.info(
                f"💾 Trade saved: {trade.trade_id} | "
                f"PnL: ${trade.realized_pnl:+.2f}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save trade: {e}")

    # ─── SIGNALS ──────────────────────────────────────────

    def save_signal(self, symbol: str, signal) -> None:
        try:
            sigs = signal.individual_signals

            def get_score(name):
                s = sigs.get(name)
                return s.score if s else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO signals (
                        timestamp, symbol, regime,
                        composite_score, signals_aligned,
                        should_trade, direction,
                        momentum_score, zscore_score,
                        order_flow_score, volume_score,
                        trend_align_score, obi_score,
                        cointegration_score
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    signal.timestamp.isoformat(),
                    symbol,
                    signal.regime.name,
                    signal.composite_score,
                    signal.signals_aligned,
                    1 if signal.should_trade else 0,
                    signal.direction.value,
                    get_score('momentum'),
                    get_score('zscore_reversion'),
                    get_score('order_flow'),
                    get_score('volume_confirmation'),
                    get_score('trend_alignment'),
                    get_score('order_book_imbalance'),
                    get_score('cointegration')
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"❌ Failed to save signal: {e}")

    # ─── CAPITAL ──────────────────────────────────────────

    def save_capital(
        self,
        capital: float,
        daily_pnl: float,
        total_pnl: float,
        drawdown: float,
        phase: int,
        note: str = ""
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO capital_history (
                        timestamp, capital, daily_pnl,
                        total_pnl, drawdown, phase, note
                    ) VALUES (?,?,?,?,?,?,?)
                """, (
                    datetime.now().isoformat(),
                    capital, daily_pnl,
                    total_pnl, drawdown, phase, note
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save capital: {e}")

    def save_daily_performance(
        self,
        date: str,
        starting_capital: float,
        ending_capital: float,
        trades: list
    ) -> None:
        try:
            closed      = [t for t in trades if t.status.value == 'CLOSED']
            wins        = [t for t in closed if t.realized_pnl > 0]
            losses      = [t for t in closed if t.realized_pnl <= 0]
            daily_pnl   = ending_capital - starting_capital
            daily_return= daily_pnl / starting_capital if starting_capital > 0 else 0
            win_rate    = len(wins) / len(closed) if closed else 0

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, starting_capital, ending_capital,
                        daily_pnl, daily_return, trades_taken,
                        trades_won, trades_lost, win_rate
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    date, starting_capital, ending_capital,
                    daily_pnl, daily_return, len(closed),
                    len(wins), len(losses), win_rate
                ))
                conn.commit()

            logger.info(
                f"📅 Daily saved: {date} | "
                f"PnL: ${daily_pnl:+.2f} | "
                f"Trades: {len(closed)}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to save daily: {e}")

    # ─── STATE ────────────────────────────────────────────

    def save_state(self, key: str, value) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_state
                    (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(value), datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def load_state(self, key: str, default=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM system_state WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return default
        except Exception as e:
            logger.error(f"❌ Failed to load state: {e}")
            return default

    # ─── QUERIES ──────────────────────────────────────────

    def get_all_trades(self) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM trades ORDER BY entry_time DESC"
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get trades: {e}")
            return []

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM trades "
                    "ORDER BY entry_time DESC LIMIT ?",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get recent trades: {e}")
            return []

    def get_performance_summary(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*)                         as total_trades,
                        SUM(realized_pnl)                as total_pnl,
                        AVG(realized_pnl)                as avg_pnl,
                        AVG(r_multiple)                  as avg_r,
                        SUM(CASE WHEN realized_pnl > 0
                            THEN 1 ELSE 0 END)           as wins,
                        SUM(CASE WHEN realized_pnl <= 0
                            THEN 1 ELSE 0 END)           as losses,
                        MIN(entry_time)                  as first_trade,
                        MAX(exit_time)                   as last_trade,
                        SUM(commission_paid)             as total_commission
                    FROM trades
                    WHERE status = 'CLOSED'
                """)
                row = cursor.fetchone()
                if row and row[0]:
                    total = row[0]
                    wins  = row[4] or 0
                    return {
                        'total_trades':     total,
                        'total_pnl':        round(row[1] or 0, 2),
                        'avg_pnl':          round(row[2] or 0, 2),
                        'avg_r':            round(row[3] or 0, 3),
                        'win_rate':         round(wins/total, 3)
                                            if total > 0 else 0,
                        'wins':             wins,
                        'losses':           row[5] or 0,
                        'first_trade':      row[6],
                        'last_trade':       row[7],
                        'total_commission': round(row[8] or 0, 2)
                    }
                return {'total_trades': 0}
        except Exception as e:
            logger.error(f"❌ Failed to get summary: {e}")
            return {}

    def get_capital_history(self, days: int = 30) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM capital_history
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp ASC
                """, (f'-{days} days',))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get capital history: {e}")
            return []

    def print_summary(self) -> None:
        summary = self.get_performance_summary()

        if summary.get('total_trades', 0) == 0:
            print("\n📊 No completed trades yet.\n")
            return

        print(f"""
╔══════════════════════════════════════════════╗
║  DATABASE PERFORMANCE SUMMARY                ║
╠══════════════════════════════════════════════╣
║  Total Trades:    {summary['total_trades']:>6}                    ║
║  Win Rate:        {summary['win_rate']:>6.1%}                    ║
║  Total PnL:      ${summary['total_pnl']:>+10.2f}                ║
║  Avg PnL/Trade:  ${summary['avg_pnl']:>+10.2f}                ║
║  Avg R Multiple:  {summary['avg_r']:>+9.3f}R                   ║
║  Total Commission:${summary['total_commission']:>9.2f}                ║
╚══════════════════════════════════════════════╝
        """)
