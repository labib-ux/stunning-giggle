"""SQLAlchemy models and logging helpers for trades and agent decisions."""

import logging
import statistics
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for ORM models."""
    pass


class Trade(Base):
    """ORM model representing an executed trade."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    adapter: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String)
    side: Mapped[str] = mapped_column(String)
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    portfolio_value: Mapped[float] = mapped_column(Float)


class AgentDecision(Base):
    """ORM model representing an agent decision record."""

    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    state_hash: Mapped[str] = mapped_column(String)
    action: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)


def init_db(db_path: str) -> sessionmaker:
    """Initialize the database and return a session factory.

    Args:
        db_path: SQLite file path or full SQLAlchemy URL.

    Returns:
        SQLAlchemy sessionmaker bound to the database engine.
    """
    if db_path.startswith("sqlite:///"):
        connection_url = db_path
    else:
        connection_url = f"sqlite:///{db_path}"

    engine = create_engine(
        connection_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    logger.info("Database initialized with connection URL: %s", connection_url)
    return SessionLocal


def log_trade(session: Session, trade_data: dict) -> None:
    """Persist a trade record to the database.

    Args:
        session: SQLAlchemy session for database operations.
        trade_data: Trade data dictionary.
    """
    trade = Trade(
        adapter=trade_data.get("adapter", ""),
        symbol=trade_data.get("symbol", ""),
        side=trade_data.get("side", ""),
        qty=float(trade_data.get("qty", 0.0)),
        price=float(trade_data.get("price", 0.0)),
        pnl=trade_data.get("pnl"),
        portfolio_value=float(trade_data.get("portfolio_value", 0.0)),
    )

    try:
        session.add(trade)
        session.commit()
        logger.debug("Trade logged: %s %s", trade.symbol, trade.side)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("log_trade failed: %s", exc, exc_info=True)
        session.rollback()


def log_decision(session: Session, decision_data: dict) -> None:
    """Persist an agent decision record to the database.

    Args:
        session: SQLAlchemy session for database operations.
        decision_data: Decision data dictionary.
    """
    decision = AgentDecision(
        state_hash=decision_data.get("state_hash", ""),
        action=int(decision_data.get("action", 0)),
        confidence=float(decision_data.get("confidence", 0.0)),
        reward=decision_data.get("reward"),
    )

    try:
        session.add(decision)
        session.commit()
        logger.debug("Decision logged: action=%d", decision.action)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("log_decision failed: %s", exc, exc_info=True)
        session.rollback()


def get_trade_history(session: Session, limit: int = 100) -> list[dict]:
    """Fetch recent trade history records.

    Args:
        session: SQLAlchemy session for database operations.
        limit: Maximum number of trades to return.

    Returns:
        List of serialized trade dictionaries.
    """
    try:
        results = (
            session.query(Trade)
            .order_by(Trade.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": trade.id,
                "timestamp": trade.timestamp.isoformat(),
                "adapter": trade.adapter,
                "symbol": trade.symbol,
                "side": trade.side,
                "qty": trade.qty,
                "price": trade.price,
                "pnl": trade.pnl,
                "portfolio_value": trade.portfolio_value,
            }
            for trade in results
        ]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("get_trade_history failed: %s", exc, exc_info=True)
        session.rollback()
        return []


def get_pnl_summary(session: Session) -> dict[str, float]:
    """Compute summary statistics over all trade records.

    Args:
        session: SQLAlchemy session for database operations.

    Returns:
        Dictionary containing PnL summary statistics.
    """
    try:
        trades = session.query(Trade).all()
        total_trades = len(trades)
        winning_trades = sum(
            1 for trade in trades if trade.pnl is not None and trade.pnl > 0
        )
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(trade.pnl if trade.pnl is not None else 0.0 for trade in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        pnl_values = [trade.pnl for trade in trades if trade.pnl is not None]
        if len(pnl_values) > 1:
            mean_pnl = statistics.mean(pnl_values)
            std_pnl = statistics.stdev(pnl_values)
            sharpe = mean_pnl / (std_pnl + 1e-9)
        else:
            sharpe = 0.0

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "total_trades": float(total_trades),
            "avg_pnl": float(avg_pnl),
            "sharpe_ratio": float(sharpe),
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("get_pnl_summary failed: %s", exc, exc_info=True)
        session.rollback()
        return {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0.0,
            "avg_pnl": 0.0,
            "sharpe_ratio": 0.0,
        }
