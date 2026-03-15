"""Streamlit dashboard for monitoring the algorithmic trading bot."""

import json
import time
import logging

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Any

from database.logger import init_db, get_trade_history, get_pnl_summary
from config import DB_PATH

logger = logging.getLogger(__name__)

CONTROL_FILE = "bot_control.json"

st.set_page_config(
    page_title="RL Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def get_db_sessionmaker() -> Any:
    """Creates and caches the SQLAlchemy sessionmaker.

    Decorated with @st.cache_resource so the database connection
    is created exactly once per Streamlit server lifetime and
    reused across all reruns and sessions.
    """
    try:
        return init_db(DB_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Database initialization failed: %s", exc, exc_info=True)
        st.error(
            f"Failed to connect to database at '{DB_PATH}'. "
            f"Ensure DB_PATH is correctly set in your .env file. "
            f"Error: {exc}"
        )
        st.stop()
        raise


def main() -> None:
    """Main Streamlit application entry point.

    Renders the full trading bot monitoring dashboard including
    live metrics, PnL charts, trade history, and bot controls.
    """
    st.title("🤖 Algorithmic Trading Dashboard")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Database path**")
    st.sidebar.code(DB_PATH, language=None)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Run the trading engine in a separate terminal:\n"
        "```bash\npython main.py\n```"
    )
    st.sidebar.markdown(
        "View TensorBoard training logs:\n"
        "```bash\ntensorboard --logdir ./tensorboard_logs\n```"
    )

    SessionLocal = get_db_sessionmaker()

    with SessionLocal() as session:
        summary = get_pnl_summary(session)
        trades = get_trade_history(session, limit=1000)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total PnL",
            value=f"${summary.get('total_pnl', 0.0):.2f}",
        )
    with col2:
        st.metric(
            label="🎯 Win Rate",
            value=f"{summary.get('win_rate', 0.0) * 100:.1f}%",
        )
    with col3:
        st.metric(
            label="📊 Total Trades",
            value=int(summary.get("total_trades", 0)),
        )
    with col4:
        st.metric(
            label="📐 Sharpe Ratio",
            value=f"{summary.get('sharpe_ratio', 0.0):.2f}",
        )

    st.divider()
    st.subheader("⚙️ Bot Controls")

    try:
        with open(CONTROL_FILE, "r") as f:
            control_state = json.load(f)
    except FileNotFoundError:
        control_state = {"paused": False}
        st.warning(
            "bot_control.json not found — defaulting to running state."
        )
    except json.JSONDecodeError:
        control_state = {"paused": False}
        logger.warning(
            "bot_control.json exists but contains invalid JSON — "
            "defaulting to running state."
        )
        st.warning(
            "bot_control.json contains invalid JSON — defaulting to running state."
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        control_state = {"paused": False}
        logger.warning(
            "Could not read bot_control.json: %s — defaulting to running state.",
            exc,
        )
        st.warning(
            f"Could not read {CONTROL_FILE}: {exc}. "
            "Defaulting to running state."
        )

    is_paused = control_state.get("paused", False)

    if is_paused:
        st.error("🔴 Bot Status: PAUSED — trading cycles are being skipped.")
    else:
        st.success("🟢 Bot Status: RUNNING — trading cycles are active.")

    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        if st.button(
            "⏸ Pause Bot",
            disabled=is_paused,
            use_container_width=True,
            help=(
                "Writes paused=true to bot_control.json. "
                "The trading engine reads this file before each cycle."
            ),
        ):
            try:
                with open(CONTROL_FILE, "w") as f:
                    json.dump({"paused": True}, f)
                logger.info("Bot paused via dashboard control panel.")
                st.rerun()
            except Exception as exc:  # pragma: no cover - defensive logging
                st.warning(
                    f"Could not write to {CONTROL_FILE}: {exc}. "
                    f"Check that the working directory is writable."
                )

    with ctrl_col2:
        if st.button(
            "▶ Resume Bot",
            disabled=not is_paused,
            use_container_width=True,
            help="Writes paused=false to bot_control.json.",
        ):
            try:
                with open(CONTROL_FILE, "w") as f:
                    json.dump({"paused": False}, f)
                logger.info("Bot resumed via dashboard control panel.")
                st.rerun()
            except Exception as exc:  # pragma: no cover - defensive logging
                st.warning(
                    f"Could not write to {CONTROL_FILE}: {exc}. "
                    f"Check that the working directory is writable."
                )

    st.divider()

    if not trades:
        st.info(
            "📭 No trades have been logged yet. "
            "The bot is waiting for its next scheduled execution cycle. "
            "Trades will appear here automatically once the engine executes."
        )
    else:
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["cumulative_pnl"] = df["pnl"].fillna(0.0).cumsum()

        st.subheader("📈 Cumulative Profit & Loss")

        fig1 = px.line(
            df,
            x="timestamp",
            y="cumulative_pnl",
            title="Cumulative PnL Over Time",
            markers=True,
            labels={
                "cumulative_pnl": "Cumulative PnL ($)",
                "timestamp": "Time",
            },
        )
        fig1.update_layout(hovermode="x unified")
        fig1.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Break even",
        )
        st.plotly_chart(fig1, use_container_width=True)

        sell_trades = df[df["pnl"].notna() & (df["side"] == "sell")].copy()

        if not sell_trades.empty:
            st.subheader("📊 Individual Trade PnL (Closed Positions Only)")
            fig2 = px.bar(
                sell_trades,
                x="timestamp",
                y="pnl",
                title="PnL per Closed Trade",
                color="pnl",
                color_continuous_scale=["red", "lightgray", "green"],
                color_continuous_midpoint=0,
                labels={
                    "pnl": "PnL ($)",
                    "timestamp": "Close Time",
                },
            )
            fig2.update_layout(hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(
                "No closed (SELL) trades yet — PnL bar chart will "
                "appear once the first position is closed."
            )

        st.subheader("📋 Recent Trade History (last 50)")

        display_df = (
            df.sort_values("timestamp", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time"),
                "price": st.column_config.NumberColumn("Price", format="$%.4f"),
                "pnl": st.column_config.NumberColumn("PnL", format="$%.4f"),
                "portfolio_value": st.column_config.NumberColumn(
                    "Portfolio",
                    format="$%.2f",
                ),
                "qty": st.column_config.NumberColumn("Qty", format="%.6f"),
            },
        )

    st.divider()
    refresh_col1, refresh_col2 = st.columns([3, 1])

    with refresh_col1:
        auto_refresh = st.checkbox(
            "🔁 Auto-refresh every 15 seconds",
            value=True,
            help=(
                "Automatically polls the database and rerenders "
                "the dashboard every 15 seconds."
            ),
        )
    with refresh_col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    if auto_refresh:
        time.sleep(15)
        st.rerun()


main()
