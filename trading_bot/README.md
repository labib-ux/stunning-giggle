# RL Trading Bot
A production-ready reinforcement learning trading bot with live execution, risk management, scheduling, and a monitoring dashboard.

**Architecture Overview**
- `data/`: Market data fetching and feature engineering
- `environment/`: Gymnasium-compatible trading environment
- `adapters/`: Broker and venue adapters (Alpaca, Polymarket, Binance stub)
- `risk/`: Risk management and position sizing logic
- `database/`: Trade and decision logging via SQLAlchemy
- `agent/`: PPO agent, feature extractor, and training utilities
- `scheduler/`: Live execution loop and job scheduling
- `dashboard/`: Streamlit monitoring UI
- `config.py`: Central configuration and environment loading
- `main.py`: Entry point for training and live execution

**Quick Start**
1. Clone / set up the repository and open a terminal in the project root.
2. Run the setup script:
```bash
./setup.sh
```
3. Add API keys and settings in `.env`.
4. Verify the installation:
```bash
python verify.py
```
5. Train the agent:
```bash
python main.py --mode train
```
6. Paper trade:
```bash
python main.py --mode paper
```
7. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```
8. View TensorBoard logs:
```bash
tensorboard --logdir ./tensorboard_logs
```

**Environment Variables**
| Name | Description | Required |
| --- | --- | --- |
| `ALPACA_API_KEY` | Alpaca API key for market access. | Required for live/paper |
| `ALPACA_SECRET_KEY` | Alpaca secret key for market access. | Required for live/paper |
| `ALPACA_BASE_URL` | Alpaca API base URL (paper or live). | Optional (defaulted) |
| `POLYMARKET_API_KEY` | Polymarket API key. | Optional |
| `POLYMARKET_PRIVATE_KEY` | Polygon wallet private key for Polymarket. | Optional |
| `POLYGON_RPC_URL` | Polygon RPC URL for on-chain signing. | Optional (defaulted) |
| `DB_PATH` | SQLite database path. | Optional (defaulted) |
| `MODEL_SAVE_PATH` | Path to saved PPO model. | Optional (defaulted) |
| `LOG_LEVEL` | Python logging level. | Optional (defaulted) |
| `INITIAL_CAPITAL` | Starting capital used for sizing and observations. | Optional (defaulted) |
| `MAX_POSITION_SIZE` | Max fraction of capital to risk per trade. | Optional (defaulted) |
| `MAX_DAILY_DRAWDOWN` | Daily drawdown limit for circuit breaker. | Optional (defaulted) |

**Project Structure**
```text
trading_bot/
├── .env.example
├── .gitignore
├── README.md
├── config.py
├── main.py
├── requirements.txt
├── setup.bat
├── setup.sh
├── verify.py
├── adapters/
│   ├── __init__.py
│   ├── alpaca_adapter.py
│   ├── base_adapter.py
│   ├── binance_adapter.py
│   └── polymarket_adapter.py
├── agent/
│   ├── __init__.py
│   ├── policy.py
│   └── trainer.py
├── dashboard/
│   ├── __init__.py
│   └── app.py
├── data/
│   ├── __init__.py
│   ├── fetcher.py
│   └── features.py
├── database/
│   ├── __init__.py
│   └── logger.py
├── environment/
│   ├── __init__.py
│   └── trading_env.py
├── risk/
│   ├── __init__.py
│   └── manager.py
└── scheduler/
    ├── __init__.py
    └── jobs.py
```

**Disclaimer**
This project is for educational purposes only and does not constitute financial advice. Trading involves risk, and you are solely responsible for your financial decisions.
