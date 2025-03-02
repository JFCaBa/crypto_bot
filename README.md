# CryptoBot: Advanced Cryptocurrency Trading Bot

CryptoBot is a robust, automated cryptocurrency trading bot that executes trades across multiple exchanges using real-time data, predefined strategies, and sophisticated risk management protocols.

## Features

### Core Functionality

- **Real-Time Data Integration**: Fetches live market data via WebSocket/API from multiple exchanges
- **Multi-Exchange Support**: Integrates with Binance, Coinbase, and Kraken using a unified interface
- **Strategy Implementation**: Includes prebuilt strategies and supports custom strategy development
- **Risk Management**: Configurable stop-loss, take-profit, trailing stop, and portfolio risk controls
- **Backtesting & Simulation**: Historical data backtesting with performance metrics and analysis
- **User Interfaces**: Command-line interface (CLI) and web dashboard for monitoring and control
- **Notifications & Alerts**: Email and Telegram notifications for important events

### Technical Highlights

- **Asynchronous Architecture**: Built with Python's `asyncio` for efficient, non-blocking operations
- **Modular Design**: Easily extensible with new exchanges, strategies, and components
- **Data Processing**: Advanced data manipulation and technical indicator calculation
- **Database Integration**: Optional database support for storing historical data and trades
- **Security**: API key encryption and secure configuration management

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL (optional, for database support)
- TA-Lib (for technical analysis)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cryptobot.git
   cd cryptobot
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:

   ```bash
   pip install -e .
   ```

5. Create a configuration file:

   ```bash
   cryptobot init --config config.json
   ```

6. Edit the configuration file with your exchange API keys and settings

## Usage

### Command-Line Interface (CLI)

The CLI provides a powerful interface for controlling and monitoring the trading bot.

```bash
# Start interactive CLI
cryptobot cli --config config.json --mode test

# Run the trading bot directly
cryptobot run --config config.json --mode test

# Show bot status
cryptobot status --config config.json

# Run a backtest
cryptobot backtest --strategy ma_crossover --start-date 2023-01-01 --end-date 2023-12-31 --timeframe 1h
```

Available commands in interactive CLI:

- `start` - Start the trading bot
- `stop` - Stop the trading bot
- `status` - Show current bot status
- `balance` - Show account balances
- `positions` - Show active positions
- `trades` - Show recent trades
- `strategies` - List or manage strategies
- `markets` - Show available markets
- `backtest` - Run a backtest
- `settings` - View or change settings
- `help` - Show help message
- `exit` - Exit the CLI

### Using Docker

You can also run CryptoBot using Docker:

1. Build the Docker image:

   ```bash
   docker-compose build
   ```

2. Start the services:

   ```bash
   docker-compose up -d
   ```

3. View logs:
   ```bash
   docker-compose logs -f cryptobot
   ```

## Project Structure

```
cryptobot/
├── backtesting/          # Backtesting engine
├── config/               # Configuration management
├── core/                 # Core engine components
├── data/                 # Data processing and storage
├── exchanges/            # Exchange connectors
├── notifications/        # Notification services
├── risk_management/      # Risk management system
├── security/             # Security and encryption
├── strategies/           # Trading strategies
├── ui/                   # User interfaces
└── utils/                # Utility functions
```

## Creating Custom Strategies

CryptoBot allows you to create custom trading strategies by extending the `BaseStrategy` class:

1. Create a new file in the `strategies` directory
2. Import the base strategy class
3. Implement required methods: `calculate_indicators` and `generate_signals`
4. Register your strategy in the configuration file

Example:

```python
from cryptobot.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, symbols, timeframes, risk_manager=None, params=None):
        super().__init__(
            name="MyCustomStrategy",
            symbols=symbols,
            timeframes=timeframes,
            params=params,
            risk_manager=risk_manager
        )

    def calculate_indicators(self, symbol, timeframe):
        # Implement your indicator calculations here
        return True

    def generate_signals(self, symbol, timeframe):
        # Implement your signal generation logic here
        return signal
```

## Configuration

The configuration file (`config.json`) contains settings for exchanges, strategies, risk management, and more. Here's an example:

```json
{
  "database": {
    "enabled": false,
    "url": "localhost",
    "port": 5432,
    "username": "user",
    "password": "password",
    "database": "cryptobot"
  },
  "exchanges": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "api_secret": "YOUR_API_SECRET",
      "encrypted": false,
      "rate_limit": true,
      "timeout": 30000
    }
  },
  "strategies": {
    "ma_crossover": {
      "enabled": true,
      "type": "MovingAverageCrossover",
      "symbols": ["BTC/USDT", "ETH/USDT"],
      "timeframes": ["1h", "4h"],
      "params": {
        "fast_period": 10,
        "slow_period": 50,
        "signal_period": 9,
        "ma_type": "ema",
        "stop_loss": 2.0,
        "take_profit": 4.0,
        "risk_per_trade": 0.01
      }
    }
  },
  "risk_management": {
    "enabled": true,
    "max_positions": 5,
    "max_daily_trades": 20,
    "max_drawdown_percent": 20.0,
    "max_risk_per_trade": 2.0,
    "max_risk_per_day": 5.0,
    "max_risk_per_symbol": 10.0,
    "default_stop_loss": 2.0,
    "default_take_profit": 4.0
  }
}
```

## Backtesting

CryptoBot includes a powerful backtesting engine to test strategies on historical data:

```bash
cryptobot backtest --strategy=ma_crossover --start=2023-01-01 --end=2023-12-31 --timeframe=1h
```

Backtesting results include:

- Profit and loss (PnL) metrics
- Win rate and profit factor
- Drawdown statistics
- Sharpe and Sortino ratios
- Comparison to buy-and-hold strategy

## Security Considerations

1. **API Keys**: Never share your API keys or commit them to version control
2. **Permissions**: Use API keys with minimal permissions (read and trade only, no withdrawal permissions)
3. **Encryption**: Enable the encryption option in the configuration to encrypt API keys
4. **VPS**: Run the bot on a secure VPS rather than your personal computer
5. **Monitoring**: Regularly check logs and set up alerts for unusual activity

## Trading Risks

**Disclaimer**: Cryptocurrency trading involves significant risk. This software is provided for educational and research purposes only. Never trade with funds you cannot afford to lose.

- Start with a small amount of capital while testing
- Test thoroughly in backtesting and paper trading modes before live trading
- Understand and configure appropriate risk management settings
- Monitor the bot regularly, especially during volatile market conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [CCXT](https://github.com/ccxt/ccxt) for exchange API integration
- [pandas](https://pandas.pydata.org/) for data manipulation
- [TA-Lib](https://ta-lib.org/) for technical analysis
- [loguru](https://github.com/Delgan/loguru) for logging
