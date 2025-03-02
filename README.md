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

## Architecture

CryptoBot is organized into several key components:

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
├── utils/                # Utility functions
├── web_dashboard.py      # Web dashboard interface
└── cli.py                # Command-line interface
```

### Key Components

1. **Trading Engine**: Orchestrates all components and manages the trading lifecycle
2. **Exchange Connectors**: Provide a unified interface to different cryptocurrency exchanges
3. **Strategy Framework**: Base classes and implementations for trading strategies
4. **Risk Management**: Controls trading risk and provides safety mechanisms
5. **Data Processing**: Handles market data processing and technical analysis
6. **Backtesting Engine**: Allows testing strategies on historical data
7. **User Interfaces**: CLI and web dashboard for monitoring and control

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL (optional, for database support)

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

4. Create a configuration file (copy from the example):

   ```bash
   cp config.example.json config.json
   ```

5. Edit the configuration file with your exchange API keys and settings

## Usage

### Command-Line Interface (CLI)

The CLI provides a powerful interface for controlling and monitoring the trading bot.

```bash
python -m cryptobot.ui.cli --config config.json --mode test
```

Available commands:

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

### Web Dashboard

The web dashboard provides a graphical interface for monitoring trading activity.

```bash
python -m cryptobot.ui.web_dashboard --config config.json --port 8050
```

Then open your browser and navigate to `http://localhost:8050` to access the dashboard.

### Running as a Service

For production use, it's recommended to run CryptoBot as a service using Docker or a process manager like Supervisor.

#### Docker

1. Build the Docker image:

   ```bash
   docker build -t cryptobot .
   ```

2. Run the container:
   ```bash
   docker run -d --name cryptobot -v $(pwd)/config.json:/app/config.json cryptobot
   ```

#### Supervisor

Example supervisor configuration:

```ini
[program:cryptobot]
command=/path/to/venv/bin/python -m cryptobot.cli --config /path/to/config.json --mode production
directory=/path/to/cryptobot
user=username
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/path/to/cryptobot/logs/supervisor.log
```

## Configuration

CryptoBot uses a JSON configuration file. Here's an overview of the key configuration sections:

### Database

```json
"database": {
    "enabled": true,
    "url": "localhost",
    "port": 5432,
    "username": "username",
    "password": "password",
    "database": "cryptobot"
}
```

### Exchanges

```json
"exchanges": {
    "binance": {
        "enabled": true,
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_API_SECRET",
        "encrypted": false,
        "rate_limit": true,
        "timeout": 30000
    }
}
```

### Strategies

```json
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
}
```

### Risk Management

```json
"risk_management": {
    "enabled": true,
    "max_positions": 5,
    "max_daily_trades": 20,
    "max_drawdown_percent": 20.0,
    "max_risk_per_trade": 2.0,
    "max_risk_per_day": 5.0,
    "max_risk_per_symbol": 10.0,
    "default_stop_loss": 2.0,
    "default_take_profit": 4.0,
    "correlation_limit": 0.7,
    "night_trading": true,
    "weekend_trading": true,
    "account_size": 10000.0
}
```

### Notifications

```json
"notifications": {
    "enabled": true,
    "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_password",
        "sender": "your_email@gmail.com",
        "recipients": ["your_email@gmail.com"]
    },
    "telegram": {
        "enabled": true,
        "token": "YOUR_BOT_TOKEN",
        "chat_ids": ["YOUR_CHAT_ID"]
    }
}
```

## Creating Custom Strategies

CryptoBot allows you to create custom trading strategies by extending the `BaseStrategy` class.

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

## Backtesting

CryptoBot includes a powerful backtesting engine to test strategies on historical data.

```bash
python -m cryptobot.ui.cli backtest --strategy=ma_crossover --start=2023-01-01 --end=2023-12-31 --timeframe=1h
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
- Implement proper monitoring and kill switch mechanisms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Keep the code modular and maintainable

## Project Roadmap

### Short-Term Goals

- [ ] Improve exchange connectivity error handling
- [ ] Add more technical indicators
- [ ] Enhance the web dashboard with custom views
- [ ] Implement more built-in strategies

### Mid-Term Goals

- [ ] Add support for more exchanges (Bybit, KuCoin, etc.)
- [ ] Implement portfolio optimization algorithms
- [ ] Develop a mobile notification app
- [ ] Create a strategy marketplace

### Long-Term Goals

- [ ] Integrate machine learning for price prediction
- [ ] Add DeFi/DEX support for decentralized trading
- [ ] Implement social trading features
- [ ] Develop an API for third-party integrations

## Troubleshooting

### Common Issues

1. **WebSocket Connection Issues**

   - Check internet connection
   - Verify API keys have correct permissions
   - Ensure the exchange is operational

2. **Strategy Not Generating Signals**

   - Check if the strategy is enabled in the configuration
   - Verify the timeframe data is being fetched correctly
   - Inspect the logs for any errors in indicator calculation

3. **Database Connection Errors**

   - Check database credentials
   - Ensure PostgreSQL service is running
   - Verify the database has been initialized

4. **High CPU/Memory Usage**
   - Reduce the number of active strategies or symbols
   - Increase the loop interval
   - Check for memory leaks in custom strategies

### Logs

Logs are stored in the `logs` directory by default. Check these logs for detailed error information and debugging insights.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [CCXT](https://github.com/ccxt/ccxt) for exchange API integration
- [pandas](https://pandas.pydata.org/) for data manipulation
- [TA-Lib](https://ta-lib.org/) for technical analysis
- [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/) for the web dashboard
- [loguru](https://github.com/Delgan/loguru) for logging
- [backtrader](https://www.backtrader.com/) for backtesting inspiration

## Contact

For questions, feedback, or support, please create an issue in the GitHub repository or contact the maintainers directly.

---

**Disclaimer**: This software is for educational and research purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred through the use of this software.
