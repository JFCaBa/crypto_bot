"""
Core Trading Engine
=================
Main engine that orchestrates all the components of the trading bot.
"""

import asyncio
import time
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
from loguru import logger

from cryptobot.exchanges.base import BaseExchange
from cryptobot.exchanges.binance import BinanceExchange
from cryptobot.exchanges.coinbase import CoinbaseExchange
from cryptobot.exchanges.kraken import KrakenExchange
from cryptobot.exchanges.mexc import MexcExchange
from cryptobot.exchanges.bybit import BybitExchange
from cryptobot.strategies.base import BaseStrategy
from cryptobot.strategies.machine_learning import MachineLearningStrategy  # Added import
from cryptobot.risk_management.manager import RiskManager
from cryptobot.data.database import DatabaseManager
from cryptobot.data.processor import DataProcessor
from cryptobot.data.historical import HistoricalDataProvider
from cryptobot.notifications.email import EmailNotifier
from cryptobot.notifications.telegram import TelegramNotifier
from cryptobot.backtesting.engine import BacktestingEngine
from cryptobot.utils.helpers import setup_logger, timeframe_to_seconds
from cryptobot.utils.helpers import setup_logger
from cryptobot.config.settings import load_config, save_config
from cryptobot.security.encryption import decrypt_api_keys
from cryptobot.core.trade import Trade


class TradingEngine:
    """
    Main trading engine that orchestrates all components.
    """
    
    def __init__(
        self,
        config_path: str,
        log_level: str = "INFO",
        mode: str = "production"  # production, test, backtest
    ):
        """
        Initialize the trading engine.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
            mode: Operation mode
        """
        # Set up logging
        setup_logger(log_level)
        logger.info(f"Initializing Trading Engine in {mode} mode")
        
        self.config_path = config_path
        self.mode = mode
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.exchanges: Dict[str, BaseExchange] = {}
        self.strategies: Dict[str, BaseStrategy] = {}
        self.risk_manager = None
        self.db_manager = None
        self.data_processor = None
        self.historical_data = None
        self.notifiers = []
        
        # Engine state
        self.is_running = False
        self.start_time = None
        self.last_heartbeat = None
        self.last_error = None
        self.account_balances = {}
        
        # Data storage for fetched historical data
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}  # Added data storage
        
        # Performance metrics
        self.performance = {}
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all engine components."""
        try:
            # Initialize database
            db_config = self.config.get('database', {})
            if db_config and db_config.get('enabled', False):
                self.db_manager = DatabaseManager(
                    url=db_config.get('url'),
                    username=db_config.get('username'),
                    password=db_config.get('password'),
                    database=db_config.get('database')
                )
                logger.info("Database manager initialized")
                
            # Initialize data processor
            self.data_processor = DataProcessor(
                cache_enabled=self.config.get('cache_enabled', True),
                cache_dir=self.config.get('cache_dir', '.cache')
            )
            logger.info("Data processor initialized")
            
            # Initialize historical data provider
            historical_config = self.config.get('historical_data', {})
            if historical_config and historical_config.get('enabled', False):
                self.historical_data = HistoricalDataProvider(
                    source=historical_config.get('source', 'csv'),
                    data_dir=historical_config.get('data_dir', 'data'),
                    api_key=historical_config.get('api_key'),
                    db_manager=self.db_manager
                )
                logger.info("Historical data provider initialized")
                
            # Initialize risk manager
            risk_config = self.config.get('risk_management', {})
            if risk_config and risk_config.get('enabled', True):
                self.risk_manager = RiskManager(
                    max_positions=risk_config.get('max_positions', 5),
                    max_daily_trades=risk_config.get('max_daily_trades', 40),
                    max_drawdown_percent=risk_config.get('max_drawdown_percent',10.0),
                    max_risk_per_trade=risk_config.get('max_risk_per_trade', 2.0),
                    max_risk_per_day=risk_config.get('max_risk_per_day', 5.0),
                    max_risk_per_symbol=risk_config.get('max_risk_per_symbol', 10.0),
                    default_stop_loss=risk_config.get('default_stop_loss', 2.0),
                    default_take_profit=risk_config.get('default_take_profit', 4.0),
                    correlation_limit=risk_config.get('correlation_limit', 0.7),
                    night_trading=risk_config.get('night_trading', True),
                    weekend_trading=risk_config.get('weekend_trading', True),
                    account_size=risk_config.get('account_size', 10000.0),
                    params=risk_config.get('params', {})
                )
                logger.info("Risk manager initialized")
                
            # Initialize notifications
            notifications_config = self.config.get('notifications', {})
            if notifications_config and notifications_config.get('enabled', False):
                # Email notifier
                email_config = notifications_config.get('email', {})
                if email_config and email_config.get('enabled', False):
                    email_notifier = EmailNotifier(
                        smtp_server=email_config.get('smtp_server'),
                        smtp_port=email_config.get('smtp_port', 587),
                        username=email_config.get('username'),
                        password=email_config.get('password'),
                        sender=email_config.get('sender'),
                        recipients=email_config.get('recipients', [])
                    )
                    self.notifiers.append(email_notifier)
                    logger.info("Email notifier initialized")
                    
                # Telegram notifier
                telegram_config = notifications_config.get('telegram', {})
                if telegram_config and telegram_config.get('enabled', False):
                    telegram_notifier = TelegramNotifier(
                        token=telegram_config.get('token'),
                        chat_ids=telegram_config.get('chat_ids', [])
                    )
                    self.notifiers.append(telegram_notifier)
                    logger.info("Telegram notifier initialized")
                    
            # Initialize exchanges
            exchanges_config = self.config.get('exchanges', {})
            for exchange_id, exchange_config in exchanges_config.items():
                if not exchange_config.get('enabled', False):
                    continue
                    
                # Get API keys
                api_key = exchange_config.get('api_key', '')
                api_secret = exchange_config.get('api_secret', '')
                
                # Decrypt API keys if encrypted
                if exchange_config.get('encrypted', False):
                    api_key = decrypt_api_keys(api_key)
                    api_secret = decrypt_api_keys(api_secret)
                    
                # Create appropriate exchange connector
                if exchange_id == 'binance':
                    exchange = BinanceExchange(
                        api_key=api_key,
                        api_secret=api_secret,
                        sandbox=self.mode != 'production',
                        rate_limit=exchange_config.get('rate_limit', True),
                        timeout=exchange_config.get('timeout', 30000)
                    )
                elif exchange_id == 'coinbase':
                    exchange = CoinbaseExchange(
                        api_key=api_key,
                        api_secret=api_secret,
                        sandbox=self.mode != 'production',
                        rate_limit=exchange_config.get('rate_limit', True),
                        timeout=exchange_config.get('timeout', 30000)
                    )
                elif exchange_id == 'kraken':
                    exchange = KrakenExchange(
                        api_key=api_key,
                        api_secret=api_secret,
                        sandbox=self.mode != 'production',
                        rate_limit=exchange_config.get('rate_limit', True),
                        timeout=exchange_config.get('timeout', 30000)
                    )
                elif exchange_id == 'mexc':
                    exchange = MexcExchange(
                        api_key=api_key,
                        api_secret=api_secret,
                        sandbox=self.mode != 'production',
                        rate_limit=exchange_config.get('rate_limit', True),
                        timeout=exchange_config.get('timeout', 30000)
                    )
                elif exchange_id == 'bybit':
                    exchange = BybitExchange(
                        api_key=api_key,
                        api_secret=api_secret,
                        sandbox=self.mode != 'production',
                        rate_limit=exchange_config.get('rate_limit', True),
                        timeout=exchange_config.get('timeout', 30000)
                    )
                else:
                    logger.warning(f"Unsupported exchange: {exchange_id}")
                    continue
                    
                self.exchanges[exchange_id] = exchange
                logger.info(f"Initialized {exchange_id} exchange connector")
                
            # Initialize strategies
            strategies_config = self.config.get('strategies', {})
            logger.info(f"Found {len(strategies_config)} strategies in config: {list(strategies_config.keys())}")
            for strategy_id, strategy_config in strategies_config.items():
                if not strategy_config.get('enabled', False):
                    logger.info(f"Strategy {strategy_id} is disabled, skipping")
                    continue
                    
                strategy_type = strategy_config.get('type')
                symbols = strategy_config.get('symbols', [])
                timeframes = strategy_config.get('timeframes', ['1h'])
                params = strategy_config.get('params', {})
                
                logger.debug(f"Loading strategy {strategy_id} with type {strategy_type}, symbols {symbols}, timeframes {timeframes}")
                
                strategy_class = self._import_strategy_class(strategy_type)
                if not strategy_class:
                    logger.warning(f"Strategy type not found for {strategy_id}: {strategy_type}")
                    continue
                    
                try:
                    strategy = strategy_class(
                        symbols=symbols,
                        timeframes=timeframes,
                        risk_manager=self.risk_manager,
                        params=params
                    )
                    self.strategies[strategy_id] = strategy
                    logger.info(f"Initialized {strategy_id} strategy as {strategy_type} (class: {strategy.__class__.__name__})")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {strategy_id}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            logger.info("All components initialized successfully")
            logger.info(f"Loaded strategies: {list(self.strategies.keys())}")
                            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _import_strategy_class(self, strategy_type: str):
        """
        Import a strategy class by type string.
        
        Args:
            strategy_type: Strategy type identifier
            
        Returns:
            class: Strategy class, or None if not found
        """
        try:
            if strategy_type == 'MovingAverageCrossover':
                from cryptobot.strategies.moving_average import MovingAverageCrossover
                return MovingAverageCrossover
            elif strategy_type == 'RSI':
                from cryptobot.strategies.rsi import RSIStrategy
                return RSIStrategy
            elif strategy_type == 'BollingerBands':
                from cryptobot.strategies.bollinger_bands import BollingerBandsStrategy
                return BollingerBandsStrategy
            elif strategy_type == 'Custom':
                from cryptobot.strategies.custom import CustomStrategy
                return CustomStrategy
            elif strategy_type == 'MachineLearning':
                from cryptobot.strategies.machine_learning import MachineLearningStrategy
                return MachineLearningStrategy
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                return None
        except ImportError as e:
            logger.error(f"Error importing strategy class {strategy_type}: {str(e)}")
            return None
        
    async def train_ml_strategy(self, strategy_id: str) -> bool:
        """
        Train the machine learning model for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy to train
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        if not isinstance(strategy, MachineLearningStrategy):
            logger.error(f"Strategy {strategy_id} is not a MachineLearningStrategy")
            return False
            
        try:
            if not self.historical_data:
                logger.error("Historical data provider not initialized")
                return False

            # Fetch historical data for all symbols and timeframes
            lookback_period = strategy.params.get('lookback_period', 100)
            end_date = datetime.now()
            
            # Calculate the lookback period in days
            smallest_timeframe_seconds = None
            if strategy.timeframes:
                try:
                    # Convert all timeframes to seconds and find the smallest
                    valid_timeframes = []
                    for tf in strategy.timeframes:
                        try:
                            tf_seconds = timeframe_to_seconds(tf)
                            valid_timeframes.append(tf_seconds)
                        except ValueError as ve:
                            logger.warning(f"Invalid timeframe '{tf}' in strategy {strategy_id}: {ve}")
                            continue
                    if not valid_timeframes:
                        logger.error(f"No valid timeframes found for strategy {strategy_id}")
                        return False
                    smallest_timeframe = min(valid_timeframes)
                    # Estimate lookback in days (assuming smallest timeframe for safety)
                    lookback_days = (lookback_period * smallest_timeframe) / (24 * 3600)
                    lookback_days = max(int(lookback_days) + 1, 1)  # Ensure at least 1 day
                except Exception as e:
                    logger.error(f"Error calculating smallest timeframe for strategy {strategy_id}: {str(e)}")
                    return False
            else:
                logger.error(f"No timeframes defined for strategy {strategy_id}")
                return False

            start_date = end_date - timedelta(days=lookback_days)

            # Ensure all required data is available
            data_available = False
            for symbol in strategy.symbols:
                if symbol not in self.data:
                    self.data[symbol] = {}
                for timeframe in strategy.timeframes:
                    logger.info(f"Checking historical data for {symbol} {timeframe} from {start_date} to {end_date}")
                    
                    # First, check if we have recent CSV data
                    csv_data = await self.historical_data.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    fetch_from_api = False
                    if not csv_data.empty:
                        # Check if CSV data covers the required date range
                        if csv_data.index.min() <= start_date and csv_data.index.max() >= end_date - timedelta(hours=1):
                            # CSV data is sufficiently recent (within 1 hour of end_date)
                            logger.info(f"Using existing CSV data for {symbol} {timeframe}")
                            data = csv_data
                        else:
                            # CSV data is outdated or doesn't cover the full range
                            logger.info(f"Existing CSV data for {symbol} {timeframe} is outdated (covers {csv_data.index.min()} to {csv_data.index.max()})")
                            fetch_from_api = True
                    else:
                        # No CSV data available
                        logger.warning(f"No CSV data found for {symbol} {timeframe}")
                        fetch_from_api = True

                    if fetch_from_api:
                        # Fetch new data from API
                        logger.info(f"Attempting to download data for {symbol} {timeframe} from API")
                        data = await self.historical_data.download_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            save_to_csv=True
                        )
                        if data.empty:
                            logger.warning(f"No data downloaded from API for {symbol} {timeframe}")
                            continue
                    
                    # Verify required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    if missing_columns:
                        logger.warning(f"Missing required columns {missing_columns} in data for {symbol} {timeframe}")
                        continue
                    
                    # Ensure enough data points
                    if len(data) < lookback_period:
                        logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(data)} rows, required {lookback_period}")
                        continue
                    
                    # Store the most recent data and update strategy
                    self.data[symbol][timeframe] = data
                    strategy.update_data(symbol, timeframe, data)
                    data_available = True
                    logger.info(f"Updated data for {symbol} {timeframe} with {len(data)} rows")

            if not data_available:
                logger.error("No data available for any symbol/timeframe combination. Cannot train model.")
                return False

            # Force training by resetting last_train_time
            for symbol in strategy.last_train_time:
                for timeframe in strategy.last_train_time[symbol]:
                    strategy.last_train_time[symbol][timeframe] = None
            
            # Trigger indicator calculation and training
            for symbol in strategy.symbols:
                for timeframe in strategy.timeframes:
                    if symbol in self.data and timeframe in self.data[symbol]:
                        strategy.calculate_indicators(symbol, timeframe)
                    else:
                        logger.warning(f"Skipping training for {symbol} {timeframe}: No data available")

            logger.info(f"Successfully trained ML model for strategy {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error training ML model for strategy {strategy_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    async def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Trading engine already running")
            return True
            
        try:
            logger.info("Starting trading engine")
            
            # Connect to exchanges
            for exchange_id, exchange in self.exchanges.items():
                if not await exchange.connect():
                    logger.error(f"Failed to connect to {exchange_id}")
                    return False
                    
                # Get account balance
                balance = await exchange.fetch_balance()
                self.account_balances[exchange_id] = balance
                
                # Update risk manager with account balance
                if self.risk_manager:
                    total_balance = sum(float(asset['free']) for asset in balance['info']['balances']
                                     if asset['asset'] in ['USDT', 'BUSD', 'USD', 'USDC'])
                    self.risk_manager.update_account_balance(total_balance)
                    
                logger.info(f"Connected to {exchange_id}, balance retrieved")
                
            # Start strategies
            for strategy_id, strategy in self.strategies.items():
                strategy.start()
                logger.info(f"Started {strategy_id} strategy")
                
            # Set engine state
            self.is_running = True
            self.start_time = datetime.now()
            self.last_heartbeat = datetime.now()
            
            # Send notification
            await self._send_notification(
                "Trading Bot Started",
                f"Trading bot started successfully at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Start main loop if in production/test mode
            if self.mode in ['production', 'test']:
                asyncio.create_task(self._main_loop())
                
            logger.info("Trading engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    async def stop(self) -> bool:
        """
        Stop the trading engine.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Trading engine already stopped")
            return True
            
        try:
            logger.info("Stopping trading engine")
            
            # Stop strategies
            for strategy_id, strategy in self.strategies.items():
                strategy.stop()
                logger.info(f"Stopped {strategy_id} strategy")
                
            # Disconnect from exchanges
            for exchange_id, exchange in self.exchanges.items():
                await exchange.disconnect()
                logger.info(f"Disconnected from {exchange_id}")
                
            # Set engine state
            self.is_running = False
            stop_time = datetime.now()
            uptime = stop_time - self.start_time if self.start_time else timedelta(0)
            
            # Send notification
            await self._send_notification(
                "Trading Bot Stopped",
                f"Trading bot stopped at {stop_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Uptime: {uptime}"
            )
            
            logger.info(f"Trading engine stopped, uptime: {uptime}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    async def _main_loop(self):
        """Main trading loop."""
        try:
            logger.info("Starting main trading loop")
            
            while self.is_running:
                try:
                    # Update heartbeat
                    self.last_heartbeat = datetime.now()
                    
                    # Get data for all symbols and timeframes
                    for exchange_id, exchange in self.exchanges.items():
                        for strategy_id, strategy in self.strategies.items():
                            for symbol in strategy.symbols:
                                if symbol not in self.data:
                                    self.data[symbol] = {}
                                for timeframe in strategy.timeframes:
                                    # Fetch OHLCV data
                                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
                                    
                                    # Convert to DataFrame
                                    df = self.data_processor.ohlcv_to_dataframe(ohlcv)
                                    
                                    # Store in self.data
                                    self.data[symbol][timeframe] = df
                                    
                                    # Update strategy data
                                    strategy.update_data(symbol, timeframe, df)
                                    
                            # Generate signals
                            all_signals = []
                            for symbol in strategy.symbols:
                                for timeframe in strategy.timeframes:
                                    signal = strategy.generate_signals(symbol, timeframe)
                                    if signal and signal.get('action') is not None:
                                        all_signals.append(signal)
                                        
                            # Execute signals
                            if all_signals:
                                executed_trades = await strategy.execute_signals(all_signals, exchange)
                                
                                # Update risk manager with trade results
                                if self.risk_manager and executed_trades:
                                    for trade in executed_trades:
                                        self.risk_manager.update_after_trade(trade)
                                        
                                    # Update account balance
                                    balance = await exchange.fetch_balance()
                                    self.account_balances[exchange_id] = balance
                                    
                                    total_balance = sum(float(asset['free']) for asset in balance['info']['balances']
                                                    if asset['asset'] in ['USDT', 'BUSD', 'USD', 'USDC'])
                                    self.risk_manager.update_account_balance(total_balance)
                                    
                                # Send notifications for trades
                                for trade in executed_trades:
                                    await self.send_formatted_trade_notification(trade)
                        
                    # Check for anomalies
                    if self.risk_manager:
                        # Get current prices for all symbols
                        all_symbols = set()
                        for strategy in self.strategies.values():
                            all_symbols.update(strategy.symbols)
                            
                        current_prices = {}
                        volatility = {}
                        
                        for exchange_id, exchange in self.exchanges.items():
                            for symbol in all_symbols:
                                ticker = await exchange.fetch_ticker(symbol)
                                current_prices[symbol] = ticker['last']
                                volatility[symbol] = ticker['percentage'] if 'percentage' in ticker else 0
                                
                        # Check for market anomalies
                        self.risk_manager.check_anomalies(current_prices, volatility)
                        
                        # If kill switch activated, stop trading
                        if self.risk_manager.kill_switch_active:
                            logger.warning(f"Kill switch activated: {self.risk_manager.kill_switch_reason}")
                            await self._send_notification(
                                "Kill Switch Activated",
                                f"Trading stopped: {self.risk_manager.kill_switch_reason}"
                            )
                            await self.stop()
                            break
                            
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Delay before next iteration
                    await asyncio.sleep(self.config.get('loop_interval', 60))  # Default: 1 minute
                    
                except Exception as e:
                    self.last_error = str(e)
                    logger.error(f"Error in main loop: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Send error notification
                    await self.send_error_notification(
                        "Trading Bot Error",
                        f"Error in main loop: {str(e)}\n{traceback.format_exc()}"
                    )
                    
                    # Sleep and continue
                    await asyncio.sleep(10)
                    
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            await self.send_error_notification(
                "Trading Bot Fatal Error",
                f"Fatal error in main loop: {str(e)}\n{traceback.format_exc()}"
            )
            

    """
    Enhanced methods for the TradingEngine class to support the new functionality
    """

    async def run_backtest(
        self, 
        strategy_id: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str = '1h',
        return_full_results: bool = False
    ) -> Dict[str, Any]:
        """
        Run a backtest for a strategy with enhanced result handling.
        
        Args:
            strategy_id: Strategy identifier
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Timeframe for backtest
            return_full_results: Whether to return the full results including equity curve and trades
                
        Returns:
            dict: Backtest results including metrics and optionally full data
        """
        try:
            if not self.historical_data:
                logger.error("Historical data provider not initialized")
                return {}
                    
            if strategy_id not in self.strategies:
                logger.error(f"Strategy not found: {strategy_id}")
                return {}
                    
            strategy = self.strategies[strategy_id]
                
            # Create backtesting engine
            backtest_engine = BacktestingEngine(
                strategy=strategy,
                data_provider=self.historical_data,
                initial_balance=self.config.get('backtest', {}).get('initial_balance', 10000.0),
                maker_fee=self.config.get('backtest', {}).get('maker_fee', 0.001),
                taker_fee=self.config.get('backtest', {}).get('taker_fee', 0.002),
                slippage=self.config.get('backtest', {}).get('slippage', 0.001),
                enable_margin=self.config.get('backtest', {}).get('enable_margin', False),
                leverage=self.config.get('backtest', {}).get('leverage', 1.0),
                debug=self.config.get('backtest', {}).get('debug', False)
            )
                
            # Run backtest
            logger.info(f"Running backtest for {strategy_id} from {start_date} to {end_date}")
            metrics = await backtest_engine.run(start_date, end_date, timeframe)
                
            # Save results to files if needed
            save_results = self.config.get('backtest', {}).get('save_results', True)
            
            if save_results:
                # Save results
                results_dir = self.config.get('backtest', {}).get('results_dir', 'backtest_results')
                os.makedirs(results_dir, exist_ok=True)
                    
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backtest_engine.export_results(f"{results_dir}/{strategy_id}_{timestamp}_results.csv")
                backtest_engine.export_trade_history(f"{results_dir}/{strategy_id}_{timestamp}_trades.csv")
                    
                # Generate and save plots if requested
                if self.config.get('backtest', {}).get('generate_plots', True):
                    try:
                        backtest_engine.plot_results(
                            benchmark=True, 
                            save_path=f"{results_dir}/{strategy_id}_{timestamp}_plot.png"
                        )
                    except Exception as plot_e:
                        logger.error(f"Error generating plot: {str(plot_e)}")
            
            # Return the results
            if return_full_results:
                results = {
                    'metrics': metrics,
                    'equity_curve': backtest_engine.get_results(),
                    'trades': backtest_engine.get_trade_history()
                }
                # Ensure equity_curve is a DataFrame with datetime index
                if isinstance(results['equity_curve'], dict):
                    equity_df = pd.DataFrame(results['equity_curve'])
                    if 'timestamp' in equity_df.columns:
                        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                        equity_df.set_index('timestamp', inplace=True)
                    results['equity_curve'] = equity_df
                elif isinstance(results['equity_curve'], pd.DataFrame) and not isinstance(results['equity_curve'].index, pd.DatetimeIndex):
                    if 'timestamp' in results['equity_curve'].columns:
                        results['equity_curve']['timestamp'] = pd.to_datetime(results['equity_curve']['timestamp'])
                        results['equity_curve'].set_index('timestamp', inplace=True)
                return results
            else:
                return metrics
                
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    async def execute_manual_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a manual trading signal, such as closing a position.
        
        Args:
            signal: Signal details including symbol, action, price, etc.
                
        Returns:
            dict: Result of signal execution
        """
        try:
            if not self.is_running:
                return {'success': False, 'error': 'Trading engine is not running'}
                
            # Extract signal components
            symbol = signal.get('symbol')
            action = signal.get('action')
            strategy_id = signal.get('strategy')
            
            if not symbol or not action:
                return {'success': False, 'error': 'Invalid signal format - missing symbol or action'}
                
            # Find the appropriate strategy
            strategy = None
            if strategy_id:
                strategy = self.strategies.get(strategy_id)
            else:
                # Find first strategy that has this symbol
                for strat in self.strategies.values():
                    if symbol in strat.symbols:
                        strategy = strat
                        break
                        
            if not strategy:
                return {'success': False, 'error': f'No strategy found for {symbol}'}
                
            # Find the appropriate exchange
            exchange = next(iter(self.exchanges.values()))  # Default to first exchange
            
            # Update signal with any missing fields
            if 'price' not in signal:
                # Fetch current price from exchange
                ticker = await exchange.fetch_ticker(symbol)
                signal['price'] = ticker['last']
                
            # Execute the signal
            if action == 'close':
                # Closing a position
                for position_symbol, position in strategy.positions.items():
                    if position_symbol == symbol and position.get('is_active', False):
                        # Execute close order based on position side
                        side = 'sell' if position.get('side') == 'long' else 'buy'
                        amount = position.get('amount', 0)
                        price = signal.get('price')
                        
                        try:
                            # Create market order to close position
                            order = await exchange.create_order(
                                symbol=symbol,
                                order_type='market',
                                side=side,
                                amount=amount,
                                price=price
                            )
                            
                            # Calculate PnL
                            entry_price = position.get('entry_price', 0)
                            
                            if position.get('side') == 'long':
                                pnl = (price - entry_price) * amount
                                pnl_percent = ((price / entry_price) - 1) * 100
                            else:  # short
                                pnl = (entry_price - price) * amount
                                pnl_percent = ((entry_price / price) - 1) * 100
                                
                            # Create trade record
                            trade = Trade(
                                id=order.get('id', str(uuid.uuid4())),
                                symbol=symbol,
                                side=side,
                                amount=amount,
                                price=price,
                                timestamp=datetime.now(),
                                strategy=strategy.name,
                                timeframe='manual',
                                status='executed',
                                pnl=pnl,
                                pnl_percent=pnl_percent,
                                related_trade_id=position.get('order_id')
                            )
                            
                            # Add trade to strategy history
                            strategy.trade_history.append(trade)
                            
                            # Reset position
                            strategy.positions[symbol] = {
                                'is_active': False,
                                'side': None,
                                'entry_price': None,
                                'amount': None,
                                'entry_time': None,
                                'order_id': None
                            }
                            
                            # Update risk manager
                            if self.risk_manager:
                                self.risk_manager.update_after_trade(trade)
                                
                            # Send notification
                            await self.send_formatted_trade_notification(trade)
                            
                            return {
                                'success': True,
                                'order': order,
                                'pnl': pnl,
                                'pnl_percent': pnl_percent
                            }
                            
                        except Exception as e:
                            logger.error(f"Error executing manual close for {symbol}: {str(e)}")
                            return {
                                'success': False,
                                'error': str(e)
                            }
                            
                # If we get here, no matching active position was found
                return {
                    'success': False,
                    'error': f'No active position found for {symbol}'
                }
                
            else:
                # Other signal types (not implemented yet)
                return {
                    'success': False,
                    'error': f'Manual signal action "{action}" not implemented'
                }
                
        except Exception as e:
            logger.error(f"Error executing manual signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def train_ml_strategy(self, strategy_id: str) -> bool:
        """
        Train a MachineLearningStrategy model with historical data.
        
        Args:
            strategy_id: Strategy identifier
                
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Check if strategy exists and is a MachineLearningStrategy
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            strategy = self.strategies[strategy_id]
            if not hasattr(strategy, 'models') or not isinstance(strategy, MachineLearningStrategy):
                logger.error(f"Strategy {strategy_id} is not a MachineLearningStrategy")
                return False
                
            # Load historical data for this strategy
            if not self.historical_data:
                logger.error("Historical data provider not initialized")
                return False
                
            logger.info(f"Training ML models for strategy {strategy_id}")
            
            # For each symbol and timeframe, load historical data and train
            for symbol in strategy.symbols:
                for timeframe in strategy.timeframes:
                    # Load historical data (last 1000 candles should be enough for training)
                    end_date = datetime.now()
                    
                    # Calculate start date based on timeframe
                    if timeframe.endswith('m'):
                        minutes = int(timeframe[:-1])
                        start_date = end_date - timedelta(minutes=minutes * 1000)
                    elif timeframe.endswith('h'):
                        hours = int(timeframe[:-1])
                        start_date = end_date - timedelta(hours=hours * 1000)
                    elif timeframe.endswith('d'):
                        days = int(timeframe[:-1])
                        start_date = end_date - timedelta(days=days * 1000)
                    else:
                        # Default to 100 days
                        start_date = end_date - timedelta(days=100)
                    
                    logger.info(f"Loading historical data for {symbol} {timeframe} from {start_date} to {end_date}")
                    
                    # Get historical data
                    df = await self.historical_data.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df.empty:
                        logger.warning(f"No historical data available for {symbol} {timeframe}")
                        continue
                        
                    # Update strategy with this data
                    strategy.update_data(symbol, timeframe, df)
                    
                    # Force model training
                    logger.info(f"Training model for {symbol} {timeframe}")
                    success = strategy._train_model(symbol, timeframe)
                    
                    if success:
                        logger.info(f"Successfully trained model for {symbol} {timeframe}")
                    else:
                        logger.error(f"Failed to train model for {symbol} {timeframe}")
                        return False
                        
            logger.info(f"Successfully trained all models for strategy {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML strategy {strategy_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current trading engine status.
        
        Returns:
            dict: Status information
        """
        status = {
            'mode': self.mode,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'last_error': self.last_error,
            'exchanges': {},
            'strategies': {},
            'account_balances': self.account_balances,
            'performance': self.performance,
            'risk_management': self.risk_manager.to_dict() if self.risk_manager else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get exchange status
        for exchange_id, exchange in self.exchanges.items():
            status['exchanges'][exchange_id] = exchange.get_exchange_status()
            
        # Get strategy status
        for strategy_id, strategy in self.strategies.items():
            status['strategies'][strategy_id] = strategy.to_dict()
            
        return status
        
    def _update_performance_metrics(self):
        """Update performance metrics for all strategies."""
        try:
            for strategy_id, strategy in self.strategies.items():
                self.performance[strategy_id] = strategy.get_performance_stats()
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            
    async def _send_notification(self, title: str, message: str, level: str = 'info'):
        """
        Send notification to all registered notifiers.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level ('info', 'warning', 'error')
        """
        try:
            for notifier in self.notifiers:
                await notifier.send(title, message, level)
                
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            
    async def send_formatted_trade_notification(self, trade):
        """
        Send notification for a trade.
        
        Args:
            trade: Trade object
        """
        try:
            # Format trade details
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                title = f"Trade Executed: {trade.side.upper()} {trade.symbol}"
                message = (
                    f"Symbol: {trade.symbol}\n"
                    f"Side: {trade.side.upper()}\n"
                    f"Price: {trade.price:.8f}\n"
                    f"Amount: {trade.amount:.8f}\n"
                    f"PnL: {trade.pnl:.2f} ({trade.pnl_percent:.2f}%)\n"
                    f"Strategy: {trade.strategy}\n"
                    f"Time: {trade.timestamp}"
                )
                level = 'info' if trade.pnl >= 0 else 'warning'
            else:
                title = f"New Position: {trade.side.upper()} {trade.symbol}"
                message = (
                    f"Symbol: {trade.symbol}\n"
                    f"Side: {trade.side.upper()}\n"
                    f"Price: {trade.price:.8f}\n"
                    f"Amount: {trade.amount:.8f}\n"
                    f"Strategy: {trade.strategy}\n"
                    f"Time: {trade.timestamp}"
                )
                level = 'info'
                
            await self._send_notification(title, message, level)
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")


    """
    Enhanced notification system for the trading bot.
    """

    async def send_formatted_trade_notification(
        self, 
        trade: Trade, 
        additional_info: Dict[str, Any] = None,
        include_screenshot: bool = False
    ) -> bool:
        """
        Send a well-formatted trade notification with enhanced information.
        
        Args:
            trade: Trade object
            additional_info: Additional information to include
            include_screenshot: Whether to include a chart screenshot
                
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            if not self.notifiers or not trade:
                return False
                
            # Determine if it's a position open or close
            is_close = trade.related_trade_id is not None
            action = "CLOSED" if is_close else "OPENED"
            
            # Determine the outcome for closed positions
            outcome = ""
            if is_close and trade.pnl is not None:
                outcome = "PROFIT" if trade.pnl > 0 else "LOSS"
                
            # Format the base message
            symbol = trade.symbol
            side = trade.side.upper()
            price = trade.price
            amount = trade.amount
            timestamp = trade.timestamp
            strategy = trade.strategy
            
            # Create emoji indicators
            action_emoji = "🔴" if side == "SELL" else "🟢"
            result_emoji = ""
            if is_close:
                result_emoji = "✅" if trade.pnl and trade.pnl > 0 else "❌"
                
            # Build the title
            title = f"{action_emoji} {symbol} {side} {action} {result_emoji}"
            
            # Build the message body
            message = (
                f"📊 *Trade Details*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Action: {action}\n"
                f"Price: {price:.8f}\n"
                f"Amount: {amount:.8f}\n"
                f"Value: ${price * amount:.2f}\n"
                f"Strategy: {strategy}\n"
                f"Time: {timestamp}\n"
            )
            
            # Add P/L information for closed trades
            if is_close and trade.pnl is not None:
                message += (
                    f"\n💰 *P/L Information*\n"
                    f"P/L: ${trade.pnl:.2f} ({trade.pnl_percent:+.2f}%)\n"
                    f"Result: {outcome}\n"
                )
                
            # Add any additional info
            if additional_info:
                message += "\n📝 *Additional Information*\n"
                for key, value in additional_info.items():
                    message += f"{key}: {value}\n"
                    
            # Add account summary if available
            try:
                account_info = await self.get_account_summary()
                if account_info:
                    message += (
                        f"\n💼 *Account Summary*\n"
                        f"Balance: ${account_info.get('balance', 0):.2f}\n"
                        f"Open Positions: {account_info.get('open_positions', 0)}\n"
                        f"Today's P/L: ${account_info.get('daily_pnl', 0):+.2f}\n"
                    )
            except:
                # Skip account summary if it fails
                pass
                
            # Determine notification level
            level = "info"
            if is_close:
                level = "info" if trade.pnl and trade.pnl > 0 else "warning"
                
            # Send the notification
            success = True
            
            for notifier in self.notifiers:
                # Special handling for Telegram to include chart screenshot
                if include_screenshot and hasattr(notifier, 'send_photo') and callable(notifier.send_photo):
                    # Try to generate and send a chart screenshot
                    try:
                        chart_path = await self.generate_chart_screenshot(symbol, timeframe="1h")
                        if chart_path:
                            # Send message with photo
                            for chat_id in notifier.chat_ids:
                                await notifier.send_photo(chat_id, chart_path, message)
                        else:
                            # Fall back to regular message
                            await notifier.send(title, message, level)
                    except:
                        # Fall back to regular message
                        await notifier.send(title, message, level)
                else:
                    # Regular notification
                    notifier_success = await notifier.send(title, message, level)
                    success = success and notifier_success
                    
            return success
            
        except Exception as e:
            logger.error(f"Error sending formatted trade notification: {str(e)}")
            return False

    async def generate_chart_screenshot(self, symbol: str, timeframe: str = "1h") -> Optional[str]:
        """
        Generate a chart screenshot for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
                
        Returns:
            str: Path to the screenshot file, or None if failed
        """
        try:
            # This is a placeholder - in a real implementation, this would use
            # a library like selenium or an external chart API to generate a chart
            
            # For now, we'll just create a simple matplotlib chart
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            
            # Get historical data
            if not self.historical_data:
                return None
                
            # Get recent data (last 100 candles)
            end_date = datetime.now()
            
            # Calculate start date based on timeframe
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                start_date = end_date - timedelta(minutes=minutes * 100)
            elif timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                start_date = end_date - timedelta(hours=hours * 100)
            elif timeframe.endswith('d'):
                days = int(timeframe[:-1])
                start_date = end_date - timedelta(days=days * 100)
            else:
                # Default to 30 days
                start_date = end_date - timedelta(days=30)
            
            # Get data
            df = await self.historical_data.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return None
                
            # Create a simple candlestick chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            
            # Plot OHLC data
            width = 0.6
            width2 = 0.1
            
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Wicks
            ax.vlines(up.index, up.low, up.high, color='green', linewidth=1)
            ax.vlines(down.index, down.low, down.high, color='red', linewidth=1)
            
            # Candle bodies
            ax.bar(up.index, up.close-up.open, width, bottom=up.open, color='green', alpha=0.7)
            ax.bar(down.index, down.close-down.open, width, bottom=down.open, color='red', alpha=0.7)
            
            # Set title and labels
            ax.set_title(f'{symbol} - {timeframe} Chart')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            # Add current price line
            current_price = df['close'].iloc[-1]
            ax.axhline(y=current_price, color='blue', linestyle='-', alpha=0.6)
            ax.text(df.index[-1], current_price, f' {current_price:.2f}', 
                    verticalalignment='center', color='blue')
            
            # Set y-limits to focus on recent price action
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_range = recent_high - recent_low
            ax.set_ylim(recent_low - price_range*0.1, recent_high + price_range*0.1)
            
            # Save the chart
            chart_dir = 'charts'
            if not os.path.exists(chart_dir):
                os.makedirs(chart_dir)
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"{chart_dir}/{symbol.replace('/', '')}_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close(fig)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating chart screenshot: {str(e)}")
            return None

    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the account status.
                
        Returns:
            dict: Account summary information
        """
        try:
            summary = {
                'balance': 0.0,
                'open_positions': 0,
                'daily_pnl': 0.0,
                'equity': 0.0
            }
            
            # Get total balance from all exchanges
            for exchange_id, exchange in self.exchanges.items():
                balance = await exchange.fetch_balance()
                
                # Calculate total in USD (simplified)
                if 'total' in balance and 'USDT' in balance['total']:
                    summary['balance'] += balance['total']['USDT']
                elif 'total' in balance and 'USD' in balance['total']:
                    summary['balance'] += balance['total']['USD']
                    
                # Add other stablecoins if available
                for stable in ['USDC', 'BUSD', 'DAI', 'TUSD']:
                    if 'total' in balance and stable in balance['total']:
                        summary['balance'] += balance['total'][stable]
            
            # Count open positions
            for strategy_id, strategy in self.strategies.items():
                for symbol, position in strategy.positions.items():
                    if position.get('is_active', False):
                        summary['open_positions'] += 1
                        
                        # Calculate unrealized PnL
                        side = position.get('side')
                        entry_price = position.get('entry_price', 0)
                        amount = position.get('amount', 0)
                        
                        if side and entry_price and amount:
                            # Get current price
                            for exchange_id, exchange in self.exchanges.items():
                                try:
                                    ticker = await exchange.fetch_ticker(symbol)
                                    current_price = ticker['last']
                                    
                                    # Calculate unrealized PnL
                                    if side == 'long':
                                        unrealized_pnl = (current_price - entry_price) * amount
                                    else:  # short
                                        unrealized_pnl = (entry_price - current_price) * amount
                                        
                                    # Add to daily PnL
                                    summary['daily_pnl'] += unrealized_pnl
                                    break
                                except:
                                    continue
            
            # Calculate today's realized PnL from closed trades
            today = datetime.now().date()
            
            for strategy_id, strategy in self.strategies.items():
                for trade in strategy.trade_history:
                    # Check if trade was closed today and has PnL
                    if (isinstance(trade.timestamp, datetime) and 
                        trade.timestamp.date() == today and 
                        trade.pnl is not None):
                        summary['daily_pnl'] += trade.pnl
            
            # Calculate equity (balance + unrealized PnL)
            summary['equity'] = summary['balance'] + summary['daily_pnl']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {
                'balance': 0.0,
                'open_positions': 0,
                'daily_pnl': 0.0,
                'equity': 0.0,
                'error': str(e)
            }

    async def send_status_update(self, include_positions: bool = True, include_trades: bool = True) -> bool:
        """
        Send a periodic status update with current account information.
                
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            if not self.notifiers:
                return False
                
            # Get account summary
            summary = await self.get_account_summary()
            
            # Create the message
            title = "📊 CryptoBot Status Update"
            
            message = (
                f"*CryptoBot Status Report*\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"*Account Summary*\n"
                f"Balance: ${summary.get('balance', 0):.2f}\n"
                f"Equity: ${summary.get('equity', 0):.2f}\n"
                f"Open Positions: {summary.get('open_positions', 0)}\n"
                f"Today's P/L: ${summary.get('daily_pnl', 0):+.2f}\n"
            )
            
            # Add active positions if requested
            if include_positions and summary.get('open_positions', 0) > 0:
                message += "\n*Open Positions*\n"
                
                position_details = []
                
                for strategy_id, strategy in self.strategies.items():
                    for symbol, position in strategy.positions.items():
                        if position.get('is_active', False):
                            side = position.get('side', 'unknown')
                            entry_price = position.get('entry_price', 0)
                            amount = position.get('amount', 0)
                            
                            # Get current price and calculate PnL if possible
                            pnl_text = ""
                            
                            for exchange_id, exchange in self.exchanges.items():
                                try:
                                    ticker = await exchange.fetch_ticker(symbol)
                                    current_price = ticker['last']
                                    
                                    # Calculate unrealized PnL
                                    if side == 'long':
                                        unrealized_pnl = (current_price - entry_price) * amount
                                        pnl_pct = ((current_price / entry_price) - 1) * 100
                                    else:  # short
                                        unrealized_pnl = (entry_price - current_price) * amount
                                        pnl_pct = ((entry_price / current_price) - 1) * 100
                                        
                                    pnl_text = f" | P/L: ${unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)"
                                    break
                                except:
                                    continue
                            
                            position_details.append(
                                f"• {symbol} {side.upper()}: {amount:.4f} @ {entry_price:.4f}{pnl_text}"
                            )
                            
                message += "\n".join(position_details) + "\n"
            
            # Add recent trades if requested
            if include_trades:
                # Collect today's trades
                today = datetime.now().date()
                today_trades = []
                
                for strategy_id, strategy in self.strategies.items():
                    for trade in strategy.trade_history:
                        if isinstance(trade.timestamp, datetime) and trade.timestamp.date() == today:
                            today_trades.append(trade)
                
                # Sort by timestamp (newest first)
                today_trades.sort(key=lambda x: x.timestamp, reverse=True)
                
                # Add recent trades to message
                if today_trades:
                    message += f"\n*Recent Trades (Today)*\n"
                    
                    # Show up to 5 most recent trades
                    for i, trade in enumerate(today_trades[:5]):
                        side = trade.side.upper()
                        action = "CLOSED" if trade.related_trade_id else "OPENED"
                        
                        # Add PnL info for closed trades
                        pnl_text = ""
                        if trade.pnl is not None:
                            pnl_text = f" | P/L: ${trade.pnl:+.2f} ({trade.pnl_percent:+.2f}%)"
                            
                        message += f"• {trade.symbol} {side} {action}{pnl_text}\n"
                        
                    if len(today_trades) > 5:
                        message += f"... and {len(today_trades) - 5} more today\n"
            
            # Add active strategies
            active_strategies = [s_id for s_id, strategy in self.strategies.items() if strategy.is_active]
            if active_strategies:
                message += f"\n*Active Strategies*\n"
                message += ", ".join(active_strategies) + "\n"
                
            # Add system info
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds // 60) % 60
            
            message += (
                f"\n*System Information*\n"
                f"Running: {self.is_running}\n"
                f"Mode: {self.mode}\n"
                f"Uptime: {days}d {hours}h {minutes}m\n"
            )
            
            # Send notification
            success = True
            for notifier in self.notifiers:
                notifier_success = await notifier.send(title, message, "info")
                success = success and notifier_success
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending status update: {str(e)}")
            return False

    async def send_error_notification(
        self, 
        error_message: str, 
        error_type: str = "Error", 
        component: str = "System",
        include_stack_trace: bool = True
    ) -> bool:
        """
        Send an error notification.
        
        Args:
            error_message: Error message
            error_type: Type of error
            component: Component where error occurred
            include_stack_trace: Whether to include the stack trace
                
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            if not self.notifiers:
                return False
                
            # Capture stack trace if requested
            stack_trace = ""
            if include_stack_trace:
                import traceback
                stack_trace = "\n```\n" + traceback.format_exc() + "\n```"
                
            # Create the message
            title = f"🚨 CryptoBot {error_type} in {component}"
            
            message = (
                f"*Error Details*\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Component: {component}\n"
                f"Type: {error_type}\n"
                f"Message: {error_message}"
            )
            
            if stack_trace:
                message += f"\n\n*Stack Trace*{stack_trace}"
                
            # Send notification
            success = True
            for notifier in self.notifiers:
                notifier_success = await notifier.send(title, message, "error")
                success = success and notifier_success
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")
            return False

    async def send_risk_alert(
        self, 
        alert_type: str, 
        message: str, 
        severity: str = "warning",
        affected_symbols: List[str] = None
    ) -> bool:
        """
        Send a risk management alert.
        
        Args:
            alert_type: Type of risk alert
            message: Alert message
            severity: Alert severity (info, warning, error)
            affected_symbols: List of affected symbols
                
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            if not self.notifiers:
                return False
                
            # Create the message
            emoji_map = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "🚨"
            }
            
            emoji = emoji_map.get(severity, "⚠️")
            title = f"{emoji} Risk Alert: {alert_type}"
            
            alert_message = (
                f"*Risk Management Alert*\n"
                f"Type: {alert_type}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Severity: {severity.upper()}\n\n"
                f"*Details*\n{message}"
            )
            
            if affected_symbols:
                alert_message += f"\n\n*Affected Symbols*\n{', '.join(affected_symbols)}"
                
            # Add risk manager state if available
            if self.risk_manager:
                risk_state = self.risk_manager.to_dict()
                
                # Format the state in a readable way
                risk_state_text = (
                    f"\n\n*Risk Manager State*\n"
                    f"Daily trades: {risk_state.get('daily_trades_count', 0)}/{risk_state.get('max_daily_trades', 0)}\n"
                    f"Current drawdown: {risk_state.get('current_drawdown', 0):.2f}%\n"
                    f"Max drawdown limit: {risk_state.get('max_drawdown_percent', 0):.2f}%\n"
                    f"Kill switch active: {risk_state.get('kill_switch_active', False)}\n"
                )
                
                if risk_state.get('kill_switch_active', False):
                    risk_state_text += f"Kill switch reason: {risk_state.get('kill_switch_reason', 'Unknown')}\n"
                    
                alert_message += risk_state_text
            
            # Send notification
            success = True
            for notifier in self.notifiers:
                notifier_success = await notifier.send(title, alert_message, severity)
                success = success and notifier_success
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {str(e)}")
            return False