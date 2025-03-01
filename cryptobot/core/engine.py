"""
Core Trading Engine
=================
Main engine that orchestrates all the components of the trading bot.
"""

import asyncio
import time
import os
import json
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
from cryptobot.risk_management.manager import RiskManager
from cryptobot.data.database import DatabaseManager
from cryptobot.data.processor import DataProcessor
from cryptobot.data.historical import HistoricalDataProvider
from cryptobot.notifications.email import EmailNotifier
from cryptobot.notifications.telegram import TelegramNotifier
from cryptobot.backtesting.engine import BacktestingEngine
from cryptobot.utils.helpers import setup_logger
from cryptobot.config.settings import load_config, save_config
from cryptobot.security.encryption import decrypt_api_keys


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
                    max_daily_trades=risk_config.get('max_daily_trades', 20),
                    max_drawdown_percent=risk_config.get('max_drawdown_percent', 20.0),
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
            for strategy_id, strategy_config in strategies_config.items():
                if not strategy_config.get('enabled', False):
                    continue
                    
                # Get strategy parameters
                strategy_type = strategy_config.get('type')
                symbols = strategy_config.get('symbols', [])
                timeframes = strategy_config.get('timeframes', ['1h'])
                params = strategy_config.get('params', {})
                
                # Import the strategy class
                strategy_class = self._import_strategy_class(strategy_type)
                if not strategy_class:
                    logger.warning(f"Strategy type not found: {strategy_type}")
                    continue
                    
                # Create the strategy instance
                strategy = strategy_class(
                    symbols=symbols,
                    timeframes=timeframes,
                    risk_manager=self.risk_manager,
                    params=params
                )
                
                self.strategies[strategy_id] = strategy
                logger.info(f"Initialized {strategy_id} strategy")
                
            logger.info("All components initialized successfully")
            
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
                                for timeframe in strategy.timeframes:
                                    # Fetch OHLCV data
                                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
                                    
                                    # Convert to DataFrame
                                    df = self.data_processor.ohlcv_to_dataframe(ohlcv)
                                    
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
                                    await self._send_trade_notification(trade)
                        
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
                    await self._send_notification(
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
            
            await self._send_notification(
                "Trading Bot Fatal Error",
                f"Fatal error in main loop: {str(e)}\n{traceback.format_exc()}"
            )
            
    async def run_backtest(
        self, 
        strategy_id: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str = '1h'
    ) -> Dict[str, Any]:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Timeframe for backtest
            
        Returns:
            dict: Backtest results
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
            
            # Save results
            results_dir = self.config.get('backtest', {}).get('results_dir', 'backtest_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_engine.export_results(f"{results_dir}/{strategy_id}_{timestamp}_results.csv")
            backtest_engine.export_trade_history(f"{results_dir}/{strategy_id}_{timestamp}_trades.csv")
            
            # Optionally, generate and save plots
            if self.config.get('backtest', {}).get('generate_plots', True):
                backtest_engine.plot_results(
                    benchmark=True, 
                    save_path=f"{results_dir}/{strategy_id}_{timestamp}_plot.png"
                )
                
            logger.info(f"Backtest completed for {strategy_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
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
            
    async def _send_trade_notification(self, trade):
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