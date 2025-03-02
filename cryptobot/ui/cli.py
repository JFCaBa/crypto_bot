"""
Command Line Interface
====================
Command-line interface for the trading bot.
"""

import os
import sys
import cmd
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from loguru import logger

from cryptobot.core.engine import TradingEngine
from cryptobot.utils.helpers import setup_logger
from cryptobot.config.settings import load_config, save_config
# Import the MachineLearningStrategy to check its type
from cryptobot.strategies.machine_learning import MachineLearningStrategy


class CryptoBotCLI(cmd.Cmd):
    """
    Interactive command-line interface for the trading bot.
    """
    
    intro = """
    ================================================
     CryptoBot - Cryptocurrency Trading Bot
    ================================================
    
    Type 'help' or '?' to list commands.
    Type 'start' to start the trading bot.
    Type 'exit' to quit.
    """
    
    prompt = 'cryptobot> '
    
    def __init__(self, engine: TradingEngine):
        """
        Initialize the CLI.
        
        Args:
            engine: Trading engine instance
        """
        super().__init__()
        self.engine = engine
        self.config_path = engine.config_path
        
    # Helper method to get strategies from engine status
    def _get_strategies(self) -> Dict[str, Any]:
        """
        Fetch strategies from the engine's status.
        
        Returns:
            dict: Strategies data
        """
        status = asyncio.run(self.engine.get_status())
        return status.get('strategies', {})
    
    def do_start(self, arg):
        """Start the trading bot."""
        if self.engine.is_running:
            print("Trading bot is already running")
            return
            
        print("Starting trading bot...")
        asyncio.run(self.engine.start())
        
    def do_stop(self, arg):
        """Stop the trading bot."""
        if not self.engine.is_running:
            print("Trading bot is not running")
            return
            
        print("Stopping trading bot...")
        asyncio.run(self.engine.stop())
        
    def do_status(self, arg):
        """Show current bot status."""
        status = asyncio.run(self.engine.get_status())
        
        print("\nBot Status:")
        print("===========")
        print(f"Running: {status.get('is_running', False)}")
        print(f"Mode: {status.get('mode', 'unknown')}")
        
        print("\nStrategies:")
        print("===========")
        for strategy_id, strategy_data in status.get('strategies', {}).items():
            print(f"{strategy_id}:")
            print(f"  Type: {strategy_data.get('name', 'Unknown')}")
            print(f"  Active: {strategy_data.get('is_active', False)}")
            print(f"  Symbols: {', '.join(strategy_data.get('symbols', []))}")
            print(f"  Timeframes: {', '.join(strategy_data.get('timeframes', []))}")
            # Add last training time for ML strategies
            if strategy_data.get('name') == "MachineLearningStrategy":
                last_train_time = strategy_data.get('last_train_time', {})
                if last_train_time:
                    print("  Last Training Times:")
                    for symbol, tf_data in last_train_time.items():
                        for tf, train_time in tf_data.items():
                            print(f"    {symbol} ({tf}): {train_time if train_time else 'Never trained'}")
                else:
                    print(f"  Last Training Time: Not available")
            print()

        print("\nExchanges:")
        print("===========")
        for exchange_id, exchange_data in status.get('exchanges', {}).items():
            print(f"{exchange_id}: Connected = {exchange_data.get('connected', False)}")
        
    def do_balance(self, arg):
        """Show account balances."""
        status = asyncio.run(self.engine.get_status())
        
        print("\nAccount Balances:")
        print("=================")
        
        for exchange_id, balance in status.get('account_balances', {}).items():
            print(f"\n{exchange_id.upper()}:")
            
            if 'info' in balance and 'balances' in balance['info']:
                # For Binance
                balances = balance['info']['balances']
                for asset in balances:
                    if float(asset['free']) > 0 or float(asset['locked']) > 0:
                        print(f"  {asset['asset']}: Free = {asset['free']}, Locked = {asset['locked']}")
            else:
                # Generic format
                for currency, data in balance.items():
                    if isinstance(data, dict) and ('free' in data or 'total' in data):
                        free = data.get('free', 0)
                        used = data.get('used', 0)
                        total = data.get('total', 0)
                        
                        if float(free) > 0 or float(used) > 0:
                            print(f"  {currency}: Free = {free}, Used = {used}, Total = {total}")
        
    def do_positions(self, arg):
        """Show active positions."""
        status = asyncio.run(self.engine.get_status())
        
        print("\nActive Positions:")
        print("================")
        
        for strategy_id, strategy_data in status.get('strategies', {}).items():
            positions = strategy_data.get('positions', {})
            active_positions = {symbol: pos for symbol, pos in positions.items() if pos.get('is_active', False)}
            
            if active_positions:
                print(f"\nStrategy: {strategy_id}")
                for symbol, position in active_positions.items():
                    side = position.get('side', 'unknown')
                    entry_price = position.get('entry_price', 0)
                    amount = position.get('amount', 0)
                    entry_time = position.get('entry_time', '')
                    
                    print(f"  {symbol}: {side.upper()} {amount} @ {entry_price} (entered: {entry_time})")
            
    def do_trades(self, arg):
        """Show recent trades."""
        status = asyncio.run(self.engine.get_status())
        
        print("\nRecent Trades:")
        print("=============")
        
        for strategy_id, strategy_data in status.get('strategies', {}).items():
            performance = strategy_data.get('performance', {})
            trade_count = performance.get('total_trades', 0)
            
            if trade_count > 0:
                print(f"\nStrategy: {strategy_id}")
                print(f"  Total Trades: {trade_count}")
                print(f"  Win Rate: {performance.get('win_rate', 0):.2f}%")
                print(f"  Avg Profit: {performance.get('avg_profit', 0):.2f}%")
                print(f"  Total PnL: {performance.get('total_pnl', 0):.2f}")
        
    def do_strategies(self, arg):
        """List or manage strategies."""
        args = arg.split()
        
        if not args:
            # List all strategies
            status = asyncio.run(self.engine.get_status())
            
            print("\nAvailable Strategies:")
            print("====================")
            
            for strategy_id, strategy_data in status.get('strategies', {}).items():
                active = "ACTIVE" if strategy_data.get('is_active', False) else "INACTIVE"
                print(f"  {strategy_id} ({active})")
                print(f"    Type: {strategy_data.get('name', 'Unknown')}")
                print(f"    Symbols: {', '.join(strategy_data.get('symbols', []))}")
                print(f"    Timeframes: {', '.join(strategy_data.get('timeframes', []))}")
                print()
        else:
            # Manage a specific strategy
            command = args[0]
            strategy_id = args[1] if len(args) > 1 else None
            
            if not strategy_id:
                print("Strategy ID is required")
                return
                
            config = load_config(self.config_path)
            strategies = config.get('strategies', {})
            
            if strategy_id not in strategies:
                print(f"Strategy {strategy_id} not found")
                return
                
            if command == 'enable':
                strategies[strategy_id]['enabled'] = True
                save_config(config, self.config_path)
                print(f"Strategy {strategy_id} enabled")
            elif command == 'disable':
                strategies[strategy_id]['enabled'] = False
                save_config(config, self.config_path)
                print(f"Strategy {strategy_id} disabled")
            else:
                print(f"Unknown command: {command}")
        
    def do_markets(self, arg):
        """Show available markets."""
        status = asyncio.run(self.engine.get_status())
        
        print("\nAvailable Markets:")
        print("=================")
        
        for exchange_id, exchange_data in status.get('exchanges', {}).items():
            print(f"\n{exchange_id.upper()}:")
            
            # In a real implementation, we would get symbols from the exchange
            # For now, just show some examples
            print("  BTC/USDT")
    
    def do_download_data(self, arg):
        """Download historical data for backtesting."""
        from cryptobot.scripts.download_historical_data import download_data, download_strategy_data, batch_download
        
        args = arg.split()
        
        if len(args) < 1:
            print("Usage: download_data <strategy_id or symbols> <start_date> [<end_date>] [--batch]")
            print("\nExamples:")
            print("  download_data ma_crossover 2023-01-01 2023-12-31")
            print("  download_data BTC/USDT,ETH/USDT 2023-01-01 2023-12-31 --batch")
            return
            
        try:
            # Parse arguments
            first_arg = args[0]
            start_date = args[1] if len(args) > 1 else None
            end_date = args[2] if len(args) > 2 else None
            use_batch = "--batch" in args
            
            if not start_date:
                print("Start date is required")
                return
                
            # Parse dates
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            # Check if first argument is a strategy or symbols
            config = load_config(self.config_path)
            strategies = config.get('strategies', {})
            
            if first_arg in strategies:
                # Download data for strategy
                print(f"Downloading data for strategy {first_arg} from {start_date} to {end_date or 'now'}...")
                asyncio.run(download_strategy_data(
                    config_path=self.config_path,
                    strategy_id=first_arg,
                    start_date=start_date_obj,
                    end_date=end_date_obj
                ))
            else:
                # Treat first argument as symbols
                symbols = first_arg.split(',')
                timeframes = ['1h', '4h', '1d']  # Default timeframes
                
                print(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date or 'now'}...")
                
                if use_batch:
                    asyncio.run(batch_download(
                        symbols=symbols,
                        timeframes=timeframes,
                        start_date=start_date_obj,
                        end_date=end_date_obj,
                        batch_days=90,
                        data_dir='data'
                    ))
                else:
                    asyncio.run(download_data(
                        symbols=symbols,
                        timeframes=timeframes,
                        start_date=start_date_obj,
                        end_date=end_date_obj,
                        data_dir='data'
                    ))
                    
            print("Data download completed")
            
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def do_backtest(self, arg):
        """Run a backtest."""
        args = arg.split()
        
        if len(args) < 3:
            print("Usage: backtest <strategy_id> <start_date> [<end_date>] [<timeframe>] [--auto-download]")
            print("\nExample: backtest ma_crossover 2023-01-01 2023-12-31 1h --auto-download")
            return
            
        strategy_id = args[0]
        start_date = args[1]
        end_date = args[2] if len(args) > 2 and not args[2].startswith('--') else None
        
        # Check for timeframe and auto-download flag
        timeframe = "1h"  # Default
        auto_download = False
        
        for i in range(2 if end_date else 3, len(args)):
            if args[i] == "--auto-download":
                auto_download = True
            elif not args[i].startswith('--'):
                timeframe = args[i]
        
        try:
            # Parse dates
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            # Check if auto-download is requested
            if auto_download:
                from cryptobot.scripts.download_historical_data import download_strategy_data
                
                print(f"Downloading data for {strategy_id} from {start_date.date()} to {end_date.date()}...")
                await_download_strategy_data = download_strategy_data(
                    config_path=self.config_path,
                    strategy_id=strategy_id,
                    start_date=start_date,
                    end_date=end_date,
                    timeframes=[timeframe]
                )
                asyncio.run(await_download_strategy_data)
            
            print(f"Running backtest for {strategy_id} from {start_date.date()} to {end_date.date()} with timeframe {timeframe}...")
            
            # Run backtest
            results = asyncio.run(self.engine.run_backtest(
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            ))
            
            print("\nBacktest Results:")
            print("================")
            print(json.dumps(results, indent=2))
            
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
        
    def do_train_ml(self, arg):
        """Train machine learning models for a specific strategy or all ML strategies."""
        args = arg.split()
        
        if not args:
            print("Usage: train_ml [<strategy_id>]")
            print("\nExamples:")
            print("  train_ml  # Train all ML strategies")
            print("  train_ml ml_strategy_1  # Train a specific ML strategy")
            return
            
        strategy_id = args[0] if args else None
        
        try:
            # Get all strategies from the engine
            strategies = self._get_strategies()
            logger.info(f"Strategies found in engine: {list(strategies.keys())}")
            
            # Log the details of each strategy's to_dict() output
            for sid, sdata in strategies.items():
                logger.debug(f"Strategy {sid} data: {sdata}")
                logger.debug(f"Strategy {sid} name field: {sdata.get('name')}")
            
            # Filter ML strategies
            ml_strategies = {
                sid: sdata for sid, sdata in strategies.items()
                if sdata.get('name') == "MachineLearningStrategy"
            }
            logger.info(f"MachineLearningStrategy instances found: {list(ml_strategies.keys())}")
            
            if not ml_strategies:
                print("No MachineLearningStrategy instances found")
                return
                
            if strategy_id:
                # Train a specific strategy
                if strategy_id not in ml_strategies:
                    print(f"Strategy {strategy_id} not found or is not a MachineLearningStrategy")
                    return
                    
                print(f"Training ML model for strategy {strategy_id}...")
                success = asyncio.run(self.engine.train_ml_strategy(strategy_id))
                if success:
                    print(f"Successfully trained ML model for {strategy_id}")
                else:
                    print(f"Failed to train ML model for {strategy_id}")
            else:
                # Train all ML strategies
                print("Training ML models for all MachineLearningStrategy instances...")
                for sid in ml_strategies.keys():
                    print(f"Training ML model for strategy {sid}...")
                    success = asyncio.run(self.engine.train_ml_strategy(sid))
                    if success:
                        print(f"Successfully trained ML model for {sid}")
                    else:
                        print(f"Failed to train ML model for {sid}")
                        
        except Exception as e:
            print(f"Error training ML model: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
    def do_settings(self, arg):
        """View or change settings."""
        args = arg.split()
        
        if not args:
            # View all settings
            config = load_config(self.config_path)
            print("\nCurrent Settings:")
            print("================")
            print(json.dumps(config, indent=2))
        else:
            # Change a specific setting
            print("Setting changes not implemented in this version")
        
    def do_help(self, arg):
        """Show help message."""
        if arg:
            # Show help for a specific command
            super().do_help(arg)
        else:
            # Show general help
            print("\nAvailable Commands:")
            print("==================")
            print("  start           - Start the trading bot")
            print("  stop            - Stop the trading bot")
            print("  status          - Show current bot status")
            print("  balance         - Show account balances")
            print("  positions       - Show active positions")
            print("  trades          - Show recent trades")
            print("  strategies      - List or manage strategies")
            print("  markets         - Show available markets")
            print("  download_data   - Download historical data for backtesting")
            print("  backtest        - Run a backtest")
            print("  train_ml        - Train machine learning models for a strategy")
            print("  settings        - View or change settings")
            print("  help            - Show this help message")
            print("  exit            - Exit the CLI")
        
    def do_exit(self, arg):
        """Exit the CLI."""
        print("Exiting...")
        return True
        
    def do_EOF(self, arg):
        """Exit on EOF (Ctrl+D)."""
        print("Exiting...")
        return True


def run_command(args):
    """
    Run a command based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Log the config path being used
    logger.info(f"Using config path: {args.config}")
    
    # Validate config path exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        raise FileNotFoundError(f"Configuration file not found at {args.config}")
    
    # Initialize trading engine
    engine = TradingEngine(
        config_path=args.config,
        log_level=args.log_level.upper(),
        mode=args.mode
    )
    
    # Execute command
    if args.command == "run":
        asyncio.run(engine.start())
        try:
            asyncio.run(asyncio.sleep(float('inf')))
        except KeyboardInterrupt:
            print("\nStopping bot...")
            asyncio.run(engine.stop())
    elif args.command == "cli":
        cli = CryptoBotCLI(engine)
        cli.cmdloop()
    elif args.command == "backtest":
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
            
            if args.auto_download:
                from cryptobot.scripts.download_historical_data import download_strategy_data
                print(f"Downloading data for {args.strategy} from {start_date.date()} to {end_date.date()}...")
                await_download_strategy_data = download_strategy_data(
                    config_path=args.config,
                    strategy_id=args.strategy,
                    start_date=start_date,
                    end_date=end_date,
                    timeframes=[args.timeframe]
                )
                asyncio.run(await_download_strategy_data)
            
            print(f"Running backtest for {args.strategy} from {start_date.date()} to {end_date.date()} with timeframe {args.timeframe}...")
            results = asyncio.run(engine.run_backtest(
                strategy_id=args.strategy,
                start_date=start_date,
                end_date=end_date,
                timeframe=args.timeframe
            ))
            print("\nBacktest Results:")
            print("================")
            print(json.dumps(results, indent=2))
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
    elif args.command == "download-data":
        try:
            from cryptobot.scripts.download_historical_data import download_data, download_strategy_data, batch_download
            
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
            
            if args.strategy:
                print(f"Downloading data for strategy {args.strategy} from {start_date.date()} to {end_date.date()}...")
                asyncio.run(download_strategy_data(
                    config_path=args.config,
                    strategy_id=args.strategy,
                    start_date=start_date,
                    end_date=end_date
                ))
            else:
                symbols = args.symbols.split(',') if args.symbols else ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                timeframes = args.timeframes.split(',') if args.timeframes else ['1h', '4h', '1d']
                print(f"Downloading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}...")
                if args.batch:
                    asyncio.run(batch_download(
                        symbols=symbols,
                        timeframes=timeframes,
                        start_date=start_date,
                        end_date=end_date,
                        batch_days=args.batch_days,
                        data_dir=args.data_dir
                    ))
                else:
                    asyncio.run(download_data(
                        symbols=symbols,
                        timeframes=timeframes,
                        start_date=start_date,
                        end_date=end_date,
                        data_dir=args.data_dir
                    ))
            print("Data download completed")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            import traceback
            print(traceback.format_exc())
    elif args.command == "status":
        status = asyncio.run(engine.get_status())
        print(json.dumps(status, indent=2))
    elif args.command == "init":
        config = load_config(args.config)
        save_config(config, args.config)
        print(f"Initialized configuration: {args.config}")
    else:
        print(f"Unknown command: {args.command}")


def create_parser():
    """
    Create argument parser.
    
    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description="CryptoBot - Cryptocurrency Trading Bot")
    
    # Global arguments
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--log-level", "-l", default="info", choices=["debug", "info", "warning", "error", "critical"],
                       help="Logging level")
    parser.add_argument("--mode", "-m", default="production", choices=["production", "test"],
                       help="Operation mode")
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run trading bot")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Start interactive CLI")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--strategy", "-s", required=True, help="Strategy ID")
    backtest_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe for backtest")
    backtest_parser.add_argument("--auto-download", "-a", action="store_true", help="Automatically download missing data")
    
    # Download data command
    download_parser = subparsers.add_parser("download-data", help="Download historical data for backtesting")
    download_parser.add_argument("--strategy", help="Strategy ID to download data for")
    download_parser.add_argument("--symbols", "-s", help="Comma-separated list of symbols")
    download_parser.add_argument("--timeframes", "-t", help="Comma-separated list of timeframes")
    download_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    download_parser.add_argument("--data-dir", default="data", help="Directory to save data")
    download_parser.add_argument("--batch", action="store_true", help="Download in batches to avoid rate limits")
    download_parser.add_argument("--batch-days", type=int, default=90, help="Number of days per batch")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show bot status")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration")
    
    return parser


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logger(args.log_level.upper())
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        # Run command
        run_command(args)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()