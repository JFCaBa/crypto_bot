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
import pandas as pd
import numpy as np
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
        """Show account balances with detailed information."""
        try:
            # Process arguments
            args = arg.strip().split()
            show_all = '--all' in args
            sort_by = 'value'  # Default sort by USD value
            
            for i, arg in enumerate(args):
                if arg == '--sort' and i + 1 < len(args):
                    sort_by = args[i + 1]
            
            # Get balance data
            balance_data = asyncio.run(self.get_account_balance())
            
            if 'error' in balance_data:
                print(f"\nError retrieving balance: {balance_data['error']}")
                return
            
            # Display total portfolio value
            print("\n" + "=" * 60)
            print(f"TOTAL PORTFOLIO VALUE: ${balance_data['total_value_usd']:,.2f} USD")
            print("=" * 60)
            
            # Display balances by exchange
            for exchange_id, exchange_data in balance_data['exchanges'].items():
                print(f"\n{exchange_id.upper()} (${exchange_data['total_value_usd']:,.2f} USD)")
                print("-" * 60)
                
                # Format header
                print(f"{'Asset':<8} {'Amount':<18} {'Value (USD)':<15} {'Allocation':<10} {'Status':<10}")
                print("-" * 60)
                
                # Sort assets by USD value by default
                assets = exchange_data['assets'].items()
                if sort_by == 'value':
                    assets = sorted(assets, key=lambda x: x[1]['usd_value'], reverse=True)
                elif sort_by == 'name':
                    assets = sorted(assets)
                
                # Filter out small balances unless --all is specified
                min_value = 0.0 if show_all else 1.0
                
                for symbol, asset_data in assets:
                    # Skip dust balances unless --all is specified
                    if asset_data['usd_value'] < min_value:
                        continue
                    
                    # Format allocation percentage
                    allocation = (asset_data['usd_value'] / exchange_data['total_value_usd']) * 100 if exchange_data['total_value_usd'] > 0 else 0
                    
                    # Format decimals based on price magnitude
                    if asset_data['total'] < 0.001:
                        amount_str = f"{asset_data['total']:.8f}"
                    elif asset_data['total'] < 1:
                        amount_str = f"{asset_data['total']:.6f}"
                    elif asset_data['total'] < 1000:
                        amount_str = f"{asset_data['total']:.4f}"
                    else:
                        amount_str = f"{asset_data['total']:,.2f}"
                    
                    # Format status (if some funds are locked/used)
                    status = ""
                    if asset_data.get('locked', 0) > 0 or asset_data.get('used', 0) > 0:
                        status = "* Locked"
                    
                    print(f"{symbol:<8} {amount_str:<18} ${asset_data['usd_value']:,.2f}".ljust(42) + 
                        f"{allocation:.2f}%".ljust(10) + f"{status:<10}")
            
            # Display global asset allocations
            print("\nASSET ALLOCATIONS (All Exchanges)")
            print("-" * 60)
            
            # Sort assets by USD value
            sorted_assets = sorted(balance_data['assets'].items(), key=lambda x: x[1]['usd_value'], reverse=True)
            
            # Get category totals
            stablecoins_total = sum(asset['usd_value'] for symbol, asset in sorted_assets 
                                if symbol in balance_data['stablecoins'])
            crypto_total = sum(asset['usd_value'] for symbol, asset in sorted_assets 
                            if symbol not in balance_data['stablecoins'] and symbol not in balance_data['fiat'])
            fiat_total = sum(asset['usd_value'] for symbol, asset in sorted_assets 
                            if symbol in balance_data['fiat'])
            
            # Print category summaries
            if stablecoins_total > 0:
                stablecoin_pct = (stablecoins_total / balance_data['total_value_usd']) * 100
                print(f"Stablecoins: ${stablecoins_total:,.2f} ({stablecoin_pct:.2f}%)")
                
            if crypto_total > 0:
                crypto_pct = (crypto_total / balance_data['total_value_usd']) * 100
                print(f"Cryptocurrencies: ${crypto_total:,.2f} ({crypto_pct:.2f}%)")
                
            if fiat_total > 0:
                fiat_pct = (fiat_total / balance_data['total_value_usd']) * 100
                print(f"Fiat: ${fiat_total:,.2f} ({fiat_pct:.2f}%)")
                
            print("-" * 60)
            
            # Print top assets
            print("\nTOP HOLDINGS")
            print("-" * 60)
            for i, (symbol, asset) in enumerate(sorted_assets):
                if i >= 10:  # Show top 10
                    break
                allocation = asset['allocation']
                print(f"{i+1}. {symbol}: ${asset['usd_value']:,.2f} ({allocation:.2f}%)")
                
            print("\nNOTE: Values are estimates and may not reflect current market prices.")
            print("Use --all to show all balances including dust amounts.")
            print("Use --sort value|name to sort by USD value or asset name.")
            
        except Exception as e:
            print(f"\nError displaying balance: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def do_positions(self, arg):
        """Show detailed active positions information."""
        try:
            # Process arguments
            args = arg.strip().split()
            show_all = '--all' in args
            sort_by = 'pnl'  # Default sort by PnL
            detail_view = '--detail' in args
            
            for i, arg in enumerate(args):
                if arg == '--sort' and i + 1 < len(args):
                    sort_by = args[i + 1]
                    
                # Filter by strategy or symbol
                elif arg == '--strategy' and i + 1 < len(args):
                    filter_strategy = args[i + 1]
                elif arg == '--symbol' and i + 1 < len(args):
                    filter_symbol = args[i + 1]
            
            # Get position data
            position_data = asyncio.run(self.get_active_positions())
            
            if 'error' in position_data:
                print(f"\nError retrieving positions: {position_data['error']}")
                return
            
            # Display summary
            print("\n" + "=" * 70)
            print(f"ACTIVE POSITIONS SUMMARY: {position_data['total_positions']} positions, ${position_data['total_value_usd']:,.2f} total value")
            print(f"  • Long positions: {position_data['positions_by_side']['long']}")
            print(f"  • Short positions: {position_data['positions_by_side']['short']}")
            print("=" * 70)
            
            # Check if there are any positions
            if position_data['total_positions'] == 0:
                print("\nNo active positions.")
                return
            
            # Get all positions in a flat list for sorting
            all_positions = []
            for strategy_data in position_data['positions_by_strategy'].values():
                all_positions.extend(strategy_data['positions'])
            
            # Sort positions
            if sort_by == 'pnl':
                all_positions.sort(key=lambda x: x['unrealized_pnl'], reverse=True)
            elif sort_by == 'pnl_pct':
                all_positions.sort(key=lambda x: x['unrealized_pnl_pct'], reverse=True)
            elif sort_by == 'value':
                all_positions.sort(key=lambda x: x['value_usd'], reverse=True)
            elif sort_by == 'time':
                all_positions.sort(key=lambda x: x['entry_time'] if x['entry_time'] else datetime.min)
            elif sort_by == 'symbol':
                all_positions.sort(key=lambda x: x['symbol'])
            
            # Display positions
            if detail_view:
                # Detailed view - one position at a time
                for position in all_positions:
                    print(f"\n{position['symbol']} - {position['side'].upper()}")
                    print("-" * 70)
                    print(f"Strategy:      {position['strategy']}")
                    print(f"Entry price:   {position['entry_price']:.8f}")
                    print(f"Current price: {position['current_price']:.8f}")
                    print(f"Amount:        {position['amount']:.8f}")
                    print(f"Value (USD):   ${position['value_usd']:,.2f}")
                    
                    if position['entry_time']:
                        print(f"Entry time:    {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                    if position['duration']:
                        days = position['duration'].days
                        hours = position['duration'].seconds // 3600
                        minutes = (position['duration'].seconds // 60) % 60
                        print(f"Duration:      {days}d {hours}h {minutes}m")
                    
                    print(f"Unrealized P/L: ${position['unrealized_pnl']:,.2f} ({position['unrealized_pnl_pct']:+.2f}%)")
                    
                    # Display P/L status with visual indicator
                    if position['unrealized_pnl'] > 0:
                        print(f"Status:        🟢 PROFIT")
                    elif position['unrealized_pnl'] < 0:
                        print(f"Status:        🔴 LOSS")
                    else:
                        print(f"Status:        ⚪ BREAKEVEN")
                        
                    print("-" * 70)
                    
                    # If showing all positions, wait for user to press Enter
                    if position != all_positions[-1]:
                        input("Press Enter to view next position...")
            else:
                # Summary view - table format
                print("\nPOSITIONS:")
                print("-" * 110)
                print(f"{'Symbol':<10} {'Side':<6} {'Entry':<12} {'Current':<12} {'Amount':<15} {'Value':<15} {'P/L':<15} {'P/L %':<8} {'Strategy':<12}")
                print("-" * 110)
                
                for position in all_positions:
                    # Format side with arrow for visual indication
                    side_str = "▲ LONG" if position['side'] == 'long' else "▼ SHORT" if position['side'] == 'short' else position['side']
                    
                    # Format P/L with color indicators
                    pnl_prefix = "+" if position['unrealized_pnl'] > 0 else ""
                    
                    print(
                        f"{position['symbol']:<10} "
                        f"{side_str:<6} "
                        f"{position['entry_price']:<12.8f} "
                        f"{position['current_price']:<12.8f} "
                        f"{position['amount']:<15.8f} "
                        f"${position['value_usd']:<14,.2f} "
                        f"${pnl_prefix}{position['unrealized_pnl']:<14,.2f} "
                        f"{position['unrealized_pnl_pct']:+.2f}% "
                        f"{position['strategy']:<12}"
                    )
            
            # Display strategy breakdown
            print("\nBREAKDOWN BY STRATEGY:")
            print("-" * 70)
            for strategy_id, strategy_data in position_data['positions_by_strategy'].items():
                total_pnl = sum(p['unrealized_pnl'] for p in strategy_data['positions'])
                avg_pnl_pct = sum(p['unrealized_pnl_pct'] for p in strategy_data['positions']) / len(strategy_data['positions']) if strategy_data['positions'] else 0
                
                print(f"{strategy_id}: {strategy_data['count']} positions, ${strategy_data['value_usd']:,.2f} value, "
                    f"P/L: ${total_pnl:,.2f} ({avg_pnl_pct:+.2f}% avg)")
            
            print("\nNOTE: Use --detail for detailed position view")
            print("      Use --sort pnl|pnl_pct|value|time|symbol to sort positions")
            
        except Exception as e:
            print(f"\nError displaying positions: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def do_close_position(self, arg):
        """Close a position manually."""
        try:
            args = arg.strip().split()
            
            if len(args) < 1:
                print("Usage: close_position <symbol> [--strategy <strategy_id>]")
                return
                
            symbol = args[0]
            strategy_id = None
            
            # Check for strategy argument
            for i, arg in enumerate(args):
                if arg == '--strategy' and i + 1 < len(args):
                    strategy_id = args[i + 1]
            
            # Get active positions
            position_data = asyncio.run(self.get_active_positions())
            
            if 'error' in position_data:
                print(f"\nError retrieving positions: {position_data['error']}")
                return
                
            # Find the position
            target_positions = []
            
            for strategy_data in position_data['positions_by_strategy'].values():
                for position in strategy_data['positions']:
                    if position['symbol'] == symbol:
                        if strategy_id is None or position['strategy'] == strategy_id:
                            target_positions.append(position)
            
            if not target_positions:
                print(f"No active position found for {symbol}" + 
                    (f" with strategy {strategy_id}" if strategy_id else ""))
                return
                
            if len(target_positions) > 1 and strategy_id is None:
                print(f"Multiple positions found for {symbol}. Please specify a strategy with --strategy.")
                print("Available positions:")
                for position in target_positions:
                    print(f"  {position['symbol']} ({position['strategy']}): {position['side'].upper()}, "
                        f"Amount: {position['amount']}, P/L: {position['unrealized_pnl_pct']:+.2f}%")
                return
            
            # Confirm closing
            position = target_positions[0]
            print(f"\nClosing position: {position['symbol']} ({position['strategy']})")
            print(f"Side: {position['side'].upper()}")
            print(f"Amount: {position['amount']}")
            print(f"Current P/L: ${position['unrealized_pnl']:,.2f} ({position['unrealized_pnl_pct']:+.2f}%)")
            
            confirm = input("\nAre you sure you want to close this position? (y/n): ")
            
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return
                
            # Close the position
            print(f"Closing position {position['symbol']}...")
            
            # Create a close signal
            signal = {
                'symbol': position['symbol'],
                'action': 'close',
                'price': position['current_price'],
                'amount': position['amount'],
                'strategy': position['strategy']
            }
            
            # Execute the close signal
            result = asyncio.run(self.engine.execute_manual_signal(signal))
            
            if result and result.get('success'):
                print(f"Position closed successfully!")
                print(f"Realized P/L: ${result.get('pnl', position['unrealized_pnl']):,.2f} ({result.get('pnl_percent', position['unrealized_pnl_pct']):+.2f}%)")
            else:
                print(f"Error closing position: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"\nError closing position: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
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
        """Run a backtest with enhanced analysis and visualization."""
        args = arg.split()
        
        if len(args) < 3:
            print("Usage: backtest <strategy_id> <start_date> [<end_date>] [<timeframe>] [--auto-download] [--detailed]")
            print("\nExample: backtest ma_crossover 2023-01-01 2023-12-31 1h --auto-download --detailed")
            return
            
        strategy_id = args[0]
        start_date_str = args[1]
        
        # Fix the end_date parsing logic
        end_date_str = None
        if len(args) > 2 and not args[2].startswith('--'):
            end_date_str = args[2]
        
        # Check for timeframe and flags
        timeframe = "1h"  # Default
        auto_download = False
        detailed_analysis = False
        
        for i in range(2 if end_date_str is None else 3, len(args)):
            if args[i] == "--auto-download":
                auto_download = True
            elif args[i] == "--detailed":
                detailed_analysis = True
            elif not args[i].startswith('--'):
                timeframe = args[i]
        
        try:
            # Parse dates with robustness
            try:
                start_date = pd.to_datetime(start_date_str).to_pydatetime()
                if end_date_str:
                    end_date = pd.to_datetime(end_date_str).to_pydatetime()
                else:
                    end_date = datetime.now()
            except ValueError as e:
                print(f"Invalid date format: {str(e)}. Use YYYY-MM-DD")
                return
            
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
                timeframe=timeframe,
                return_full_results=True  # Get equity curve and trade history as well
            ))
            
            if results is None or not isinstance(results, dict) or 'metrics' not in results:
                print("Error: Backtest did not return valid results.")
                return
                
            # Extract components from results
            metrics = results.get('metrics', {})
            equity_curve = results.get('equity_curve')
            trades = results.get('trades')
            
            # Validate equity_curve
            if equity_curve is not None and isinstance(equity_curve, pd.DataFrame):
                logger.debug(f"Equity curve shape: {equity_curve.shape}, columns: {equity_curve.columns}")
                if not isinstance(equity_curve.index, pd.DatetimeIndex):
                    logger.warning("Equity curve index is not datetime. Attempting to fix...")
            else:
                logger.warning("Equity curve is not a DataFrame or is None")
            
            # Convert trades to DataFrame if needed
            if trades and not isinstance(trades, pd.DataFrame):
                try:
                    trades = pd.DataFrame(trades)
                    if 'timestamp' in trades.columns:
                        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
                except:
                    trades = None
                    
            # Run enhanced analysis
            print("\nAnalyzing backtest results...")
            analysis = asyncio.run(self.analyze_backtest_results(
                metrics, equity_curve, trades, start_date, end_date
            ))
            
            if 'error' in analysis:
                print(f"Error analyzing results: {analysis['error']}")
                
            # Display basic results
            print("\n" + "=" * 70)
            print(f"BACKTEST RESULTS: {strategy_id} ({start_date.date()} to {end_date.date()}, {timeframe})")
            print("=" * 70)
            
            # Basic metrics
            basic_metrics = metrics
            print(f"\nSTRATEGY PERFORMANCE:")
            print(f"Initial balance:    ${basic_metrics.get('start_balance', 0):,.2f}")
            print(f"Final balance:      ${basic_metrics.get('end_balance', 0):,.2f}")
            print(f"Total return:       {basic_metrics.get('strategy_return', 0):+.2f}%")
            print(f"Buy & hold return:  {basic_metrics.get('buy_hold_return', 0):+.2f}%")
            print(f"Alpha:              {basic_metrics.get('alpha', 0):+.2f}")
            print(f"Beta:               {basic_metrics.get('beta', 0):.2f}")
            print(f"Sharpe ratio:       {basic_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max drawdown:       {basic_metrics.get('max_drawdown_pct', 0):.2f}%")
            
            # Trade statistics
            print(f"\nTRADE STATISTICS:")
            print(f"Total trades:       {basic_metrics.get('total_trades', 0)}")
            print(f"Win rate:           {basic_metrics.get('win_rate', 0):.2f}%")
            print(f"Profit factor:      {basic_metrics.get('profit_factor', 0):.2f}")
            print(f"Avg profit:         {basic_metrics.get('avg_profit', 0):+.2f}%")
            print(f"Avg loss:           {basic_metrics.get('avg_loss', 0):+.2f}%")
            
            # If we have the extended analysis
            if 'trade_analysis' in analysis:
                trade_analysis = analysis['trade_analysis']
                print(f"Expectancy:         ${trade_analysis.get('expectancy', 0):+.2f}")
                print(f"Largest win:        ${trade_analysis.get('largest_win', 0):+,.2f}")
                print(f"Largest loss:       ${trade_analysis.get('largest_loss', 0):+,.2f}")
                print(f"Consecutive wins:   {trade_analysis.get('max_consecutive_wins', 0)}")
                print(f"Consecutive losses: {trade_analysis.get('max_consecutive_losses', 0)}")
                
            if 'monthly_returns' in analysis and analysis['monthly_returns']:
                print(f"\nMONTHLY RETURNS:")
                monthly_returns = analysis['monthly_returns']
                
                # Group by year and format as a table
                current_year = None
                for month, ret in sorted(monthly_returns.items()):
                    year, month_num = month.split('-')
                    if year != current_year:
                        if current_year is not None:
                            print()  # Add line between years
                        print(f"{year}:", end="")
                        current_year = year
                    
                    # Print month return with color coding
                    month_name = datetime(2023, int(month_num), 1).strftime('%b')
                    print(f" {month_name}: {ret:+.2f}%", end="")
                print()  # End the line
                
            if detailed_analysis and 'drawdown_analysis' in analysis and 'top_drawdowns' in analysis['drawdown_analysis']:
                print(f"\nTOP DRAWDOWNS:")
                print(f"{'Rank':<4} {'Start Date':<12} {'End Date':<12} {'Depth':<8} {'Duration':<10} {'Recovery':<10}")
                print("-" * 60)
                
                for dd in analysis['drawdown_analysis']['top_drawdowns']:
                    recovery = f"{dd['recovery_days']} days" if dd['recovery_days'] is not None else "Ongoing"
                    print(f"{dd['rank']:<4} {dd['start_date']:<12} {dd['end_date']:<12} "
                        f"{dd['depth_pct']:.2f}%".ljust(8) + 
                        f"{dd['duration_days']} days".ljust(10) + 
                        f"{recovery:<10}")
                    
            if detailed_analysis and 'trade_analysis' in analysis and 'by_symbol' in analysis['trade_analysis']:
                print(f"\nPERFORMANCE BY SYMBOL:")
                print(f"{'Symbol':<10} {'Trades':<8} {'Win Rate':<10} {'Total P/L':<12} {'Avg P/L':<10}")
                print("-" * 60)
                
                symbol_analysis = analysis['trade_analysis']['by_symbol']
                # Sort by total P/L
                for symbol, stats in sorted(symbol_analysis.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                    print(f"{symbol:<10} {stats['total_trades']:<8} {stats['win_rate']:.2f}%".ljust(19) + 
                        f"${stats['total_pnl']:+,.2f}".ljust(12) + 
                        f"${stats['avg_pnl']:+,.2f}")
                    
            if detailed_analysis and 'benchmark_comparison' in analysis:
                benchmark = analysis['benchmark_comparison']
                print(f"\nBENCHMARK COMPARISON:")
                print(f"Alpha:             {benchmark.get('alpha', 0):+.2f}")
                print(f"Beta:              {benchmark.get('beta', 0):.2f}")
                print(f"Correlation:       {benchmark.get('correlation', 0):.2f}")
                print(f"Information ratio: {benchmark.get('information_ratio', 0):.2f}")
            
            # Visualize results
            self._visualize_backtest(
                strategy_id, 
                start_date, 
                end_date, 
                timeframe, 
                equity_curve, 
                trades, 
                analysis
            )
            
        except ValueError as e:
            print(f"Invalid date format in '{start_date_str}' or '{end_date_str}'. Use YYYY-MM-DD")
            print("Invalid date format. Use YYYY-MM-DD")
            print(f"Error details: {str(e)}")
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _visualize_backtest(
        self, 
        strategy_id, 
        start_date, 
        end_date, 
        timeframe, 
        equity_curve=None, 
        trades=None, 
        analysis=None
    ):
        """
        Visualize backtest results using Matplotlib.
        
        Args:
            strategy_id: Strategy identifier
            start_date: Start date of backtest
            end_date: End date of backtest
            timeframe: Timeframe used
            equity_curve: Equity curve DataFrame
            trades: Trade history DataFrame
            analysis: Enhanced analysis results
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            from matplotlib.ticker import FuncFormatter
            from matplotlib.gridspec import GridSpec
            
            if equity_curve is None:
                print("No equity curve data available for visualization.")
                return
                
            # Create figure
            plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
            
            # Plot equity curve
            ax1 = plt.subplot(gs[0, :])
            equity_curve['equity'].plot(ax=ax1, color='blue', linewidth=2, label='Strategy')
            
            # Plot benchmark if available
            if 'benchmark_equity' in equity_curve.columns:
                equity_curve['benchmark_equity'].plot(ax=ax1, color='gray', linewidth=1, alpha=0.7, linestyle='--', label='Buy & Hold')
            
            # Format the equity curve plot
            ax1.set_title(f'Equity Curve: {strategy_id} ({start_date.date()} to {end_date.date()}, {timeframe})')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Annotate trade points if available
            if trades is not None and len(trades) > 0 and 'timestamp' in trades.columns:
                # Only plot a reasonable number of trades to avoid clutter
                max_trades_to_plot = 50
                plot_interval = max(1, len(trades) // max_trades_to_plot)
                
                buy_trades = trades[trades['side'] == 'buy'].iloc[::plot_interval]
                sell_trades = trades[trades['side'] == 'sell'].iloc[::plot_interval]
                
                # Plot trade points on the equity curve
                if not buy_trades.empty:
                    buy_times = pd.to_datetime(buy_trades['timestamp'])
                    buy_values = [equity_curve['equity'].asof(t) if t in equity_curve.index 
                                else equity_curve['equity'].iloc[0] for t in buy_times]
                    ax1.scatter(buy_times, buy_values, marker='^', color='green', s=50, alpha=0.7, label='Buy')
                    
                if not sell_trades.empty:
                    sell_times = pd.to_datetime(sell_trades['timestamp'])
                    sell_values = [equity_curve['equity'].asof(t) if t in equity_curve.index 
                                else equity_curve['equity'].iloc[0] for t in sell_times]
                    ax1.scatter(sell_times, sell_values, marker='v', color='red', s=50, alpha=0.7, label='Sell')
                    
                ax1.legend()
            
            # Plot drawdown
            if 'drawdown_pct' in equity_curve.columns:
                ax2 = plt.subplot(gs[1, :], sharex=ax1)
                equity_curve['drawdown_pct'].plot(ax=ax2, color='red', linewidth=1, alpha=0.5)
                ax2.fill_between(equity_curve.index, 0, equity_curve['drawdown_pct'], color='red', alpha=0.3)
                ax2.set_title('Drawdown (%)')
                ax2.set_ylabel('Drawdown %')
                ax2.set_ylim(bottom=0, top=max(equity_curve['drawdown_pct']) * 1.1)
                ax2.invert_yaxis()  # Invert so that drawdowns go down
                ax2.grid(True, alpha=0.3)
                
                # Add annotations for major drawdowns
                if analysis and 'drawdown_analysis' in analysis and 'top_drawdowns' in analysis['drawdown_analysis']:
                    top_dd = analysis['drawdown_analysis']['top_drawdowns'][0]  # Get worst drawdown
                    # Find the timestamp of the maximum drawdown
                    max_dd_date = equity_curve['drawdown_pct'].idxmax()
                    if max_dd_date:
                        max_dd = equity_curve.loc[max_dd_date, 'drawdown_pct']
                        ax2.annotate(f"Max DD: {max_dd:.2f}%", 
                                    xy=(max_dd_date, max_dd),
                                    xytext=(10, 10),
                                    textcoords='offset points',
                                    arrowprops=dict(arrowstyle='->', color='black'),
                                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            # Plot monthly returns
            if analysis and 'monthly_returns' in analysis and analysis['monthly_returns']:
                ax3 = plt.subplot(gs[2, 0])
                monthly_returns = pd.Series(analysis['monthly_returns'])
                monthly_returns = monthly_returns.sort_index()
                colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                
                monthly_returns.plot(kind='bar', ax=ax3, color=colors)
                ax3.set_title('Monthly Returns (%)')
                ax3.set_ylabel('Return %')
                ax3.set_xlabel('')
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Rotate x-axis labels for better readability
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Only show every 3rd month label to avoid crowding
                for i, label in enumerate(ax3.xaxis.get_ticklabels()):
                    if i % 3 != 0:
                        label.set_visible(False)
            
            # Plot trade analysis
            if analysis and 'trade_analysis' in analysis and 'by_symbol' in analysis['trade_analysis']:
                ax4 = plt.subplot(gs[2, 1])
                
                # Extract symbol data
                symbols = list(analysis['trade_analysis']['by_symbol'].keys())
                pnls = [data['total_pnl'] for data in analysis['trade_analysis']['by_symbol'].values()]
                win_rates = [data['win_rate'] for data in analysis['trade_analysis']['by_symbol'].values()]
                
                # Sort by PnL
                sorted_indices = np.argsort(pnls)[::-1]  # Reverse to get descending
                sorted_symbols = [symbols[i] for i in sorted_indices[:10]]  # Take top 10
                sorted_pnls = [pnls[i] for i in sorted_indices[:10]]
                colors = ['green' if x > 0 else 'red' for x in sorted_pnls]
                
                # Create bar chart
                bars = ax4.bar(sorted_symbols, sorted_pnls, color=colors)
                ax4.set_title('P/L by Symbol ($)')
                ax4.set_ylabel('Profit/Loss ($)')
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add win rate as text on top of bars
                for i, (bar, win_rate) in enumerate(zip(bars, [win_rates[sorted_indices[i]] for i in range(min(10, len(sorted_indices)))])):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1) * 0.5,
                            f"{win_rate:.1f}%",
                            ha='center', va='bottom' if height > 0 else 'top', rotation=0,
                            fontsize=8)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure if results directory exists
            results_dir = 'backtest_results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{results_dir}/{strategy_id}_{timestamp}_backtest.png"
            plt.savefig(filename)
            print(f"\nBacktest visualization saved to {filename}")
            
            # Show the figure
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing backtest results: {str(e)}")
            import traceback
            print(traceback.format_exc())
       
        
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
    
    async def get_account_balance(self):
        """
        Fetch and process account balances from all exchanges.
        
        Returns:
            dict: Processed balance information with totals and allocations
        """
        try:
            status = await self.engine.get_status()
            
            # Process balances
            result = {
                'exchanges': {},
                'total_value_usd': 0.0,
                'assets': {},
                'stablecoins': ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'GUSD'],
                'fiat': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD']
            }
            
            # Default pricing if we don't have access to real-time prices
            default_prices = {
                'BTC': 65000.0,
                'ETH': 3500.0,
                'BNB': 600.0,
                'SOL': 140.0,
                'XRP': 0.50,
                'ADA': 0.45,
                'AVAX': 35.0,
                'DOT': 7.0,
                'MATIC': 0.80,
                'LINK': 18.0
                # Add more as needed
            }
            
            # Process each exchange
            for exchange_id, balance in status.get('account_balances', {}).items():
                exchange_data = {
                    'total_value_usd': 0.0,
                    'assets': {}
                }
                
                # Process Binance-specific format
                if 'info' in balance and 'balances' in balance['info']:
                    balances = balance['info']['balances']
                    for asset in balances:
                        symbol = asset['asset']
                        free = float(asset['free'])
                        locked = float(asset['locked'])
                        total = free + locked
                        
                        # Skip zero balances
                        if total <= 0:
                            continue
                        
                        # Calculate USD value
                        usd_value = 0.0
                        if symbol in result['stablecoins'] or symbol in result['fiat']:
                            usd_value = total  # Assume 1:1 for stablecoins and USD
                        elif symbol in default_prices:
                            usd_value = total * default_prices[symbol]
                        
                        # Store asset data
                        asset_data = {
                            'free': free,
                            'locked': locked,
                            'total': total,
                            'usd_value': usd_value
                        }
                        
                        exchange_data['assets'][symbol] = asset_data
                        exchange_data['total_value_usd'] += usd_value
                        
                        # Add to global assets
                        if symbol not in result['assets']:
                            result['assets'][symbol] = {
                                'total': 0.0,
                                'usd_value': 0.0,
                                'allocation': 0.0
                            }
                        
                        result['assets'][symbol]['total'] += total
                        result['assets'][symbol]['usd_value'] += usd_value
                
                # Process generic format
                else:
                    for currency, data in balance.items():
                        if not isinstance(data, dict):
                            continue
                        
                        free = float(data.get('free', 0))
                        used = float(data.get('used', 0))
                        total = float(data.get('total', 0))
                        
                        # Skip zero balances and non-asset fields
                        if total <= 0 or currency == 'info':
                            continue
                        
                        # Calculate USD value
                        usd_value = 0.0
                        if currency in result['stablecoins'] or currency in result['fiat']:
                            usd_value = total  # Assume 1:1 for stablecoins and USD
                        elif currency in default_prices:
                            usd_value = total * default_prices[currency]
                        
                        # Store asset data
                        asset_data = {
                            'free': free,
                            'used': used,
                            'total': total,
                            'usd_value': usd_value
                        }
                        
                        exchange_data['assets'][currency] = asset_data
                        exchange_data['total_value_usd'] += usd_value
                        
                        # Add to global assets
                        if currency not in result['assets']:
                            result['assets'][currency] = {
                                'total': 0.0,
                                'usd_value': 0.0,
                                'allocation': 0.0
                            }
                        
                        result['assets'][currency]['total'] += total
                        result['assets'][currency]['usd_value'] += usd_value
                
                result['exchanges'][exchange_id] = exchange_data
                result['total_value_usd'] += exchange_data['total_value_usd']
            
            # Calculate allocations
            if result['total_value_usd'] > 0:
                for symbol, data in result['assets'].items():
                    data['allocation'] = (data['usd_value'] / result['total_value_usd']) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving account balance: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'exchanges': {},
                'total_value_usd': 0.0,
                'assets': {},
                'error': str(e)
            }
        
    async def analyze_backtest_results(self, metrics, equity_curve, trades, start_date, end_date):
        """
        Analyze backtest results and generate additional metrics.

        Args:
            metrics (dict): Raw backtest metrics.
            equity_curve (pd.DataFrame): Equity curve data.
            trades (pd.DataFrame): Trade history data.
            start_date (datetime): Start date of the backtest.
            end_date (datetime): End date of the backtest.

        Returns:
            dict: Enhanced backtest analysis.
        """
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime
            
            analysis = {
                'raw_results': metrics,
                'extended_metrics': {},
                'monthly_returns': {},
                'drawdown_analysis': {},
                'trade_analysis': {},
                'risk_metrics': {},
                'benchmark_comparison': {}
            }
            
            # Extract base metrics
            base_metrics = metrics
            
            # Calculate additional risk-adjusted return metrics
            if 'sharpe_ratio' in base_metrics:
                # Calculate Sortino ratio from downside deviation
                downside_dev = base_metrics.get('downside_deviation', base_metrics.get('volatility', 0) * 0.7)
                if downside_dev > 0:
                    annual_return = base_metrics.get('annual_return', 0)
                    risk_free_rate = 0.02  # Assumed 2% risk-free rate
                    analysis['risk_metrics']['sortino_ratio'] = (annual_return - risk_free_rate) / (downside_dev * 100)
                
                # Calculate Calmar ratio (return / max drawdown)
                max_dd = base_metrics.get('max_drawdown_pct', 0)
                if max_dd > 0:
                    analysis['risk_metrics']['calmar_ratio'] = base_metrics.get('annual_return', 0) / max_dd
            
            # Process equity curve if provided
            if equity_curve is not None and isinstance(equity_curve, pd.DataFrame):
                # Ensure datetime index
                if not isinstance(equity_curve.index, pd.DatetimeIndex):
                    if 'timestamp' in equity_curve.columns:
                        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
                        equity_curve.set_index('timestamp', inplace=True)
                    else:
                        # If no timestamp column, assume index is sequential and map to dates
                        logger.warning("No timestamp column found in equity_curve. Creating synthetic datetime index.")
                        date_range = pd.date_range(start=start_date, end=end_date, periods=len(equity_curve))
                        equity_curve.index = date_range

                # Calculate daily returns
                if 'equity' in equity_curve.columns:
                    equity_curve['daily_return'] = equity_curve['equity'].pct_change()
                    
                    # Calculate monthly returns
                    equity_curve['year_month'] = equity_curve.index.strftime('%Y-%m')
                    monthly_returns = equity_curve.groupby('year_month')['daily_return'].apply(
                        lambda x: ((1 + x).prod() - 1) * 100
                    )
                    analysis['monthly_returns'] = monthly_returns.to_dict()
                    
                    # Calculate winning days vs losing days
                    winning_days = (equity_curve['daily_return'] > 0).sum()
                    losing_days = (equity_curve['daily_return'] < 0).sum()
                    flat_days = (equity_curve['daily_return'] == 0).sum()
                    
                    analysis['extended_metrics']['winning_days'] = int(winning_days)
                    analysis['extended_metrics']['losing_days'] = int(losing_days)
                    analysis['extended_metrics']['flat_days'] = int(flat_days)
                    analysis['extended_metrics']['winning_days_pct'] = float(winning_days / len(equity_curve) * 100)
                    
                    # Calculate average winning day vs average losing day
                    avg_win = equity_curve.loc[equity_curve['daily_return'] > 0, 'daily_return'].mean() * 100
                    avg_loss = equity_curve.loc[equity_curve['daily_return'] < 0, 'daily_return'].mean() * 100
                    
                    analysis['extended_metrics']['avg_winning_day_pct'] = float(avg_win if not np.isnan(avg_win) else 0)
                    analysis['extended_metrics']['avg_losing_day_pct'] = float(avg_loss if not np.isnan(avg_loss) else 0)
                    
                    # Calculate max consecutive winning and losing days
                    streaks = equity_curve['daily_return'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
                    pos_streaks = []
                    neg_streaks = []
                    current_streak = 0
                    
                    for s in streaks:
                        if s == 0:
                            continue
                        if s == 1:
                            if current_streak > 0:
                                current_streak += 1
                            else:
                                if current_streak < 0:
                                    neg_streaks.append(abs(current_streak))
                                current_streak = 1
                        else:  # s == -1
                            if current_streak < 0:
                                current_streak -= 1
                            else:
                                if current_streak > 0:
                                    pos_streaks.append(current_streak)
                                current_streak = -1
                    
                    # Add the last streak
                    if current_streak > 0:
                        pos_streaks.append(current_streak)
                    elif current_streak < 0:
                        neg_streaks.append(abs(current_streak))
                    
                    analysis['extended_metrics']['max_win_streak'] = max(pos_streaks) if pos_streaks else 0
                    analysis['extended_metrics']['max_lose_streak'] = max(neg_streaks) if neg_streaks else 0
                    
                    # Analyze drawdowns
                    if 'drawdown_pct' in equity_curve.columns:
                        drawdown_starts = []
                        drawdown_ends = []
                        drawdown_depths = []
                        in_drawdown = False
                        drawdown_start = None
                        max_depth = 0
                        
                        for idx, row in equity_curve.iterrows():
                            dd = row['drawdown_pct']
                            
                            if not in_drawdown and dd > 0:
                                # Start of a drawdown
                                in_drawdown = True
                                drawdown_start = idx
                                max_depth = dd
                            elif in_drawdown:
                                if dd > max_depth:
                                    max_depth = dd
                                
                                if dd == 0:
                                    # End of a drawdown
                                    in_drawdown = False
                                    drawdown_starts.append(drawdown_start)
                                    drawdown_ends.append(idx)
                                    drawdown_depths.append(max_depth)
                                    drawdown_start = None
                                    max_depth = 0
                        
                        # If still in drawdown at the end
                        if in_drawdown:
                            drawdown_starts.append(drawdown_start)
                            drawdown_ends.append(equity_curve.index[-1])
                            drawdown_depths.append(max_depth)
                        
                        # Sort drawdowns by depth
                        if drawdown_depths:
                            drawdowns = sorted(zip(drawdown_starts, drawdown_ends, drawdown_depths), 
                                            key=lambda x: x[2], reverse=True)
                            
                            analysis['drawdown_analysis']['top_drawdowns'] = []
                            for i, (start, end, depth) in enumerate(drawdowns[:5]):
                                duration = (end - start).days
                                recovery = None
                                
                                # Find recovery date if available
                                if i < len(drawdowns) - 1:
                                    next_start = drawdowns[i+1][0]
                                    recovery_duration = (next_start - end).days
                                    recovery = recovery_duration
                                
                                analysis['drawdown_analysis']['top_drawdowns'].append({
                                    'rank': i + 1,
                                    'start_date': start.strftime('%Y-%m-%d'),
                                    'end_date': end.strftime('%Y-%m-%d'),
                                    'depth_pct': float(depth),
                                    'duration_days': duration,
                                    'recovery_days': recovery
                                })
                            
                            # Calculate average drawdown
                            analysis['drawdown_analysis']['avg_drawdown_pct'] = float(np.mean(drawdown_depths))
                            analysis['drawdown_analysis']['avg_drawdown_duration'] = float(np.mean(
                                [(end - start).days for start, end, _ in drawdowns]
                            ))
                    
                    # Benchmark comparison
                    if 'benchmark_equity' in equity_curve.columns:
                        # Calculate benchmark returns
                        equity_curve['benchmark_return'] = equity_curve['benchmark_equity'].pct_change()
                        
                        # Calculate correlation with benchmark
                        correlation = equity_curve['daily_return'].corr(equity_curve['benchmark_return'])
                        analysis['benchmark_comparison']['correlation'] = float(correlation)
                        
                        # Calculate beta (using covariance / variance)
                        cov = equity_curve['daily_return'].cov(equity_curve['benchmark_return'])
                        var = equity_curve['benchmark_return'].var()
                        beta = cov / var if var > 0 else 0
                        analysis['benchmark_comparison']['beta'] = float(beta)
                        
                        # Calculate alpha (excess return over benchmark)
                        strategy_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0] - 1) * 100
                        benchmark_return = (equity_curve['benchmark_equity'].iloc[-1] / equity_curve['benchmark_equity'].iloc[0] - 1) * 100
                        alpha = strategy_return - (beta * benchmark_return)
                        analysis['benchmark_comparison']['alpha'] = float(alpha)
                        
                        # Calculate information ratio
                        excess_returns = equity_curve['daily_return'] - equity_curve['benchmark_return']
                        tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
                        info_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
                        analysis['benchmark_comparison']['information_ratio'] = float(info_ratio)
            
            # Process trade history if provided
            if trades is not None and isinstance(trades, pd.DataFrame):
                # Ensure trades are sorted by timestamp
                if 'timestamp' in trades.columns:
                    trades = trades.sort_values('timestamp')
                    
                    # Calculate basic trade statistics
                    total_trades = len(trades)
                    winning_trades = len(trades[trades['pnl'] > 0])
                    losing_trades = len(trades[trades['pnl'] <= 0])
                    
                    analysis['trade_analysis']['total_trades'] = total_trades
                    analysis['trade_analysis']['winning_trades'] = winning_trades
                    analysis['trade_analysis']['losing_trades'] = losing_trades
                    analysis['trade_analysis']['win_rate'] = float(winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    # Calculate profit metrics
                    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
                    gross_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
                    
                    analysis['trade_analysis']['gross_profit'] = float(gross_profit)
                    analysis['trade_analysis']['gross_loss'] = float(gross_loss)
                    analysis['trade_analysis']['profit_factor'] = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
                    
                    # Calculate average metrics
                    analysis['trade_analysis']['avg_profit'] = float(trades[trades['pnl'] > 0]['pnl'].mean()) if winning_trades > 0 else 0
                    analysis['trade_analysis']['avg_loss'] = float(trades[trades['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
                    analysis['trade_analysis']['avg_profit_pct'] = float(trades[trades['pnl'] > 0]['pnl_percent'].mean()) if winning_trades > 0 else 0
                    analysis['trade_analysis']['avg_loss_pct'] = float(trades[trades['pnl'] <= 0]['pnl_percent'].mean()) if losing_trades > 0 else 0
                    
                    # Calculate expectancy
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                    avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
                    
                    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
                    analysis['trade_analysis']['expectancy'] = float(expectancy)
                    
                    # Calculate trade duration statistics if possible
                    if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
                        trades['duration'] = (trades['exit_time'] - trades['entry_time']).apply(lambda x: x.total_seconds() / 3600)  # in hours
                        
                        analysis['trade_analysis']['avg_trade_duration_hours'] = float(trades['duration'].mean())
                        analysis['trade_analysis']['max_trade_duration_hours'] = float(trades['duration'].max())
                        analysis['trade_analysis']['min_trade_duration_hours'] = float(trades['duration'].min())
                    
                    # Calculate largest win and loss
                    analysis['trade_analysis']['largest_win'] = float(trades['pnl'].max())
                    analysis['trade_analysis']['largest_loss'] = float(trades['pnl'].min())
                    
                    # Calculate consecutive wins and losses
                    trade_results = trades['pnl'].apply(lambda x: 1 if x > 0 else 0)
                    win_streaks = []
                    lose_streaks = []
                    current_streak = 0
                    current_streak_type = None
                    
                    for result in trade_results:
                        if current_streak_type is None:
                            current_streak_type = result
                            current_streak = 1
                        elif result == current_streak_type:
                            current_streak += 1
                        else:
                            if current_streak_type == 1:
                                win_streaks.append(current_streak)
                            else:
                                lose_streaks.append(current_streak)
                            current_streak_type = result
                            current_streak = 1
                    
                    # Add the last streak
                    if current_streak_type == 1:
                        win_streaks.append(current_streak)
                    else:
                        lose_streaks.append(current_streak)
                    
                    analysis['trade_analysis']['max_consecutive_wins'] = max(win_streaks) if win_streaks else 0
                    analysis['trade_analysis']['max_consecutive_losses'] = max(lose_streaks) if lose_streaks else 0
                    
                    # Analyze by symbol
                    symbol_analysis = {}
                    for symbol in trades['symbol'].unique():
                        symbol_trades = trades[trades['symbol'] == symbol]
                        wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                        losses = len(symbol_trades[symbol_trades['pnl'] <= 0])
                        total = len(symbol_trades)
                        
                        symbol_analysis[symbol] = {
                            'total_trades': total,
                            'winning_trades': wins,
                            'losing_trades': losses,
                            'win_rate': float(wins / total * 100) if total > 0 else 0,
                            'total_pnl': float(symbol_trades['pnl'].sum()),
                            'avg_pnl': float(symbol_trades['pnl'].mean()),
                            'largest_win': float(symbol_trades['pnl'].max()) if not symbol_trades.empty else 0,
                            'largest_loss': float(symbol_trades['pnl'].min()) if not symbol_trades.empty else 0
                        }
                    
                    analysis['trade_analysis']['by_symbol'] = symbol_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'raw_results': metrics,
                'error': str(e)
            }
    
    async def get_active_positions(self):
        """
        Get detailed information about active positions across all strategies.
        
        Returns:
            dict: Detailed position information
        """
        try:
            status = await self.engine.get_status()
            
            # Process positions
            result = {
                'total_positions': 0,
                'total_value_usd': 0.0,
                'positions_by_strategy': {},
                'positions_by_symbol': {},
                'positions_by_side': {
                    'long': 0,
                    'short': 0
                }
            }
            
            # Default pricing if we don't have access to real-time prices
            default_prices = {
                'BTC': 65000.0,
                'ETH': 3500.0,
                'BNB': 600.0,
                'SOL': 140.0,
                'XRP': 0.50,
                'ADA': 0.45,
                'AVAX': 35.0,
                'DOT': 7.0,
                'MATIC': 0.80,
                'LINK': 18.0
                # Add more as needed
            }
            
            # Get current market prices (in a real implementation, this would be fetched from exchanges)
            # For now, we'll use the default prices and the current price from position data
            current_prices = {}
            
            # Process positions from each strategy
            for strategy_id, strategy_data in status.get('strategies', {}).items():
                positions = strategy_data.get('positions', {})
                active_positions = {symbol: pos for symbol, pos in positions.items() if pos.get('is_active', False)}
                
                if not active_positions:
                    continue
                    
                # Initialize strategy entry in result
                result['positions_by_strategy'][strategy_id] = {
                    'count': 0,
                    'value_usd': 0.0,
                    'positions': []
                }
                
                # Process each active position
                for symbol, position in active_positions.items():
                    side = position.get('side', 'unknown')
                    entry_price = position.get('entry_price', 0)
                    amount = position.get('amount', 0)
                    entry_time = position.get('entry_time', '')
                    
                    # Parse entry time if it's a string
                    if isinstance(entry_time, str):
                        try:
                            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        except:
                            entry_time = None
                    
                    # Get current price
                    if symbol not in current_prices:
                        base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
                        current_prices[symbol] = default_prices.get(base_currency, entry_price)
                        
                    current_price = current_prices[symbol]
                    
                    # Calculate position value and P/L
                    position_value = amount * current_price
                    
                    if side == 'long':
                        unrealized_pnl = (current_price - entry_price) * amount
                        unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
                    elif side == 'short':
                        unrealized_pnl = (entry_price - current_price) * amount
                        unrealized_pnl_pct = ((entry_price / current_price) - 1) * 100
                    else:
                        unrealized_pnl = 0
                        unrealized_pnl_pct = 0
                    
                    # Calculate duration
                    duration = None
                    if entry_time:
                        duration = datetime.now() - entry_time
                    
                    # Create position record
                    position_record = {
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'amount': amount,
                        'value_usd': position_value,
                        'entry_time': entry_time,
                        'duration': duration,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'strategy': strategy_id
                    }
                    
                    # Add to strategy positions
                    result['positions_by_strategy'][strategy_id]['positions'].append(position_record)
                    result['positions_by_strategy'][strategy_id]['count'] += 1
                    result['positions_by_strategy'][strategy_id]['value_usd'] += position_value
                    
                    # Add to symbol positions
                    if symbol not in result['positions_by_symbol']:
                        result['positions_by_symbol'][symbol] = {
                            'count': 0,
                            'value_usd': 0.0,
                            'positions': []
                        }
                    
                    result['positions_by_symbol'][symbol]['positions'].append(position_record)
                    result['positions_by_symbol'][symbol]['count'] += 1
                    result['positions_by_symbol'][symbol]['value_usd'] += position_value
                    
                    # Add to side counter
                    if side in ['long', 'short']:
                        result['positions_by_side'][side] += 1
                    
                    # Add to totals
                    result['total_positions'] += 1
                    result['total_value_usd'] += position_value
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving active positions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'total_positions': 0,
                'total_value_usd': 0.0,
                'positions_by_strategy': {},
                'positions_by_symbol': {},
                'positions_by_side': {
                    'long': 0,
                    'short': 0
                },
                'error': str(e)
            }

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