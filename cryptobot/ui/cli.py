"""
Command Line Interface
=====================
CLI for controlling and monitoring the trading bot.
"""

import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
from tabulate import tabulate
from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from cryptobot.core.engine import TradingEngine
from cryptobot.utils.helpers import setup_logger


class CLI:
    """Command Line Interface for the trading bot."""
    
    def __init__(self, trading_engine: TradingEngine):
        """
        Initialize the CLI.
        
        Args:
            trading_engine: TradingEngine instance
        """
        self.trading_engine = trading_engine
        self.running = False
        
        # Setup prompt session
        self.commands = [
            'start', 'stop', 'status', 'balance', 'positions', 'trades',
            'strategies', 'markets', 'backtest', 'settings', 'help', 'exit'
        ]
        self.command_completer = WordCompleter(self.commands)
        self.session = PromptSession(completer=self.command_completer)
        
        # Setup prompt style
        self.style = Style.from_dict({
            'prompt': 'fg:ansigreen bold',
            'command': 'fg:ansibrightcyan',
            'param': 'fg:ansiyellow',
            'error': 'fg:ansired bold',
            'success': 'fg:ansigreen',
            'warning': 'fg:ansiyellow',
            'info': 'fg:ansiblue',
        })
        
        logger.info("CLI initialized")
        
    async def start(self):
        """Start the CLI."""
        self.running = True
        clear()
        self._print_header()
        
        while self.running:
            try:
                # Get command from user
                command_input = await self.session.prompt_async(
                    HTML('<prompt>cryptobot></prompt> '),
                    style=self.style
                )
                
                # Parse command
                command_parts = command_input.strip().split()
                if not command_parts:
                    continue
                    
                command = command_parts[0].lower()
                args = command_parts[1:]
                
                # Process command
                if command == 'exit':
                    await self._handle_exit()
                    break
                elif command == 'help':
                    self._handle_help()
                elif command == 'start':
                    await self._handle_start()
                elif command == 'stop':
                    await self._handle_stop()
                elif command == 'status':
                    await self._handle_status()
                elif command == 'balance':
                    await self._handle_balance()
                elif command == 'positions':
                    await self._handle_positions()
                elif command == 'trades':
                    await self._handle_trades(args)
                elif command == 'strategies':
                    await self._handle_strategies(args)
                elif command == 'markets':
                    await self._handle_markets(args)
                elif command == 'backtest':
                    await self._handle_backtest(args)
                elif command == 'settings':
                    await self._handle_settings(args)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                await self._handle_exit()
                break
            except Exception as e:
                logger.error(f"Error processing command: {str(e)}")
                print(f"Error: {str(e)}")
                
    async def _handle_exit(self):
        """Handle exit command."""
        if await self._confirm_action("Are you sure you want to exit?"):
            # Stop the bot if it's running
            status = await self.trading_engine.get_status()
            if status.get("is_running", False):
                print("Stopping trading bot before exit...")
                await self.trading_engine.stop()
                
            self.running = False
            print("Exiting CLI. Goodbye!")
            
    def _handle_help(self):
        """Handle help command."""
        help_text = """
Available Commands:
------------------
start       - Start the trading bot
stop        - Stop the trading bot
status      - Show current bot status
balance     - Show account balances
positions   - Show active positions
trades      - Show recent trades
             Options: --limit=N, --strategy=NAME
strategies  - List or manage strategies
             Sub-commands: list, info NAME, enable NAME, disable NAME
markets     - Show available markets
             Options: --exchange=NAME
backtest    - Run a backtest
             Required: --strategy=NAME --start=YYYY-MM-DD
             Optional: --end=YYYY-MM-DD --timeframe=1h
settings    - View or change settings
             Sub-commands: view, set PARAM VALUE
help        - Show this help message
exit        - Exit the CLI
        """
        print(help_text)
        
    async def _handle_start(self):
        """Handle start command."""
        status = await self.trading_engine.get_status()
        if status.get("is_running", False):
            print("Trading bot is already running.")
            return
            
        print("Starting trading bot...")
        success = await self.trading_engine.start()
        if success:
            print("Trading bot started successfully.")
        else:
            print("Failed to start trading bot. Check logs for details.")
            
    async def _handle_stop(self):
        """Handle stop command."""
        status = await self.trading_engine.get_status()
        if not status.get("is_running", False):
            print("Trading bot is not running.")
            return
            
        if await self._confirm_action("Are you sure you want to stop the trading bot?"):
            print("Stopping trading bot...")
            success = await self.trading_engine.stop()
            if success:
                print("Trading bot stopped successfully.")
            else:
                print("Failed to stop trading bot. Check logs for details.")
                
    async def _handle_status(self):
        """Handle status command."""
        print("Fetching trading bot status...")
        status = await self.trading_engine.get_status()
        
        # Display basic status
        print("\n=== Bot Status ===")
        print(f"Mode: {status.get('mode', 'Unknown')}")
        print(f"Running: {'Yes' if status.get('is_running', False) else 'No'}")
        if status.get('start_time'):
            print(f"Start time: {status.get('start_time')}")
            print(f"Uptime: {status.get('uptime', 'Unknown')}")
        print(f"Last heartbeat: {status.get('last_heartbeat', 'Unknown')}")
        if status.get('last_error'):
            print(f"Last error: {status.get('last_error')}")
            
        # Display exchange status
        print("\n=== Exchange Status ===")
        for exchange_id, exchange_status in status.get('exchanges', {}).items():
            print(f"{exchange_id}: {'Connected' if exchange_status.get('connected', False) else 'Disconnected'}")
            print(f"  API calls: {exchange_status.get('api_calls', 0)}")
            
        # Display active strategies
        print("\n=== Active Strategies ===")
        active_strategies = 0
        for strategy_id, strategy_info in status.get('strategies', {}).items():
            if strategy_info.get('is_active', False):
                active_strategies += 1
                print(f"- {strategy_id} ({strategy_info.get('name', 'Unknown')})")
                
        if active_strategies == 0:
            print("No active strategies")
            
        # Display risk management status
        if status.get('risk_management'):
            print("\n=== Risk Management ===")
            risk_info = status.get('risk_management', {})
            print(f"Account size: ${risk_info.get('account_size', 0):.2f}")
            print(f"Current drawdown: {risk_info.get('current_drawdown', 0):.2f}%")
            print(f"Daily trades count: {risk_info.get('daily_trades_count', 0)}/{risk_info.get('max_daily_trades', 0)}")
            if risk_info.get('kill_switch_active', False):
                print(f"Kill switch: ACTIVE - {risk_info.get('kill_switch_reason', 'Unknown')}")
                
    async def _handle_balance(self):
        """Handle balance command."""
        print("Fetching account balances...")
        status = await self.trading_engine.get_status()
        account_balances = status.get('account_balances', {})
        
        if not account_balances:
            print("No account balance information available.")
            return
            
        # Process and display balances
        for exchange_id, balance_info in account_balances.items():
            print(f"\n=== {exchange_id} Balance ===")
            
            if "total" in balance_info:
                # CCXT standard balance format
                assets = []
                for currency, amount in balance_info.get("total", {}).items():
                    free = balance_info.get("free", {}).get(currency, 0)
                    used = balance_info.get("used", {}).get(currency, 0)
                    if amount > 0:
                        assets.append({
                            "Asset": currency,
                            "Total": amount,
                            "Free": free,
                            "In Use": used
                        })
                
                if assets:
                    print(tabulate(assets, headers="keys", tablefmt="pretty"))
                else:
                    print("No assets with non-zero balance.")
            elif "info" in balance_info and "balances" in balance_info["info"]:
                # Binance-specific balance format
                assets = []
                for asset in balance_info["info"]["balances"]:
                    free = float(asset["free"])
                    locked = float(asset["locked"])
                    total = free + locked
                    if total > 0:
                        assets.append({
                            "Asset": asset["asset"],
                            "Total": total,
                            "Free": free,
                            "Locked": locked
                        })
                
                if assets:
                    print(tabulate(assets, headers="keys", tablefmt="pretty"))
                else:
                    print("No assets with non-zero balance.")
            else:
                print("Unknown balance format.")
                
    async def _handle_positions(self):
        """Handle positions command."""
        print("Fetching active positions...")
        status = await self.trading_engine.get_status()
        
        active_positions = []
        for strategy_id, strategy_info in status.get('strategies', {}).items():
            for symbol, position in strategy_info.get('positions', {}).items():
                if position.get('is_active', False):
                    active_positions.append({
                        "Symbol": symbol,
                        "Strategy": strategy_id,
                        "Side": position.get('side', '').title(),
                        "Entry Price": position.get('entry_price', 0),
                        "Amount": position.get('amount', 0),
                        "Entry Time": position.get('entry_time', '')
                    })
        
        if active_positions:
            print("\n=== Active Positions ===")
            print(tabulate(active_positions, headers="keys", tablefmt="pretty"))
        else:
            print("No active positions.")
            
    async def _handle_trades(self, args):
        """
        Handle trades command.
        
        Args:
            args: Command arguments
        """
        # Parse arguments
        limit = 10  # Default limit
        strategy = None
        
        for arg in args:
            if arg.startswith('--limit='):
                try:
                    limit = int(arg.split('=')[1])
                except ValueError:
                    print("Invalid limit value. Using default limit of 10.")
            elif arg.startswith('--strategy='):
                strategy = arg.split('=')[1]
        
        print(f"Fetching last {limit} trades{' for strategy ' + strategy if strategy else ''}...")
        status = await self.trading_engine.get_status()
        
        all_trades = []
        for strategy_id, strategy_info in status.get('strategies', {}).items():
            if strategy and strategy_id != strategy:
                continue
                
            # In a real implementation, we would get the actual trade history
            for trade in strategy_info.get('trade_history', []):
                trade_info = {
                    "Symbol": trade.get('symbol', ''),
                    "Strategy": strategy_id,
                    "Side": trade.get('side', '').title(),
                    "Price": trade.get('price', 0),
                    "Amount": trade.get('amount', 0),
                    "Time": trade.get('timestamp', ''),
                    "PnL": trade.get('pnl', 0),
                    "PnL %": trade.get('pnl_percent', 0)
                }
                all_trades.append(trade_info)
        
        # Sort by timestamp (most recent first)
        all_trades = sorted(all_trades, key=lambda x: x.get("Time", ""), reverse=True)
        
        # Limit to the requested number of trades
        all_trades = all_trades[:limit]
        
        if all_trades:
            print("\n=== Recent Trades ===")
            print(tabulate(all_trades, headers="keys", tablefmt="pretty"))
        else:
            print("No trade history found.")
            
    async def _handle_strategies(self, args):
        """
        Handle strategies command.
        
        Args:
            args: Command arguments
        """
        if not args:
            # Default: list all strategies
            await self._list_strategies()
            return
            
        subcommand = args[0].lower()
        
        if subcommand == 'list':
            await self._list_strategies()
        elif subcommand == 'info' and len(args) > 1:
            await self._show_strategy_info(args[1])
        elif subcommand == 'enable' and len(args) > 1:
            await self._enable_strategy(args[1])
        elif subcommand == 'disable' and len(args) > 1:
            await self._disable_strategy(args[1])
        else:
            print("Invalid strategy subcommand. Use: list, info NAME, enable NAME, disable NAME")
            
    async def _list_strategies(self):
        """List all available strategies."""
        print("Fetching strategies...")
        status = await self.trading_engine.get_status()
        
        strategies = []
        for strategy_id, strategy_info in status.get('strategies', {}).items():
            strategies.append({
                "ID": strategy_id,
                "Name": strategy_info.get('name', 'Unknown'),
                "Symbols": ', '.join(strategy_info.get('symbols', [])),
                "Timeframes": ', '.join(strategy_info.get('timeframes', [])),
                "Active": 'Yes' if strategy_info.get('is_active', False) else 'No',
                "Trades": strategy_info.get('trade_count', 0)
            })
        
        if strategies:
            print("\n=== Available Strategies ===")
            print(tabulate(strategies, headers="keys", tablefmt="pretty"))
        else:
            print("No strategies configured.")
            
    async def _show_strategy_info(self, strategy_id):
        """
        Show detailed information about a strategy.
        
        Args:
            strategy_id: Strategy identifier
        """
        print(f"Fetching information for strategy {strategy_id}...")
        status = await self.trading_engine.get_status()
        
        if strategy_id not in status.get('strategies', {}):
            print(f"Strategy {strategy_id} not found.")
            return
            
        strategy_info = status['strategies'][strategy_id]
        
        print(f"\n=== Strategy: {strategy_id} ===")
        print(f"Name: {strategy_info.get('name', 'Unknown')}")
        print(f"Active: {'Yes' if strategy_info.get('is_active', False) else 'No'}")
        print(f"Symbols: {', '.join(strategy_info.get('symbols', []))}")
        print(f"Timeframes: {', '.join(strategy_info.get('timeframes', []))}")
        
        # Parameters
        print("\nParameters:")
        params = strategy_info.get('params', {})
        for param, value in params.items():
            print(f"  {param}: {value}")
            
        # Performance
        print("\nPerformance:")
        perf = strategy_info.get('performance', {})
        print(f"  Total trades: {perf.get('total_trades', 0)}")
        print(f"  Win rate: {perf.get('win_rate', 0):.2f}%")
        print(f"  Average profit: {perf.get('avg_profit', 0):.2f}%")
        print(f"  Total PnL: {perf.get('total_pnl', 0):.2f}")
        print(f"  Profit factor: {perf.get('profit_factor', 0):.2f}")
        
    async def _enable_strategy(self, strategy_id):
        """
        Enable a strategy.
        
        Args:
            strategy_id: Strategy identifier
        """
        print(f"Enabling strategy {strategy_id}...")
        # In a real implementation, this would update the strategy in the engine
        # For now, we'll just print a message
        print(f"Strategy {strategy_id} enabled. (Not actually implemented in this demo)")
        
    async def _disable_strategy(self, strategy_id):
        """
        Disable a strategy.
        
        Args:
            strategy_id: Strategy identifier
        """
        print(f"Disabling strategy {strategy_id}...")
        # In a real implementation, this would update the strategy in the engine
        # For now, we'll just print a message
        print(f"Strategy {strategy_id} disabled. (Not actually implemented in this demo)")
        
    async def _handle_markets(self, args):
        """
        Handle markets command.
        
        Args:
            args: Command arguments
        """
        # Parse arguments
        exchange = None
        
        for arg in args:
            if arg.startswith('--exchange='):
                exchange = arg.split('=')[1]
        
        print(f"Fetching available markets{' for exchange ' + exchange if exchange else ''}...")
        
        if not self.trading_engine.exchanges:
            print("No exchanges connected.")
            return
            
        for exchange_id, exchange_instance in self.trading_engine.exchanges.items():
            if exchange and exchange_id != exchange:
                continue
                
            print(f"\n=== {exchange_id} Markets ===")
            
            # In a real implementation, we would get the actual markets
            # For now, we'll just show a subset of common symbols
            symbols = exchange_instance.get_supported_symbols()
            if not symbols:
                symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]
                
            # Limit to first 20 symbols
            if len(symbols) > 20:
                print(f"Showing first 20 of {len(symbols)} symbols")
                symbols = symbols[:20]
                
            symbol_data = []
            for symbol in symbols:
                symbol_data.append({"Symbol": symbol})
                
            print(tabulate(symbol_data, headers="keys", tablefmt="pretty"))
            
    async def _handle_backtest(self, args):
        """
        Handle backtest command.
        
        Args:
            args: Command arguments
        """
        # Parse arguments
        strategy = None
        start_date = None
        end_date = None
        timeframe = "1h"  # Default timeframe
        
        for arg in args:
            if arg.startswith('--strategy='):
                strategy = arg.split('=')[1]
            elif arg.startswith('--start='):
                start_date = arg.split('=')[1]
            elif arg.startswith('--end='):
                end_date = arg.split('=')[1]
            elif arg.startswith('--timeframe='):
                timeframe = arg.split('=')[1]
        
        if not strategy or not start_date:
            print("Required parameters missing. Usage: backtest --strategy=NAME --start=YYYY-MM-DD [--end=YYYY-MM-DD] [--timeframe=1h]")
            return
            
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            return
            
        print(f"Running backtest for strategy {strategy} from {start_date} to {end_date or 'now'} with timeframe {timeframe}...")
        
        # Run backtest
        results = await self.trading_engine.run_backtest(
            strategy_id=strategy,
            start_date=start_date_obj,
            end_date=end_date_obj,
            timeframe=timeframe
        )
        
        if not results:
            print("Backtest failed or returned no results.")
            return
            
        # Display backtest results
        print("\n=== Backtest Results ===")
        print(f"Strategy: {strategy}")
        print(f"Period: {start_date} to {end_date or datetime.now().strftime('%Y-%m-%d')}")
        print(f"Timeframe: {timeframe}")
        print(f"Start balance: ${results.get('start_balance', 0):.2f}")
        print(f"End balance: ${results.get('end_balance', 0):.2f}")
        print(f"Return: {((results.get('end_balance', 0) / results.get('start_balance', 1)) - 1) * 100:.2f}%")
        print(f"Buy & Hold return: {results.get('buy_hold_return', 0):.2f}%")
        print(f"Total trades: {results.get('total_trades', 0)}")
        print(f"Win rate: {results.get('win_rate', 0):.2f}%")
        print(f"Profit factor: {results.get('profit_factor', 0):.2f}")
        print(f"Max drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        print(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
        
        # Indicate where results are saved
        results_dir = self.trading_engine.config.get('backtest', {}).get('results_dir', 'backtest_results')
        print(f"\nDetailed results saved to {results_dir}/{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_*.csv")
        
    async def _handle_settings(self, args):
        """
        Handle settings command.
        
        Args:
            args: Command arguments
        """
        if not args:
            # Default: view all settings
            await self._view_settings()
            return
            
        subcommand = args[0].lower()
        
        if subcommand == 'view':
            await self._view_settings()
        elif subcommand == 'set' and len(args) > 2:
            await self._set_setting(args[1], args[2])
        else:
            print("Invalid settings subcommand. Use: view, set PARAM VALUE")
            
    async def _view_settings(self):
        """View all settings."""
        print("Fetching configuration settings...")
        
        # Display basic settings
        print("\n=== Configuration Settings ===")
        print(f"Config path: {self.trading_engine.config_path}")
        print(f"Mode: {self.trading_engine.mode}")
        
        # Display database settings
        db_config = self.trading_engine.config.get('database', {})
        print("\nDatabase:")
        print(f"  Enabled: {db_config.get('enabled', False)}")
        if db_config.get('enabled', False):
            print(f"  URL: {db_config.get('url', 'localhost')}")
            print(f"  Database: {db_config.get('database', 'cryptobot')}")
            
        # Display cache settings
        print("\nCache:")
        print(f"  Enabled: {self.trading_engine.config.get('cache_enabled', True)}")
        print(f"  Directory: {self.trading_engine.config.get('cache_dir', '.cache')}")
        
        # Display historical data settings
        hist_config = self.trading_engine.config.get('historical_data', {})
        print("\nHistorical Data:")
        print(f"  Enabled: {hist_config.get('enabled', True)}")
        print(f"  Source: {hist_config.get('source', 'csv')}")
        print(f"  Directory: {hist_config.get('data_dir', 'data')}")
        
        # Display risk management settings
        risk_config = self.trading_engine.config.get('risk_management', {})
        print("\nRisk Management:")
        print(f"  Enabled: {risk_config.get('enabled', True)}")
        print(f"  Max positions: {risk_config.get('max_positions', 5)}")
        print(f"  Max daily trades: {risk_config.get('max_daily_trades', 20)}")
        print(f"  Max drawdown: {risk_config.get('max_drawdown_percent', 20.0)}%")
        print(f"  Max risk per trade: {risk_config.get('max_risk_per_trade', 2.0)}%")
        
        # Display notification settings
        notif_config = self.trading_engine.config.get('notifications', {})
        print("\nNotifications:")
        print(f"  Enabled: {notif_config.get('enabled', False)}")
        print(f"  Email: {notif_config.get('email', {}).get('enabled', False)}")
        print(f"  Telegram: {notif_config.get('telegram', {}).get('enabled', False)}")
        
        # Display loop interval
        print(f"\nLoop interval: {self.trading_engine.config.get('loop_interval', 60)} seconds")
        
    async def _set_setting(self, param, value):
        """
        Set a configuration parameter.
        
        Args:
            param: Parameter name
            value: Parameter value
        """
        print(f"Setting {param} to {value}...")
        # In a real implementation, this would update the configuration
        # For now, we'll just print a message
        print(f"Configuration updated. (Not actually implemented in this demo)")
        
    async def _confirm_action(self, message) -> bool:
        """
        Ask for confirmation before executing an action.
        
        Args:
            message: Confirmation message
            
        Returns:
            bool: True if confirmed, False otherwise
        """
        response = await self.session.prompt_async(
            HTML(f'{message} (y/n): '),
            style=self.style
        )
        return response.lower() in ['y', 'yes']
        
    def _print_header(self):
        """Print the CLI header."""
        header = """
        ╔═══════════════════════════════════════════════════╗
        ║                                                   ║
        ║   ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗     ║
        ║  ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗    ║
        ║  ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║    ║
        ║  ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║    ║
        ║  ╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝    ║
        ║   ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝     ║
        ║                                                   ║
        ║         Automated Cryptocurrency Trading Bot         ║
        ║                                                   ║
        ╚═══════════════════════════════════════════════════╝
        
        Type 'help' for available commands.
        """
        print(header)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CryptoBot CLI")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--mode", "-m", default="test", choices=["production", "test"],
                       help="Trading mode")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logger(args.log_level)
    
    # Initialize trading engine
    trading_engine = TradingEngine(
        config_path=args.config,
        log_level=args.log_level,
        mode=args.mode
    )
    
    # Initialize CLI
    cli = CLI(trading_engine)
    
    # Run CLI
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\nExiting CLI. Goodbye!")
    except Exception as e:
        logger.error(f"Error running CLI: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()