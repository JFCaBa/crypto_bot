#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot
========================
Main entry point for the trading bot application.
"""

import asyncio
import argparse
import json
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from loguru import logger

from cryptobot.core.engine import TradingEngine
from cryptobot.utils.helpers import setup_logger


# Global variables
trading_engine = None
shutdown_event = None


async def start_bot(args):
    """
    Start the trading bot.
    
    Args:
        args: Command line arguments
    """
    global trading_engine, shutdown_event
    
    # Initialize shutdown event
    shutdown_event = asyncio.Event()
    
    # Create config directory if it doesn't exist
    config_dir = os.path.dirname(args.config)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        logger.info(f"Creating default configuration file: {args.config}")
        create_default_config(args.config)
        
    # Initialize trading engine
    trading_engine = TradingEngine(
        config_path=args.config,
        log_level=args.log_level.upper(),
        mode=args.mode
    )
    
    # Start trading engine
    if await trading_engine.start():
        logger.info("Trading bot started successfully")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Stop trading engine
        await trading_engine.stop()
        
    else:
        logger.error("Failed to start trading bot")
        sys.exit(1)
        
    logger.info("Trading bot stopped")
    

async def run_backtest(args):
    """
    Run backtest mode.
    
    Args:
        args: Command line arguments
    """
    # Initialize trading engine in backtest mode
    engine = TradingEngine(
        config_path=args.config,
        log_level=args.log_level.upper(),
        mode='backtest'
    )
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Run backtest
    results = await engine.run_backtest(
        strategy_id=args.strategy,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe
    )
    
    # Display results
    print(json.dumps(results, indent=2))
    

def handle_signal(signal_num, frame):
    """
    Handle termination signals.
    
    Args:
        signal_num: Signal number
        frame: Current frame
    """
    logger.info(f"Received signal {signal_num}, shutting down...")
    
    if shutdown_event:
        shutdown_event.set()
    

def create_default_config(config_path: str):
    """
    Create default configuration file.
    
    Args:
        config_path: Path to configuration file
    """
    default_config = {
        "database": {
            "enabled": False,
            "url": "localhost",
            "port": 5432,
            "username": "user",
            "password": "password",
            "database": "cryptobot"
        },
        "cache_enabled": True,
        "cache_dir": ".cache",
        "historical_data": {
            "enabled": True,
            "source": "csv",
            "data_dir": "data",
            "api_key": ""
        },
        "exchanges": {
            "binance": {
                "enabled": True,
                "api_key": "",
                "api_secret": "",
                "encrypted": False,
                "rate_limit": True,
                "timeout": 30000
            },
            "coinbase": {
                "enabled": False,
                "api_key": "",
                "api_secret": "",
                "encrypted": False,
                "rate_limit": True,
                "timeout": 30000
            },
            "kraken": {
                "enabled": False,
                "api_key": "",
                "api_secret": "",
                "encrypted": False,
                "rate_limit": True,
                "timeout": 30000
            }
        },
        "strategies": {
            "ma_crossover": {
                "enabled": True,
                "type": "MovingAverageCrossover",
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "timeframes": ["1h", "4h"],
                "params": {
                    "fast_period": 10,
                    "slow_period": 50,
                    "signal_period": 9,
                    "ma_type": "ema",
                    "use_macd": False,
                    "entry_threshold": 0.0,
                    "exit_threshold": 0.0,
                    "trailing_stop": 0.0,
                    "stop_loss": 2.0,
                    "take_profit": 4.0,
                    "risk_per_trade": 0.01
                }
            },
            "rsi": {
                "enabled": False,
                "type": "RSI",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1h"],
                "params": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "stop_loss": 2.0,
                    "take_profit": 4.0,
                    "risk_per_trade": 0.01
                }
            }
        },
        "risk_management": {
            "enabled": True,
            "max_positions": 5,
            "max_daily_trades": 20,
            "max_drawdown_percent": 20.0,
            "max_risk_per_trade": 2.0,
            "max_risk_per_day": 5.0,
            "max_risk_per_symbol": 10.0,
            "default_stop_loss": 2.0,
            "default_take_profit": 4.0,
            "correlation_limit": 0.7,
            "night_trading": True,
            "weekend_trading": True,
            "account_size": 10000.0,
            "params": {
                "volatility_threshold": 5.0,
                "price_change_threshold": 10.0
            }
        },
        "notifications": {
            "enabled": True,
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your_email@gmail.com",
                "password": "",
                "sender": "your_email@gmail.com",
                "recipients": ["your_email@gmail.com"]
            },
            "telegram": {
                "enabled": False,
                "token": "",
                "chat_ids": []
            }
        },
        "loop_interval": 60,
        "backtest": {
            "initial_balance": 10000.0,
            "maker_fee": 0.001,
            "taker_fee": 0.002,
            "slippage": 0.001,
            "enable_margin": False,
            "leverage": 1.0,
            "debug": False,
            "generate_plots": True,
            "results_dir": "backtest_results"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
        

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    
    # Base arguments
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--log-level", "-l", default="info", choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Run bot parser
    run_parser = subparsers.add_parser("run", help="Run trading bot")
    run_parser.add_argument("--mode", "-m", default="production", choices=["production", "test"],
                          help="Trading mode")
    
    # Backtest parser
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--strategy", "-s", required=True, help="Strategy ID")
    backtest_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe for backtest (e.g., 1m, 5m, 1h, 1d)")
    
    # Initialize configuration parser
    init_parser = subparsers.add_parser("init", help="Initialize configuration")
    
    # Status parser
    status_parser = subparsers.add_parser("status", help="Show trading bot status")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Execute command
    if args.command == "run":
        asyncio.run(start_bot(args))
    elif args.command == "backtest":
        asyncio.run(run_backtest(args))
    elif args.command == "init":
        create_default_config(args.config)
        print(f"Initialized configuration file: {args.config}")
    elif args.command == "status":
        # Initialize engine to get status
        engine = TradingEngine(
            config_path=args.config,
            log_level=args.log_level.upper(),
            mode="test"
        )
        status = asyncio.run(engine.get_status())
        print(json.dumps(status, indent=2))
    else:
        parser.print_help()
        

if __name__ == "__main__":
    main()