"""
Configuration Settings
===================
Handles loading, saving, and validating configuration settings.
"""

import os
import json
import copy
from typing import Dict, Any

from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration as a dictionary
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return create_default_config(config_path)
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.info("Creating default configuration")
        return create_default_config(config_path)
        
        
def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False
        
        
def create_default_config(config_path: str = None) -> Dict[str, Any]:
    """
    Create default configuration.
    
    Args:
        config_path: Path to save the configuration file (optional)
        
    Returns:
        dict: Default configuration
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
            "symbols": ["BTC/USDT"],
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
            "enabled": True,
            "type": "RSI",
            "symbols": ["BTC/USDT"],
            "timeframes": ["1h"],
            "params": {
                "period": 14,
                "overbought": 70,
                "oversold": 30,
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "risk_per_trade": 0.01
            }
            },
            "bollinger_bands": {
            "enabled": True,
            "type": "BollingerBands",
            "symbols": ["BTC/USDT"],
            "timeframes": ["1h"],
            "params": {
                "period": 14,
                "overbought": 70,
                "oversold": 30,
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "risk_per_trade": 0.01
            }
            },
            "ml_strategy_1": {
            "enabled": True,
            "type": "MachineLearning",
            "symbols": ["BTC/USDT"],
            "timeframes": ["1h"],
            "params": {
                "lookback_period": 200,
                "train_interval": 720
            }
            }
        },
        "risk_management": {
            "enabled": True,
            "max_positions": 5,
            "max_daily_trades": 40,
            "max_drawdown_percent": 10.0,
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
    
    # Save config if path is provided
    if config_path:
        save_config(default_config, config_path)
        
    return default_config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and fill in missing values with defaults.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Validated configuration
    """
    default_config = create_default_config()
    validated_config = copy.deepcopy(default_config)
    
    # Helper function to recursively merge dictionaries
    def merge_dicts(source, destination):
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                merge_dicts(value, destination[key])
            else:
                destination[key] = value
                
    # Merge user config with default config
    merge_dicts(config, validated_config)
    
    # Additional validation logic can be added here
    
    return validated_config


def get_exchange_config(config: Dict[str, Any], exchange_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific exchange.
    
    Args:
        config: Configuration dictionary
        exchange_id: Exchange identifier
        
    Returns:
        dict: Exchange configuration
    """
    exchanges = config.get('exchanges', {})
    return exchanges.get(exchange_id, {})


def get_strategy_config(config: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific strategy.
    
    Args:
        config: Configuration dictionary
        strategy_id: Strategy identifier
        
    Returns:
        dict: Strategy configuration
    """
    strategies = config.get('strategies', {})
    return strategies.get(strategy_id, {})