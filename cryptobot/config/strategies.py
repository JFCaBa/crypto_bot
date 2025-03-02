"""
Strategies Configuration
=====================
Configuration for trading strategies.
"""

import importlib
from typing import Dict, List, Optional, Any, Type

from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


# Supported strategy types and their class and module paths
STRATEGY_TYPES = {
    'MovingAverageCrossover': {
        'class': 'MovingAverageCrossover',
        'module': 'cryptobot.strategies.moving_average'
    },
    'RSI': {
        'class': 'RSIStrategy',
        'module': 'cryptobot.strategies.rsi'
    },
    'BollingerBands': {
        'class': 'BollingerBandsStrategy',
        'module': 'cryptobot.strategies.bollinger_bands'
    },
    'Custom': {
        'class': 'CustomStrategy',
        'module': 'cryptobot.strategies.custom'
    },
    'MachineLearning': {
        'class': 'MachineLearningStrategy',
        'module': 'cryptobot.strategies.machine_learning'
    }
}

# Default parameters for each strategy type
DEFAULT_STRATEGY_PARAMS = {
    'MovingAverageCrossover': {
        'fast_period': 10,
        'slow_period': 50,
        'signal_period': 9,
        'ma_type': 'ema',
        'use_macd': False,
        'entry_threshold': 0.0,
        'exit_threshold': 0.0,
        'trailing_stop': 0.0,
        'stop_loss': 2.0,
        'take_profit': 4.0,
        'risk_per_trade': 0.01
    },
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'stop_loss': 2.0,
        'take_profit': 4.0,
        'risk_per_trade': 0.01
    },
    'BollingerBands': {
        'period': 20,
        'std_dev': 2.0,
        'ma_type': 'sma',
        'entry_trigger': 'touch',
        'exit_trigger': 'middle',
        'use_volume': False,
        'volume_threshold': 1.5,
        'stop_loss': 2.0,
        'take_profit': 4.0,
        'trailing_stop': 0.0,
        'risk_per_trade': 0.01
    }
}


def get_strategy_class(strategy_type: str) -> Tuple[Optional[Type[BaseStrategy]], Optional[str]]:
    """
    Get the strategy class for a strategy type.
    
    Args:
        strategy_type: Strategy type
        
    Returns:
        tuple: (Strategy class, error message) or (None, error message) if not found
    """
    if strategy_type not in STRATEGY_TYPES:
        return None, f"Unsupported strategy type: {strategy_type}"
        
    strategy_info = STRATEGY_TYPES[strategy_type]
    module_path = strategy_info['module']
    class_name = strategy_info['class']
    
    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        return strategy_class, None
    except ImportError:
        return None, f"Could not import module: {module_path}"
    except AttributeError:
        return None, f"Could not find class {class_name} in module {module_path}"
    except Exception as e:
        return None, f"Error loading strategy class: {str(e)}"


def get_default_params(strategy_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a strategy type.
    
    Args:
        strategy_type: Strategy type
        
    Returns:
        dict: Default parameters
    """
    return DEFAULT_STRATEGY_PARAMS.get(strategy_type, {}).copy()


def get_enabled_strategies(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get all enabled strategies from the configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        dict: Dictionary of enabled strategies
    """
    strategies_config = config.get('strategies', {})
    enabled_strategies = {}
    
    for strategy_id, strategy_config in strategies_config.items():
        if strategy_config.get('enabled', False):
            enabled_strategies[strategy_id] = strategy_config
            
    return enabled_strategies


def validate_strategy_config(strategy_id: str, strategy_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Validate strategy configuration.
    
    Args:
        strategy_id: Strategy identifier
        strategy_config: Strategy configuration
        
    Returns:
        tuple: (Validated configuration, error message) or (config, None) if valid
    """
    strategy_type = strategy_config.get('type')
    if not strategy_type:
        return strategy_config, "Strategy type not specified"
        
    if strategy_type not in STRATEGY_TYPES:
        return strategy_config, f"Unsupported strategy type: {strategy_type}"
        
    # Check required fields
    required_fields = ['symbols', 'timeframes']
    for field in required_fields:
        if field not in strategy_config or not strategy_config[field]:
            return strategy_config, f"Required field missing: {field}"
            
    # Merge with default parameters
    default_params = get_default_params(strategy_type)
    params = strategy_config.get('params', {})
    
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
            
    strategy_config['params'] = params
    
    return strategy_config, None


def initialize_strategies(config: Dict[str, Any], risk_manager: Optional[RiskManager] = None) -> Dict[str, BaseStrategy]:
    """
    Initialize strategy instances based on configuration.
    
    Args:
        config: Application configuration
        risk_manager: Risk manager instance
        
    Returns:
        dict: Dictionary of strategy instances
    """
    strategies = {}
    enabled_strategies = get_enabled_strategies(config)
    
    for strategy_id, strategy_config in enabled_strategies.items():
        try:
            # Validate strategy configuration
            validated_config, error = validate_strategy_config(strategy_id, strategy_config)
            if error:
                logger.error(f"Invalid configuration for strategy {strategy_id}: {error}")
                continue
                
            # Get strategy class
            strategy_type = validated_config['type']
            strategy_class, error = get_strategy_class(strategy_type)
            
            if error:
                logger.error(error)
                continue
                
            # Initialize strategy
            strategy = strategy_class(
                symbols=validated_config['symbols'],
                timeframes=validated_config['timeframes'],
                risk_manager=risk_manager,
                params=validated_config['params']
            )
            
            strategies[strategy_id] = strategy
            logger.info(f"Initialized {strategy_id} strategy")
            
        except Exception as e:
            logger.error(f"Error initializing strategy {strategy_id}: {str(e)}")
            
    return strategies