"""
Exchanges Configuration
=====================
Configuration for multiple exchanges.
"""

from typing import Dict, List, Optional, Any

from loguru import logger

from cryptobot.config.exchange import (
    is_exchange_supported,
    get_default_exchange_params,
    validate_exchange_config
)


def get_enabled_exchanges(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get all enabled exchanges from the configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        dict: Dictionary of enabled exchanges
    """
    exchanges_config = config.get('exchanges', {})
    enabled_exchanges = {}
    
    for exchange_id, exchange_config in exchanges_config.items():
        if exchange_config.get('enabled', False) and is_exchange_supported(exchange_id):
            enabled_exchanges[exchange_id] = exchange_config
            
    return enabled_exchanges


def get_exchange_class(exchange_id: str) -> Optional[str]:
    """
    Get the exchange class name for an exchange.
    
    Args:
        exchange_id: Exchange identifier
        
    Returns:
        str: Exchange class name or None if not found
    """
    exchange_classes = {
        'binance': 'BinanceExchange',
        'coinbase': 'CoinbaseExchange',
        'kraken': 'KrakenExchange'
    }
    
    return exchange_classes.get(exchange_id)


def get_exchange_module(exchange_id: str) -> Optional[str]:
    """
    Get the exchange module path for an exchange.
    
    Args:
        exchange_id: Exchange identifier
        
    Returns:
        str: Exchange module path or None if not found
    """
    exchange_modules = {
        'binance': 'cryptobot.exchanges.binance',
        'coinbase': 'cryptobot.exchanges.coinbase',
        'kraken': 'cryptobot.exchanges.kraken',
        'mexc': 'cryptobot.exchanges.mexc',
        'bybit': 'cryptobot.exchanges.bybit'
    }
    
    return exchange_modules.get(exchange_id)


def validate_exchanges_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the exchanges configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        dict: Validated configuration
    """
    exchanges_config = config.get('exchanges', {})
    
    # Validate each exchange
    for exchange_id, exchange_config in exchanges_config.items():
        if is_exchange_supported(exchange_id):
            if exchange_config.get('enabled', False):
                exchanges_config[exchange_id] = validate_exchange_config(exchange_id, exchange_config)
        else:
            logger.warning(f"Unsupported exchange: {exchange_id}")
            exchange_config['enabled'] = False
            
    # Make sure at least one exchange is enabled
    enabled_exchanges = [exchange_id for exchange_id, exchange_config in exchanges_config.items() 
                        if exchange_config.get('enabled', False)]
    
    if not enabled_exchanges:
        logger.warning("No enabled exchanges found")
        
    return config


def initialize_exchange_connectors(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize exchange connectors based on configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        dict: Dictionary of exchange instances
    """
    import importlib
    
    exchanges = {}
    enabled_exchanges = get_enabled_exchanges(config)
    
    for exchange_id, exchange_config in enabled_exchanges.items():
        try:
            # Get exchange class and module
            exchange_class_name = get_exchange_class(exchange_id)
            exchange_module_path = get_exchange_module(exchange_id)
            
            if not exchange_class_name or not exchange_module_path:
                logger.error(f"Exchange information not found for {exchange_id}")
                continue
                
            # Import the module
            module = importlib.import_module(exchange_module_path)
            
            # Get the class
            exchange_class = getattr(module, exchange_class_name)
            
            # Create the exchange instance
            exchanges[exchange_id] = exchange_class(
                api_key=exchange_config.get('api_key', ''),
                api_secret=exchange_config.get('api_secret', ''),
                sandbox=config.get('mode', 'production') != 'production',
                rate_limit=exchange_config.get('rate_limit', True),
                timeout=exchange_config.get('timeout', 30000)
            )
            
            logger.info(f"Initialized {exchange_id} exchange connector")
            
        except Exception as e:
            logger.error(f"Error initializing exchange {exchange_id}: {str(e)}")
            
    return exchanges


def get_common_symbols(exchanges: Dict[str, Any]) -> List[str]:
    """
    Get common symbols available across all exchanges.
    
    Args:
        exchanges: Dictionary of exchange instances
        
    Returns:
        list: List of common symbols
    """
    if not exchanges:
        return []
        
    # Get symbols for each exchange
    exchange_symbols = {}
    for exchange_id, exchange in exchanges.items():
        try:
            symbols = exchange.get_supported_symbols()
            exchange_symbols[exchange_id] = set(symbols)
        except Exception as e:
            logger.error(f"Error getting symbols for {exchange_id}: {str(e)}")
            exchange_symbols[exchange_id] = set()
            
    # Find common symbols
    common_symbols = set.intersection(*exchange_symbols.values()) if exchange_symbols else set()
    
    return sorted(list(common_symbols))


def get_common_timeframes(exchanges: Dict[str, Any]) -> List[str]:
    """
    Get common timeframes available across all exchanges.
    
    Args:
        exchanges: Dictionary of exchange instances
        
    Returns:
        list: List of common timeframes
    """
    if not exchanges:
        return []
        
    # Get timeframes for each exchange
    exchange_timeframes = {}
    for exchange_id, exchange in exchanges.items():
        try:
            timeframes = exchange.get_supported_timeframes()
            exchange_timeframes[exchange_id] = set(timeframes.keys())
        except Exception as e:
            logger.error(f"Error getting timeframes for {exchange_id}: {str(e)}")
            exchange_timeframes[exchange_id] = set()
            
    # Find common timeframes
    common_timeframes = set.intersection(*exchange_timeframes.values()) if exchange_timeframes else set()
    
    return sorted(list(common_timeframes))