"""
Helper Utilities
==============
Utility functions for the trading bot.
"""

import os
import sys
import functools
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, TypeVar

import pandas as pd
from loguru import logger


# Type variables for generic function decorators
T = TypeVar('T')
R = TypeVar('R')


def setup_logger(log_level: str = "INFO"):
    """
    Set up logger configuration.
    
    Args:
        log_level: Logging level
    """
    # Remove default loggers
    logger.remove()
    
    # Set up loggers
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"cryptobot_{datetime.now().strftime('%Y%m%d')}.log")
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",  # New file at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip"  # Compress rotated logs
    )
    
    logger.info(f"Logger initialized with level {log_level}")
    

def retry(max_retries: int = 3, delay: int = 1, backoff: int = 2, exceptions: tuple = (Exception,)):
    """
    Retry decorator for synchronous functions.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (how much to increase delay after each retry)
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            retry_count = 0
            current_delay = delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                        
                    logger.warning(f"Retry {retry_count}/{max_retries} for {func.__name__}: {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
        return wrapper
    return decorator
    

def retry_async(max_retries: int = 3, delay: int = 1, backoff: int = 2, exceptions: tuple = (Exception,)):
    """
    Retry decorator for asynchronous functions.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (how much to increase delay after each retry)
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            current_delay = delay
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                        
                    logger.warning(f"Retry {retry_count}/{max_retries} for {func.__name__}: {str(e)}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    
        return wrapper
    return decorator
    

def safe_execute(default_return=None, log_exception=True):
    """
    Safe execution decorator to catch and handle exceptions.
    
    Args:
        default_return: Default return value on exception
        log_exception: Whether to log the exception
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., Optional[R]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[R]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
                
        return wrapper
    return decorator
    

def safe_execute_async(default_return=None, log_exception=True):
    """
    Safe execution decorator for asynchronous functions.
    
    Args:
        default_return: Default return value on exception
        log_exception: Whether to log the exception
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
                
        return wrapper
    return decorator
    

def format_price(price: float, precision: int = 8) -> str:
    """
    Format price with specified precision.
    
    Args:
        price: Price value
        precision: Decimal precision
        
    Returns:
        str: Formatted price
    """
    return f"{price:.{precision}f}"
    

def calculate_change_percent(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0.0
        
    return ((new_value - old_value) / old_value) * 100
    

def normalize_symbol(symbol: str, exchange: str = None) -> str:
    """
    Normalize trading pair symbol format.
    
    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        
    Returns:
        str: Normalized symbol
    """
    if exchange == 'binance':
        # Binance uses BTCUSDT format
        return symbol.replace('/', '')
    elif exchange == 'coinbase':
        # Coinbase uses BTC-USD format
        return symbol.replace('/', '-')
    elif exchange == 'kraken':
        # Kraken uses XXBTUSDT or special symbols
        if symbol.startswith('BTC/'):
            return symbol.replace('BTC/', 'XXBT')
        return symbol.replace('/', '')
    else:
        # Default format: BTC/USDT
        return symbol
    

def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., 1m, 5m, 1h, 1d)
        
    Returns:
        int: Timeframe in seconds
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60 * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 60 * 24
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 60 * 60 * 24 * 7
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    

def parse_timeframe(timeframe: str) -> Dict[str, int]:
    """
    Parse timeframe string into a dictionary.
    
    Args:
        timeframe: Timeframe string (e.g., 1m, 5m, 1h, 1d)
        
    Returns:
        dict: Timeframe components
    """
    if timeframe.endswith('m'):
        return {'minutes': int(timeframe[:-1])}
    elif timeframe.endswith('h'):
        return {'hours': int(timeframe[:-1])}
    elif timeframe.endswith('d'):
        return {'days': int(timeframe[:-1])}
    elif timeframe.endswith('w'):
        return {'weeks': int(timeframe[:-1])}
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    

def round_to_tick_size(value: float, tick_size: float) -> float:
    """
    Round a value to the nearest tick size.
    
    Args:
        value: Value to round
        tick_size: Tick size
        
    Returns:
        float: Rounded value
    """
    return round(value / tick_size) * tick_size
    

def to_camel_case(snake_str: str) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        snake_str: String in snake_case
        
    Returns:
        str: String in camelCase
    """
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])
    

def to_snake_case(camel_str: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        camel_str: String in camelCase
        
    Returns:
        str: String in snake_case
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    

def truncate_float(value: float, decimals: int) -> float:
    """
    Truncate a float to a specific number of decimal places.
    
    Args:
        value: Float value
        decimals: Number of decimal places
        
    Returns:
        float: Truncated float
    """
    factor = 10 ** decimals
    return int(value * factor) / factor