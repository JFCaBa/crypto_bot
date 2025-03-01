"""
Input Validation Utilities
=======================
Provides validation functions for various inputs to ensure data integrity
and security throughout the application.
"""

import re
import json
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from loguru import logger


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading pair symbol format.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(symbol, str):
        return False
        
    # Common formats: BTC/USDT, BTC-USDT, BTCUSDT
    pattern = r'^[A-Za-z0-9]+(/|-)?[A-Za-z0-9]+$'
    
    return bool(re.match(pattern, symbol))


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(timeframe, str):
        return False
        
    # Pattern: number followed by m, h, d, w, M
    pattern = r'^[1-9][0-9]*[mhdwM]$'
    
    return bool(re.match(pattern, timeframe))


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(api_key, str):
        return False
        
    # Check if it's not empty
    if not api_key:
        return False
        
    # Most API keys are alphanumeric and at least 16 characters
    if len(api_key) < 16:
        return False
        
    # Check for valid characters
    pattern = r'^[A-Za-z0-9\-_]+$'
    
    return bool(re.match(pattern, api_key))


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(email, str):
        return False
        
    # Simple email pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return bool(re.match(pattern, email))


def validate_price(price: Union[float, str, Decimal]) -> bool:
    """
    Validate price value.
    
    Args:
        price: Price value
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert to Decimal for precise comparison
        price_decimal = Decimal(str(price))
        
        # Price must be positive
        return price_decimal > Decimal('0')
    except (InvalidOperation, ValueError, TypeError):
        return False


def validate_amount(amount: Union[float, str, Decimal]) -> bool:
    """
    Validate amount value.
    
    Args:
        amount: Amount value
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert to Decimal for precise comparison
        amount_decimal = Decimal(str(amount))
        
        # Amount must be positive
        return amount_decimal > Decimal('0')
    except (InvalidOperation, ValueError, TypeError):
        return False


def validate_percentage(percentage: Union[float, str, Decimal]) -> bool:
    """
    Validate percentage value.
    
    Args:
        percentage: Percentage value
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert to Decimal for precise comparison
        percentage_decimal = Decimal(str(percentage))
        
        # Percentage must be between 0 and 100
        return Decimal('0') <= percentage_decimal <= Decimal('100')
    except (InvalidOperation, ValueError, TypeError):
        return False


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict]]:
    """
    Validate JSON string.
    
    Args:
        json_str: JSON string
        
    Returns:
        tuple: (is_valid, parsed_json)
    """
    try:
        # Try to parse JSON
        parsed = json.loads(json_str)
        return True, parsed
    except json.JSONDecodeError:
        return False, None
    except Exception:
        return False, None


def validate_date(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
    """
    Validate date string format.
    
    Args:
        date_str: Date string
        format_str: Expected date format
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Try to parse date
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False


def validate_timestamp(timestamp: Union[int, float]) -> bool:
    """
    Validate timestamp value.
    
    Args:
        timestamp: Unix timestamp (seconds or milliseconds)
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert to float
        ts = float(timestamp)
        
        # Check if it's a reasonable timestamp (between 2010 and 2050)
        min_ts = datetime(2010, 1, 1).timestamp()
        max_ts = datetime(2050, 1, 1).timestamp()
        
        # Check if it's in seconds
        if min_ts <= ts <= max_ts:
            return True
            
        # Check if it's in milliseconds
        if min_ts * 1000 <= ts <= max_ts * 1000:
            return True
            
        return False
    except (ValueError, TypeError):
        return False


def validate_uuid(uuid_str: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_str: UUID string
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Try to parse UUID
        uuid_obj = uuid.UUID(uuid_str)
        return str(uuid_obj) == uuid_str
    except (ValueError, AttributeError, TypeError):
        return False


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(url, str):
        return False
        
    # URL pattern
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    
    return bool(re.match(pattern, url))


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(ip, str):
        return False
        
    # IPv4 pattern
    ipv4_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'
    
    if re.match(ipv4_pattern, ip):
        # Check each octet
        octets = ip.split('.')
        for octet in octets:
            if not (0 <= int(octet) <= 255):
                return False
        return True
        
    # IPv6 pattern (simplified)
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    
    return bool(re.match(ipv6_pattern, ip))


def validate_strategy_id(strategy_id: str) -> bool:
    """
    Validate strategy ID format.
    
    Args:
        strategy_id: Strategy ID string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(strategy_id, str):
        return False
        
    # Check if it's not empty
    if not strategy_id:
        return False
        
    # Strategy ID format: alphanumeric plus underscore, 1-50 chars
    pattern = r'^[A-Za-z0-9_]{1,50}$'
    
    return bool(re.match(pattern, strategy_id))


def validate_order_type(order_type: str) -> bool:
    """
    Validate order type.
    
    Args:
        order_type: Order type string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(order_type, str):
        return False
        
    # Common order types
    valid_types = [
        'market', 'limit', 'stop', 'stop_limit',
        'take_profit', 'take_profit_limit',
        'trailing_stop', 'fill_or_kill', 'immediate_or_cancel'
    ]
    
    return order_type.lower() in valid_types


def validate_order_side(side: str) -> bool:
    """
    Validate order side.
    
    Args:
        side: Order side string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(side, str):
        return False
        
    # Valid sides: buy or sell
    valid_sides = ['buy', 'sell']
    
    return side.lower() in valid_sides


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check if exchanges are properly configured
    if 'exchanges' in config:
        exchanges = config['exchanges']
        
        if not isinstance(exchanges, dict):
            errors.append("Exchanges must be a dictionary")
        else:
            for exchange_id, exchange_config in exchanges.items():
                if not isinstance(exchange_config, dict):
                    errors.append(f"Exchange {exchange_id} configuration must be a dictionary")
                    continue
                    
                # Check required fields
                if exchange_config.get('enabled', False):
                    if 'api_key' not in exchange_config or not exchange_config['api_key']:
                        errors.append(f"API key missing for enabled exchange {exchange_id}")
                    if 'api_secret' not in exchange_config or not exchange_config['api_secret']:
                        errors.append(f"API secret missing for enabled exchange {exchange_id}")
    
    # Check if strategies are properly configured
    if 'strategies' in config:
        strategies = config['strategies']
        
        if not isinstance(strategies, dict):
            errors.append("Strategies must be a dictionary")
        else:
            for strategy_id, strategy_config in strategies.items():
                if not isinstance(strategy_config, dict):
                    errors.append(f"Strategy {strategy_id} configuration must be a dictionary")
                    continue
                    
                # Check required fields
                if strategy_config.get('enabled', False):
                    if 'type' not in strategy_config or not strategy_config['type']:
                        errors.append(f"Type missing for enabled strategy {strategy_id}")
                    if 'symbols' not in strategy_config or not strategy_config['symbols']:
                        errors.append(f"Symbols missing for enabled strategy {strategy_id}")
                    if 'timeframes' not in strategy_config or not strategy_config['timeframes']:
                        errors.append(f"Timeframes missing for enabled strategy {strategy_id}")
    
    # Check database configuration if enabled
    if 'database' in config and config['database'].get('enabled', False):
        db_config = config['database']
        
        if 'url' not in db_config or not db_config['url']:
            errors.append("Database URL missing for enabled database")
        if 'username' not in db_config or not db_config['username']:
            errors.append("Database username missing for enabled database")
        if 'password' not in db_config or not db_config['password']:
            errors.append("Database password missing for enabled database")
    
    # Check notifications configuration if enabled
    if 'notifications' in config and config['notifications'].get('enabled', False):
        notif_config = config['notifications']
        
        # Check email configuration if enabled
        if 'email' in notif_config and notif_config['email'].get('enabled', False):
            email_config = notif_config['email']
            
            if 'smtp_server' not in email_config or not email_config['smtp_server']:
                errors.append("SMTP server missing for enabled email notifications")
            if 'username' not in email_config or not email_config['username']:
                errors.append("Username missing for enabled email notifications")
            if 'password' not in email_config or not email_config['password']:
                errors.append("Password missing for enabled email notifications")
        
        # Check Telegram configuration if enabled
        if 'telegram' in notif_config and notif_config['telegram'].get('enabled', False):
            telegram_config = notif_config['telegram']
            
            if 'token' not in telegram_config or not telegram_config['token']:
                errors.append("Token missing for enabled Telegram notifications")
            if 'chat_ids' not in telegram_config or not telegram_config['chat_ids']:
                errors.append("Chat IDs missing for enabled Telegram notifications")
    
    return len(errors) == 0, errors


def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_str: Input string
        
    Returns:
        str: Sanitized string
    """
    if not isinstance(input_str, str):
        return ""
        
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[;<>"\'\(\)&|]', '', input_str)
    
    return sanitized


def validate_trade_signal(signal: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a trading signal.
    
    Args:
        signal: Trading signal dictionary
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    if 'symbol' not in signal or not signal['symbol']:
        errors.append("Symbol is required")
    elif not validate_symbol(signal['symbol']):
        errors.append("Invalid symbol format")
        
    if 'action' not in signal or not signal['action']:
        errors.append("Action is required")
    elif signal['action'] not in ['buy', 'sell', 'close']:
        errors.append("Invalid action")
        
    if signal.get('action') in ['buy', 'sell']:
        if 'amount' not in signal or not signal['amount']:
            errors.append("Amount is required for buy/sell actions")
        elif not validate_amount(signal['amount']):
            errors.append("Invalid amount")
            
        if 'price' not in signal or not signal['price']:
            errors.append("Price is required for buy/sell actions")
        elif not validate_price(signal['price']):
            errors.append("Invalid price")
    
    # Validate optional fields if present
    if 'stop_loss' in signal and signal['stop_loss']:
        if not validate_price(signal['stop_loss']):
            errors.append("Invalid stop loss price")
            
    if 'take_profit' in signal and signal['take_profit']:
        if not validate_price(signal['take_profit']):
            errors.append("Invalid take profit price")
            
    if 'timeframe' in signal and signal['timeframe']:
        if not validate_timeframe(signal['timeframe']):
            errors.append("Invalid timeframe format")
    
    return len(errors) == 0, errors