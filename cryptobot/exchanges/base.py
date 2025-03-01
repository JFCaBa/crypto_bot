"""
Base Exchange Connector
======================
Abstract base class for all exchange connectors that handles common
functionality and defines the interface that all exchange-specific
implementations must follow.
"""

import abc
import asyncio
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import ccxt
import ccxt.async_support as ccxt_async
from loguru import logger

from cryptobot.core.orderbook import OrderBook
from cryptobot.data.websocket import WebSocketManager
from cryptobot.utils.helpers import retry_async


class BaseExchange(abc.ABC):
    """Abstract base class for all exchange connectors."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        rate_limit: bool = True,
        timeout: int = 30000,
        enableRateLimit: bool = True,
        exchange_id: str = None,
    ):
        """
        Initialize the exchange connector.
        
        Args:
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            sandbox: Whether to use the sandbox/testnet API
            rate_limit: Whether to enable rate limiting
            timeout: Request timeout in milliseconds
            enableRateLimit: CCXT rate limiting
            exchange_id: Exchange identifier string
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.exchange_id = exchange_id
        
        # Will be initialized in the connect method
        self.exchange = None
        self.exchange_async = None
        self.ws_manager = None
        
        # Track order book data
        self.orderbooks: Dict[str, OrderBook] = {}
        
        # Track connected status
        self.is_connected = False
        
        # API call statistics
        self.api_calls = 0
        self.last_api_call_time = 0
        
    async def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize the CCXT exchange object
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_class_async = getattr(ccxt_async, self.exchange_id)
            
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'timeout': self.timeout,
                'enableRateLimit': self.rate_limit,
            })
            
            self.exchange_async = exchange_class_async({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'timeout': self.timeout,
                'enableRateLimit': self.rate_limit,
            })
            
            # Use sandbox if enabled
            if self.sandbox and self.exchange.has['test']:
                self.exchange.set_sandbox_mode(True)
                self.exchange_async.set_sandbox_mode(True)
                
            # Initialize WebSocket manager if exchange supports it
            if self.has_websocket_support():
                self.ws_manager = WebSocketManager(self.exchange_id, self.api_key, self.api_secret)
                await self.ws_manager.connect()
            
            # Test connection with a basic API call
            await self.exchange_async.load_markets()
            
            self.is_connected = True
            logger.info(f"Successfully connected to {self.exchange_id}.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {str(e)}")
            self.is_connected = False
            return False
            
    async def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.ws_manager:
                await self.ws_manager.disconnect()
                
            if self.exchange_async:
                await self.exchange_async.close()
                
            self.is_connected = False
            logger.info(f"Successfully disconnected from {self.exchange_id}.")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.exchange_id}: {str(e)}")
            return False
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the current ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: Ticker data including bid, ask, last price, etc.
        """
        self._update_api_stats()
        ticker = await self.exchange_async.fetch_ticker(symbol)
        return ticker
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = None
    ) -> List[List[Union[int, float]]]:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            since: Timestamp in ms for start time
            limit: Number of candles to fetch
            
        Returns:
            list: List of OHLCV candles
        """
        self._update_api_stats()
        ohlcv = await self.exchange_async.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_order_book(self, symbol: str, limit: int = None) -> Dict[str, Any]:
        """
        Fetch the order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Limit the number of bids/asks returned
            
        Returns:
            dict: Order book with bids and asks
        """
        self._update_api_stats()
        order_book = await self.exchange_async.fetch_order_book(symbol, limit)
        
        # Update the stored order book
        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = OrderBook(symbol)
        self.orderbooks[symbol].update(order_book)
        
        return order_book
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def create_order(
        self, 
        symbol: str, 
        order_type: str, 
        side: str, 
        amount: float, 
        price: float = None, 
        params: Dict = None
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            order_type: Order type (e.g., 'limit', 'market')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            params: Additional parameters specific to the exchange
            
        Returns:
            dict: Order information
        """
        self._update_api_stats()
        
        if params is None:
            params = {}
            
        try:
            order = await self.exchange_async.create_order(
                symbol, order_type, side, amount, price, params
            )
            logger.info(
                f"Created {order_type} {side} order for {amount} {symbol} "
                f"at price {price if price else 'market'}"
            )
            return order
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            raise
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def cancel_order(self, id: str, symbol: str, params: Dict = None) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            id: Order ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            params: Additional parameters specific to the exchange
            
        Returns:
            dict: Canceled order information
        """
        self._update_api_stats()
        
        if params is None:
            params = {}
        
        try:
            result = await self.exchange_async.cancel_order(id, symbol, params)
            logger.info(f"Canceled order {id} for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Error canceling order {id}: {str(e)}")
            raise
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            dict: Account balance information
        """
        self._update_api_stats()
        balance = await self.exchange_async.fetch_balance()
        return balance
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            list: List of open orders
        """
        self._update_api_stats()
        orders = await self.exchange_async.fetch_open_orders(symbol)
        return orders
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_closed_orders(self, symbol: str = None, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch closed orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms for start time
            limit: Number of orders to fetch
            
        Returns:
            list: List of closed orders
        """
        self._update_api_stats()
        orders = await self.exchange_async.fetch_closed_orders(symbol, since, limit)
        return orders
    
    @retry_async(max_retries=3, delay=1, backoff=2)
    async def fetch_order(self, id: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch a specific order by ID.
        
        Args:
            id: Order ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: Order information
        """
        self._update_api_stats()
        order = await self.exchange_async.fetch_order(id, symbol)
        return order
    
    async def subscribe_to_ticker(self, symbol: str, callback):
        """
        Subscribe to ticker updates via WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        if not self.ws_manager:
            logger.warning(f"WebSocket not supported for {self.exchange_id}")
            return False
            
        await self.ws_manager.subscribe_ticker(symbol, callback)
        return True
    
    async def subscribe_to_orderbook(self, symbol: str, callback):
        """
        Subscribe to order book updates via WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        if not self.ws_manager:
            logger.warning(f"WebSocket not supported for {self.exchange_id}")
            return False
            
        await self.ws_manager.subscribe_orderbook(symbol, callback)
        return True
    
    async def subscribe_to_trades(self, symbol: str, callback):
        """
        Subscribe to trade updates via WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        if not self.ws_manager:
            logger.warning(f"WebSocket not supported for {self.exchange_id}")
            return False
            
        await self.ws_manager.subscribe_trades(symbol, callback)
        return True
    
    async def subscribe_to_ohlcv(self, symbol: str, timeframe: str, callback):
        """
        Subscribe to OHLCV updates via WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            callback: Callback function to handle updates
        """
        if not self.ws_manager:
            logger.warning(f"WebSocket not supported for {self.exchange_id}")
            return False
            
        await self.ws_manager.subscribe_ohlcv(symbol, timeframe, callback)
        return True
    
    def has_websocket_support(self) -> bool:
        """
        Check if the exchange has WebSocket support.
        
        Returns:
            bool: True if WebSocket is supported, False otherwise
        """
        # Override in exchange-specific implementations
        return False
        
    def get_supported_timeframes(self) -> Dict[str, int]:
        """
        Get supported timeframes for the exchange.
        
        Returns:
            dict: Dictionary of supported timeframes and their values in seconds
        """
        return self.exchange.timeframes
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get supported trading pairs for the exchange.
        
        Returns:
            list: List of supported trading pair symbols
        """
        return list(self.exchange.markets.keys())
    
    def get_trading_fees(self) -> Dict[str, Any]:
        """
        Get trading fees for the exchange.
        
        Returns:
            dict: Trading fees information
        """
        return self.exchange.fees
    
    def get_exchange_status(self) -> Dict[str, Any]:
        """
        Get the current status of the exchange.
        
        Returns:
            dict: Exchange status information
        """
        return {
            'id': self.exchange_id,
            'connected': self.is_connected,
            'websocket_status': self.ws_manager.is_connected if self.ws_manager else None,
            'api_calls': self.api_calls,
            'timestamp': time.time()
        }
        
    def _update_api_stats(self):
        """Update API call statistics."""
        self.api_calls += 1
        self.last_api_call_time = time.time()