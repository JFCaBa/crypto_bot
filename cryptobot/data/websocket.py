"""
WebSocket Manager
===============
Manages WebSocket connections to cryptocurrency exchanges for real-time data streaming.
Provides a unified interface for subscribing to various data streams.
"""

import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Union, Any, Callable
import websockets
from loguru import logger


class WebSocketManager:
    """Manages WebSocket connections to exchanges."""
    
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None):
        """
        Initialize the WebSocket manager.
        
        Args:
            exchange_id: Exchange identifier string
            api_key: API key for authenticated WebSocket streams
            api_secret: API secret for authenticated WebSocket streams
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize connection variables
        self.connections = {}
        self.callbacks = {}
        self.is_connected = False
        self.keepalive_task = None
        
        # Set appropriate WebSocket URLs based on exchange
        # Set appropriate WebSocket URLs based on exchange
        if exchange_id == 'binance':
            self.spot_ws_url = 'wss://stream.binance.com:9443/ws'
            self.futures_ws_url = 'wss://fstream.binance.com/ws'
            self.spot_combined_url = 'wss://stream.binance.com:9443/stream'
            self.futures_combined_url = 'wss://fstream.binance.com/stream'
        elif exchange_id == 'coinbase':
            self.ws_url = 'wss://ws-feed.pro.coinbase.com'
        elif exchange_id == 'kraken':
            self.ws_url = 'wss://ws.kraken.com'
        elif exchange_id == 'mexc':
            self.spot_ws_url = 'wss://wbs.mexc.com/ws'
            self.futures_ws_url = 'wss://contract.mexc.com/ws'
        elif exchange_id == 'bybit':
            self.spot_ws_url = 'wss://stream.bybit.com/v5/public/spot'
            self.futures_ws_url = 'wss://stream.bybit.com/v5/public/linear'
            self.private_ws_url = 'wss://stream.bybit.com/v5/private'
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
            
    async def connect(self) -> bool:
        """
        Connect to exchange WebSocket.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.is_connected = True
            # Start the keepalive task
            self.keepalive_task = asyncio.create_task(self._keepalive())
            logger.info(f"WebSocket manager for {self.exchange_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {str(e)}")
            self.is_connected = False
            return False
            
    async def disconnect(self) -> bool:
        """
        Disconnect from all WebSocket connections.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.keepalive_task:
                self.keepalive_task.cancel()
                
            # Close all active connections
            for stream_name, ws in self.connections.items():
                logger.info(f"Closing WebSocket connection for {stream_name}")
                await ws.close()
                
            self.connections = {}
            self.callbacks = {}
            self.is_connected = False
            logger.info(f"All WebSocket connections for {self.exchange_id} closed")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket connections: {str(e)}")
            return False
            
    async def subscribe_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to ticker updates.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        if self.exchange_id == 'binance':
            # For Binance, convert symbol format from BTC/USDT to btcusdt
            formatted_symbol = symbol.lower().replace('/', '')
            stream_name = f"{formatted_symbol}@ticker"
            await self._subscribe_binance(stream_name, callback)
        elif self.exchange_id == 'coinbase':
            await self._subscribe_coinbase('ticker', symbol, callback)
        elif self.exchange_id == 'kraken':
            await self._subscribe_kraken('ticker', symbol, callback)
            
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[Dict[str, Any]], None], depth: str = '20'):
        """
        Subscribe to order book updates.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
            depth: Order book depth ('5', '10', '20', etc.)
        """
        if self.exchange_id == 'binance':
            formatted_symbol = symbol.lower().replace('/', '')
            stream_name = f"{formatted_symbol}@depth{depth}"
            await self._subscribe_binance(stream_name, callback)
        elif self.exchange_id == 'coinbase':
            await self._subscribe_coinbase('level2', symbol, callback)
        elif self.exchange_id == 'kraken':
            await self._subscribe_kraken('book', symbol, callback, depth=depth)
            
    async def subscribe_trades(self, symbol: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to trade updates.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        if self.exchange_id == 'binance':
            formatted_symbol = symbol.lower().replace('/', '')
            stream_name = f"{formatted_symbol}@trade"
            await self._subscribe_binance(stream_name, callback)
        elif self.exchange_id == 'coinbase':
            await self._subscribe_coinbase('matches', symbol, callback)
        elif self.exchange_id == 'kraken':
            await self._subscribe_kraken('trade', symbol, callback)
            
    async def subscribe_ohlcv(self, symbol: str, timeframe: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to OHLCV (candlestick) updates.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            callback: Callback function to handle updates
        """
        if self.exchange_id == 'binance':
            formatted_symbol = symbol.lower().replace('/', '')
            # Convert timeframe to Binance format
            tf_mapping = {'1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                         '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                         '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'}
            binance_tf = tf_mapping.get(timeframe, '1h')
            stream_name = f"{formatted_symbol}@kline_{binance_tf}"
            await self._subscribe_binance(stream_name, callback)
        elif self.exchange_id == 'kraken':
            await self._subscribe_kraken('ohlc', symbol, callback, interval=self._convert_timeframe_to_kraken(timeframe))
            
    async def subscribe_user_data(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to user data updates (requires authentication).
        
        Args:
            callback: Callback function to handle updates
        """
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret required for user data streams")
            return False
            
        if self.exchange_id == 'binance':
            await self._subscribe_binance_user_data(callback)
        elif self.exchange_id == 'coinbase':
            await self._subscribe_coinbase_user_data(callback)
        elif self.exchange_id == 'kraken':
            await self._subscribe_kraken_user_data(callback)
            
    async def _subscribe_binance(self, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to a Binance WebSocket stream.
        
        Args:
            stream_name: Binance stream name
            callback: Callback function to handle updates
        """
        try:
            # Check if we're already subscribed
            if stream_name in self.connections:
                logger.info(f"Already subscribed to {stream_name}")
                # Update callback if it's different
                if self.callbacks[stream_name] != callback:
                    self.callbacks[stream_name] = callback
                return True
                
            url = f"{self.spot_ws_url}/{stream_name}"
            logger.info(f"Connecting to Binance WebSocket: {url}")
            
            ws = await websockets.connect(url)
            self.connections[stream_name] = ws
            self.callbacks[stream_name] = callback
            
            # Start listener task
            asyncio.create_task(self._binance_listener(ws, stream_name, callback))
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Binance stream {stream_name}: {str(e)}")
            return False
            
    async def _subscribe_binance_user_data(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to Binance user data stream.
        
        Args:
            callback: Callback function to handle updates
        """
        try:
            # First, get a listen key from Binance API
            from aiohttp import ClientSession
            
            async with ClientSession() as session:
                # For spot
                headers = {'X-MBX-APIKEY': self.api_key}
                async with session.post('https://api.binance.com/api/v3/userDataStream', headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get listen key: {await response.text()}")
                        return False
                        
                    data = await response.json()
                    listen_key = data['listenKey']
                    
                    # Connect to user data WebSocket
                    url = f"{self.spot_ws_url}/{listen_key}"
                    logger.info(f"Connecting to Binance user data stream")
                    
                    ws = await websockets.connect(url)
                    stream_name = 'user_data'
                    self.connections[stream_name] = ws
                    self.callbacks[stream_name] = callback
                    
                    # Start listener and keepalive tasks
                    asyncio.create_task(self._binance_listener(ws, stream_name, callback))
                    asyncio.create_task(self._binance_user_data_keepalive(listen_key))
                    return True
        except Exception as e:
            logger.error(f"Error subscribing to Binance user data stream: {str(e)}")
            return False
            
    async def _binance_user_data_keepalive(self, listen_key: str):
        """
        Keep the Binance user data listen key alive.
        
        Args:
            listen_key: Binance listen key
        """
        from aiohttp import ClientSession
        
        while 'user_data' in self.connections:
            try:
                async with ClientSession() as session:
                    headers = {'X-MBX-APIKEY': self.api_key}
                    url = f"https://api.binance.com/api/v3/userDataStream?listenKey={listen_key}"
                    async with session.put(url, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Failed to refresh listen key: {await response.text()}")
            except Exception as e:
                logger.error(f"Error refreshing listen key: {str(e)}")
                
            # Refresh every 30 minutes (Binance requires refresh every 60 minutes)
            await asyncio.sleep(30 * 60)
            
    async def _binance_listener(self, ws, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Listen for WebSocket messages and call the callback.
        
        Args:
            ws: WebSocket connection
            stream_name: Stream name
            callback: Callback function
        """
        try:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                await callback(data)
        except asyncio.CancelledError:
            logger.info(f"Binance listener for {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error in Binance listener for {stream_name}: {str(e)}")
            # Try to reconnect
            await self._reconnect_binance(stream_name, callback)
            
    async def _reconnect_binance(self, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Reconnect to a Binance WebSocket stream.
        
        Args:
            stream_name: Stream name
            callback: Callback function
        """
        try:
            # Remove old connection
            if stream_name in self.connections:
                old_ws = self.connections.pop(stream_name)
                await old_ws.close()
                
            # Wait before reconnecting
            await asyncio.sleep(1)
            
            # Reconnect
            logger.info(f"Reconnecting to Binance stream {stream_name}")
            if stream_name == 'user_data':
                await self._subscribe_binance_user_data(callback)
            else:
                await self._subscribe_binance(stream_name, callback)
        except Exception as e:
            logger.error(f"Error reconnecting to Binance stream {stream_name}: {str(e)}")
            # Try again after a delay
            await asyncio.sleep(5)
            asyncio.create_task(self._reconnect_binance(stream_name, callback))
            
    async def _subscribe_coinbase(self, channel: str, symbol: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to a Coinbase WebSocket stream.
        
        Args:
            channel: Channel name ('ticker', 'level2', 'matches', etc.)
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
        """
        try:
            # Format the symbol for Coinbase (e.g., BTC-USD)
            formatted_symbol = symbol.replace('/', '-')
            stream_name = f"{channel}_{formatted_symbol}"
            
            # Check if we're already subscribed
            if stream_name in self.connections:
                logger.info(f"Already subscribed to Coinbase {stream_name}")
                if self.callbacks[stream_name] != callback:
                    self.callbacks[stream_name] = callback
                return True
                
            # Connect to Coinbase WebSocket
            logger.info(f"Connecting to Coinbase WebSocket for {channel}:{formatted_symbol}")
            ws = await websockets.connect(self.ws_url)
            
            # Subscription message
            subscribe_msg = {
                "type": "subscribe",
                "channels": [
                    {
                        "name": channel,
                        "product_ids": [formatted_symbol]
                    }
                ]
            }
            
            await ws.send(json.dumps(subscribe_msg))
            
            # Store the connection and callback
            self.connections[stream_name] = ws
            self.callbacks[stream_name] = callback
            
            # Start listener task
            asyncio.create_task(self._coinbase_listener(ws, stream_name, callback))
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Coinbase {channel} for {symbol}: {str(e)}")
            return False
            
    async def _coinbase_listener(self, ws, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Listen for Coinbase WebSocket messages and call the callback.
        
        Args:
            ws: WebSocket connection
            stream_name: Stream name
            callback: Callback function
        """
        try:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                # Skip subscription confirmation messages
                if data.get('type') == 'subscriptions':
                    continue
                    
                await callback(data)
        except asyncio.CancelledError:
            logger.info(f"Coinbase listener for {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error in Coinbase listener for {stream_name}: {str(e)}")
            # Try to reconnect
            await self._reconnect_coinbase(stream_name, callback)
            
    async def _reconnect_coinbase(self, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Reconnect to a Coinbase WebSocket stream.
        
        Args:
            stream_name: Stream name
            callback: Callback function
        """
        try:
            # Parse stream name back to channel and symbol
            parts = stream_name.split('_', 1)
            if len(parts) != 2:
                logger.error(f"Invalid Coinbase stream name: {stream_name}")
                return
                
            channel, formatted_symbol = parts
            symbol = formatted_symbol.replace('-', '/')
            
            # Remove old connection
            if stream_name in self.connections:
                old_ws = self.connections.pop(stream_name)
                await old_ws.close()
                
            # Wait before reconnecting
            await asyncio.sleep(1)
            
            # Reconnect
            logger.info(f"Reconnecting to Coinbase stream {stream_name}")
            await self._subscribe_coinbase(channel, symbol, callback)
        except Exception as e:
            logger.error(f"Error reconnecting to Coinbase stream {stream_name}: {str(e)}")
            # Try again after a delay
            await asyncio.sleep(5)
            asyncio.create_task(self._reconnect_coinbase(stream_name, callback))
            
    async def _subscribe_kraken(self, channel: str, symbol: str, callback: Callable[[Dict[str, Any]], None], **kwargs):
        """
        Subscribe to a Kraken WebSocket stream.
        
        Args:
            channel: Channel name ('ticker', 'book', 'trade', etc.)
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Callback function to handle updates
            **kwargs: Additional subscription parameters
        """
        try:
            # Format the symbol for Kraken (e.g., XBT/USD)
            # For Kraken, some symbols need specific formatting
            formatted_symbol = symbol
            if symbol.startswith('BTC/'):
                formatted_symbol = symbol.replace('BTC/', 'XBT/')
                
            stream_name = f"{channel}_{formatted_symbol}"
            
            # Check if we're already subscribed
            if stream_name in self.connections:
                logger.info(f"Already subscribed to Kraken {stream_name}")
                if self.callbacks[stream_name] != callback:
                    self.callbacks[stream_name] = callback
                return True
                
            # Connect to Kraken WebSocket
            logger.info(f"Connecting to Kraken WebSocket for {channel}:{formatted_symbol}")
            ws = await websockets.connect(self.ws_url)
            
            # Subscription message
            subscribe_msg = {
                "name": "subscribe",
                "reqid": int(time.time()),
                "pair": [formatted_symbol],
                "subscription": {
                    "name": channel,
                    **kwargs
                }
            }
            
            await ws.send(json.dumps(subscribe_msg))
            
            # Store the connection and callback
            self.connections[stream_name] = ws
            self.callbacks[stream_name] = callback
            
            # Start listener task
            asyncio.create_task(self._kraken_listener(ws, stream_name, callback))
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Kraken {channel} for {symbol}: {str(e)}")
            return False
            
    async def _kraken_listener(self, ws, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Listen for Kraken WebSocket messages and call the callback.
        
        Args:
            ws: WebSocket connection
            stream_name: Stream name
            callback: Callback function
        """
        try:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                # Skip subscription confirmation messages
                if isinstance(data, dict) and 'event' in data and data['event'] == 'subscriptionStatus':
                    continue
                    
                await callback(data)
        except asyncio.CancelledError:
            logger.info(f"Kraken listener for {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error in Kraken listener for {stream_name}: {str(e)}")
            # Try to reconnect
            await self._reconnect_kraken(stream_name, callback)
            
    async def _reconnect_kraken(self, stream_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Reconnect to a Kraken WebSocket stream.
        
        Args:
            stream_name: Stream name
            callback: Callback function
        """
        try:
            # Parse stream name back to channel and symbol
            parts = stream_name.split('_', 1)
            if len(parts) != 2:
                logger.error(f"Invalid Kraken stream name: {stream_name}")
                return
                
            channel, formatted_symbol = parts
            
            # Convert XBT back to BTC if needed
            symbol = formatted_symbol
            if formatted_symbol.startswith('XBT/'):
                symbol = formatted_symbol.replace('XBT/', 'BTC/')
                
            # Remove old connection
            if stream_name in self.connections:
                old_ws = self.connections.pop(stream_name)
                await old_ws.close()
                
            # Wait before reconnecting
            await asyncio.sleep(1)
            
            # Reconnect
            logger.info(f"Reconnecting to Kraken stream {stream_name}")
            await self._subscribe_kraken(channel, symbol, callback)
        except Exception as e:
            logger.error(f"Error reconnecting to Kraken stream {stream_name}: {str(e)}")
            # Try again after a delay
            await asyncio.sleep(5)
            asyncio.create_task(self._reconnect_kraken(stream_name, callback))
            
    async def _keepalive(self):
        """Periodically check WebSocket connections and reconnect if needed."""
        while self.is_connected:
            try:
                for stream_name, ws in list(self.connections.items()):
                    # Check if connection is still open
                    if not ws.open:
                        logger.warning(f"WebSocket for {stream_name} is closed, attempting to reconnect")
                        callback = self.callbacks.get(stream_name)
                        if callback:
                            if self.exchange_id == 'binance':
                                await self._reconnect_binance(stream_name, callback)
                            elif self.exchange_id == 'coinbase':
                                await self._reconnect_coinbase(stream_name, callback)
                            elif self.exchange_id == 'kraken':
                                await self._reconnect_kraken(stream_name, callback)
            except Exception as e:
                logger.error(f"Error in WebSocket keepalive: {str(e)}")
                
            # Check every 30 seconds
            await asyncio.sleep(30)
            
    def _convert_timeframe_to_kraken(self, timeframe: str) -> int:
        """
        Convert timeframe string to Kraken interval in minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            int: Interval in minutes
        """
        if timeframe == '1m':
            return 1
        elif timeframe == '5m':
            return 5
        elif timeframe == '15m':
            return 15
        elif timeframe == '30m':
            return 30
        elif timeframe == '1h':
            return 60
        elif timeframe == '4h':
            return 240
        elif timeframe == '1d':
            return 1440
        elif timeframe == '1w':
            return 10080
        else:
            # Default to 1 hour
            return 60