"""
Binance Exchange Connector
=========================
Implementation of the BaseExchange class for the Binance exchange.
Handles Binance-specific functionality and WebSocket connections.
"""

import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from loguru import logger

from cryptobot.exchanges.base import BaseExchange
from cryptobot.utils.helpers import retry_async


class BinanceExchange(BaseExchange):
    """Binance exchange connector implementation."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        rate_limit: bool = True,
        timeout: int = 30000
    ):
        """
        Initialize the Binance exchange connector.
        
        Args:
            api_key: API key for Binance
            api_secret: API secret for Binance
            sandbox: Whether to use the testnet API
            rate_limit: Whether to enable rate limiting
            timeout: Request timeout in milliseconds
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit,
            timeout=timeout,
            exchange_id='binance'
        )
        
    def has_websocket_support(self) -> bool:
        """
        Check if Binance has WebSocket support.
        
        Returns:
            bool: True as Binance supports WebSockets
        """
        return True
        
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
        Create a new order on Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            order_type: Order type ('limit', 'market', 'stop_loss', 'take_profit')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            params: Additional parameters for the order
            
        Returns:
            dict: Order information
        """
        if params is None:
            params = {}
            
        # Convert order types to Binance-specific format
        if order_type == 'stop_loss':
            order_type = 'stop_loss_limit' if price else 'stop_loss'
            if 'stopPrice' not in params:
                raise ValueError("stopPrice parameter is required for stop_loss orders")
                
        elif order_type == 'take_profit':
            order_type = 'take_profit_limit' if price else 'take_profit'
            if 'stopPrice' not in params:
                raise ValueError("stopPrice parameter is required for take_profit orders")
                
        # Add trailing stop if specified
        if 'trailingDelta' in params or 'callback_rate' in params:
            # Convert callback_rate to trailingDelta if provided
            if 'callback_rate' in params and 'trailingDelta' not in params:
                callback_rate = params.pop('callback_rate')
                params['trailingDelta'] = int(callback_rate * 100)  # Convert percentage to basis points
                
            # Ensure we're using the correct order type for trailing stops
            if order_type in ['limit', 'market']:
                logger.info("Adding trailing stop to order")
        
        return await super().create_order(symbol, order_type, side, amount, price, params)
        
    async def fetch_deposit_address(self, currency: str, params: Dict = None) -> Dict[str, Any]:
        """
        Fetch the deposit address for a currency.
        
        Args:
            currency: Currency code (e.g., 'BTC')
            params: Additional parameters
            
        Returns:
            dict: Deposit address information
        """
        self._update_api_stats()
        
        if params is None:
            params = {}
            
        try:
            address = await self.exchange_async.fetch_deposit_address(currency, params)
            return address
        except Exception as e:
            logger.error(f"Error fetching deposit address for {currency}: {str(e)}")
            raise
            
    async def fetch_deposits(self, currency: str = None, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch deposit history.
        
        Args:
            currency: Currency code (e.g., 'BTC')
            since: Timestamp in ms for start time
            limit: Number of deposits to fetch
            
        Returns:
            list: List of deposits
        """
        self._update_api_stats()
        
        try:
            deposits = await self.exchange_async.fetch_deposits(currency, since, limit)
            return deposits
        except Exception as e:
            logger.error(f"Error fetching deposits: {str(e)}")
            raise
            
    async def fetch_withdrawals(self, currency: str = None, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch withdrawal history.
        
        Args:
            currency: Currency code (e.g., 'BTC')
            since: Timestamp in ms for start time
            limit: Number of withdrawals to fetch
            
        Returns:
            list: List of withdrawals
        """
        self._update_api_stats()
        
        try:
            withdrawals = await self.exchange_async.fetch_withdrawals(currency, since, limit)
            return withdrawals
        except Exception as e:
            logger.error(f"Error fetching withdrawals: {str(e)}")
            raise
            
    async def fetch_my_trades(self, symbol: str = None, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch trade history.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms for start time
            limit: Number of trades to fetch
            
        Returns:
            list: List of trades
        """
        self._update_api_stats()
        
        try:
            trades = await self.exchange_async.fetch_my_trades(symbol, since, limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades: {str(e)}")
            raise
            
    async def fetch_funding_history(self, symbol: str = None, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Fetch funding history for futures trading.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms for start time
            limit: Number of funding payments to fetch
            
        Returns:
            list: List of funding payments
        """
        self._update_api_stats()
        
        try:
            # Binance-specific endpoint for funding history
            params = {}
            if symbol:
                params['symbol'] = self.exchange.market_id(symbol)
            if since:
                params['startTime'] = since
            if limit:
                params['limit'] = limit
                
            funding = await self.exchange_async.fapiPrivateGetIncome({
                'incomeType': 'FUNDING_FEE',
                **params
            })
            return funding
        except Exception as e:
            logger.error(f"Error fetching funding history: {str(e)}")
            raise
            
    async def subscribe_to_user_data(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to user data updates via WebSocket.
        
        Args:
            callback: Callback function to handle updates
        """
        if not self.ws_manager:
            logger.warning("WebSocket not initialized")
            return False
            
        await self.ws_manager.subscribe_user_data(callback)
        return True
        
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for futures trading.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            leverage: Leverage value (1-125)
            
        Returns:
            dict: Response from the exchange
        """
        self._update_api_stats()
        
        try:
            market_id = self.exchange.market_id(symbol)
            response = await self.exchange_async.fapiPrivatePostLeverage({
                'symbol': market_id,
                'leverage': leverage
            })
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return response
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            raise
            
    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        Set margin type for futures trading.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            margin_type: Margin type ('ISOLATED' or 'CROSSED')
            
        Returns:
            dict: Response from the exchange
        """
        self._update_api_stats()
        
        try:
            market_id = self.exchange.market_id(symbol)
            response = await self.exchange_async.fapiPrivatePostMarginType({
                'symbol': market_id,
                'marginType': margin_type.upper()
            })
            logger.info(f"Set margin type for {symbol} to {margin_type}")
            return response
        except Exception as e:
            logger.error(f"Error setting margin type for {symbol}: {str(e)}")
            raise
            
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: Funding rate information
        """
        self._update_api_stats()
        
        try:
            market_id = self.exchange.market_id(symbol)
            response = await self.exchange_async.fapiPublicGetPremiumIndex({
                'symbol': market_id
            })
            return response
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {str(e)}")
            raise
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            dict: Exchange information
        """
        self._update_api_stats()
        
        try:
            info = await self.exchange_async.fetch_markets()
            return info
        except Exception as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            raise
            
    async def create_conditional_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float = None,
        stop_price: float = None,
        order_type: str = 'STOP',  # 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
        time_in_force: str = 'GTC',
        params: Dict = None
    ) -> Dict[str, Any]:
        """
        Create a conditional order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            stop_price: Trigger price
            order_type: Order type
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            params: Additional parameters
            
        Returns:
            dict: Order information
        """
        self._update_api_stats()
        
        if params is None:
            params = {}
            
        try:
            market_id = self.exchange.market_id(symbol)
            
            # Build request based on order type
            request = {
                'symbol': market_id,
                'side': side.upper(),
                'quantity': self.exchange.amount_to_precision(symbol, amount),
                'type': order_type,
                'timeInForce': time_in_force,
                'stopPrice': self.exchange.price_to_precision(symbol, stop_price),
            }
            
            # Add price for limit orders
            if price and order_type in ['STOP', 'TAKE_PROFIT']:
                request['price'] = self.exchange.price_to_precision(symbol, price)
                
            # Add any additional parameters
            request.update(params)
            
            # Create the order
            response = await self.exchange_async.fapiPrivatePostOrder(request)
            logger.info(f"Created conditional {order_type} {side} order for {amount} {symbol} at stop price {stop_price}")
            return response
        except Exception as e:
            logger.error(f"Error creating conditional order: {str(e)}")
            raise