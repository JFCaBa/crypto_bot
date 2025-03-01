"""
Bybit Exchange Connector
======================
Implementation of the BaseExchange class for the Bybit exchange.
Handles Bybit-specific functionality and WebSocket connections.
"""

import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from loguru import logger

from cryptobot.exchanges.base import BaseExchange
from cryptobot.utils.helpers import retry_async


class BybitExchange(BaseExchange):
    """Bybit exchange connector implementation."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        rate_limit: bool = True,
        timeout: int = 30000
    ):
        """
        Initialize the Bybit exchange connector.
        
        Args:
            api_key: API key for Bybit
            api_secret: API secret for Bybit
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
            exchange_id='bybit'
        )
        
    def has_websocket_support(self) -> bool:
        """
        Check if Bybit has WebSocket support.
        
        Returns:
            bool: True as Bybit supports WebSockets
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
        Create a new order on Bybit.
        
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
            
        # Convert order types to Bybit-specific format
        if order_type == 'stop_loss':
            # For Bybit, we need to specify if it's a market or limit order with a stop loss
            order_type = 'limit' if price else 'market'
            params['stop_loss'] = params.get('stopPrice') or price
            
        elif order_type == 'take_profit':
            # For Bybit, we need to specify if it's a market or limit order with a take profit
            order_type = 'limit' if price else 'market'
            params['take_profit'] = params.get('stopPrice') or price
            
        # Bybit supports trailing stops differently than some other exchanges
        if 'trailingDelta' in params or 'callback_rate' in params:
            # Convert callback_rate to Bybit's trailing_stop format if provided
            if 'callback_rate' in params:
                callback_rate = params.pop('callback_rate')
                params['trailing_stop'] = callback_rate
            elif 'trailingDelta' in params:
                # Convert basis points to percentage for Bybit
                trailing_delta = params.pop('trailingDelta')
                params['trailing_stop'] = trailing_delta / 100
                
            logger.info(f"Adding trailing stop with rate {params.get('trailing_stop')}")
            
        # Map order types to Bybit-specific values
        bybit_order_types = {
            'limit': 'Limit',
            'market': 'Market',
            'stop': 'Stop',
            'stop_loss': 'StopLoss',
            'take_profit': 'TakeProfit',
            'trailing_stop': 'TrailingStop'
        }
        
        if order_type in bybit_order_types:
            params['type'] = bybit_order_types[order_type]
            
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
            leverage: Leverage value
            
        Returns:
            dict: Response from the exchange
        """
        self._update_api_stats()
        
        try:
            # For Bybit, set leverage
            market_id = self.exchange.market_id(symbol)
            response = await self.exchange_async.v2PrivatePostPrivateLinearPositionSetLeverage({
                'symbol': market_id,
                'buy_leverage': leverage,
                'sell_leverage': leverage
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
            # Map margin types to Bybit's expected values
            bybit_margin_type = 0 if margin_type.upper() == 'CROSSED' else 1
            
            response = await self.exchange_async.v2PrivatePostPrivateLinearPositionSwitchIsolated({
                'symbol': market_id,
                'is_isolated': bybit_margin_type,
                'buy_leverage': 1,  # Default values, will be overridden by set_leverage
                'sell_leverage': 1
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
            response = await self.exchange_async.v2PublicGetPublicLinearFundingPrevFundingRate({
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
        order_type: str = 'LIMIT',  # 'LIMIT' or 'MARKET'
        trigger_type: str = 'LAST_PRICE',  # 'LAST_PRICE', 'INDEX_PRICE', 'MARK_PRICE'
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        params: Dict = None
    ) -> Dict[str, Any]:
        """
        Create a conditional order on Bybit.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            stop_price: Trigger price
            order_type: Order type ('LIMIT' or 'MARKET')
            trigger_type: Trigger price type
            reduce_only: Whether the order should only reduce position size
            close_on_trigger: Whether to close position on trigger
            params: Additional parameters
            
        Returns:
            dict: Order information
        """
        self._update_api_stats()
        
        if params is None:
            params = {}
            
        try:
            market_id = self.exchange.market_id(symbol)
            
            # Build request
            request = {
                'symbol': market_id,
                'side': side.upper(),
                'order_type': order_type,
                'qty': self.exchange.amount_to_precision(symbol, amount),
                'stop_px': self.exchange.price_to_precision(symbol, stop_price),
                'base_price': self.exchange.price_to_precision(symbol, self.exchange_async.fetch_ticker(symbol)['last']),
                'trigger_by': trigger_type,
                'reduce_only': reduce_only,
                'close_on_trigger': close_on_trigger,
                'time_in_force': 'GoodTillCancel'
            }
            
            # Add price for limit orders
            if price and order_type == 'LIMIT':
                request['price'] = self.exchange.price_to_precision(symbol, price)
                
            # Add any additional parameters
            request.update(params)
            
            # Create the conditional order
            response = await self.exchange_async.v2PrivatePostPrivateLinearStopOrderCreate(request)
            logger.info(f"Created conditional {order_type} {side} order for {amount} {symbol} at stop price {stop_price}")
            return response
        except Exception as e:
            logger.error(f"Error creating conditional order: {str(e)}")
            raise