"""
Take Profit Handler
=================
Implements various take profit strategies for profit management.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from loguru import logger


class TakeProfitHandler:
    """
    Handles calculation and management of take profit orders.
    """
    
    def __init__(self):
        """Initialize the take profit handler."""
        self.active_take_profits = {}
        
    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        strategy: str = 'fixed',
        params: Dict[str, Any] = None
    ) -> Union[float, List[Dict[str, Any]]]:
        """
        Calculate the take profit price based on strategy and parameters.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            strategy: Take profit strategy ('fixed', 'percent', 'fibonacci', 'tiered')
            params: Additional parameters for the strategy
            
        Returns:
            float or list: Calculated take profit price(s)
        """
        if params is None:
            params = {}
            
        if strategy == 'fixed':
            return self._calculate_fixed_take_profit(entry_price, side, params)
        elif strategy == 'percent':
            return self._calculate_percent_take_profit(entry_price, side, params)
        elif strategy == 'fibonacci':
            return self._calculate_fibonacci_take_profit(entry_price, side, params)
        elif strategy == 'tiered':
            return self._calculate_tiered_take_profit(entry_price, side, params)
        else:
            logger.warning(f"Unknown take profit strategy: {strategy}, using fixed")
            return self._calculate_fixed_take_profit(entry_price, side, params)
            
    def _calculate_fixed_take_profit(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate take profit based on a fixed price distance.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Take profit price
        """
        # Get distance in price units
        distance = params.get('distance', 0)
        
        if side == 'long':
            return entry_price + distance
        else:  # short
            return entry_price - distance
            
    def _calculate_percent_take_profit(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate take profit based on a percentage of entry price.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Take profit price
        """
        # Get percentage (e.g. 5.0 for 5%)
        percentage = params.get('percentage', 5.0)
        
        if side == 'long':
            return entry_price * (1 + percentage / 100)
        else:  # short
            return entry_price * (1 - percentage / 100)
            
    def _calculate_fibonacci_take_profit(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Calculate multiple take profit levels using Fibonacci retracement.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            list: Take profit levels
        """
        # Get price range for Fibonacci calculation
        price_range = params.get('price_range', 0)
        if price_range == 0:
            # Default to a percentage of entry price
            price_range = entry_price * 0.1  # 10% of entry price
            
        # Get Fibonacci levels (common levels: 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
        levels = params.get('levels', [0.5, 0.618, 0.786, 1.0])
        
        result = []
        for level in levels:
            if side == 'long':
                price = entry_price + (price_range * level)
                percentage = ((price / entry_price) - 1) * 100
            else:  # short
                price = entry_price - (price_range * level)
                percentage = ((entry_price / price) - 1) * 100
                
            result.append({
                'level': level,
                'price': price,
                'percentage': percentage
            })
            
        return result
        
    def _calculate_tiered_take_profit(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Calculate tiered take profit levels with position sizing.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            list: Take profit levels with allocation percentages
        """
        # Get tiers as a list of dicts with percentage and allocation
        # e.g. [{'percentage': 2.0, 'allocation': 0.3}, {'percentage': 5.0, 'allocation': 0.7}]
        tiers = params.get('tiers', [
            {'percentage': 2.0, 'allocation': 0.3},
            {'percentage': 5.0, 'allocation': 0.7}
        ])
        
        result = []
        for tier in tiers:
            percentage = tier.get('percentage', 0)
            allocation = tier.get('allocation', 0)
            
            if side == 'long':
                price = entry_price * (1 + percentage / 100)
            else:  # short
                price = entry_price * (1 - percentage / 100)
                
            result.append({
                'price': price,
                'percentage': percentage,
                'allocation': allocation
            })
            
        return result
        
    def register_take_profit(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        take_profit_price: Union[float, List[Dict[str, Any]]],
        side: str,
        amount: float,
        strategy: str = 'fixed',
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a take profit for tracking.
        
        Args:
            symbol: Trading pair symbol
            position_id: Position identifier
            entry_price: Entry price of the position
            take_profit_price: Take profit price or levels
            side: Position side ('long' or 'short')
            amount: Position size
            strategy: Take profit strategy
            params: Additional parameters
            
        Returns:
            dict: Take profit information
        """
        if params is None:
            params = {}
            
        take_profit_info = {
            'symbol': symbol,
            'position_id': position_id,
            'entry_price': entry_price,
            'take_profit_price': take_profit_price,
            'side': side,
            'amount': amount,
            'strategy': strategy,
            'params': params,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'is_active': True,
            'triggered_levels': []  # For tracking which levels have been triggered
        }
        
        # Store by position ID
        self.active_take_profits[position_id] = take_profit_info
        
        logger.info(f"Registered take profit for {symbol} position {position_id}")
        return take_profit_info
        
    def update_take_profit(
        self,
        position_id: str,
        new_take_profit_price: Union[float, List[Dict[str, Any]]],
        params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing take profit.
        
        Args:
            position_id: Position identifier
            new_take_profit_price: New take profit price or levels
            params: Updated parameters
            
        Returns:
            dict: Updated take profit information, or None if not found
        """
        if position_id not in self.active_take_profits:
            logger.warning(f"Take profit for position {position_id} not found")
            return None
            
        take_profit_info = self.active_take_profits[position_id]
        take_profit_info['take_profit_price'] = new_take_profit_price
        take_profit_info['updated_at'] = datetime.now()
        
        if params:
            take_profit_info['params'].update(params)
            
        logger.info(f"Updated take profit for position {position_id}")
        return take_profit_info
        
    def cancel_take_profit(self, position_id: str) -> bool:
        """
        Cancel an active take profit.
        
        Args:
            position_id: Position identifier
            
        Returns:
            bool: True if canceled, False if not found
        """
        if position_id not in self.active_take_profits:
            logger.warning(f"Take profit for position {position_id} not found")
            return False
            
        take_profit_info = self.active_take_profits.pop(position_id)
        logger.info(f"Canceled take profit for position {position_id}")
        return True
        
    def check_take_profit_trigger(
        self,
        symbol: str,
        current_price: float
    ) -> List[Dict[str, Any]]:
        """
        Check if any take profits should be triggered at the current price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            list: Triggered take profit information
        """
        triggered = []
        
        for position_id, take_profit in list(self.active_take_profits.items()):
            if take_profit['symbol'] != symbol or not take_profit['is_active']:
                continue
                
            # Handle simple fixed take profit
            if isinstance(take_profit['take_profit_price'], (int, float)):
                if (take_profit['side'] == 'long' and current_price >= take_profit['take_profit_price']) or \
                   (take_profit['side'] == 'short' and current_price <= take_profit['take_profit_price']):
                    # Trigger take profit
                    take_profit['triggered_price'] = current_price
                    take_profit['triggered_at'] = datetime.now()
                    take_profit['is_active'] = False
                    
                    triggered.append(take_profit)
                    logger.info(f"Take profit triggered for {symbol} position {position_id} at {current_price}")
                    
            # Handle tiered take profit levels
            elif isinstance(take_profit['take_profit_price'], list):
                triggered_level = None
                
                # Check each level
                for level in take_profit['take_profit_price']:
                    level_price = level['price']
                    level_idx = take_profit['take_profit_price'].index(level)
                    
                    # Skip already triggered levels
                    if level_idx in take_profit['triggered_levels']:
                        continue
                        
                    if (take_profit['side'] == 'long' and current_price >= level_price) or \
                       (take_profit['side'] == 'short' and current_price <= level_price):
                        # Trigger this level
                        triggered_level = {
                            **take_profit,
                            'triggered_price': current_price,
                            'triggered_at': datetime.now(),
                            'level_idx': level_idx,
                            'level_info': level,
                            'is_final_level': level_idx == len(take_profit['take_profit_price']) - 1
                        }
                        
                        # Mark this level as triggered
                        take_profit['triggered_levels'].append(level_idx)
                        
                        # If all levels are triggered or it's the final level with 100% allocation,
                        # mark the take profit as inactive
                        if level_idx == len(take_profit['take_profit_price']) - 1 or \
                           len(take_profit['triggered_levels']) == len(take_profit['take_profit_price']):
                            take_profit['is_active'] = False
                            
                        logger.info(f"Take profit level {level_idx} triggered for {symbol} position {position_id} at {current_price}")
                        triggered.append(triggered_level)
                        
                        # Only trigger one level at a time
                        break
                    
        return triggered
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of active take profits.
        
        Returns:
            dict: Active take profits
        """
        return {
            'active_take_profits': self.active_take_profits
        }