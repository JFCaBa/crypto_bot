"""
Trailing Stop Handler
===================
Implements trailing stop functionality for dynamic stop loss management.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np
from loguru import logger


class TrailingStopHandler:
    """
    Handles calculation and management of trailing stops.
    """
    
    def __init__(self):
        """Initialize the trailing stop handler."""
        self.active_trailing_stops = {}
        
    def register_trailing_stop(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        side: str,
        amount: float,
        initial_stop_price: float,
        strategy: str = 'percent',
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a trailing stop for tracking.
        
        Args:
            symbol: Trading pair symbol
            position_id: Position identifier
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            amount: Position size
            initial_stop_price: Initial stop price
            strategy: Trailing stop strategy ('percent', 'atr', 'fixed')
            params: Additional parameters
            
        Returns:
            dict: Trailing stop information
        """
        if params is None:
            params = {}
            
        trailing_stop_info = {
            'symbol': symbol,
            'position_id': position_id,
            'entry_price': entry_price,
            'initial_stop_price': initial_stop_price,
            'current_stop_price': initial_stop_price,
            'side': side,
            'amount': amount,
            'strategy': strategy,
            'params': params,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'is_active': True,
            'activated': False,  # Track if trailing has been activated
            'activation_price': params.get('activation_price', entry_price),
            'trail_distance': self._calculate_trail_distance(
                entry_price, initial_stop_price, side, strategy, params
            )
        }
        
        # Store by position ID
        self.active_trailing_stops[position_id] = trailing_stop_info
        
        logger.info(f"Registered trailing stop for {symbol} position {position_id}")
        return trailing_stop_info

    def update_trailing_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Update a trailing stop based on current price.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            dict: Updated trailing stop information, or None if not applicable
        """
        if position_id not in self.active_trailing_stops:
            return None
            
        stop_info = self.active_trailing_stops[position_id]
        
        if not stop_info['is_active']:
            return None
            
        # Check if price has moved favorably enough to activate trailing
        side = stop_info['side']
        entry_price = stop_info['entry_price']
        activation_price = stop_info['activation_price']
        
        # Determine if trailing should be activated
        is_favorable_move = False
        if side == 'long' and current_price > activation_price:
            is_favorable_move = True
        elif side == 'short' and current_price < activation_price:
            is_favorable_move = True
            
        # If already activated or favorable move detected, update stop price
        if stop_info['activated'] or is_favorable_move:
            stop_info['activated'] = True
            
            strategy = stop_info['strategy']
            trail_distance = stop_info['trail_distance']
            
            # Calculate new stop price
            new_stop_price = self._calculate_new_stop_price(
                current_price, trail_distance, side, strategy
            )
            
            # Only update if it would move the stop in the favorable direction
            if side == 'long' and new_stop_price > stop_info['current_stop_price']:
                stop_info['current_stop_price'] = new_stop_price
                stop_info['updated_at'] = datetime.now()
                logger.debug(f"Updated trailing stop for {stop_info['symbol']} to {new_stop_price:.8f}")
                return stop_info
            elif side == 'short' and new_stop_price < stop_info['current_stop_price']:
                stop_info['current_stop_price'] = new_stop_price
                stop_info['updated_at'] = datetime.now()
                logger.debug(f"Updated trailing stop for {stop_info['symbol']} to {new_stop_price:.8f}")
                return stop_info
                
        return None
        
    def cancel_trailing_stop(self, position_id: str) -> bool:
        """
        Cancel an active trailing stop.
        
        Args:
            position_id: Position identifier
            
        Returns:
            bool: True if canceled, False if not found
        """
        if position_id not in self.active_trailing_stops:
            logger.warning(f"Trailing stop for position {position_id} not found")
            return False
            
        stop_info = self.active_trailing_stops.pop(position_id)
        logger.info(f"Canceled trailing stop for position {position_id}")
        return True
        
    def check_stop_triggered(
        self,
        symbol: str,
        current_price: float
    ) -> List[Dict[str, Any]]:
        """
        Check if any trailing stops have been triggered.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            list: Triggered trailing stop information
        """
        triggered = []
        
        for position_id, stop_info in list(self.active_trailing_stops.items()):
            if stop_info['symbol'] != symbol or not stop_info['is_active']:
                continue
                
            side = stop_info['side']
            stop_price = stop_info['current_stop_price']
            
            if (side == 'long' and current_price <= stop_price) or \
               (side == 'short' and current_price >= stop_price):
                # Trigger stop
                stop_info['triggered_price'] = current_price
                stop_info['triggered_at'] = datetime.now()
                stop_info['is_active'] = False
                
                triggered.append(stop_info)
                logger.info(f"Trailing stop triggered for {symbol} position {position_id} at {current_price}")
                
        return triggered
        
    def _calculate_trail_distance(
        self,
        entry_price: float,
        initial_stop_price: float,
        side: str,
        strategy: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate the trailing distance based on strategy.
        
        Args:
            entry_price: Entry price of the position
            initial_stop_price: Initial stop price
            side: Position side ('long' or 'short')
            strategy: Trailing stop strategy
            params: Strategy parameters
            
        Returns:
            float: Trailing distance
        """
        if strategy == 'percent':
            # Use percentage of current price
            percent = params.get('trail_percent', 1.0)
            return entry_price * (percent / 100)
        elif strategy == 'atr':
            # Use ATR multiple
            atr = params.get('atr_value', 0)
            atr_multiple = params.get('atr_multiple', 1.5)
            return atr * atr_multiple
        elif strategy == 'fixed':
            # Use fixed distance
            return params.get('trail_distance', abs(entry_price - initial_stop_price))
        else:
            # Default to the initial distance from entry
            return abs(entry_price - initial_stop_price)
            
    def _calculate_new_stop_price(
        self,
        current_price: float,
        trail_distance: float,
        side: str,
        strategy: str
    ) -> float:
        """
        Calculate new stop price based on current price and trail distance.
        
        Args:
            current_price: Current market price
            trail_distance: Trailing distance
            side: Position side ('long' or 'short')
            strategy: Trailing stop strategy
            
        Returns:
            float: New stop price
        """
        if strategy == 'percent':
            # Recalculate based on percentage of current price
            if side == 'long':
                return current_price * (1 - trail_distance / current_price)
            else:  # short
                return current_price * (1 + trail_distance / current_price)
        else:
            # Use fixed distance
            if side == 'long':
                return current_price - trail_distance
            else:  # short
                return current_price + trail_distance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of active trailing stops.
        
        Returns:
            dict: Active trailing stops
        """
        return {
            'active_trailing_stops': self.active_trailing_stops
        }