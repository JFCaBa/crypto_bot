"""
Stop Loss Handler
===============
Implements various stop loss strategies for risk management.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from loguru import logger


class StopLossHandler:
    """
    Handles calculation and management of stop loss orders.
    """
    
    def __init__(self):
        """Initialize the stop loss handler."""
        self.active_stop_losses = {}
        
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        strategy: str = 'fixed',
        params: Dict[str, Any] = None
    ) -> float:
        """
        Calculate the stop loss price based on strategy and parameters.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            strategy: Stop loss strategy ('fixed', 'atr', 'support_resistance', 'percent')
            params: Additional parameters for the strategy
            
        Returns:
            float: Calculated stop loss price
        """
        if params is None:
            params = {}
            
        if strategy == 'fixed':
            return self._calculate_fixed_stop_loss(entry_price, side, params)
        elif strategy == 'atr':
            return self._calculate_atr_stop_loss(symbol, entry_price, side, params)
        elif strategy == 'support_resistance':
            return self._calculate_support_resistance_stop_loss(symbol, entry_price, side, params)
        elif strategy == 'percent':
            return self._calculate_percent_stop_loss(entry_price, side, params)
        else:
            logger.warning(f"Unknown stop loss strategy: {strategy}, using fixed")
            return self._calculate_fixed_stop_loss(entry_price, side, params)
            
    def _calculate_fixed_stop_loss(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate stop loss based on a fixed price distance.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Stop loss price
        """
        # Get distance in price units
        distance = params.get('distance', 0)
        
        if side == 'long':
            return entry_price - distance
        else:  # short
            return entry_price + distance
            
    def _calculate_percent_stop_loss(
        self,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate stop loss based on a percentage of entry price.
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Stop loss price
        """
        # Get percentage (e.g. 2.0 for 2%)
        percentage = params.get('percentage', 2.0)
        
        if side == 'long':
            return entry_price * (1 - percentage / 100)
        else:  # short
            return entry_price * (1 + percentage / 100)
            
    def _calculate_atr_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate stop loss based on Average True Range (ATR).
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Stop loss price
        """
        # ATR value should be provided in params
        atr_value = params.get('atr_value', 0)
        # Multiplier for ATR
        multiplier = params.get('multiplier', 2.0)
        
        atr_distance = atr_value * multiplier
        
        if side == 'long':
            return entry_price - atr_distance
        else:  # short
            return entry_price + atr_distance
            
    def _calculate_support_resistance_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        params: Dict[str, Any]
    ) -> float:
        """
        Calculate stop loss based on support/resistance levels.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            params: Additional parameters
            
        Returns:
            float: Stop loss price
        """
        # Support/resistance levels should be provided
        levels = params.get('levels', [])
        default_percentage = params.get('default_percentage', 2.0)
        
        if not levels:
            # Fallback to percentage-based stop loss
            return self._calculate_percent_stop_loss(entry_price, side, {'percentage': default_percentage})
            
        if side == 'long':
            # Find the closest support level below entry price
            supports = [level for level in levels if level < entry_price]
            if supports:
                return max(supports)  # Highest support below entry
            else:
                # No support found, use percentage
                return self._calculate_percent_stop_loss(entry_price, side, {'percentage': default_percentage})
        else:  # short
            # Find the closest resistance level above entry price
            resistances = [level for level in levels if level > entry_price]
            if resistances:
                return min(resistances)  # Lowest resistance above entry
            else:
                # No resistance found, use percentage
                return self._calculate_percent_stop_loss(entry_price, side, {'percentage': default_percentage})
                
    def register_stop_loss(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        amount: float,
        strategy: str = 'fixed',
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a stop loss for tracking.
        
        Args:
            symbol: Trading pair symbol
            position_id: Position identifier
            entry_price: Entry price of the position
            stop_loss_price: Stop loss price
            side: Position side ('long' or 'short')
            amount: Position size
            strategy: Stop loss strategy
            params: Additional parameters
            
        Returns:
            dict: Stop loss information
        """
        if params is None:
            params = {}
            
        stop_loss_info = {
            'symbol': symbol,
            'position_id': position_id,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'side': side,
            'amount': amount,
            'strategy': strategy,
            'params': params,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'is_active': True
        }
        
        # Store by position ID
        self.active_stop_losses[position_id] = stop_loss_info
        
        logger.info(f"Registered stop loss for {symbol} position {position_id} at {stop_loss_price}")
        return stop_loss_info
        
    def update_stop_loss(
        self,
        position_id: str,
        new_stop_loss_price: float,
        params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing stop loss.
        
        Args:
            position_id: Position identifier
            new_stop_loss_price: New stop loss price
            params: Updated parameters
            
        Returns:
            dict: Updated stop loss information, or None if not found
        """
        if position_id not in self.active_stop_losses:
            logger.warning(f"Stop loss for position {position_id} not found")
            return None
            
        stop_loss_info = self.active_stop_losses[position_id]
        stop_loss_info['stop_loss_price'] = new_stop_loss_price
        stop_loss_info['updated_at'] = datetime.now()
        
        if params:
            stop_loss_info['params'].update(params)
            
        logger.info(f"Updated stop loss for position {position_id} to {new_stop_loss_price}")
        return stop_loss_info
        
    def cancel_stop_loss(self, position_id: str) -> bool:
        """
        Cancel an active stop loss.
        
        Args:
            position_id: Position identifier
            
        Returns:
            bool: True if canceled, False if not found
        """
        if position_id not in self.active_stop_losses:
            logger.warning(f"Stop loss for position {position_id} not found")
            return False
            
        stop_loss_info = self.active_stop_losses.pop(position_id)
        logger.info(f"Canceled stop loss for position {position_id}")
        return True
        
    def check_stop_loss_trigger(
        self,
        symbol: str,
        current_price: float
    ) -> List[Dict[str, Any]]:
        """
        Check if any stop losses should be triggered at the current price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            list: Triggered stop loss information
        """
        triggered = []
        
        for position_id, stop_loss in list(self.active_stop_losses.items()):
            if stop_loss['symbol'] != symbol or not stop_loss['is_active']:
                continue
                
            if (stop_loss['side'] == 'long' and current_price <= stop_loss['stop_loss_price']) or \
               (stop_loss['side'] == 'short' and current_price >= stop_loss['stop_loss_price']):
                # Trigger stop loss
                stop_loss['triggered_price'] = current_price
                stop_loss['triggered_at'] = datetime.now()
                stop_loss['is_active'] = False
                
                triggered.append(stop_loss)
                logger.info(f"Stop loss triggered for {symbol} position {position_id} at {current_price}")
                
        return triggered
        
    def apply_trailing_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Apply trailing stop logic to adjust stop loss based on price movement.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            dict: Updated stop loss information, or None if not applicable
        """
        if position_id not in self.active_stop_losses:
            return None
            
        stop_loss = self.active_stop_losses[position_id]
        
        # Check if trailing stop is enabled
        if 'trailing' not in stop_loss['params'] or not stop_loss['params']['trailing']:
            return None
            
        # Get trailing parameters
        activation_price = stop_loss['params'].get('activation_price', stop_loss['entry_price'])
        distance = stop_loss['params'].get('trailing_distance', 0)
        
        # Check if trailing should be activated
        if stop_loss['side'] == 'long':
            if current_price > activation_price:
                # Calculate new stop loss (current price - distance)
                new_stop_price = current_price - distance
                # Only update if it would raise the stop loss
                if new_stop_price > stop_loss['stop_loss_price']:
                    return self.update_stop_loss(position_id, new_stop_price)
        else:  # short
            if current_price < activation_price:
                # Calculate new stop loss (current price + distance)
                new_stop_price = current_price + distance
                # Only update if it would lower the stop loss
                if new_stop_price < stop_loss['stop_loss_price']:
                    return self.update_stop_loss(position_id, new_stop_price)
                    
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of active stop losses.
        
        Returns:
            dict: Active stop losses
        """
        return {
            'active_stop_losses': self.active_stop_losses
        }