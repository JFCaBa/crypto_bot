"""
Risk Management
==============
Manages risk for trading strategies by enforcing various risk controls.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

from loguru import logger

from cryptobot.core.trade import Trade
from cryptobot.risk_management.stop_loss import StopLossHandler
from cryptobot.risk_management.take_profit import TakeProfitHandler


class RiskManager:
    """
    Risk Manager class for enforcing trading risk controls.
    """
    
    def __init__(
        self,
        max_positions: int = 5,
        max_daily_trades: int = 20,
        max_drawdown_percent: float = 20.0,
        max_risk_per_trade: float = 2.0,
        max_risk_per_day: float = 5.0,
        max_risk_per_symbol: float = 10.0,
        default_stop_loss: float = 2.0,
        default_take_profit: float = 4.0,
        correlation_limit: float = 0.7,
        night_trading: bool = True,
        weekend_trading: bool = True,
        account_size: float = 10000.0,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Risk Manager.
        
        Args:
            max_positions: Maximum number of open positions
            max_daily_trades: Maximum number of trades per day
            max_drawdown_percent: Maximum drawdown as percentage of account size
            max_risk_per_trade: Maximum risk per trade as percentage of account size
            max_risk_per_day: Maximum risk per day as percentage of account size
            max_risk_per_symbol: Maximum risk per symbol as percentage of account size
            default_stop_loss: Default stop loss percentage
            default_take_profit: Default take profit percentage
            correlation_limit: Correlation limit for concurrent positions
            night_trading: Whether to allow trading during night hours
            weekend_trading: Whether to allow trading during weekends
            account_size: Account size for risk calculations
            params: Additional risk parameters
        """
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_percent = max_drawdown_percent
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_per_day = max_risk_per_day
        self.max_risk_per_symbol = max_risk_per_symbol
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        self.correlation_limit = correlation_limit
        self.night_trading = night_trading
        self.weekend_trading = weekend_trading
        self.account_size = account_size
        
        # Additional parameters
        self.params = params or {}
        
        # Internal state
        self.current_drawdown = 0.0
        self.daily_trades_count = 0
        self.daily_risk_used = 0.0
        self.symbol_risk: Dict[str, float] = {}
        self.last_date = None  # Last observed date, will be set on first data point
        
        # Initialize stop loss and take profit handlers
        self.stop_loss_handler = StopLossHandler()
        self.take_profit_handler = TakeProfitHandler()
        
        # Kill switch
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        logger.info("Risk Manager initialized")
        
    def validate_signal(
        self, 
        signal: Dict[str, Any], 
        positions: Dict[str, Dict[str, Any]], 
        trade_history: List[Trade]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a trading signal against risk rules.
        
        Args:
            signal: Trading signal
            positions: Current open positions
            trade_history: Trade history
            
        Returns:
            tuple: (approval_status, adjusted_signal)
        """
        # Check if kill switch is active
        if self.kill_switch_active:
            logger.warning(f"Kill switch active: {self.kill_switch_reason}")
            return False, signal
            
        # Extract timestamp from signal
        timestamp = self._get_timestamp(signal.get('timestamp'))
        
        # Check if date has changed and reset metrics if needed
        if timestamp:
            self._check_date_change(timestamp)
            
        # Extract signal details
        symbol = signal.get('symbol')
        action = signal.get('action')
        side = signal.get('side', 'buy' if action == 'buy' else 'sell' if action == 'sell' else None)
        amount = signal.get('amount', 0)
        price = signal.get('price', 0)
        
        if not symbol or not action or not amount or not price:
            logger.warning(f"Invalid signal format: {signal}")
            return False, signal
            
        # Check trading hours restrictions
        if timestamp and not self._check_trading_hours(timestamp):
            logger.warning(f"Trading not allowed during current hours ({timestamp})")
            return False, signal
            
        # Check if opening new position
        if action in ['buy', 'sell'] and not positions[symbol]['is_active']:
            # Check maximum positions limit
            active_positions = sum(1 for pos in positions.values() if pos['is_active'])
            if active_positions >= self.max_positions:
                logger.warning(f"Maximum positions limit reached: {active_positions}/{self.max_positions}")
                return False, signal
                
            # Check daily trade count limit
            if self.daily_trades_count >= self.max_daily_trades:
                logger.warning(f"Maximum daily trades limit reached: {self.daily_trades_count}/{self.max_daily_trades}")
                return False, signal
                
            # Calculate risk for this trade
            trade_risk = self._calculate_trade_risk(signal)
            
            # Check risk per trade limit
            max_risk_amount = self.account_size * (self.max_risk_per_trade / 100)
            if trade_risk > max_risk_amount:
                logger.warning(f"Risk per trade limit exceeded: {trade_risk:.2f}/{max_risk_amount:.2f}")
                
                # Adjust position size to comply with risk limit
                adjusted_amount = amount * (max_risk_amount / trade_risk)
                signal['amount'] = adjusted_amount
                logger.info(f"Adjusted position size from {amount} to {adjusted_amount}")
                
                # Recalculate trade risk
                trade_risk = self._calculate_trade_risk(signal)
                
            # Check risk per day limit
            if self.daily_risk_used + trade_risk > self.account_size * (self.max_risk_per_day / 100):
                logger.warning("Daily risk limit would be exceeded")
                return False, signal
                
            # Check risk per symbol limit
            symbol_risk = self.symbol_risk.get(symbol, 0) + trade_risk
            if symbol_risk > self.account_size * (self.max_risk_per_symbol / 100):
                logger.warning(f"Risk per symbol limit exceeded for {symbol}")
                return False, signal
                
            # Check correlation with existing positions
            if not self._check_correlation(symbol, side, positions):
                logger.warning(f"Correlation limit exceeded for {symbol}")
                return False, signal
                
            # Ensure stop loss and take profit are set
            if 'stop_loss' not in signal and self.default_stop_loss > 0:
                # Use the stop loss handler to calculate the stop loss
                stop_loss_price = self.stop_loss_handler.calculate_stop_loss(
                    symbol=symbol,
                    entry_price=price,
                    side=side,
                    strategy='percent',
                    params={'percentage': self.default_stop_loss}
                )
                signal['stop_loss'] = stop_loss_price
                logger.info(f"Added default stop loss at {signal['stop_loss']}")
                
            if 'take_profit' not in signal and self.default_take_profit > 0:
                # Use the take profit handler to calculate the take profit
                take_profit_price = self.take_profit_handler.calculate_take_profit(
                    symbol=symbol,
                    entry_price=price,
                    side=side,
                    strategy='percent',
                    params={'percentage': self.default_take_profit}
                )
                signal['take_profit'] = take_profit_price
                logger.info(f"Added default take profit at {signal['take_profit']}")
                
            # Update risk metrics
            self.daily_trades_count += 1
            self.daily_risk_used += trade_risk
            self.symbol_risk[symbol] = symbol_risk
            
            return True, signal
            
        # For closing positions, always allow
        elif action == 'close' and positions[symbol]['is_active']:
            return True, signal
            
        else:
            logger.warning(f"Invalid action {action} for current position state")
            return False, signal
            
    def update_account_balance(self, balance: float, timestamp=None):
        """
        Update account balance.
        
        Args:
            balance: New account balance
            timestamp: Timestamp for the update (for date tracking)
        """
        previous_balance = self.account_size
        self.account_size = balance
        
        # Calculate current drawdown
        peak_balance = max(previous_balance, balance)
        if peak_balance > 0:
            self.current_drawdown = (peak_balance - balance) / peak_balance * 100
            
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_percent:
            self._activate_kill_switch(f"Max drawdown exceeded: {self.current_drawdown:.2f}%")
            
        logger.info(f"Account balance updated: {balance:.2f}, Drawdown: {self.current_drawdown:.2f}%")
        
        # Check for date change if timestamp provided
        if timestamp:
            timestamp_obj = self._get_timestamp(timestamp)
            if timestamp_obj:
                self._check_date_change(timestamp_obj)
        
    def update_after_trade(self, trade: Trade):
        """
        Update risk metrics after a trade.
        
        Args:
            trade: Completed trade
        """
        # Check for date change using trade timestamp
        if hasattr(trade, 'timestamp') and trade.timestamp:
            timestamp = self._get_timestamp(trade.timestamp)
            if timestamp:
                self._check_date_change(timestamp)
        
        # Update metrics based on trade result
        if trade.pnl is not None:
            # Update account size
            self.account_size += trade.pnl
            
            # Calculate current drawdown
            if trade.pnl < 0:
                # Check if this increases our current drawdown
                peak_balance = self.account_size - trade.pnl  # Previous balance
                self.current_drawdown = max(
                    self.current_drawdown,
                    (peak_balance - self.account_size) / peak_balance * 100
                )
                
                # Check drawdown limit
                if self.current_drawdown > self.max_drawdown_percent:
                    self._activate_kill_switch(f"Max drawdown exceeded: {self.current_drawdown:.2f}%")
                    
        # For closed positions, update symbol risk
        if trade.symbol in self.symbol_risk:
            # Reduce the risk for this symbol
            self.symbol_risk[trade.symbol] = max(0, self.symbol_risk[trade.symbol] - self._calculate_trade_risk_from_trade(trade))
            
        logger.info(f"Risk metrics updated after trade: {trade.id}")
        
    def check_anomalies(self, current_prices: Dict[str, float], volatility: Dict[str, float], timestamp=None):
        """
        Check for market anomalies and activate kill switch if needed.
        
        Args:
            current_prices: Current prices by symbol
            volatility: Current volatility by symbol
            timestamp: Current timestamp for date tracking
        """
        # Check for date change if timestamp provided
        if timestamp:
            timestamp_obj = self._get_timestamp(timestamp)
            if timestamp_obj:
                self._check_date_change(timestamp_obj)
        
        # Check for extreme volatility
        volatility_threshold = self.params.get('volatility_threshold', 5.0)  # Default 5%
        
        for symbol, vol in volatility.items():
            if vol > volatility_threshold:
                self._activate_kill_switch(f"Extreme volatility detected for {symbol}: {vol:.2f}%")
                return
                
        # Check for extreme price movements
        price_change_threshold = self.params.get('price_change_threshold', 10.0)  # Default 10%
        
        # This requires historical comparison - in a real implementation,
        # we would compare to prices from previous period
        
        logger.debug("Anomaly check completed")
        
    def register_position(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        side: str,
        amount: float,
        stop_loss_price: float = None,
        take_profit_price: float = None,
        stop_loss_strategy: str = 'percent',
        take_profit_strategy: str = 'percent',
        stop_loss_params: Dict[str, Any] = None,
        take_profit_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a new position with stop loss and take profit.
        
        Args:
            symbol: Trading pair symbol
            position_id: Position identifier
            entry_price: Entry price of the position
            side: Position side ('long' or 'short')
            amount: Position size
            stop_loss_price: Stop loss price (optional)
            take_profit_price: Take profit price (optional)
            stop_loss_strategy: Stop loss strategy
            take_profit_strategy: Take profit strategy
            stop_loss_params: Stop loss parameters
            take_profit_params: Take profit parameters
            
        Returns:
            dict: Position registration information
        """
        if stop_loss_params is None:
            stop_loss_params = {}
        if take_profit_params is None:
            take_profit_params = {}
            
        position_info = {
            'symbol': symbol,
            'position_id': position_id,
            'entry_price': entry_price,
            'side': side,
            'amount': amount,
            'stop_loss': None,
            'take_profit': None,
            'created_at': datetime.now()
        }
        
        # Register stop loss if provided or calculate default
        if stop_loss_price:
            stop_loss = self.stop_loss_handler.register_stop_loss(
                symbol=symbol,
                position_id=position_id,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                side=side,
                amount=amount,
                strategy=stop_loss_strategy,
                params=stop_loss_params
            )
            position_info['stop_loss'] = stop_loss
        elif self.default_stop_loss > 0:
            # Calculate default stop loss
            stop_loss_price = self.stop_loss_handler.calculate_stop_loss(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                strategy='percent',
                params={'percentage': self.default_stop_loss}
            )
            stop_loss = self.stop_loss_handler.register_stop_loss(
                symbol=symbol,
                position_id=position_id,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                side=side,
                amount=amount,
                strategy='percent',
                params={'percentage': self.default_stop_loss}
            )
            position_info['stop_loss'] = stop_loss
            
        # Register take profit if provided or calculate default
        if take_profit_price:
            take_profit = self.take_profit_handler.register_take_profit(
                symbol=symbol,
                position_id=position_id,
                entry_price=entry_price,
                take_profit_price=take_profit_price,
                side=side,
                amount=amount,
                strategy=take_profit_strategy,
                params=take_profit_params
            )
            position_info['take_profit'] = take_profit
        elif self.default_take_profit > 0:
            # Calculate default take profit
            take_profit_price = self.take_profit_handler.calculate_take_profit(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                strategy='percent',
                params={'percentage': self.default_take_profit}
            )
            take_profit = self.take_profit_handler.register_take_profit(
                symbol=symbol,
                position_id=position_id,
                entry_price=entry_price,
                take_profit_price=take_profit_price,
                side=side,
                amount=amount,
                strategy='percent',
                params={'percentage': self.default_take_profit}
            )
            position_info['take_profit'] = take_profit
            
        logger.info(f"Registered position {position_id} for {symbol} with stop loss and take profit")
        return position_info
        
    def update_position_exit_points(
        self,
        position_id: str,
        stop_loss_price: float = None,
        take_profit_price: float = None,
        trailing_stop: bool = False,
        trailing_distance: float = None
    ) -> Dict[str, Any]:
        """
        Update the stop loss and take profit levels for an existing position.
        
        Args:
            position_id: Position identifier
            stop_loss_price: New stop loss price
            take_profit_price: New take profit price
            trailing_stop: Whether to enable trailing stop
            trailing_distance: Distance for trailing stop
            
        Returns:
            dict: Updated position information
        """
        position_info = {
            'position_id': position_id,
            'stop_loss_updated': False,
            'take_profit_updated': False,
            'trailing_stop_updated': False
        }
        
        # Update stop loss if provided
        if stop_loss_price is not None:
            updated_stop_loss = self.stop_loss_handler.update_stop_loss(
                position_id=position_id,
                new_stop_loss_price=stop_loss_price
            )
            position_info['stop_loss_updated'] = updated_stop_loss is not None
            
        # Update take profit if provided
        if take_profit_price is not None:
            updated_take_profit = self.take_profit_handler.update_take_profit(
                position_id=position_id,
                new_take_profit_price=take_profit_price
            )
            position_info['take_profit_updated'] = updated_take_profit is not None
            
        # Enable trailing stop if requested
        if trailing_stop:
            # Get the stop loss information
            if position_id in self.stop_loss_handler.active_stop_losses:
                stop_loss_info = self.stop_loss_handler.active_stop_losses[position_id]
                
                # Update parameters for trailing stop
                trailing_params = {
                    'trailing': True,
                    'trailing_distance': trailing_distance or (stop_loss_info['entry_price'] - stop_loss_info['stop_loss_price'])
                }
                
                # Update the stop loss with trailing parameters
                updated_stop_loss = self.stop_loss_handler.update_stop_loss(
                    position_id=position_id,
                    new_stop_loss_price=stop_loss_info['stop_loss_price'],
                    params=trailing_params
                )
                
                position_info['trailing_stop_updated'] = updated_stop_loss is not None
                logger.info(f"Enabled trailing stop for position {position_id}")
                
        return position_info
        
    def close_position(self, position_id: str) -> bool:
        """
        Close a position and remove its stop loss and take profit orders.
        
        Args:
            position_id: Position identifier
            
        Returns:
            bool: True if position was closed successfully
        """
        # Cancel stop loss
        stop_loss_cancelled = self.stop_loss_handler.cancel_stop_loss(position_id)
        
        # Cancel take profit
        take_profit_cancelled = self.take_profit_handler.cancel_take_profit(position_id)
        
        return stop_loss_cancelled or take_profit_cancelled
        
    def check_exit_conditions(self, current_prices: Dict[str, float], timestamp=None) -> List[Dict[str, Any]]:
        """
        Check if any positions should be closed based on stop loss or take profit triggers.
        
        Args:
            current_prices: Current prices by symbol
            timestamp: Current timestamp for date tracking
            
        Returns:
            list: List of exit signals
        """
        # Check for date change if timestamp provided
        if timestamp:
            timestamp_obj = self._get_timestamp(timestamp)
            if timestamp_obj:
                self._check_date_change(timestamp_obj)
        
        exit_signals = []
        
        for symbol, price in current_prices.items():
            # Check stop losses
            triggered_stop_losses = self.stop_loss_handler.check_stop_loss_trigger(symbol, price)
            for stop_loss in triggered_stop_losses:
                exit_signal = {
                    'symbol': symbol,
                    'action': 'close',
                    'side': 'sell' if stop_loss['side'] == 'long' else 'buy',
                    'amount': stop_loss['amount'],
                    'price': price,
                    'position_id': stop_loss['position_id'],
                    'reason': 'stop_loss',
                    'triggered_price': stop_loss['triggered_price']
                }
                exit_signals.append(exit_signal)
                logger.info(f"Stop loss triggered for {symbol} at {price}")
                
            # Check take profits
            triggered_take_profits = self.take_profit_handler.check_take_profit_trigger(symbol, price)
            for take_profit in triggered_take_profits:
                # For tiered take profits, calculate the amount to close
                amount = take_profit['amount']
                if 'level_info' in take_profit and 'allocation' in take_profit['level_info']:
                    amount = take_profit['amount'] * take_profit['level_info']['allocation']
                
                exit_signal = {
                    'symbol': symbol,
                    'action': 'close',
                    'side': 'sell' if take_profit['side'] == 'long' else 'buy',
                    'amount': amount,
                    'price': price,
                    'position_id': take_profit['position_id'],
                    'reason': 'take_profit',
                    'triggered_price': take_profit['triggered_price'],
                    'is_partial': 'level_info' in take_profit and not take_profit['is_final_level']
                }
                exit_signals.append(exit_signal)
                logger.info(f"Take profit triggered for {symbol} at {price}")
                
            # Update trailing stops
            for position_id in list(self.stop_loss_handler.active_stop_losses.keys()):
                stop_loss = self.stop_loss_handler.active_stop_losses[position_id]
                if stop_loss['symbol'] == symbol:
                    self.stop_loss_handler.apply_trailing_stop(position_id, price)
                    
        return exit_signals
        
    def reset_kill_switch(self):
        """Reset the kill switch."""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        logger.info("Kill switch reset")
        
    def _activate_kill_switch(self, reason: str):
        """
        Activate the kill switch.
        
        Args:
            reason: Reason for activation
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        logger.warning(f"Kill switch activated: {reason}")
        
    def _reset_daily_metrics(self):
        """Reset daily risk metrics."""
        self.daily_trades_count = 0
        self.daily_risk_used = 0.0
        logger.info("Daily risk metrics reset")
        
    def _get_timestamp(self, timestamp) -> Optional[datetime]:
        """
        Convert various timestamp formats to datetime.
        
        Args:
            timestamp: Timestamp in various possible formats
            
        Returns:
            datetime or None: Converted timestamp or None if conversion failed
        """
        if timestamp is None:
            return None
            
        # If already a datetime, return it
        if isinstance(timestamp, datetime):
            return timestamp
            
        # Handle pandas Timestamp
        try:
            import pandas as pd
            if isinstance(timestamp, pd.Timestamp):
                return timestamp.to_pydatetime()
        except ImportError:
            pass
            
        # Handle string timestamp
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                pass
                
            try:
                import dateutil.parser
                return dateutil.parser.parse(timestamp)
            except (ImportError, ValueError):
                pass
                
        # Handle integer/float timestamp (assume milliseconds)
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp / 1000)
            except (ValueError, OverflowError):
                try:
                    return datetime.fromtimestamp(timestamp)
                except (ValueError, OverflowError):
                    pass
                    
        logger.warning(f"Could not parse timestamp: {timestamp}")
        return None
        
    def _check_date_change(self, current_timestamp: datetime):
        """
        Check if the date has changed and reset daily metrics if needed.
        
        Args:
            current_timestamp: Current datetime
        """
        if current_timestamp:
            current_date = current_timestamp.date()
            
            # If this is the first timestamp we've seen, just store the date
            if self.last_date is None:
                self.last_date = current_date
                return
                
            # If date has changed, reset daily metrics
            if current_date > self.last_date:
                logger.info(f"Date changed from {self.last_date} to {current_date}, resetting daily metrics")
                self._reset_daily_metrics()
                self.last_date = current_date
                
    def _calculate_trade_risk(self, signal: Dict[str, Any]) -> float:
        """
        Calculate risk amount for a trade signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            float: Risk amount
        """
        price = signal.get('price', 0)
        amount = signal.get('amount', 0)
        stop_loss = signal.get('stop_loss', 0)
        
        if price <= 0 or amount <= 0 or stop_loss <= 0:
            return 0.0
            
        # For buy orders, risk is (entry - stop_loss) * amount
        if signal.get('action') == 'buy' or signal.get('side') == 'buy':
            return abs(price - stop_loss) * amount
        # For sell orders, risk is (stop_loss - entry) * amount
        elif signal.get('action') == 'sell' or signal.get('side') == 'sell':
            return abs(stop_loss - price) * amount
        else:
            return 0.0
            
    def _calculate_trade_risk_from_trade(self, trade: Trade) -> float:
        """
        Calculate risk amount from a trade.
        
        Args:
            trade: Trade object
            
        Returns:
            float: Risk amount
        """
        # In a real implementation, this would use the actual stop loss from the trade
        # For this implementation, we'll use a simplified approach
        return trade.amount * trade.price * (self.default_stop_loss / 100)
        
    def _check_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if trading is allowed during the hours of the given timestamp.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        # Check weekend trading
        if not self.weekend_trading and timestamp.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
            
        # Check night trading (define night as 10 PM to 6 AM)
        if not self.night_trading and (timestamp.hour >= 22 or timestamp.hour < 6):
            return False
            
        return True
        
    def _check_correlation(self, symbol: str, side: str, positions: Dict[str, Dict[str, Any]]) -> bool:
        """
        Check correlation between new position and existing positions.
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('buy' or 'sell')
            positions: Current open positions
            
        Returns:
            bool: True if correlation is acceptable, False otherwise
        """
        # Count how many positions have the same direction
        same_direction_count = 0
        for pos in positions.values():
            if pos['is_active'] and pos['side'] == ('long' if side == 'buy' else 'short'):
                same_direction_count += 1
                
        # If too many positions in the same direction, reject
        correlation_threshold = self.max_positions // 2
        if same_direction_count >= correlation_threshold:
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert risk manager to dictionary.
        
        Returns:
            dict: Risk manager information
        """
        return {
            'max_positions': self.max_positions,
            'max_daily_trades': self.max_daily_trades,
            'max_drawdown_percent': self.max_drawdown_percent,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_risk_per_day': self.max_risk_per_day,
            'max_risk_per_symbol': self.max_risk_per_symbol,
            'default_stop_loss': self.default_stop_loss,
            'default_take_profit': self.default_take_profit,
            'correlation_limit': self.correlation_limit,
            'night_trading': self.night_trading,
            'weekend_trading': self.weekend_trading,
            'account_size': self.account_size,
            'current_drawdown': self.current_drawdown,
            'daily_trades_count': self.daily_trades_count,
            'daily_risk_used': self.daily_risk_used,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'last_date': self.last_date.isoformat() if self.last_date else None,
            'stop_loss_handler': self.stop_loss_handler.to_dict() if self.stop_loss_handler else None,
            'take_profit_handler': self.take_profit_handler.to_dict() if self.take_profit_handler else None
        }