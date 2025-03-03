"""
Grid Trading Strategy
====================
Implementation of a grid trading strategy that places orders at regular intervals
within a specified price range to profit from price oscillations.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager
from cryptobot.core.trade import Trade


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy.
    
    This strategy places buy and sell orders at predetermined price levels, creating
    a grid. As price moves up and down, the strategy buys at lower levels and sells
    at higher levels, generating profits from the oscillations.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Grid Trading strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            risk_manager: Risk manager instance
            params: Strategy parameters
        """
        # Initialize default parameters
        default_params = {
            'grid_levels': 10,             # Number of grid levels
            'grid_spacing': 1.0,           # Spacing between grid levels (percentage)
            'grid_range_low': 5.0,         # Lower bound of grid (percentage below current price)
            'grid_range_high': 5.0,        # Upper bound of grid (percentage above current price)
            'auto_set_range': True,        # Auto-set range based on recent volatility
            'volatility_period': 20,       # Period for volatility calculation (days)
            'volatility_factor': 2.0,      # Multiplier for volatility range
            'total_investment': 1000.0,    # Total investment amount
            'reinvest_profits': True,      # Whether to reinvest profits
            'rebalance_interval': 24,      # Hours between grid rebalancing
            'order_type': 'limit',         # Order type: 'limit' or 'market'
            'stop_loss_percent': 0.0,      # Overall stop loss (percentage)
            'max_active_grids': 1,         # Maximum active grids per symbol
            'min_profit_per_grid': 0.3,    # Minimum profit per grid (percentage)
            'enable_dynamic_grid': False,  # Dynamically adjust grid levels based on volatility
            'grid_order_lifetime': 24,     # Order lifetime in hours before recalculation
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="GridTrading",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        
        # Grid-specific attributes
        self.grid_levels: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.grid_info: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.last_rebalance: Dict[str, Dict[str, Optional[datetime]]] = {}
        
        # Initialize grids for each symbol
        for symbol in symbols:
            self.grid_levels[symbol] = {tf: [] for tf in timeframes}
            self.grid_info[symbol] = {tf: {} for tf in timeframes}
            self.last_rebalance[symbol] = {tf: None for tf in timeframes}
    
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate indicators for the grid strategy.
        For grid trading, we primarily need volatility measures to set grid ranges.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = self.data[symbol][timeframe]
            
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return False
                
            # Ensure we have OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns in data for {symbol} {timeframe}")
                return False
                
            # Calculate True Range for volatility
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Calculate ATR (Average True Range)
            df['atr'] = df['tr'].rolling(window=self.params['volatility_period']).mean()
            
            # Calculate percentage volatility
            df['volatility_pct'] = (df['atr'] / df['close']) * 100
            
            # Calculate other useful metrics for grid trading
            df['price_change_pct'] = df['close'].pct_change() * 100
            df['daily_range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
            
            # Store calculated indicators
            indicators = {
                'atr': df['atr'],
                'volatility_pct': df['volatility_pct'],
                'price_change_pct': df['price_change_pct'],
                'daily_range_pct': df['daily_range_pct']
            }
            
            # Update indicators
            self.indicators[symbol][timeframe] = indicators
            
            # Update the data with calculated values
            self.data[symbol][timeframe] = df
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals for grid strategy.
        
        For grid trading, this function:
        1. Checks if grid needs to be established or rebalanced
        2. Checks if any grid levels have been triggered
        3. Generates signals for new orders and closes
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            dict: Signal information
        """
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': None,
            'price': None,
            'amount': None,
            'params': {},
            'timestamp': pd.Timestamp.now()
        }
        
        try:
            # Ensure indicators are calculated
            if not self.calculate_indicators(symbol, timeframe):
                return signal
                
            df = self.data[symbol][timeframe]
            if df.empty:
                return signal
                
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Check if we need to establish or rebalance the grid
            if symbol not in self.grid_levels or timeframe not in self.grid_levels[symbol] or not self.grid_levels[symbol][timeframe]:
                # No grid established yet, create a new one
                self._create_grid(symbol, timeframe, current_price)
                return signal  # Return without action after grid setup
            
            # Check if grid needs rebalancing
            if self._should_rebalance_grid(symbol, timeframe, current_price):
                # Rebalance the grid
                self._rebalance_grid(symbol, timeframe, current_price)
                return signal  # Return without action after rebalance
            
            # Check each grid level to see if it's triggered
            position = self.positions[symbol]
            is_active = position['is_active']
            
            # Find the grid level closest to current price
            closest_level = self._find_closest_grid_level(symbol, timeframe, current_price)
            
            if closest_level:
                if current_price <= closest_level['buy_price'] and not is_active:
                    # Buy signal - price crossed below a buy level
                    signal['action'] = 'buy'
                    signal['price'] = closest_level['buy_price']
                    signal['amount'] = closest_level['amount']
                    
                    # Set stop loss if configured
                    if self.params['stop_loss_percent'] > 0:
                        stop_loss_price = closest_level['buy_price'] * (1 - self.params['stop_loss_percent'] / 100)
                        signal['stop_loss'] = stop_loss_price
                    
                    # Set take profit
                    take_profit_price = closest_level['sell_price']
                    signal['take_profit'] = take_profit_price
                    
                    # Mark this level as active
                    closest_level['active'] = True
                    closest_level['entry_price'] = signal['price']
                    closest_level['entry_time'] = pd.Timestamp.now()
                    
                    logger.info(f"Grid buy signal for {symbol} at level {closest_level['level']}: {signal['price']}")
                
                elif current_price >= closest_level['sell_price'] and is_active:
                    # Sell signal - price crossed above a sell level
                    signal['action'] = 'close'
                    signal['price'] = closest_level['sell_price']
                    signal['amount'] = position['amount']
                    
                    # Mark this level as inactive
                    closest_level['active'] = False
                    closest_level['exit_price'] = signal['price']
                    closest_level['exit_time'] = pd.Timestamp.now()
                    
                    # Calculate profit
                    profit_pct = ((closest_level['sell_price'] / position['entry_price']) - 1) * 100
                    logger.info(f"Grid sell signal for {symbol} at level {closest_level['level']}: {signal['price']} (Profit: {profit_pct:.2f}%)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating grid signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return signal
    
    def _create_grid(self, symbol: str, timeframe: str, current_price: float) -> None:
        """
        Create a new grid for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            current_price: Current market price
        """
        try:
            # Get parameters
            grid_levels = self.params['grid_levels']
            grid_spacing = self.params['grid_spacing']
            total_investment = self.params['total_investment']
            
            # Auto-set grid range based on volatility if enabled
            if self.params['auto_set_range']:
                # Get volatility from indicators
                volatility = self.indicators[symbol][timeframe]['volatility_pct'].iloc[-1]
                if pd.notna(volatility) and volatility > 0:
                    volatility_factor = self.params['volatility_factor']
                    grid_range = volatility * volatility_factor
                    self.params['grid_range_low'] = grid_range / 2
                    self.params['grid_range_high'] = grid_range / 2
                    logger.info(f"Auto-set grid range based on volatility: {grid_range:.2f}% for {symbol} {timeframe}")
            
            # Calculate grid price range
            lower_bound = current_price * (1 - self.params['grid_range_low'] / 100)
            upper_bound = current_price * (1 + self.params['grid_range_high'] / 100)
            price_range = upper_bound - lower_bound
            
            # Calculate price step
            price_step = price_range / (grid_levels - 1) if grid_levels > 1 else price_range
            
            # Calculate investment per grid level
            investment_per_level = total_investment / grid_levels
            
            # Create grid levels
            grid = []
            for i in range(grid_levels):
                # Calculate prices for this level
                buy_price = lower_bound + (i * price_step)
                sell_price = buy_price * (1 + grid_spacing / 100)
                
                # Calculate order amount based on investment
                amount = investment_per_level / buy_price
                
                # Create grid level
                grid_level = {
                    'level': i,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'amount': amount,
                    'active': False,  # Not active initially
                    'entry_price': None,
                    'exit_price': None,
                    'entry_time': None,
                    'exit_time': None,
                    'order_id': None
                }
                
                grid.append(grid_level)
            
            # Store grid information
            self.grid_levels[symbol][timeframe] = grid
            self.grid_info[symbol][timeframe] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'price_step': price_step,
                'current_price': current_price,
                'created_at': pd.Timestamp.now()
            }
            self.last_rebalance[symbol][timeframe] = pd.Timestamp.now()
            
            logger.info(f"Created grid for {symbol} {timeframe} with {grid_levels} levels from {lower_bound} to {upper_bound}")
            
        except Exception as e:
            logger.error(f"Error creating grid for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _should_rebalance_grid(self, symbol: str, timeframe: str, current_price: float) -> bool:
        """
        Determine if the grid needs rebalancing.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            current_price: Current market price
            
        Returns:
            bool: True if grid should be rebalanced
        """
        try:
            # Check if grid exists
            if (symbol not in self.grid_levels or 
                timeframe not in self.grid_levels[symbol] or 
                not self.grid_levels[symbol][timeframe]):
                return True
                
            # Get grid info
            grid_info = self.grid_info[symbol][timeframe]
            last_rebalance = self.last_rebalance[symbol][timeframe]
            
            # Check if price is outside grid bounds
            if current_price < grid_info['lower_bound'] or current_price > grid_info['upper_bound']:
                logger.info(f"Price {current_price} is outside grid bounds [{grid_info['lower_bound']}, {grid_info['upper_bound']}] for {symbol} {timeframe}")
                return True
                
            # Check if rebalance interval has passed
            if last_rebalance:
                hours_since_rebalance = (pd.Timestamp.now() - last_rebalance).total_seconds() / 3600
                if hours_since_rebalance >= self.params['rebalance_interval']:
                    logger.info(f"Rebalance interval reached for {symbol} {timeframe}: {hours_since_rebalance:.2f} hours")
                    return True
                    
            # Check for dynamic grid adjustment if enabled
            if self.params['enable_dynamic_grid']:
                # Check if volatility has changed significantly
                current_volatility = self.indicators[symbol][timeframe]['volatility_pct'].iloc[-1]
                if pd.notna(current_volatility) and 'initial_volatility' in grid_info:
                    volatility_change = abs(current_volatility - grid_info['initial_volatility']) / grid_info['initial_volatility']
                    if volatility_change > 0.5:  # 50% change in volatility
                        logger.info(f"Significant volatility change for {symbol} {timeframe}: {volatility_change:.2f}%")
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Error checking if grid should be rebalanced for {symbol} {timeframe}: {str(e)}")
            return False
    
    def _rebalance_grid(self, symbol: str, timeframe: str, current_price: float) -> None:
        """
        Rebalance the grid around current price.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            current_price: Current market price
        """
        try:
            # Close any open positions before rebalancing
            position = self.positions[symbol]
            if position['is_active']:
                logger.info(f"Active position found during rebalance for {symbol} {timeframe}. This will be closed by the strategy controller.")
                # Note: The actual closing of the position happens at the engine level
            
            # Re-create the grid
            self._create_grid(symbol, timeframe, current_price)
            logger.info(f"Grid rebalanced for {symbol} {timeframe} around price {current_price}")
            
        except Exception as e:
            logger.error(f"Error rebalancing grid for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _find_closest_grid_level(self, symbol: str, timeframe: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Find the grid level closest to the current price.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            current_price: Current market price
            
        Returns:
            dict: Closest grid level or None
        """
        try:
            grid = self.grid_levels[symbol][timeframe]
            
            closest_level = None
            min_distance = float('inf')
            
            for level in grid:
                # Check buy level distance
                buy_distance = abs(current_price - level['buy_price'])
                if buy_distance < min_distance:
                    min_distance = buy_distance
                    closest_level = level
                
                # Check sell level distance
                sell_distance = abs(current_price - level['sell_price'])
                if sell_distance < min_distance:
                    min_distance = sell_distance
                    closest_level = level
            
            return closest_level
            
        except Exception as e:
            logger.error(f"Error finding closest grid level for {symbol} {timeframe}: {str(e)}")
            return None
    
    async def execute_signals(self, signals: List[Dict[str, Any]], exchange) -> List[Trade]:
        """
        Execute grid trading signals through an exchange.
        
        This extends the base implementation to handle grid-specific logic.
        
        Args:
            signals: List of signals to execute
            exchange: Exchange instance for trade execution
            
        Returns:
            list: List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            if not self.is_active:
                logger.warning("Strategy is not active. Skipping signal execution.")
                break
                
            symbol = signal.get('symbol')
            action = signal.get('action')
            
            if not symbol or not action:
                logger.warning(f"Invalid signal: {signal}")
                continue
                
            try:
                # Apply risk management rules
                if self.risk_manager:
                    approved, adjusted_signal = self.risk_manager.validate_signal(signal, self.positions, self.trade_history)
                    if not approved:
                        logger.warning(f"Signal rejected by risk manager: {signal}")
                        continue
                    signal = adjusted_signal
                
                # Execute the signal based on the action
                if action == 'buy' and not self.positions[symbol]['is_active']:
                    # Open long position for grid level
                    amount = signal.get('amount')
                    price = signal.get('price')
                    params = signal.get('params', {})
                    
                    # Add stop loss and take profit if specified
                    if 'stop_loss' in signal:
                        params['stopLoss'] = signal['stop_loss']
                    if 'take_profit' in signal:
                        params['takeProfit'] = signal['take_profit']
                        
                    # Create market or limit order
                    order_type = self.params.get('order_type', 'limit')
                    result = await exchange.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side='buy',
                        amount=amount,
                        price=price if order_type == 'limit' else None,
                        params=params
                    )
                    
                    # Update grid level with order information
                    if result and 'id' in result:
                        timeframe = signal.get('timeframe')
                        for level in self.grid_levels[symbol][timeframe]:
                            if level['buy_price'] == price:
                                level['order_id'] = result['id']
                                level['active'] = True
                                level['entry_time'] = pd.Timestamp.now()
                                level['entry_price'] = price
                    
                    # Update position status
                    self.positions[symbol] = {
                        'is_active': True,
                        'side': 'long',
                        'entry_price': result.get('price') or price,
                        'amount': amount,
                        'entry_time': datetime.now(),
                        'order_id': result.get('id')
                    }
                    
                    # Create trade record
                    trade = Trade(
                        id=result.get('id'),
                        symbol=symbol,
                        side='buy',
                        amount=amount,
                        price=result.get('price') or price,
                        timestamp=datetime.now(),
                        strategy=self.name,
                        timeframe=signal.get('timeframe'),
                        status='executed'
                    )
                    
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    logger.info(f"Executed grid buy for {symbol}: {trade}")
                    
                elif action == 'close' and self.positions[symbol]['is_active']:
                    # Close grid position
                    position = self.positions[symbol]
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = position['amount']
                    params = signal.get('params', {})
                    
                    # Create market or limit order
                    order_type = self.params.get('order_type', 'limit')
                    price = signal.get('price')
                    
                    result = await exchange.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side=side,
                        amount=amount,
                        price=price if order_type == 'limit' else None,
                        params=params
                    )
                    
                    # Update grid level with order information
                    if result and 'id' in result:
                        timeframe = signal.get('timeframe')
                        for level in self.grid_levels[symbol][timeframe]:
                            if level['order_id'] == position.get('order_id'):
                                level['active'] = False
                                level['exit_time'] = pd.Timestamp.now()
                                level['exit_price'] = price
                    
                    # Reset position status
                    self.positions[symbol] = {
                        'is_active': False,
                        'side': None,
                        'entry_price': None,
                        'amount': None,
                        'entry_time': None,
                        'order_id': None
                    }
                    
                    # Calculate profit/loss
                    exit_price = result.get('price') or price
                    if position['side'] == 'long':
                        pnl = (exit_price - position['entry_price']) * amount
                        pnl_percent = (exit_price / position['entry_price'] - 1) * 100
                    else:
                        pnl = (position['entry_price'] - exit_price) * amount
                        pnl_percent = (position['entry_price'] / exit_price - 1) * 100
                        
                    # Create trade record
                    trade = Trade(
                        id=result.get('id'),
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=exit_price,
                        timestamp=datetime.now(),
                        strategy=self.name,
                        timeframe=signal.get('timeframe'),
                        status='executed',
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        related_trade_id=position.get('order_id')
                    )
                    
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    logger.info(f"Closed grid position for {symbol}: {trade}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
                    
                    # Reinvest profits if enabled
                    if self.params['reinvest_profits'] and pnl > 0:
                        logger.info(f"Reinvesting profits of {pnl:.2f} for {symbol}")
                        # This will be reflected in the next grid rebalance
                    
            except Exception as e:
                logger.error(f"Error executing grid signal for {symbol}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
        return executed_trades
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate grid trading specific performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        # Get base performance stats
        stats = super().get_performance_stats()
        
        # Add grid-specific stats
        grid_stats = {
            'total_grids': 0,
            'active_grids': 0,
            'grid_trades': 0,
            'grid_pnl': 0.0,
            'grid_pnl_percent': 0.0,
            'grid_win_rate': 0.0,
            'grid_info': {}
        }
        
        # Aggregate grid statistics
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                if symbol in self.grid_levels and timeframe in self.grid_levels[symbol]:
                    grid = self.grid_levels[symbol][timeframe]
                    grid_stats['total_grids'] += 1
                    
                    active_levels = sum(1 for level in grid if level['active'])
                    grid_stats['active_grids'] += 1 if active_levels > 0 else 0
                    
                    # Count completed grid trades
                    completed_trades = sum(1 for level in grid if level['exit_time'] is not None)
                    grid_stats['grid_trades'] += completed_trades
                    
                    # Calculate grid PnL
                    grid_pnl = 0.0
                    grid_pnl_percent = 0.0
                    winning_trades = 0
                    
                    for level in grid:
                        if level['entry_price'] is not None and level['exit_price'] is not None:
                            level_pnl = (level['exit_price'] - level['entry_price']) * level['amount']
                            level_pnl_percent = (level['exit_price'] / level['entry_price'] - 1) * 100
                            
                            grid_pnl += level_pnl
                            grid_pnl_percent += level_pnl_percent
                            
                            if level_pnl > 0:
                                winning_trades += 1
                    
                    grid_stats['grid_pnl'] += grid_pnl
                    
                    # Calculate win rate
                    if completed_trades > 0:
                        grid_stats['grid_win_rate'] = (winning_trades / completed_trades) * 100
                        grid_stats['grid_pnl_percent'] = grid_pnl_percent / completed_trades
                    
                    # Store grid info
                    key = f"{symbol}_{timeframe}"
                    grid_info = self.grid_info.get(symbol, {}).get(timeframe, {})
                    grid_stats['grid_info'][key] = {
                        'levels': len(grid),
                        'lower_bound': grid_info.get('lower_bound'),
                        'upper_bound': grid_info.get('upper_bound'),
                        'created_at': grid_info.get('created_at')
                    }
        
        # Merge with base stats
        stats.update({
            'grid_stats': grid_stats
        })
        
        return stats
    
    def reset(self) -> bool:
        """
        Reset the strategy state.
        
        Returns:
            bool: True if reset successfully, False otherwise
        """
        try:
            # Reset base strategy state
            super().reset()
            
            # Reset grid-specific state
            self.grid_levels = {symbol: {tf: [] for tf in self.timeframes} for symbol in self.symbols}
            self.grid_info = {symbol: {tf: {} for tf in self.timeframes} for symbol in self.symbols}
            self.last_rebalance = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            
            logger.info(f"Reset grid strategy {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting grid strategy {self.name}: {str(e)}")
            return False