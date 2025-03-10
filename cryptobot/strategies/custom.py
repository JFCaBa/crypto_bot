"""
Custom Strategy for Cryptocurrency Range Breakout
================================================
Implementation of a range breakout strategy with ATR-based
stop loss, take profit, and trailing stop functionality.
Uses a rolling time window for range calculation, suitable for 24/7 crypto markets.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from loguru import logger
# logger.add(sys.stderr, level="DEBUG")
from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class CustomStrategy(BaseStrategy):
    """
    Custom strategy for cryptocurrency trading that identifies price ranges
    within specific time windows and trades breakouts with ATR-based risk management.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Custom Strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            risk_manager: Risk manager instance
            params: Strategy parameters
        """
        # Default parameters
        default_params = {
            "position_size": 0.02,        # of account per trade
            "range_hours": 4,             # Keep 4 hours (suits volatility)
            "min_range_pct": 1.0,         # Min range (matches hourly moves)
            "max_range_pct": 8.0,         # Max (captures typical daily ranges)
            "max_trades_per_day": 5,      # Limit to 5 trades/day (realistic)
            "min_wait_minutes": 60,       # 1 hour wait (matches timeframe)
            "atr_period": 7,              # Keep 7 (good balance)
            "atr_multiplier": 1.0,        # SL = 1.5 * ATR (tighter)
            "rr_multiplier": 1.5,         # TP = 2 * SL (keep 2:1)
            "stop_loss_pct": 0.0015,       # SL if no ATR
            "take_profit_pct": 0.01,      # TP if no ATR
            "trailing_stop_pct": 0.01,    # trailing stop
            "recalculate_after_trade": True
        }
        
        # Initialize base strategy first
        super().__init__(
            name="CustomStrategy",
            symbols=symbols,
            timeframes=timeframes,
            params=params if params else default_params,
            risk_manager=risk_manager
        )
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        self.params = default_params
        
        # Initialize range variables
        self.high_range = {symbol: {tf: 0.0 for tf in timeframes} for symbol in symbols}
        self.low_range = {symbol: {tf: 0.0 for tf in timeframes} for symbol in symbols}
        
        # Initialize state tracking
        self.price_returned_to_range = {symbol: {tf: True for tf in timeframes} for symbol in symbols}
        self.daily_trades = {symbol: {tf: 0 for tf in timeframes} for symbol in symbols}
        self.last_trade_time = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.active_trades = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.stop_loss_levels = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.take_profit_levels = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.range_defined = {symbol: {tf: False for tf in timeframes} for symbol in symbols}
        
        # Track current day for resetting daily counters
        self.current_day = datetime.utcnow().date()
        
        # Track the earliest time in the dataset for each symbol/timeframe
        self.data_start_time = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        
        # Track the last time range was calculated for each symbol/timeframe
        self.last_range_calc_time = {symbol: {tf: None for tf in timeframes} for symbol in symbols}

        logger.info(f"CustomStrategy initialized with {len(symbols)} symbols and {len(timeframes)} timeframes")
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range for a dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.Series: ATR values
        """
        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        
        # Get true range by taking the maximum of the three
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR as rolling average of true range
        atr = df['tr'].rolling(window=self.params['atr_period']).mean()
        
        return atr
    
    def reset_daily_counters(self):
        """Reset daily trade counters if a new day has started."""
        current_date = datetime.utcnow().date()
        if current_date != self.current_day:
            logger.info(f"New day detected, resetting daily trade counters")
            self.current_day = current_date
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    self.daily_trades[symbol][timeframe] = 0
    
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        try:
            df = self.data[symbol][timeframe]
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return False
            
            self.reset_daily_counters()
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            if self.data_start_time[symbol][timeframe] is None:
                self.data_start_time[symbol][timeframe] = df.index[0]
            
            latest_time = df.index[-1]
            range_hours = self.params['range_hours']
            min_start_time = self.data_start_time[symbol][timeframe] + timedelta(hours=range_hours)
            
            if latest_time < min_start_time:
                logger.debug(f"Waiting for {range_hours}h data: {latest_time} < {min_start_time}")
                return False
            
            last_calc_time = self.last_range_calc_time[symbol][timeframe]
            if last_calc_time is None or (latest_time >= last_calc_time + timedelta(hours=range_hours)):
                start_time = latest_time - timedelta(hours=range_hours)
                df_filtered = df.loc[(df.index >= start_time) & (df.index <= latest_time)]
                if df_filtered.empty or len(df_filtered) < 3:
                    logger.warning(f"Insufficient data for range calc: {symbol} {timeframe}")
                    return False
                
                high = df_filtered['high'].max()
                low = df_filtered['low'].min()
                range_percent = ((high - low) / low) * 100 if low > 0 else 0
                
                if not (self.params['min_range_pct'] <= range_percent <= self.params['max_range_pct']):
                    # logger.debug(f"Range {range_percent:.2f}% outside bounds")
                    return False
                
                self.high_range[symbol][timeframe] = high
                self.low_range[symbol][timeframe] = low
                self.range_defined[symbol][timeframe] = True
                self.last_range_calc_time[symbol][timeframe] = latest_time
                logger.info(f"Updated range for {symbol} {timeframe}: High={high:.8f}, Low={low:.8f}, Range%={range_percent:.2f}%")
            
            atr_series = self.calculate_atr(df)
            if not atr_series.isnull().all():
                self.indicators[symbol][timeframe]['atr'] = atr_series.iloc[-1]
            
            return True
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {str(e)}")
            return False
    
    def can_trade(self, symbol: str, timeframe: str) -> bool:
        if not self.range_defined[symbol][timeframe]:
            logger.debug(f"Range not defined for {symbol} {timeframe}")
            return False
            
        if self.daily_trades[symbol][timeframe] >= self.params['max_trades_per_day']:
            logger.debug(f"Daily trade limit reached for {symbol} {timeframe}")
            return False
            
        # if self.last_trade_time[symbol][timeframe] is not None:
        #     df = self.data[symbol][timeframe]
        #     if df.empty:
        #         logger.debug(f"No data available in can_trade for {symbol} {timeframe}")
        #         return False
        #     time_since_last = (df.index[-1] - self.last_trade_time[symbol][timeframe]).total_seconds() / 60
        #     logger.debug(f"Time since last trade: {time_since_last:.1f} min")
        #     if time_since_last < self.params['min_wait_minutes']:
        #         logger.debug(f"Min wait time not elapsed for {symbol} {timeframe}: {time_since_last:.1f} min < {self.params['min_wait_minutes']} min")
        #         return False
        
        # Check if price has returned to range
        if not self.price_returned_to_range[symbol][timeframe]:
            logger.debug(f"Price has not returned to range for {symbol} {timeframe}")
            return False
                
        # All checks passed
        return True
    
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals based on breakout strategy.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            dict: Signal information including action, price, etc.
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
            # Get data from the strategy's data storage
            df = self.data[symbol][timeframe]
            current_time = df.index[-1]
            
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe} to generate signals.")
                return signal
            
            # Check if range is defined
            if not self.range_defined[symbol][timeframe]:
                # Try to calculate indicators if not already done
                if not self.calculate_indicators(symbol, timeframe):
                    # If we can't calculate indicators, return empty signal
                    return signal
            
            # Get current price
            current_price = df.iloc[-1]['close']
            
            # Get range levels
            high = self.high_range.get(symbol, {}).get(timeframe, None)
            low = self.low_range.get(symbol, {}).get(timeframe, None)
            
            if high is None or low is None:
                return signal
            
            # Check if price is inside range (for tracking price return to range)
            if low <= current_price <= high:
                self.price_returned_to_range[symbol][timeframe] = True
                logger.debug(f"{symbol} {timeframe}: Price returned to range at {current_price:.8f}")
            
            # No signal if we can't trade
            if not self.can_trade(symbol, timeframe):
                return signal
            
            # Check if position is active
            position = self.positions[symbol]
            is_active = position['is_active']

            # if is_active:
            #     logger.debug(f"{symbol} {timeframe}: Position active, side={position['side']}")
            
            if not is_active:
                # Generate breakout signals
                if current_price > high:
                    signal['action'] = 'buy'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(symbol, current_price)
                    
                    # Calculate and add stop loss and take profit
                    stop_loss, take_profit = self._calculate_sl_tp(df, symbol, timeframe, "buy", current_price)
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    logger.debug(f"BUY at {current_price:.8f}, SL={stop_loss:.8f}, TP={take_profit:.8f}")
                    
                    # Add trailing stop if enabled
                    if self.params['trailing_stop_pct'] > 0:
                        signal['params']['trailing_stop'] = True
                        signal['params']['trailing_stop_percent'] = self.params['trailing_stop_pct']
                    
                    # Update tracking variables
                    self.price_returned_to_range[symbol][timeframe] = False
                    self.daily_trades[symbol][timeframe] += 1
                    self.last_trade_time[symbol][timeframe] = current_time
                    self.stop_loss_levels[symbol][timeframe] = stop_loss
                    self.take_profit_levels[symbol][timeframe] = take_profit
                    
                    logger.info(f"{symbol} {timeframe}: BUY signal triggered at {current_price:.8f}")
                    
                elif current_price < low:
                    signal['action'] = 'sell'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(symbol, current_price)
                    
                    # Calculate and add stop loss and take profit
                    stop_loss, take_profit = self._calculate_sl_tp(df, symbol, timeframe, "sell", current_price)
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    
                    # Add trailing stop if enabled
                    if self.params['trailing_stop_pct'] > 0:
                        signal['params']['trailing_stop'] = True
                        signal['params']['trailing_stop_percent'] = self.params['trailing_stop_pct']
                    
                    # Update tracking variables
                    self.price_returned_to_range[symbol][timeframe] = False
                    self.daily_trades[symbol][timeframe] += 1
                    self.last_trade_time[symbol][timeframe] = current_time
                    self.stop_loss_levels[symbol][timeframe] = stop_loss
                    self.take_profit_levels[symbol][timeframe] = take_profit
                    
                    logger.info(f"{symbol} {timeframe}: SELL signal triggered at {current_price:.8f}")
            
            else:
                # Check for exit signals
                if position['side'] == 'long':
                    # Exit long position if price falls below stop loss
                    stop_loss = self.stop_loss_levels.get(symbol, {}).get(timeframe)
                    take_profit = self.take_profit_levels.get(symbol, {}).get(timeframe)
                    
                    if stop_loss and current_price <= stop_loss:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                        logger.info(f"{symbol} {timeframe}: Stop loss hit for LONG position at {current_price:.8f}")
                    
                    elif take_profit and current_price >= take_profit:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                        logger.info(f"{symbol} {timeframe}: Take profit hit for LONG position at {current_price:.8f}")
                
                elif position['side'] == 'short':
                    # Exit short position if price rises above stop loss
                    stop_loss = self.stop_loss_levels.get(symbol, {}).get(timeframe)
                    take_profit = self.take_profit_levels.get(symbol, {}).get(timeframe)
                    
                    if stop_loss and current_price >= stop_loss:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                        logger.info(f"{symbol} {timeframe}: Stop loss hit for SHORT position at {current_price:.8f}")
                    
                    elif take_profit and current_price <= take_profit:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                        logger.info(f"{symbol} {timeframe}: Take profit hit for SHORT position at {current_price:.8f}")
            
            # If a position is being closed, update state
            if signal['action'] == 'close':
                self.stop_loss_levels[symbol][timeframe] = None
                self.take_profit_levels[symbol][timeframe] = None
                self.last_trade_time[symbol][timeframe] = current_time
                
                # Recalculate range if configured
                if self.params.get('recalculate_after_trade', True):
                    self.range_defined[symbol][timeframe] = False
                    self.last_range_calc_time[symbol][timeframe] = None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return signal
    
    def _calculate_sl_tp(self, df: pd.DataFrame, symbol: str, timeframe: str, side: str, price: float) -> tuple:
        """
        Calculate stop loss and take profit levels based on ATR or percentage.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe
            side: Trade side ("buy" or "sell")
            price: Entry price
            
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        # Get the current ATR value if available
        atr = self.indicators.get(symbol, {}).get(timeframe, {}).get('atr')
        
        if atr is not None and not np.isnan(atr):
            # ATR-based stop loss and take profit
            if side == "buy":
                stop_loss = price - (atr * self.params['atr_multiplier'])
                take_profit = price + (atr * self.params['atr_multiplier'] * self.params['rr_multiplier'])
            else:  # side == "sell"
                stop_loss = price + (atr * self.params['atr_multiplier'])
                take_profit = price - (atr * self.params['atr_multiplier'] * self.params['rr_multiplier'])
        else:
            # Percentage-based stop loss and take profit
            stop_loss_pct = self.params['stop_loss_pct'] / 100
            take_profit_pct = self.params['take_profit_pct'] / 100
            
            if side == "buy":
                stop_loss = price * (1 - stop_loss_pct)
                take_profit = price * (1 + take_profit_pct)
            else:  # side == "sell"
                stop_loss = price * (1 + stop_loss_pct)
                take_profit = price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Get account balance from risk manager if available
            account_balance = 10000.0  # Default fallback value
            if self.risk_manager:
                account_balance = self.risk_manager.account_size
            
            # Simple percentage-based position sizing
            position_size = account_balance * self.params['position_size'] / price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default fallback
    
    def update_trailing_stops(self, symbol: str, timeframe: str, current_price: float):
        """
        Update trailing stop loss for active positions.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            current_price: Current price
        """
        try:
            position = self.positions[symbol]
            if not position['is_active']:
                return
            
            stop_loss = self.stop_loss_levels.get(symbol, {}).get(timeframe)
            if not stop_loss:
                return
            
            # Get ATR if available
            atr = self.indicators.get(symbol, {}).get(timeframe, {}).get('atr')
            
            # Update trailing stop for long positions
            if position['side'] == 'long':
                # Only trail if price has moved in our favor
                if current_price > position['entry_price']:
                    if atr and not np.isnan(atr):
                        # ATR-based trailing stop
                        new_stop = current_price - (atr * self.params['atr_multiplier'])
                    else:
                        # Percentage-based trailing stop
                        trailing_pct = self.params['trailing_stop_pct'] / 100
                        new_stop = current_price * (1 - trailing_pct)
                    
                    # Only update if the new stop is higher than current stop
                    if new_stop > stop_loss:
                        self.stop_loss_levels[symbol][timeframe] = new_stop
                        logger.debug(f"{symbol} {timeframe}: Updated trailing stop to {new_stop:.8f}")
            
            # Update trailing stop for short positions
            elif position['side'] == 'short':
                # Only trail if price has moved in our favor
                if current_price < position['entry_price']:
                    if atr and not np.isnan(atr):
                        # ATR-based trailing stop
                        new_stop = current_price + (atr * self.params['atr_multiplier'])
                    else:
                        # Percentage-based trailing stop
                        trailing_pct = self.params['trailing_stop_pct'] / 100
                        new_stop = current_price * (1 + trailing_pct)
                    
                    # Only update if the new stop is lower than current stop
                    if new_stop < stop_loss:
                        self.stop_loss_levels[symbol][timeframe] = new_stop
                        logger.debug(f"{symbol} {timeframe}: Updated trailing stop to {new_stop:.8f}")
                        
        except Exception as e:
            logger.error(f"Error updating trailing stops for {symbol} {timeframe}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary for status reporting.
        
        Returns:
            dict: Strategy information
        """
        base_dict = super().to_dict()
        
        # Add custom strategy information
        active_positions = sum(1 for symbol in self.symbols 
                             for pos in self.positions.values() 
                             if pos.get('is_active', False))
        
        custom_info = {
            'active_positions': active_positions,
            'daily_trades': {symbol: {tf: count for tf, count in timeframe_data.items()} 
                            for symbol, timeframe_data in self.daily_trades.items()},
            'ranges': {symbol: {tf: {'high': self.high_range.get(symbol, {}).get(tf),
                                    'low': self.low_range.get(symbol, {}).get(tf)}
                              for tf in self.timeframes if self.range_defined.get(symbol, {}).get(tf, False)}
                      for symbol in self.symbols}
        }
        
        # Update the base dictionary with custom information
        base_dict.update(custom_info)
        
        return base_dict