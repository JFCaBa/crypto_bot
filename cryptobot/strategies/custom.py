"""
Custom Strategy
===============
Implementation of a breakout trading strategy based on a defined time range.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class CustomStrategy(BaseStrategy):
    """
    Custom Strategy for breakout trading.
    
    This strategy identifies a price range within a specified time window and trades breakouts,
    with configurable SL, TP, and TS. It calculates daily ranges for each day in the data period
    and generates signals based on price movements relative to these ranges.
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
        default_params = {
            "lot_size": 0.1,
            "risk_percentage": 1.0,
            "start_hour": 8,
            "start_minute": 0,
            "end_hour": 15,
            "end_minute": 0,
            "min_pips_range": 50,
            "max_pips_range": 5000,
            "no_open_hours": 3,
            "trade_london": True,
            "trade_ny": True,
            "confirmation_candles": 1,
            "max_trades_per_day": 3,
            "min_wait_hours": 1,
            "use_sl": True,
            "use_tp": True,
            "use_ts": True,
            "stop_loss_pips": 50,
            "take_profit_pips": 100,
            "trailing_stop_pips": 20
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(
            name="CustomStrategy",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        
        # Strategy-specific attributes
        self.high_range = {}
        self.low_range = {}
        self.range_defined = {}
        self.start_time = {}
        self.end_time = {}
        self.last_trade_time = {}
        self.daily_trades = {}
        self.operation_open = {}
        self.price_outside_range = {}
        self.last_outside_time = {}
        self.last_inside_time = {}
        self.last_closed_time = {}
        
        # Initialize for each symbol and timeframe
        for symbol in self.symbols:
            self.high_range[symbol] = {tf: {} for tf in self.timeframes}
            self.low_range[symbol] = {tf: {} for tf in self.timeframes}
            self.range_defined[symbol] = {tf: {} for tf in self.timeframes}
            self.start_time[symbol] = {tf: {} for tf in self.timeframes}
            self.end_time[symbol] = {tf: {} for tf in self.timeframes}
            self.last_trade_time[symbol] = {tf: None for tf in self.timeframes}
            self.daily_trades[symbol] = {tf: 0 for tf in self.timeframes}
            self.operation_open[symbol] = {tf: False for tf in self.timeframes}
            self.price_outside_range[symbol] = {tf: False for tf in self.timeframes}
            self.last_outside_time[symbol] = {tf: None for tf in self.timeframes}
            self.last_inside_time[symbol] = {tf: None for tf in self.timeframes}
            self.last_closed_time[symbol] = {tf: None for tf in self.timeframes}
    
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate the price range (high/low) for each day’s specified time window.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if calculated successfully for at least one day, False otherwise
        """
        try:
            df = self.data[symbol][timeframe]
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return False
            
            # Log total data fetched
            logger.debug(f"Fetched {len(df)} total candles for {symbol} {timeframe} from {df.index.min()} to {df.index.max()}")
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns in data for {symbol} {timeframe}")
                return False
            
            # Make df.index timezone-aware (UTC) if it's naive
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Group data by date to define daily ranges
            df['date'] = df.index.date
            unique_dates = sorted(df['date'].unique())
            logger.debug(f"Found {len(unique_dates)} unique dates in data for {symbol} {timeframe}: {unique_dates}")
            
            start_hour = self.params['start_hour']
            start_minute = self.params['start_minute']
            end_hour = self.params['end_hour']
            end_minute = self.params['end_minute']
            
            ranges_defined = 0
            for date in unique_dates:
                # Define start and end times for this date
                date_dt = pd.Timestamp(date).tz_localize('UTC')
                start_time = date_dt + pd.Timedelta(hours=start_hour, minutes=start_minute)
                end_time = date_dt + pd.Timedelta(hours=end_hour, minutes=end_minute)
                
                # Filter data within the time window for this date
                range_data = df[(df.index >= start_time) & (df.index <= end_time)]
                if len(range_data) < 1:
                    logger.debug(f"Insufficient data ({len(range_data)} candles) within time window {start_time} to {end_time} for {symbol} {timeframe} on {date}")
                    continue
                
                # Calculate high and low of the range
                high = range_data['high'].max()
                low = range_data['low'].min()
                
                if pd.isna(high) or pd.isna(low):
                    logger.debug(f"Could not calculate range for {symbol} {timeframe} on {date}: high={high}, low={low}")
                    continue
                
                # Store the range for this date
                date_key = date
                self.high_range[symbol][timeframe][date_key] = high
                self.low_range[symbol][timeframe][date_key] = low
                self.range_defined[symbol][timeframe][date_key] = True
                self.start_time[symbol][timeframe][date_key] = start_time
                self.end_time[symbol][timeframe][date_key] = end_time
                
                # Log range details for debugging
                pip_value = 0.0001
                range_pips = (high - low) / pip_value
                logger.debug(f"Range calculated for {symbol} {timeframe} on {date}: High={high}, Low={low}, Range_pips={range_pips}, Candles={len(range_data)}")
                ranges_defined += 1
            
            if ranges_defined == 0:
                logger.warning(f"No ranges could be calculated for {symbol} {timeframe}")
                return False
            
            logger.info(f"Calculated ranges for {ranges_defined} days for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            return False
    
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals based on breakout and confirmation.
        
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
            'timestamp': pd.Timestamp.now(tz='UTC')
        }
        
        try:
            df = self.data[symbol][timeframe]
            if df.empty or len(df) < 2:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return signal
            
            # Get latest candle and its timestamp
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            current_time = pd.Timestamp(latest.name)  # Use existing tzinfo
            if current_time.tzinfo is not None:
                current_time = current_time.tz_convert('UTC')
            else:
                current_time = current_time.tz_localize('UTC')
            current_price = latest['close']
            previous_price = previous['close']
            
            # Determine the date key for today's range
            date_key = current_time.date()
            
            # Check if a range is defined for this date
            if not self.range_defined.get(symbol, {}).get(timeframe, {}).get(date_key, False):
                logger.debug(f"No range defined for {symbol} {timeframe} on {date_key}")
                return signal
            
            # Get range values
            high_range = self.high_range[symbol][timeframe].get(date_key)
            low_range = self.low_range[symbol][timeframe].get(date_key)
            start_time = self.start_time[symbol][timeframe].get(date_key)
            end_time = self.end_time[symbol][timeframe].get(date_key)
            
            # Validate range against min/max pips constraints
            pip_value = 0.0001  # Adjust based on symbol
            range_pips = (high_range - low_range) / pip_value
            if range_pips < self.params['min_pips_range'] or range_pips > self.params['max_pips_range']:
                logger.debug(f"Range {range_pips} pips for {symbol} {timeframe} on {date_key} outside allowed bounds ({self.params['min_pips_range']}-{self.params['max_pips_range']})")
                return signal
            
            # Check if we're within the trading window
            if current_time < start_time or current_time > end_time:
                logger.debug(f"Outside trading window for {symbol} {timeframe}")
                return signal
            
            # Check session permissions (London/NY)
            current_hour = current_time.hour
            london_session = 8 <= current_hour < 16
            ny_session = 13 <= current_hour < 21
            if not ((self.params['trade_london'] and london_session) or (self.params['trade_ny'] and ny_session)):
                logger.debug(f"Not allowed to trade in current session for {symbol} {timeframe}")
                return signal
            
            # Check daily trade limit
            today = current_time.date()
            if self.last_trade_time[symbol][timeframe] and self.last_trade_time[symbol][timeframe].date() != today:
                self.daily_trades[symbol][timeframe] = 0  # Reset daily counter
            if self.daily_trades[symbol][timeframe] >= self.params['max_trades_per_day']:
                logger.debug(f"Max daily trades ({self.params['max_trades_per_day']}) reached for {symbol} {timeframe}")
                return signal
            
            # Check minimum time between trades
            if self.last_trade_time[symbol][timeframe]:
                time_since_last_trade = (current_time - self.last_trade_time[symbol][timeframe]).total_seconds() / 3600
                if time_since_last_trade < self.params['min_wait_hours']:
                    logger.debug(f"Too soon since last trade for {symbol} {timeframe} ({time_since_last_trade:.2f} hours since last)")
                    return signal
            
            # Check time since last operation closed
            if self.last_closed_time[symbol][timeframe]:
                time_since_closed = (current_time - self.last_closed_time[symbol][timeframe]).total_seconds() / 3600
                if time_since_closed < self.params['no_open_hours']:
                    logger.debug(f"Too soon since last close for {symbol} {timeframe} ({time_since_closed:.2f} hours since close)")
                    return signal
            
            # Detect breakout (price moves outside the range)
            logger.debug(f"Processing signal for {symbol} {timeframe}: current_price={current_price}, high_range={high_range}, low_range={low_range}")
            if current_price > high_range:
                logger.debug("Potential bullish breakout detected")
                # Bullish breakout
                self.price_outside_range[symbol][timeframe] = True
                self.last_outside_time[symbol][timeframe] = current_time
                
                # Confirmation: Check subsequent candles (simplified to previous candle for now)
                if self.params['confirmation_candles'] <= 1 or previous_price > high_range:
                    signal['action'] = 'buy'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(symbol, current_price)
                    if self.params['use_sl']:
                        signal['stop_loss'] = current_price - (self.params['stop_loss_pips'] * 0.0001)
                    if self.params['use_tp']:
                        signal['take_profit'] = current_price + (self.params['take_profit_pips'] * 0.0001)
                    if self.params['use_ts']:
                        signal['params']['trailing_stop'] = self.params['trailing_stop_pips'] * 0.0001
                    
                    self.daily_trades[symbol][timeframe] += 1
                    self.last_trade_time[symbol][timeframe] = current_time
                    self.operation_open[symbol][timeframe] = True
            
            elif current_price < low_range:
                logger.debug("Potential bearish breakout detected")
                # Bearish breakout
                self.price_outside_range[symbol][timeframe] = True
                self.last_outside_time[symbol][timeframe] = current_time
                
                if self.params['confirmation_candles'] <= 1 or previous_price < low_range:
                    signal['action'] = 'sell'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(symbol, current_price)
                    if self.params['use_sl']:
                        signal['stop_loss'] = current_price + (self.params['stop_loss_pips'] * 0.0001)
                    if self.params['use_tp']:
                        signal['take_profit'] = current_price - (self.params['take_profit_pips'] * 0.0001)
                    if self.params['use_ts']:
                        signal['params']['trailing_stop'] = self.params['trailing_stop_pips'] * 0.0001
                    
                    self.daily_trades[symbol][timeframe] += 1
                    self.last_trade_time[symbol][timeframe] = current_time
                    self.operation_open[symbol][timeframe] = True
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            return signal
    
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
            lot_size = self.params['lot_size']
            if self.risk_manager and self.params['risk_percentage'] > 0:
                account_balance = self.risk_manager.account_size if self.risk_manager else 10000.0
                risk_amount = account_balance * (self.params['risk_percentage'] / 100.0)
                pip_value = 0.0001
                risk_per_pip = price * pip_value
                if risk_per_pip > 0:
                    position_size = risk_amount / (self.params['stop_loss_pips'] * risk_per_pip)
                    return max(position_size, lot_size)
            return lot_size
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return self.params['lot_size']