"""
Breakout Trading Strategy
========================
Implementation of a breakout trading strategy that identifies when price
breaks out of a consolidation pattern and enters a new trend.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager
from cryptobot.core.trade import Trade


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Trading Strategy.
    
    This strategy identifies consolidation patterns (like ranges, triangles, or channels)
    and generates signals when price breaks out, indicating the start of a potential new trend.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Breakout strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            risk_manager: Risk manager instance
            params: Strategy parameters
        """
        # Initialize default parameters
        default_params = {
            'lookback_period': 20,       # Period to identify consolidation (candles)
            'breakout_type': 'range',    # Type of breakout: 'range', 'channel', 'triangle'
            'min_consolidation': 5,      # Minimum number of candles in consolidation
            'volatility_factor': 1.5,    # Factor to determine significant breakout (ATR multiplier)
            'volume_confirm': True,      # Require volume confirmation for breakout
            'volume_factor': 1.5,        # Volume increase factor for confirmation
            'max_trade_duration': 48,    # Maximum trade duration in hours
            'stop_loss_atr': 2.0,        # Stop loss in ATR multiples
            'take_profit_atr': 4.0,      # Take profit in ATR multiples
            'trailing_stop': 2.0,        # Trailing stop in ATR multiples (0 to disable)
            'risk_reward_ratio': 2.0,    # Minimum risk/reward ratio to enter trade
            'max_positions': 3,          # Maximum concurrent positions
            'risk_per_trade': 1.0,       # Risk per trade (percentage of account)
            'consolidation_atr': 0.5,    # Maximum price movement (in ATR) for consolidation
            'allow_inside_bars': True,   # Allow inside bars as consolidation
            'require_close_beyond': True, # Require close price beyond the range for breakout
            'min_range_atr': 1.0,        # Minimum range size in ATR multiples
            'false_breakout_filter': True, # Enable false breakout filtering
            'retest_entry': False,       # Wait for retest of breakout level to enter
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="BreakoutStrategy",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        
        # Strategy-specific attributes
        self.consolidation_patterns = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.atr_values = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.breakout_levels = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.pattern_start_times = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.traded_breakouts = {symbol: {tf: [] for tf in timeframes} for symbol in symbols}
        self.active_trades = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate technical indicators for breakout detection.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if calculated successfully, False otherwise
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
                
            # Check if we have enough data
            if len(df) < self.params['lookback_period'] + 10:
                logger.warning(f"Not enough data for {symbol} {timeframe}, need at least {self.params['lookback_period'] + 10}")
                return False
                
            # Calculate ATR for volatility measurement
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            # Store ATR value
            self.atr_values[symbol][timeframe] = df['atr'].iloc[-1] if not df['atr'].empty else None
            
            # Calculate volume metrics
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Identify potential consolidation patterns
            self._identify_consolidation_pattern(symbol, timeframe, df)
            
            # Store calculated indicators
            indicators = {
                'atr': df['atr'],
                'volume_sma': df['volume_sma'],
                'volume_ratio': df['volume_ratio']
            }
            
            # Save indicators to instance
            self.indicators[symbol][timeframe] = indicators
            
            # Update data with calculated indicators
            self.data[symbol][timeframe] = df
            
            return True
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _identify_consolidation_pattern(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Identify consolidation patterns in the price data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            df: DataFrame with price data
        """
        try:
            # Get parameters
            lookback_period = self.params['lookback_period']
            breakout_type = self.params['breakout_type']
            min_consolidation = self.params['min_consolidation']
            consolidation_atr = self.params['consolidation_atr']
            min_range_atr = self.params['min_range_atr']
            
            # Get recent data
            recent_data = df.iloc[-lookback_period:]
            
            # Check if ATR is available
            if 'atr' not in recent_data.columns or recent_data['atr'].isna().all():
                return
                
            current_atr = recent_data['atr'].iloc[-1]
            
            # Different pattern identification based on breakout type
            if breakout_type == 'range':
                # Identify price range consolidation
                high = recent_data['high'].max()
                low = recent_data['low'].min()
                range_size = high - low
                
                # Check if range is significant enough
                if range_size < current_atr * min_range_atr:
                    # Range too small
                    self.consolidation_patterns[symbol][timeframe] = None
                    self.breakout_levels[symbol][timeframe] = None
                    return
                
                # Check if price has been consolidating (limited volatility)
                recent_movement = recent_data['high'].max() - recent_data['low'].min()
                if recent_movement > current_atr * lookback_period * consolidation_atr:
                    # Too volatile for consolidation
                    self.consolidation_patterns[symbol][timeframe] = None
                    self.breakout_levels[symbol][timeframe] = None
                    return
                
                # Find how many candles have been in this range
                candles_in_range = 0
                for i in range(len(recent_data)-1, -1, -1):
                    candle = recent_data.iloc[i]
                    if candle['high'] <= high and candle['low'] >= low:
                        candles_in_range += 1
                    else:
                        break
                
                if candles_in_range >= min_consolidation:
                    # We have a valid range consolidation
                    self.consolidation_patterns[symbol][timeframe] = {
                        'type': 'range',
                        'high': high,
                        'low': low,
                        'candles': candles_in_range,
                        'timestamp': recent_data.index[-1]
                    }
                    
                    # Set breakout levels
                    self.breakout_levels[symbol][timeframe] = {
                        'upper': high,
                        'lower': low
                    }
                    self.pattern_start_times[symbol][timeframe] = recent_data.index[-candles_in_range]
                else:
                    self.consolidation_patterns[symbol][timeframe] = None
                    self.breakout_levels[symbol][timeframe] = None
                
            elif breakout_type == 'channel':
                # Identify channel patterns (parallel support and resistance)
                # This is a simplified approach - in reality, would use linear regression
                
                # Upper channel line (connect highs)
                highs = recent_data['high'].values
                # Lower channel line (connect lows)
                lows = recent_data['low'].values
                
                # Calculate slope of the channel
                x = np.arange(len(highs))
                slope_high, intercept_high = np.polyfit(x, highs, 1)
                slope_low, intercept_low = np.polyfit(x, lows, 1)
                
                # Check if slopes are roughly parallel
                if abs(slope_high - slope_low) / (abs(slope_high) + 1e-10) > 0.3:
                    # Not parallel enough for a channel
                    self.consolidation_patterns[symbol][timeframe] = None
                    self.breakout_levels[symbol][timeframe] = None
                    return
                
                # Project channel lines to current candle
                current_x = len(highs) - 1
                upper_level = slope_high * current_x + intercept_high
                lower_level = slope_low * current_x + intercept_low
                
                # Set channel breakout levels
                self.consolidation_patterns[symbol][timeframe] = {
                    'type': 'channel',
                    'slope': slope_high,
                    'upper_intercept': intercept_high,
                    'lower_intercept': intercept_low,
                    'candles': len(highs),
                    'timestamp': recent_data.index[-1]
                }
                
                self.breakout_levels[symbol][timeframe] = {
                    'upper': upper_level,
                    'lower': lower_level
                }
                self.pattern_start_times[symbol][timeframe] = recent_data.index[0]
                
            elif breakout_type == 'triangle':
                # Identify triangle patterns (converging support and resistance)
                highs = recent_data['high'].values
                lows = recent_data['low'].values
                
                x = np.arange(len(highs))
                slope_high, intercept_high = np.polyfit(x, highs, 1)
                slope_low, intercept_low = np.polyfit(x, lows, 1)
                
                # Triangles have converging slopes (one positive, one negative)
                if slope_high > 0 and slope_low < 0:
                    # Not converging as expected
                    self.consolidation_patterns[symbol][timeframe] = None
                    self.breakout_levels[symbol][timeframe] = None
                    return
                
                # Project triangle lines to current candle
                current_x = len(highs) - 1
                upper_level = slope_high * current_x + intercept_high
                lower_level = slope_low * current_x + intercept_low
                
                # Set triangle breakout levels
                self.consolidation_patterns[symbol][timeframe] = {
                    'type': 'triangle',
                    'slope_high': slope_high,
                    'slope_low': slope_low,
                    'upper_intercept': intercept_high,
                    'lower_intercept': intercept_low,
                    'candles': len(highs),
                    'timestamp': recent_data.index[-1]
                }
                
                self.breakout_levels[symbol][timeframe] = {
                    'upper': upper_level,
                    'lower': lower_level
                }
                self.pattern_start_times[symbol][timeframe] = recent_data.index[0]
            
            else:
                logger.warning(f"Unknown breakout type: {breakout_type}")
                self.consolidation_patterns[symbol][timeframe] = None
                self.breakout_levels[symbol][timeframe] = None
                
        except Exception as e:
            logger.error(f"Error identifying consolidation pattern for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.consolidation_patterns[symbol][timeframe] = None
            self.breakout_levels[symbol][timeframe] = None
            
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals based on breakout patterns.
        
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
            # Calculate indicators
            if not self.calculate_indicators(symbol, timeframe):
                return signal
                
            df = self.data[symbol][timeframe]
            if df.empty:
                return signal
                
            # Get current price and volume data
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            current_price = latest['close']
            current_volume = latest['volume']
            volume_ratio = latest['volume_ratio'] if 'volume_ratio' in latest else 1.0
            
            # Get current ATR
            current_atr = self.atr_values[symbol][timeframe]
            if current_atr is None:
                return signal
                
            # Check if we have identified a consolidation pattern
            pattern = self.consolidation_patterns[symbol][timeframe]
            breakout_levels = self.breakout_levels[symbol][timeframe]
            
            if pattern is None or breakout_levels is None:
                return signal
                
            # Check if this pattern has already been traded
            pattern_time = pattern['timestamp']
            if pattern_time in self.traded_breakouts[symbol][timeframe]:
                return signal
                
            # Check if position is already active
            position = self.positions[symbol]
            is_active = position['is_active']
            
            if is_active:
                # Check for exit signals
                if self.active_trades[symbol][timeframe] is not None:
                    trade_info = self.active_trades[symbol][timeframe]
                    
                    # Calculate price distance from entry
                    entry_price = position['entry_price']
                    price_distance = abs(current_price - entry_price) / entry_price * 100
                    
                    # Check for max trade duration
                    entry_time = position['entry_time']
                    if entry_time is not None:
                        hours_in_trade = (datetime.now() - entry_time).total_seconds() / 3600
                        if hours_in_trade > self.params['max_trade_duration']:
                            # Close due to time limit
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                            logger.info(f"Closing breakout trade for {symbol} due to time limit: {hours_in_trade:.1f} hours")
                            
                            # Clean up
                            self.active_trades[symbol][timeframe] = None
                            self.traded_breakouts[symbol][timeframe].append(pattern_time)
                            return signal
                    
                    # Check for trailing stop if enabled
                    if self.params['trailing_stop'] > 0 and trade_info.get('trailing_stop') is not None:
                        trailing_stop = trade_info['trailing_stop']
                        
                        if position['side'] == 'long' and current_price < trailing_stop:
                            # Trigger trailing stop for long
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                            logger.info(f"Trailing stop triggered for long {symbol} at {current_price}")
                            
                            # Clean up
                            self.active_trades[symbol][timeframe] = None
                            self.traded_breakouts[symbol][timeframe].append(pattern_time)
                            return signal
                            
                        elif position['side'] == 'short' and current_price > trailing_stop:
                            # Trigger trailing stop for short
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                            logger.info(f"Trailing stop triggered for short {symbol} at {current_price}")
                            
                            # Clean up
                            self.active_trades[symbol][timeframe] = None
                            self.traded_breakouts[symbol][timeframe].append(pattern_time)
                            return signal
                            
                        # Update trailing stop if price moves favorably
                        if position['side'] == 'long' and current_price > trade_info.get('highest_price', entry_price):
                            # Update highest price and trailing stop for long
                            new_trailing_stop = current_price * (1 - self.params['trailing_stop'] * current_atr / current_price)
                            if new_trailing_stop > trade_info.get('trailing_stop', 0):
                                trade_info['trailing_stop'] = new_trailing_stop
                                trade_info['highest_price'] = current_price
                                logger.debug(f"Updated trailing stop for {symbol} to {new_trailing_stop}")
                                
                        elif position['side'] == 'short' and current_price < trade_info.get('lowest_price', entry_price):
                            # Update lowest price and trailing stop for short
                            new_trailing_stop = current_price * (1 + self.params['trailing_stop'] * current_atr / current_price)
                            if new_trailing_stop < trade_info.get('trailing_stop', float('inf')):
                                trade_info['trailing_stop'] = new_trailing_stop
                                trade_info['lowest_price'] = current_price
                                logger.debug(f"Updated trailing stop for {symbol} to {new_trailing_stop}")
                
                return signal
            
            # Check for breakout signals
            volatility_factor = self.params['volatility_factor']
            volume_confirm = self.params['volume_confirm']
            volume_factor = self.params['volume_factor']
            require_close_beyond = self.params['require_close_beyond']
            false_breakout_filter = self.params['false_breakout_filter']
            
            # Check for bullish breakout (price breaks above upper level)
            bullish_breakout = False
            if latest['high'] > breakout_levels['upper']:
                # Price has broken above the upper level
                
                # Check if we require close beyond the level
                if require_close_beyond and latest['close'] <= breakout_levels['upper']:
                    pass  # Not a valid breakout yet
                else:
                    # Check if breakout is significant enough
                    breakout_size = (latest['high'] - breakout_levels['upper']) / current_atr
                    if breakout_size >= volatility_factor:
                        # Significant breakout
                        
                        # Check volume confirmation if required
                        if volume_confirm and volume_ratio < volume_factor:
                            pass  # Volume not confirming
                        else:
                            bullish_breakout = True
            
            # Check for bearish breakout (price breaks below lower level)
            bearish_breakout = False
            if latest['low'] < breakout_levels['lower']:
                # Price has broken below the lower level
                
                # Check if we require close beyond the level
                if require_close_beyond and latest['close'] >= breakout_levels['lower']:
                    pass  # Not a valid breakout yet
                else:
                    # Check if breakout is significant enough
                    breakout_size = (breakout_levels['lower'] - latest['low']) / current_atr
                    if breakout_size >= volatility_factor:
                        # Significant breakout
                        
                        # Check volume confirmation if required
                        if volume_confirm and volume_ratio < volume_factor:
                            pass  # Volume not confirming
                        else:
                            bearish_breakout = True
            
            # Filter for false breakouts if enabled
            if false_breakout_filter:
                # Check if price quickly reversed after breaking out
                if bullish_breakout and latest['close'] < previous['close']:
                    # Price closed lower after breaking out - might be a false breakout
                    bullish_breakout = False
                    
                if bearish_breakout and latest['close'] > previous['close']:
                    # Price closed higher after breaking out - might be a false breakout
                    bearish_breakout = False
            
            # Generate signal for valid breakout
            if bullish_breakout:
                # Calculate stop loss and take profit levels
                stop_loss = latest['low']  # Default to current candle low
                
                # More conservative stop below the breakout candle
                if self.params['stop_loss_atr'] > 0:
                    stop_loss = min(stop_loss, current_price - (self.params['stop_loss_atr'] * current_atr))
                
                # Calculate take profit based on ATR
                take_profit = current_price + (self.params['take_profit_atr'] * current_atr)
                
                # Check risk-reward ratio
                risk = current_price - stop_loss
                reward = take_profit - current_price
                risk_reward = reward / risk if risk > 0 else 0
                
                if risk_reward >= self.params['risk_reward_ratio']:
                    # Valid trade with good risk-reward
                    signal['action'] = 'buy'
                    signal['price'] = current_price
                    
                    # Calculate position size based on risk per trade
                    signal['amount'] = self._calculate_position_size(symbol, current_price, stop_loss)
                    
                    # Set stop loss and take profit
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    
                    # Store trade information
                    self.active_trades[symbol][timeframe] = {
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(),
                        'pattern_time': pattern_time,
                        'trailing_stop': current_price * (1 - self.params['trailing_stop'] * current_atr / current_price) if self.params['trailing_stop'] > 0 else None,
                        'highest_price': current_price,
                        'risk_reward': risk_reward
                    }
                    
                    logger.info(f"Bullish breakout signal for {symbol} at {current_price}, RR: {risk_reward:.2f}")
                    
            elif bearish_breakout:
                # Calculate stop loss and take profit levels
                stop_loss = latest['high']  # Default to current candle high
                
                # More conservative stop above the breakout candle
                if self.params['stop_loss_atr'] > 0:
                    stop_loss = max(stop_loss, current_price + (self.params['stop_loss_atr'] * current_atr))
                
                # Calculate take profit based on ATR
                take_profit = current_price - (self.params['take_profit_atr'] * current_atr)
                
                # Check risk-reward ratio
                risk = stop_loss - current_price
                reward = current_price - take_profit
                risk_reward = reward / risk if risk > 0 else 0
                
                if risk_reward >= self.params['risk_reward_ratio']:
                    # Valid trade with good risk-reward
                    signal['action'] = 'sell'
                    signal['price'] = current_price
                    
                    # Calculate position size based on risk per trade
                    signal['amount'] = self._calculate_position_size(symbol, current_price, stop_loss)
                    
                    # Set stop loss and take profit
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    
                    # Store trade information
                    self.active_trades[symbol][timeframe] = {
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(),
                        'pattern_time': pattern_time,
                        'trailing_stop': current_price * (1 + self.params['trailing_stop'] * current_atr / current_price) if self.params['trailing_stop'] > 0 else None,
                        'lowest_price': current_price,
                        'risk_reward': risk_reward
                    }
                    
                    logger.info(f"Bearish breakout signal for {symbol} at {current_price}, RR: {risk_reward:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating breakout signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return signal
            
    def _calculate_position_size(self, symbol: str, price: float, stop_loss: float) -> float:
        """
        Calculate position size based on account size and risk per trade.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            stop_loss: Stop loss price
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Get risk percentage per trade
            risk_percent = self.params['risk_per_trade']
            
            # Get account balance (from risk manager if available)
            if self.risk_manager:
                account_size = self.risk_manager.account_size
            else:
                account_size = 10000.0  # Default account size
                
            # Calculate risk amount in USD
            risk_amount = account_size * (risk_percent / 100)
            
            # Calculate risk per unit
            risk_per_unit = abs(price - stop_loss)
            
            # Calculate position size
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = 0
                logger.warning(f"Invalid risk calculation for {symbol}: price={price}, stop_loss={stop_loss}")
                
            # Limit position size based on max positions
            max_position_size = account_size * 0.5 / price  # Maximum 50% of account per position
            position_size = min(position_size, max_position_size)
            
            logger.info(f"Calculated position size for {symbol}: {position_size} units at price {price}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 1.0  # Default fallback
            
    async def execute_signals(self, signals: List[Dict[str, Any]], exchange) -> List[Trade]:
        """
        Execute the generated signals through an exchange.
        
        Args:
            signals: List of signals to execute
            exchange: Exchange instance for trade execution
            
        Returns:
            list: List of executed trades
        """
        executed_trades = await super().execute_signals(signals, exchange)
        
        # Additional processing for breakout strategy
        for trade in executed_trades:
            symbol = trade.symbol
            
            # Find timeframe for this trade
            for timeframe in self.timeframes:
                if self.active_trades[symbol][timeframe] is not None:
                    # Mark pattern as traded if it's a new position
                    if trade.related_trade_id is None:  # This is an entry, not exit
                        pattern_time = self.active_trades[symbol][timeframe]['pattern_time']
                        self.traded_breakouts[symbol][timeframe].append(pattern_time)
                    else:  # This is an exit
                        # Clean up after trade
                        self.active_trades[symbol][timeframe] = None
                        
        return executed_trades
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate strategy-specific performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        # Get base performance stats
        stats = super().get_performance_stats()
        
        # Add breakout-specific stats
        breakout_stats = {
            'total_patterns_identified': 0,
            'patterns_traded': 0,
            'successful_breakouts': 0,
            'failed_breakouts': 0,
            'avg_risk_reward': 0.0,
            'pattern_types': {
                'range': 0,
                'channel': 0,
                'triangle': 0
            }
        }
        
        # Count patterns by type
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                pattern = self.consolidation_patterns[symbol][timeframe]
                if pattern is not None:
                    breakout_stats['total_patterns_identified'] += 1
                    pattern_type = pattern.get('type')
                    if pattern_type in breakout_stats['pattern_types']:
                        breakout_stats['pattern_types'][pattern_type] += 1
                
                # Count traded patterns
                breakout_stats['patterns_traded'] += len(self.traded_breakouts[symbol][timeframe])
        
        # Analyze trade history for successful vs failed breakouts
        winning_trades = [t for t in self.trade_history if t.pnl is not None and t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl is not None and t.pnl <= 0]
        
        # Count successful and failed breakouts
        breakout_stats['successful_breakouts'] = len(winning_trades)
        breakout_stats['failed_breakouts'] = len(losing_trades)
        
        # Calculate average risk-reward ratio from active trades
        risk_reward_values = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                trade_info = self.active_trades[symbol][timeframe]
                if trade_info is not None and 'risk_reward' in trade_info:
                    risk_reward_values.append(trade_info['risk_reward'])
        
        if risk_reward_values:
            breakout_stats['avg_risk_reward'] = sum(risk_reward_values) / len(risk_reward_values)
        
        # Add to base stats
        stats.update({'breakout_stats': breakout_stats})
        
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
            
            # Reset breakout-specific state
            self.consolidation_patterns = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            self.atr_values = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            self.breakout_levels = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            self.pattern_start_times = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            self.traded_breakouts = {symbol: {tf: [] for tf in self.timeframes} for symbol in self.symbols}
            self.active_trades = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
            
            logger.info(f"Reset breakout strategy {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting breakout strategy {self.name}: {str(e)}")
            return False
