"""
Moving Average Crossover Strategy
================================
Implementation of a moving average crossover trading strategy.
"""

from typing import Dict, List, Any

import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        fast_period: int = 10,
        slow_period: int = 50,
        signal_period: int = 9,
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            signal_period: Signal line period (for MACD)
            risk_manager: Risk manager instance
            params: Additional strategy parameters
        """
        # Initialize base strategy
        default_params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'ma_type': 'ema',  # 'sma', 'ema', 'wma'
            'use_macd': False,  # Whether to use MACD or simple MA crossover
            'entry_threshold': 0.0,  # Minimum crossover strength
            'exit_threshold': 0.0,  # Minimum crossover strength for exit
            'trailing_stop': 0.0,  # Trailing stop (% of price)
            'stop_loss': 0.0,  # Stop loss (% of entry price)
            'take_profit': 0.0,  # Take profit (% of entry price)
            'risk_per_trade': 0.01,  # Risk per trade (% of account)
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="MovingAverageCrossover",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate technical indicators for a symbol and timeframe.
        
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
                
            # Extract parameters
            fast_period = self.params['fast_period']
            slow_period = self.params['slow_period']
            signal_period = self.params['signal_period']
            ma_type = self.params['ma_type']
            use_macd = self.params['use_macd']
            
            # Calculate moving averages
            if ma_type == 'sma':
                df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
                df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
            elif ma_type == 'ema':
                df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
                df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
            elif ma_type == 'wma':
                # Weighted moving average
                weights_fast = np.arange(1, fast_period + 1)
                weights_slow = np.arange(1, slow_period + 1)
                
                df['fast_ma'] = df['close'].rolling(window=fast_period).apply(
                    lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True
                )
                df['slow_ma'] = df['close'].rolling(window=slow_period).apply(
                    lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True
                )
            else:
                logger.warning(f"Invalid MA type: {ma_type}, using EMA")
                df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
                df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
                
            # Calculate crossover
            df['ma_diff'] = df['fast_ma'] - df['slow_ma']
            df['ma_crossover'] = np.sign(df['ma_diff']).diff()
            
            # MACD indicators if enabled
            if use_macd:
                # Calculate MACD
                df['macd'] = df['fast_ma'] - df['slow_ma']
                df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                df['macd_crossover'] = np.sign(df['macd'] - df['macd_signal']).diff()
                
            # Store calculated indicators
            indicators = {}
            indicators['fast_ma'] = df['fast_ma']
            indicators['slow_ma'] = df['slow_ma']
            indicators['ma_diff'] = df['ma_diff']
            indicators['ma_crossover'] = df['ma_crossover']
            
            if use_macd:
                indicators['macd'] = df['macd']
                indicators['macd_signal'] = df['macd_signal']
                indicators['macd_hist'] = df['macd_hist']
                indicators['macd_crossover'] = df['macd_crossover']
                
            # Save indicators to instance
            self.indicators[symbol][timeframe] = indicators
            
            # Update data with calculated indicators
            self.data[symbol][timeframe] = df
            
            return True
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            return False
            
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals for a symbol and timeframe.
        
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
            # Get latest data point
            df = self.data[symbol][timeframe]
            
            if df.empty or len(df) < self.params['slow_period'] + 10:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return signal
                
            # Extract parameters
            use_macd = self.params['use_macd']
            entry_threshold = self.params['entry_threshold']
            exit_threshold = self.params['exit_threshold']
            stop_loss = self.params['stop_loss']
            take_profit = self.params['take_profit']
            trailing_stop = self.params['trailing_stop']
            risk_per_trade = self.params['risk_per_trade']
            
            # Get latest indicators
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            
            # Check if position is active
            position = self.positions[symbol]
            is_active = position['is_active']
            
            # Get current price
            current_price = latest['close']
            
            if not is_active:
                # Check for entry signal
                if use_macd:
                    # Use MACD crossover for entry
                    crossover = latest['macd_crossover']
                    strength = abs(latest['macd_hist'])
                    
                    if crossover > 0 and strength > entry_threshold:
                        # Bullish signal: MACD crosses above signal line
                        signal['action'] = 'buy'
                        signal['price'] = current_price
                        
                        # Calculate position size based on risk
                        signal['amount'] = self._calculate_position_size(
                            symbol, current_price, risk_per_trade, stop_loss
                        )
                        
                        # Add stop loss and take profit
                        if stop_loss > 0:
                            signal['stop_loss'] = current_price * (1 - stop_loss)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 + take_profit)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
                            
                    elif crossover < 0 and strength > entry_threshold:
                        # Bearish signal: MACD crosses below signal line
                        signal['action'] = 'sell'
                        signal['price'] = current_price
                        
                        # Calculate position size based on risk
                        signal['amount'] = self._calculate_position_size(
                            symbol, current_price, risk_per_trade, stop_loss
                        )
                        
                        # Add stop loss and take profit
                        if stop_loss > 0:
                            signal['stop_loss'] = current_price * (1 + stop_loss)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 - take_profit)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
                else:
                    # Use simple MA crossover for entry
                    crossover = latest['ma_crossover']
                    strength = abs(latest['ma_diff'])
                    
                    if crossover > 0 and strength > entry_threshold:
                        # Bullish signal: fast MA crosses above slow MA
                        signal['action'] = 'buy'
                        signal['price'] = current_price
                        
                        # Calculate position size based on risk
                        signal['amount'] = self._calculate_position_size(
                            symbol, current_price, risk_per_trade, stop_loss
                        )
                        
                        # Add stop loss and take profit
                        if stop_loss > 0:
                            signal['stop_loss'] = current_price * (1 - stop_loss)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 + take_profit)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
                            
                    elif crossover < 0 and strength > entry_threshold:
                        # Bearish signal: fast MA crosses below slow MA
                        signal['action'] = 'sell'
                        signal['price'] = current_price
                        
                        # Calculate position size based on risk
                        signal['amount'] = self._calculate_position_size(
                            symbol, current_price, risk_per_trade, stop_loss
                        )
                        
                        # Add stop loss and take profit
                        if stop_loss > 0:
                            signal['stop_loss'] = current_price * (1 + stop_loss)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 - take_profit)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
            else:
                # Check for exit signals
                if use_macd:
                    # Use MACD crossover for exit
                    if position['side'] == 'long':
                        crossover = latest['macd_crossover']
                        strength = abs(latest['macd_hist'])
                        
                        if crossover < 0 and strength > exit_threshold:
                            # Bearish signal: MACD crosses below signal line
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                    elif position['side'] == 'short':
                        crossover = latest['macd_crossover']
                        strength = abs(latest['macd_hist'])
                        
                        if crossover > 0 and strength > exit_threshold:
                            # Bullish signal: MACD crosses above signal line
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                else:
                    # Use simple MA crossover for exit
                    if position['side'] == 'long':
                        crossover = latest['ma_crossover']
                        strength = abs(latest['ma_diff'])
                        
                        if crossover < 0 and strength > exit_threshold:
                            # Bearish signal: fast MA crosses below slow MA
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                    elif position['side'] == 'short':
                        crossover = latest['ma_crossover']
                        strength = abs(latest['ma_diff'])
                        
                        if crossover > 0 and strength > exit_threshold:
                            # Bullish signal: fast MA crosses above slow MA
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
            
            return signal
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            return signal
            
    def _calculate_position_size(self, symbol: str, price: float, risk_percent: float, stop_loss_percent: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            risk_percent: Risk per trade as percentage of account
            stop_loss_percent: Stop loss percentage
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Default position size (1 unit)
            if risk_percent <= 0 or stop_loss_percent <= 0:
                return 1.0
                
            # Calculate position size based on risk
            # risk_amount = account_balance * risk_percent
            # position_size = risk_amount / (price * stop_loss_percent)
            
            # For now, we'll use a simple placeholder
            # In a real implementation, this would use the actual account balance
            account_balance = 10000.0  # Placeholder
            risk_amount = account_balance * risk_percent
            position_size = risk_amount / (price * stop_loss_percent)
            
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default fallback