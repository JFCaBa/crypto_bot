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
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in data for {symbol} {timeframe}: {missing_columns}")
                return False
                
            # Check if we have enough data for calculation
            if len(df) < self.params['slow_period'] + 10:
                logger.warning(f"Not enough data points for {symbol} {timeframe}. Need at least {self.params['slow_period'] + 10}, got {len(df)}.")
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
            df['ma_crossover'] = np.sign(df['ma_diff']).diff().fillna(0)
            
            # MACD indicators if enabled
            if use_macd:
                # Calculate MACD
                df['macd'] = df['fast_ma'] - df['slow_ma']
                df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                df['macd_crossover'] = np.sign(df['macd'] - df['macd_signal']).diff().fillna(0)
                
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
            import traceback
            logger.error(traceback.format_exc())
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
            
            # Ensure indicators are calculated
            if not self.calculate_indicators(symbol, timeframe):
                logger.warning(f"Failed to calculate indicators for {symbol} {timeframe}")
                return signal
                
            # Check if required indicators exist
            if 'fast_ma' not in df.columns or 'slow_ma' not in df.columns:
                logger.warning(f"Required indicators not found for {symbol} {timeframe}")
                return signal
                
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
                if use_macd and 'macd_crossover' in df.columns:
                    # Use MACD crossover for entry
                    crossover = latest['macd_crossover']
                    strength = abs(latest['macd_hist']) if 'macd_hist' in df.columns else 0
                    
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
                            signal['stop_loss'] = current_price * (1 - stop_loss / 100)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 + take_profit / 100)
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
                            signal['stop_loss'] = current_price * (1 + stop_loss / 100)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 - take_profit / 100)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
                elif 'ma_crossover' in df.columns:
                    # Use simple MA crossover for entry
                    crossover = latest['ma_crossover']
                    strength = abs(latest['ma_diff']) if 'ma_diff' in df.columns else 0
                    
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
                            signal['stop_loss'] = current_price * (1 - stop_loss / 100)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 + take_profit / 100)
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
                            signal['stop_loss'] = current_price * (1 + stop_loss / 100)
                        if take_profit > 0:
                            signal['take_profit'] = current_price * (1 - take_profit / 100)
                        if trailing_stop > 0:
                            signal['params']['trailingDelta'] = trailing_stop * 100  # Convert to basis points
            else:
                # Check for exit signals
                if use_macd and 'macd_crossover' in df.columns:
                    # Use MACD crossover for exit
                    if position['side'] == 'long':
                        crossover = latest['macd_crossover']
                        strength = abs(latest['macd_hist']) if 'macd_hist' in df.columns else 0
                        
                        if crossover < 0 and strength > exit_threshold:
                            # Bearish signal: MACD crosses below signal line
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                    elif position['side'] == 'short':
                        crossover = latest['macd_crossover']
                        strength = abs(latest['macd_hist']) if 'macd_hist' in df.columns else 0
                        
                        if crossover > 0 and strength > exit_threshold:
                            # Bullish signal: MACD crosses above signal line
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                elif 'ma_crossover' in df.columns:
                    # Use simple MA crossover for exit
                    if position['side'] == 'long':
                        crossover = latest['ma_crossover']
                        strength = abs(latest['ma_diff']) if 'ma_diff' in df.columns else 0
                        
                        if crossover < 0 and strength > exit_threshold:
                            # Bearish signal: fast MA crosses below slow MA
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
                    elif position['side'] == 'short':
                        crossover = latest['ma_crossover']
                        strength = abs(latest['ma_diff']) if 'ma_diff' in df.columns else 0
                        
                        if crossover > 0 and strength > exit_threshold:
                            # Bullish signal: fast MA crosses above slow MA
                            signal['action'] = 'close'
                            signal['price'] = current_price
                            signal['amount'] = position['amount']
            
            return signal
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            # Input validation
            if not price or price <= 0:
                logger.warning(f"Invalid price for position size calculation: {price}")
                return 1.0
                
            # Default position size (1 unit) if risk parameters are invalid
            if risk_percent <= 0 or stop_loss_percent <= 0:
                logger.info(f"Using default position size due to invalid risk parameters: risk_percent={risk_percent}, stop_loss_percent={stop_loss_percent}")
                return 1.0
                
            # Calculate position size based on risk
            # risk_amount = account_balance * risk_percent
            # position_size = risk_amount / (price * stop_loss_percent)
            
            # Get account balance from backtesting engine or use default
            account_balance = getattr(self, 'account_size', 10000.0)
            
            # Calculate max amount to risk
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate potential loss per unit based on stop loss
            loss_per_unit = price * (stop_loss_percent / 100)
            
            # Safeguard against division by zero or very small numbers
            if loss_per_unit < 0.000001:
                logger.warning(f"Stop loss too small for calculation: {loss_per_unit}")
                return 1.0
                
            # Calculate position size
            position_size = risk_amount / loss_per_unit
            
            # Cap position size if it's unreasonably large
            max_size = account_balance / price * 0.5  # Max 50% of account in a single position
            if position_size > max_size:
                logger.warning(f"Position size capped from {position_size} to {max_size}")
                position_size = max_size
                
            logger.info(f"Calculated position size: {position_size} units at price {price}")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default fallback