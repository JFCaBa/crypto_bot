"""
RSI Strategy
===========
Implementation of a trading strategy based on the Relative Strength Index (RSI) indicator.
"""

from typing import Dict, List, Any

import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy.
    
    This strategy generates buy signals when RSI is below oversold threshold,
    and sell signals when RSI is above overbought threshold.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the RSI strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            risk_manager: Risk manager instance
            params: Additional strategy parameters
        """
        # Initialize base strategy
        default_params = {
            'period': 14,  # RSI period
            'overbought': 70,  # Overbought threshold
            'oversold': 30,  # Oversold threshold
            'exit_overbought': 65,  # Exit threshold for short positions
            'exit_oversold': 35,  # Exit threshold for long positions
            'use_ema': False,  # Whether to use EMA for RSI calculation
            'ema_period': 9,  # EMA period for RSI smoothing
            'stop_loss': 2.0,  # Stop loss (% of entry price)
            'take_profit': 4.0,  # Take profit (% of entry price)
            'trailing_stop': 0.0,  # Trailing stop (% of price)
            'risk_per_trade': 0.01,  # Risk per trade (% of account)
            'use_divergence': False,  # Whether to use RSI divergence for additional signals
            'divergence_period': 5,  # Period for divergence detection
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="RSI",
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
            period = self.params['period']
            use_ema = self.params['use_ema']
            ema_period = self.params['ema_period']
            use_divergence = self.params['use_divergence']
            divergence_period = self.params['divergence_period']
            
            # Calculate RSI
            # Get price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            if use_ema:
                # Use exponential moving average for smoother RSI
                avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
                avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
            else:
                # First value is just the simple average
                avg_gain = pd.Series([gain[:period].mean()], index=[df.index[period-1]])
                avg_loss = pd.Series([loss[:period].mean()], index=[df.index[period-1]])
                
                # Calculate subsequent values
                for i in range(period, len(df)):
                    avg_gain.loc[df.index[i]] = (avg_gain.iloc[-1] * (period - 1) + gain.iloc[i]) / period
                    avg_loss.loc[df.index[i]] = (avg_loss.iloc[-1] * (period - 1) + loss.iloc[i]) / period
            
            # Calculate RS (Relative Strength)
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Apply EMA smoothing to RSI if requested
            if use_ema:
                df['rsi_smooth'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()
            else:
                df['rsi_smooth'] = df['rsi']
            
            # Calculate RSI divergence if enabled
            if use_divergence:
                # Look for price highs and lows
                df['price_high'] = df['close'].rolling(window=divergence_period*2+1, center=True).apply(
                    lambda x: x[divergence_period] == max(x), raw=True
                ).fillna(0).astype(bool)
                
                df['price_low'] = df['close'].rolling(window=divergence_period*2+1, center=True).apply(
                    lambda x: x[divergence_period] == min(x), raw=True
                ).fillna(0).astype(bool)
                
                # Look for RSI highs and lows
                df['rsi_high'] = df['rsi'].rolling(window=divergence_period*2+1, center=True).apply(
                    lambda x: x[divergence_period] == max(x), raw=True
                ).fillna(0).astype(bool)
                
                df['rsi_low'] = df['rsi'].rolling(window=divergence_period*2+1, center=True).apply(
                    lambda x: x[divergence_period] == min(x), raw=True
                ).fillna(0).astype(bool)
                
                # Initialize divergence columns
                df['bullish_divergence'] = False
                df['bearish_divergence'] = False
                
                # Detect bullish divergence (price making lower lows but RSI making higher lows)
                for i in range(divergence_period, len(df)-divergence_period):
                    if df['price_low'].iloc[i]:
                        # Look back for previous low
                        for j in range(i-divergence_period, i):
                            if df['price_low'].iloc[j]:
                                # Check for bullish divergence
                                if df['close'].iloc[i] < df['close'].iloc[j] and df['rsi'].iloc[i] > df['rsi'].iloc[j]:
                                    df['bullish_divergence'].iloc[i] = True
                                break
                
                # Detect bearish divergence (price making higher highs but RSI making lower highs)
                for i in range(divergence_period, len(df)-divergence_period):
                    if df['price_high'].iloc[i]:
                        # Look back for previous high
                        for j in range(i-divergence_period, i):
                            if df['price_high'].iloc[j]:
                                # Check for bearish divergence
                                if df['close'].iloc[i] > df['close'].iloc[j] and df['rsi'].iloc[i] < df['rsi'].iloc[j]:
                                    df['bearish_divergence'].iloc[i] = True
                                break
            
            # Store calculated indicators
            indicators = {}
            indicators['rsi'] = df['rsi']
            indicators['rsi_smooth'] = df['rsi_smooth']
            
            if use_divergence:
                indicators['bullish_divergence'] = df['bullish_divergence']
                indicators['bearish_divergence'] = df['bearish_divergence']
            
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
            
            if df.empty or 'rsi' not in df.columns:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return signal
                
            # Extract parameters
            overbought = self.params['overbought']
            oversold = self.params['oversold']
            exit_overbought = self.params['exit_overbought']
            exit_oversold = self.params['exit_oversold']
            stop_loss = self.params['stop_loss']
            take_profit = self.params['take_profit']
            trailing_stop = self.params['trailing_stop']
            risk_per_trade = self.params['risk_per_trade']
            use_divergence = self.params['use_divergence']
            
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
                signal_strength = 0  # Base signal strength
                
                # RSI below oversold level (bullish)
                if latest['rsi_smooth'] < oversold:
                    signal_strength += 1
                    
                # Bullish divergence (if enabled)
                if use_divergence and latest.get('bullish_divergence', False):
                    signal_strength += 1
                
                # RSI crossing back above oversold level (stronger bullish signal)
                if previous is not None and previous['rsi_smooth'] < oversold and latest['rsi_smooth'] >= oversold:
                    signal_strength += 1
                    
                # Strong bullish signal
                if signal_strength >= 1:  # Adjust threshold as needed
                    # Buy signal
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
                        
                # RSI above overbought level (bearish)
                signal_strength = 0  # Reset signal strength
                
                if latest['rsi_smooth'] > overbought:
                    signal_strength += 1
                    
                # Bearish divergence (if enabled)
                if use_divergence and latest.get('bearish_divergence', False):
                    signal_strength += 1
                
                # RSI crossing back below overbought level (stronger bearish signal)
                if previous is not None and previous['rsi_smooth'] > overbought and latest['rsi_smooth'] <= overbought:
                    signal_strength += 1
                    
                # Strong bearish signal
                if signal_strength >= 1:  # Adjust threshold as needed
                    # Sell signal
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
                if position['side'] == 'long':
                    # Exit long position when RSI crosses above exit_overbought
                    if latest['rsi_smooth'] > exit_overbought:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                elif position['side'] == 'short':
                    # Exit short position when RSI crosses below exit_oversold
                    if latest['rsi_smooth'] < exit_oversold:
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
            # Default position size (1 unit)
            if risk_percent <= 0 or stop_loss_percent <= 0:
                return 1.0
                
            # Calculate position size based on risk
            account_balance = 10000.0  # Placeholder, in a real implementation, this would use the actual account balance
            
            # Calculate max amount to risk
            risk_amount = account_balance * risk_percent / 100
            
            # Calculate potential loss per unit based on stop loss
            loss_per_unit = price * (stop_loss_percent / 100)
            
            # Calculate position size
            position_size = risk_amount / loss_per_unit
            
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default fallback