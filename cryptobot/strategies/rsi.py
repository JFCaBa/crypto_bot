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
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in data for {symbol} {timeframe}: {missing_columns}")
                return False
                
            # Check if we have enough data for calculation
            period = self.params['period']
            if len(df) < period + 10:
                logger.warning(f"Not enough data points for {symbol} {timeframe}. Need at least {period + 10}, got {len(df)}.")
                return False
                
            # Extract parameters
            use_ema = self.params['use_ema']
            ema_period = self.params['ema_period']
            use_divergence = self.params['use_divergence']
            divergence_period = self.params['divergence_period']
            
            # Calculate RSI
            # Get price changes
            delta = df['close'].diff().fillna(0)
            
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
            
            # Calculate RS (Relative Strength) with handling for divide by zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Replace zeros with small number to avoid division by zero
            
            # Calculate RSI
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Apply EMA smoothing to RSI if requested
            if use_ema:
                df['rsi_smooth'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()
            else:
                df['rsi_smooth'] = df['rsi']
            
            # Fill any NaN values with default values
            df['rsi'] = df['rsi'].fillna(50)
            df['rsi_smooth'] = df['rsi_smooth'].fillna(50)
            
            # Calculate RSI divergence if enabled
            if use_divergence:
                try:
                    # Look for price highs and lows
                    df['price_high'] = df['close'].rolling(window=min(divergence_period*2+1, len(df)), center=True).apply(
                        lambda x: x[min(divergence_period, len(x)-1)] == max(x), raw=True
                    ).fillna(0).astype(bool)
                    
                    df['price_low'] = df['close'].rolling(window=min(divergence_period*2+1, len(df)), center=True).apply(
                        lambda x: x[min(divergence_period, len(x)-1)] == min(x), raw=True
                    ).fillna(0).astype(bool)
                    
                    # Look for RSI highs and lows
                    df['rsi_high'] = df['rsi'].rolling(window=min(divergence_period*2+1, len(df)), center=True).apply(
                        lambda x: x[min(divergence_period, len(x)-1)] == max(x), raw=True
                    ).fillna(0).astype(bool)
                    
                    df['rsi_low'] = df['rsi'].rolling(window=min(divergence_period*2+1, len(df)), center=True).apply(
                        lambda x: x[min(divergence_period, len(x)-1)] == min(x), raw=True
                    ).fillna(0).astype(bool)
                    
                    # Initialize divergence columns
                    df['bullish_divergence'] = False
                    df['bearish_divergence'] = False
                    
                    # Detect bullish divergence (price making lower lows but RSI making higher lows)
                    for i in range(divergence_period, len(df)-divergence_period):
                        if df['price_low'].iloc[i]:
                            # Look back for previous low
                            for j in range(max(0, i-divergence_period), i):
                                if df['price_low'].iloc[j]:
                                    # Check for bullish divergence
                                    if df['close'].iloc[i] < df['close'].iloc[j] and df['rsi'].iloc[i] > df['rsi'].iloc[j]:
                                        df.loc[df.index[i], 'bullish_divergence'] = True
                                    break
                    
                    # Detect bearish divergence (price making higher highs but RSI making lower highs)
                    for i in range(divergence_period, len(df)-divergence_period):
                        if df['price_high'].iloc[i]:
                            # Look back for previous high
                            for j in range(max(0, i-divergence_period), i):
                                if df['price_high'].iloc[j]:
                                    # Check for bearish divergence
                                    if df['close'].iloc[i] > df['close'].iloc[j] and df['rsi'].iloc[i] < df['rsi'].iloc[j]:
                                        df.loc[df.index[i], 'bearish_divergence'] = True
                                    break
                except Exception as e:
                    logger.warning(f"Error calculating divergence for {symbol} {timeframe}: {str(e)}")
                    # If divergence calculation fails, disable it for this run
                    use_divergence = False
                    df['bullish_divergence'] = False
                    df['bearish_divergence'] = False
            
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
            
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return signal
                
            # Ensure indicators are calculated
            if not self.calculate_indicators(symbol, timeframe):
                logger.warning(f"Failed to calculate indicators for {symbol} {timeframe}")
                return signal
                
            # Check if required indicators exist
            if 'rsi' not in df.columns or 'rsi_smooth' not in df.columns:
                logger.warning(f"Required indicators not found for {symbol} {timeframe}")
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
            if len(df) < 2:
                logger.warning(f"Not enough data points for signal generation for {symbol} {timeframe}")
                return signal
                
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
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
                if use_divergence and 'bullish_divergence' in latest and latest['bullish_divergence']:
                    signal_strength += 1
                
                # RSI crossing back above oversold level (stronger bullish signal)
                if previous['rsi_smooth'] < oversold and latest['rsi_smooth'] >= oversold:
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
                        
                # Check bearish signals only if we didn't already generate a bullish signal
                if signal['action'] is None:
                    # RSI above overbought level (bearish)
                    signal_strength = 0  # Reset signal strength
                    
                    if latest['rsi_smooth'] > overbought:
                        signal_strength += 1
                        
                    # Bearish divergence (if enabled)
                    if use_divergence and 'bearish_divergence' in latest and latest['bearish_divergence']:
                        signal_strength += 1
                    
                    # RSI crossing back below overbought level (stronger bearish signal)
                    if previous['rsi_smooth'] > overbought and latest['rsi_smooth'] <= overbought:
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
            # Input validation
            if not price or price <= 0:
                logger.warning(f"Invalid price for position size calculation: {price}")
                return 1.0
                
            # Default position size (1 unit) if risk parameters are invalid
            if risk_percent <= 0 or stop_loss_percent <= 0:
                logger.info(f"Using default position size due to invalid risk parameters: risk_percent={risk_percent}, stop_loss_percent={stop_loss_percent}")
                return 1.0
                
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