"""
Bollinger Bands Strategy
======================
Implementation of a trading strategy based on Bollinger Bands indicator.
"""

from typing import Dict, List, Any

import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    This strategy generates signals based on price movements relative to Bollinger Bands.
    Buy signals are generated when price touches or crosses below the lower band,
    and sell signals when price touches or crosses above the upper band.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            risk_manager: Risk manager instance
            params: Additional strategy parameters
        """
        # Initialize base strategy
        default_params = {
            'period': 20,  # Moving average period
            'std_dev': 2.0,  # Number of standard deviations for bands
            'ma_type': 'sma',  # Moving average type: 'sma', 'ema', 'wma'
            'entry_trigger': 'touch',  # Entry trigger: 'touch', 'cross', 'close'
            'exit_trigger': 'middle',  # Exit trigger: 'middle', 'opposite', 'target'
            'use_volume': False,  # Whether to use volume for confirmation
            'volume_threshold': 1.5,  # Volume threshold as multiplier of average
            'stop_loss': 2.0,  # Stop loss (% of entry price)
            'take_profit': 4.0,  # Take profit (% of entry price)
            'trailing_stop': 0.0,  # Trailing stop (% of price)
            'risk_per_trade': 0.01,  # Risk per trade (% of account)
            'use_bandwidth': False,  # Whether to use bandwidth for signal filtering
            'min_bandwidth': 0.0,  # Minimum bandwidth for valid signals
            'band_value': 'close',  # Which price to use for band comparison ('close', 'high', 'low')
            'squeeze_exit': False,  # Exit when bands squeeze (volatility decreases)
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="BollingerBands",
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
            std_dev = self.params['std_dev']
            ma_type = self.params['ma_type']
            band_value = self.params['band_value']
            use_volume = self.params['use_volume']
            use_bandwidth = self.params['use_bandwidth']
            volume_threshold = self.params['volume_threshold']
            
            # Calculate the middle band (SMA or EMA)
            if ma_type == 'sma':
                df['middle_band'] = df['close'].rolling(window=period).mean()
            elif ma_type == 'ema':
                df['middle_band'] = df['close'].ewm(span=period, adjust=False).mean()
            elif ma_type == 'wma':
                # Weighted moving average
                weights = np.arange(1, period + 1)
                df['middle_band'] = df['close'].rolling(window=period).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
            else:
                logger.warning(f"Invalid MA type: {ma_type}, using SMA")
                df['middle_band'] = df['close'].rolling(window=period).mean()
                
            # Calculate standard deviation
            df['std_dev'] = df['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df['upper_band'] = df['middle_band'] + (df['std_dev'] * std_dev)
            df['lower_band'] = df['middle_band'] - (df['std_dev'] * std_dev)
            
            # Calculate bandwidth
            if use_bandwidth:
                df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
                
            # Calculate %B (Percent Bandwidth)
            # %B = (Price - Lower Band) / (Upper Band - Lower Band)
            # 0.0 = price at lower band, 1.0 = price at upper band, 0.5 = price at middle band
            df['percent_b'] = (df[band_value] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
            
            # Calculate band touches and crosses
            df['lower_band_touch'] = (df['low'] <= df['lower_band'])
            df['upper_band_touch'] = (df['high'] >= df['upper_band'])
            
            df['lower_band_cross'] = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'])
            df['upper_band_cross'] = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'])
            
            df['middle_band_cross_up'] = (df['close'] > df['middle_band']) & (df['close'].shift(1) <= df['middle_band'])
            df['middle_band_cross_down'] = (df['close'] < df['middle_band']) & (df['close'].shift(1) >= df['middle_band'])
            
            # Volume analysis for confirmation
            if use_volume:
                df['volume_sma'] = df['volume'].rolling(window=period).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['high_volume'] = df['volume_ratio'] > volume_threshold
                
            # Band squeeze detection (for decreasing volatility)
            if use_bandwidth:
                # Calculate rolling min/max of bandwidth to determine if bands are squeezing
                df['bandwidth_min'] = df['bandwidth'].rolling(window=period).min()
                df['bandwidth_is_narrowing'] = df['bandwidth'] <= df['bandwidth_min']
                
            # Store calculated indicators
            indicators = {}
            indicators['middle_band'] = df['middle_band']
            indicators['upper_band'] = df['upper_band']
            indicators['lower_band'] = df['lower_band']
            indicators['percent_b'] = df['percent_b']
            indicators['lower_band_touch'] = df['lower_band_touch']
            indicators['upper_band_touch'] = df['upper_band_touch']
            indicators['lower_band_cross'] = df['lower_band_cross']
            indicators['upper_band_cross'] = df['upper_band_cross']
            indicators['middle_band_cross_up'] = df['middle_band_cross_up']
            indicators['middle_band_cross_down'] = df['middle_band_cross_down']
            
            if use_volume:
                indicators['volume_ratio'] = df['volume_ratio']
                indicators['high_volume'] = df['high_volume']
                
            if use_bandwidth:
                indicators['bandwidth'] = df['bandwidth']
                indicators['bandwidth_is_narrowing'] = df['bandwidth_is_narrowing']
            
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
            
            if df.empty or 'middle_band' not in df.columns:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return signal
                
            # Extract parameters
            entry_trigger = self.params['entry_trigger']
            exit_trigger = self.params['exit_trigger']
            use_volume = self.params['use_volume']
            use_bandwidth = self.params['use_bandwidth']
            min_bandwidth = self.params['min_bandwidth']
            squeeze_exit = self.params['squeeze_exit']
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
            
            # Check bandwidth if enabled
            if use_bandwidth and latest['bandwidth'] < min_bandwidth:
                # Skip trading when bands are too narrow (low volatility)
                return signal
                
            if not is_active:
                # Entry signals
                
                # Buy signal - when price touches/crosses lower band
                buy_signal = False
                
                if entry_trigger == 'touch' and latest['lower_band_touch']:
                    buy_signal = True
                elif entry_trigger == 'cross' and latest['lower_band_cross']:
                    buy_signal = True
                elif entry_trigger == 'close' and latest['close'] <= latest['lower_band']:
                    buy_signal = True
                    
                # Volume confirmation if enabled
                if buy_signal and use_volume and not latest['high_volume']:
                    buy_signal = False  # Cancel signal if volume is not high enough
                    
                if buy_signal:
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
                
                # Sell signal - when price touches/crosses upper band
                sell_signal = False
                
                if entry_trigger == 'touch' and latest['upper_band_touch']:
                    sell_signal = True
                elif entry_trigger == 'cross' and latest['upper_band_cross']:
                    sell_signal = True
                elif entry_trigger == 'close' and latest['close'] >= latest['upper_band']:
                    sell_signal = True
                    
                # Volume confirmation if enabled
                if sell_signal and use_volume and not latest['high_volume']:
                    sell_signal = False  # Cancel signal if volume is not high enough
                    
                if sell_signal:
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
                # Exit signals
                if position['side'] == 'long':
                    # Exit long position
                    exit_long = False
                    
                    if exit_trigger == 'middle' and latest['middle_band_cross_down']:
                        # Exit when price crosses down middle band
                        exit_long = True
                    elif exit_trigger == 'opposite' and latest['upper_band_touch']:
                        # Exit when price touches opposite band
                        exit_long = True
                    elif exit_trigger == 'target' and latest['close'] >= position['entry_price'] * (1 + take_profit / 100):
                        # Exit when price reaches target
                        exit_long = True
                    
                    # Check for band squeeze exit
                    if squeeze_exit and use_bandwidth and latest['bandwidth_is_narrowing']:
                        exit_long = True
                        
                    if exit_long:
                        signal['action'] = 'close'
                        signal['price'] = current_price
                        signal['amount'] = position['amount']
                        
                elif position['side'] == 'short':
                    # Exit short position
                    exit_short = False
                    
                    if exit_trigger == 'middle' and latest['middle_band_cross_up']:
                        # Exit when price crosses up middle band
                        exit_short = True
                    elif exit_trigger == 'opposite' and latest['lower_band_touch']:
                        # Exit when price touches opposite band
                        exit_short = True
                    elif exit_trigger == 'target' and latest['close'] <= position['entry_price'] * (1 - take_profit / 100):
                        # Exit when price reaches target
                        exit_short = True
                    
                    # Check for band squeeze exit
                    if squeeze_exit and use_bandwidth and latest['bandwidth_is_narrowing']:
                        exit_short = True
                        
                    if exit_short:
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