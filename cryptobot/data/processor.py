"""
Data Processor
============
Processes market data for analysis and strategy execution.
"""

import os
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataProcessor:
    """
    Processes market data for analysis and strategy execution.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: str = '.cache',
        cache_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize data processor.
        
        Args:
            cache_enabled: Whether to enable data caching
            cache_dir: Directory for cache files
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        
        # Create cache directory if enabled
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        logger.info(f"Data processor initialized with cache {'enabled' if cache_enabled else 'disabled'}")
        
    def ohlcv_to_dataframe(self, ohlcv_data: List[List[float]]) -> pd.DataFrame:
        """
        Convert raw OHLCV data to pandas DataFrame.
        
        Args:
            ohlcv_data: List of OHLCV candles [timestamp, open, high, low, close, volume]
            
        Returns:
            pd.DataFrame: OHLCV data as DataFrame
        """
        if not ohlcv_data:
            return pd.DataFrame()
            
        # Create DataFrame with appropriate columns
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.
        
        Args:
            df: OHLCV data as DataFrame
            timeframe: Target timeframe (e.g., '1h', '1d')
            
        Returns:
            pd.DataFrame: Resampled OHLCV data
        """
        if df.empty:
            return df
            
        # Map to pandas resample rule
        timeframe_map = {
            '1m': '1T',
            '3m': '3T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '8h': '8H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W',
            '1M': '1M'
        }
        
        # Get resample rule
        rule = timeframe_map.get(timeframe)
        if not rule:
            logger.warning(f"Unsupported timeframe for resampling: {timeframe}")
            return df
            
        # Resample data
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop rows with NaN values
        resampled = resampled.dropna()
        
        return resampled
        
    def calculate_indicators(self, df: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: OHLCV data as DataFrame
            indicators: Dictionary of indicators to calculate
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if df.empty:
            return df
            
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Calculate each indicator
            for indicator_name, params in indicators.items():
                indicator_type = params.get('type', '').lower()
                
                if indicator_type == 'sma':
                    # Simple Moving Average
                    period = params.get('period', 14)
                    column = params.get('column', 'close')
                    result[indicator_name] = result[column].rolling(window=period).mean()
                    
                elif indicator_type == 'ema':
                    # Exponential Moving Average
                    period = params.get('period', 14)
                    column = params.get('column', 'close')
                    result[indicator_name] = result[column].ewm(span=period, adjust=False).mean()
                    
                elif indicator_type == 'rsi':
                    # Relative Strength Index
                    period = params.get('period', 14)
                    column = params.get('column', 'close')
                    
                    delta = result[column].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    
                    rs = avg_gain / avg_loss
                    result[indicator_name] = 100 - (100 / (1 + rs))
                    
                elif indicator_type == 'macd':
                    # Moving Average Convergence Divergence
                    fast_period = params.get('fast_period', 12)
                    slow_period = params.get('slow_period', 26)
                    signal_period = params.get('signal_period', 9)
                    column = params.get('column', 'close')
                    
                    fast_ema = result[column].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = result[column].ewm(span=slow_period, adjust=False).mean()
                    
                    result[f"{indicator_name}_line"] = fast_ema - slow_ema
                    result[f"{indicator_name}_signal"] = result[f"{indicator_name}_line"].ewm(span=signal_period, adjust=False).mean()
                    result[f"{indicator_name}_hist"] = result[f"{indicator_name}_line"] - result[f"{indicator_name}_signal"]
                    
                elif indicator_type == 'bollinger':
                    # Bollinger Bands
                    period = params.get('period', 20)
                    std_dev = params.get('std_dev', 2)
                    column = params.get('column', 'close')
                    
                    result[f"{indicator_name}_middle"] = result[column].rolling(window=period).mean()
                    result[f"{indicator_name}_std"] = result[column].rolling(window=period).std()
                    result[f"{indicator_name}_upper"] = result[f"{indicator_name}_middle"] + (result[f"{indicator_name}_std"] * std_dev)
                    result[f"{indicator_name}_lower"] = result[f"{indicator_name}_middle"] - (result[f"{indicator_name}_std"] * std_dev)
                    
                elif indicator_type == 'atr':
                    # Average True Range
                    period = params.get('period', 14)
                    
                    high_low = result['high'] - result['low']
                    high_close = np.abs(result['high'] - result['close'].shift())
                    low_close = np.abs(result['low'] - result['close'].shift())
                    
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    
                    result[indicator_name] = true_range.rolling(window=period).mean()
                    
                elif indicator_type == 'stoch':
                    # Stochastic Oscillator
                    k_period = params.get('k_period', 14)
                    d_period = params.get('d_period', 3)
                    
                    low_min = result['low'].rolling(window=k_period).min()
                    high_max = result['high'].rolling(window=k_period).max()
                    
                    result[f"{indicator_name}_k"] = 100 * ((result['close'] - low_min) / (high_max - low_min))
                    result[f"{indicator_name}_d"] = result[f"{indicator_name}_k"].rolling(window=d_period).mean()
                    
                elif indicator_type == 'adx':
                    # Average Directional Index
                    period = params.get('period', 14)
                    
                    # Directional Movement
                    plus_dm = result['high'].diff()
                    minus_dm = result['low'].diff()
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm > 0] = 0
                    minus_dm = minus_dm.abs()
                    
                    # True Range calculation for DI normalization
                    tr1 = result['high'] - result['low']
                    tr2 = (result['high'] - result['close'].shift()).abs()
                    tr3 = (result['low'] - result['close'].shift()).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Smoothed DM and TR
                    smoothed_plus_dm = plus_dm.rolling(window=period).sum()
                    smoothed_minus_dm = minus_dm.rolling(window=period).sum()
                    smoothed_tr = tr.rolling(window=period).sum()
                    
                    # Directional Indicators
                    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
                    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
                    
                    # Directional Index
                    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                    
                    # Average Directional Index
                    result[indicator_name] = dx.rolling(window=period).mean()
                    result[f"{indicator_name}_plus_di"] = plus_di
                    result[f"{indicator_name}_minus_di"] = minus_di
                    
                elif indicator_type == 'vol':
                    # Volatility (Standard Deviation)
                    period = params.get('period', 14)
                    column = params.get('column', 'close')
                    result[indicator_name] = result[column].rolling(window=period).std()
                    
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing NaN values and outliers.
        
        Args:
            df: OHLCV data as DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy
        result = df.copy()
        
        # Forward fill NaN values in essential columns
        for col in ['open', 'high', 'low', 'close']:
            if col in result.columns:
                result[col] = result[col].ffill()
                
        # Replace NaN volume with 0
        if 'volume' in result.columns:
            result['volume'] = result['volume'].fillna(0)
            
        # Filter out rows with abnormal extreme values (e.g., prices of 0 or infinity)
        for col in ['open', 'high', 'low', 'close']:
            if col in result.columns:
                result = result[(result[col] > 0) & (result[col] < 1e10)]
                
        return result
        
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize data for machine learning purposes.
        
        Args:
            df: Data as DataFrame
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy
        result = df.copy()
        
        # Select numeric columns
        numeric_cols = result.select_dtypes(include=np.number).columns
        
        # Apply normalization
        if method == 'minmax':
            for col in numeric_cols:
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
                    
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = result[col].mean()
                std_val = result[col].std()
                if std_val > 0:
                    result[col] = (result[col] - mean_val) / std_val
                    
        return result
        
    def detect_anomalies(self, df: pd.DataFrame, window: int = 20, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in price data.
        
        Args:
            df: OHLCV data as DataFrame
            window: Window size for moving statistics
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            pd.DataFrame: DataFrame with anomaly flags
        """
        if df.empty:
            return df
            
        # Make a copy
        result = df.copy()
        
        # Calculate rolling mean and standard deviation
        result['rolling_mean'] = result['close'].rolling(window=window).mean()
        result['rolling_std'] = result['close'].rolling(window=window).std()
        
        # Calculate Z-score
        result['z_score'] = (result['close'] - result['rolling_mean']) / result['rolling_std']
        
        # Flag anomalies
        result['anomaly'] = result['z_score'].abs() > threshold
        
        return result
        
    def extract_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features for machine learning.
        
        Args:
            df: Data as DataFrame
            feature_config: Feature extraction configuration
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        if df.empty:
            return df
            
        # Make a copy
        result = df.copy()
        
        # Price-based features
        if feature_config.get('price_features', True):
            # Log returns
            result['log_return'] = np.log(result['close'] / result['close'].shift(1))
            
            # Price changes
            result['price_change'] = result['close'].pct_change()
            
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                result[f'ma_{period}'] = result['close'].rolling(window=period).mean()
                
            # Price velocity (rate of change)
            for period in [5, 10, 20]:
                result[f'roc_{period}'] = result['close'].pct_change(periods=period) * 100
                
            # Price acceleration
            for period in [5, 10]:
                result[f'acc_{period}'] = result[f'roc_{period}'] - result[f'roc_{period}'].shift(1)
                
            # Bollinger Bands
            bb_period = 20
            result['bb_middle'] = result['close'].rolling(window=bb_period).mean()
            result['bb_std'] = result['close'].rolling(window=bb_period).std()
            result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * 2)
            result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * 2)
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
            result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
            
        # Volume-based features
        if feature_config.get('volume_features', True):
            # Volume changes
            result['volume_change'] = result['volume'].pct_change()
            
            # Volume moving averages
            for period in [5, 10, 20]:
                result[f'volume_ma_{period}'] = result['volume'].rolling(window=period).mean()
                
            # Volume relative to moving average
            result['volume_ratio'] = result['volume'] / result['volume_ma_10']
            
            # On-balance volume
            result['obv'] = np.where(
                result['close'] > result['close'].shift(1),
                result['volume'],
                np.where(
                    result['close'] < result['close'].shift(1),
                    -result['volume'],
                    0
                )
            ).cumsum()
            
        # Volatility-based features
        if feature_config.get('volatility_features', True):
            # Daily high-low range
            result['daily_range'] = (result['high'] - result['low']) / result['low'] * 100
            
            # Average true range
            high_low = result['high'] - result['low']
            high_close = np.abs(result['high'] - result['close'].shift())
            low_close = np.abs(result['low'] - result['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result['atr_14'] = true_range.rolling(window=14).mean()
            
            # Historical volatility
            for period in [5, 20]:
                result[f'volatility_{period}'] = result['log_return'].rolling(window=period).std() * np.sqrt(period)
                
        # Trend-based features
        if feature_config.get('trend_features', True):
            # ADX - Average Directional Index
            period = 14
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = abs(minus_dm.where(minus_dm < 0, 0))
            tr = pd.concat([
                result['high'] - result['low'],
                abs(result['high'] - result['close'].shift(1)),
                abs(result['low'] - result['close'].shift(1))
            ], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
            minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
            adx_raw = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=period).mean()
            result['adx'] = adx_raw
            
            # Moving average crossovers
            result['ma_cross_5_20'] = np.where(
                result['ma_5'] > result['ma_20'],
                1,
                np.where(
                    result['ma_5'] < result['ma_20'],
                    -1,
                    0
                )
            )
            
        # Momentum-based features
        if feature_config.get('momentum_features', True):
            # RSI
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            result['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            fast = 12
            slow = 26
            signal = 9
            
            result['macd_line'] = result['close'].ewm(span=fast, adjust=False).mean() - \
                                  result['close'].ewm(span=slow, adjust=False).mean()
            result['macd_signal'] = result['macd_line'].ewm(span=signal, adjust=False).mean()
            result['macd_hist'] = result['macd_line'] - result['macd_signal']
            
            # Stochastic oscillator
            k_period = 14
            d_period = 3
            
            low_min = result['low'].rolling(window=k_period).min()
            high_max = result['high'].rolling(window=k_period).max()
            
            result['stoch_k'] = 100 * ((result['close'] - low_min) / (high_max - low_min))
            result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
            
        # Time-based features
        if feature_config.get('time_features', True) and isinstance(result.index, pd.DatetimeIndex):
            # Day of week, hour, etc.
            result['day_of_week'] = result.index.dayofweek
            result['hour_of_day'] = result.index.hour
            result['month'] = result.index.month
            result['is_weekend'] = result.index.dayofweek >= 5
            
            # Time since specific events
            # e.g., time since last price movement of 5%
            big_move = abs(result['price_change']) > 0.05
            result['days_since_big_move'] = big_move.cumsum()
            result['days_since_big_move'] = result.groupby('days_since_big_move').cumcount()
            
        return result
        
    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        window_size: int = 10,
        prediction_horizon: int = 1,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning models.
        
        Args:
            df: Data as DataFrame
            target_column: Column to predict
            feature_columns: Columns to use as features
            window_size: Number of time steps to use for prediction
            prediction_horizon: How many steps ahead to predict
            train_split: Fraction of data to use for training
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if df.empty:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        # Select features and target
        data = df[feature_columns + [target_column]].copy()
        
        # Drop rows with NaN values
        data = data.dropna()
        
        if len(data) <= window_size + prediction_horizon:
            logger.warning("Not enough data for ML preparation")
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        # Create sequences
        X, y = [], []
        for i in range(len(data) - window_size - prediction_horizon + 1):
            X.append(data[feature_columns].iloc[i:i+window_size].values)
            y.append(data[target_column].iloc[i+window_size+prediction_horizon-1])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
        
    def cache_data(self, key: str, data: Any, ttl: int = None) -> bool:
        """
        Cache data for later use.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (overrides instance TTL if provided)
            
        Returns:
            bool: True if cached successfully, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            # Create key hash
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            # Set expiration time
            expires_at = datetime.now() + timedelta(seconds=ttl or self.cache_ttl)
            
            # Create cache object
            cache_obj = {
                'data': data,
                'expires_at': expires_at.timestamp(),
                'created_at': datetime.now().timestamp(),
                'key': key
            }
            
            # Create cache file path
            cache_file = os.path.join(self.cache_dir, f"{key_hash}.cache")
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_obj, f)
                
            return True
            
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            return False
            
    def get_cached_data(self, key: str) -> Tuple[bool, Any]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            tuple: (success, data)
        """
        if not self.cache_enabled:
            return False, None
            
        try:
            # Create key hash
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            # Create cache file path
            cache_file = os.path.join(self.cache_dir, f"{key_hash}.cache")
            
            # Check if file exists
            if not os.path.exists(cache_file):
                return False, None
                
            # Load from file
            with open(cache_file, 'rb') as f:
                cache_obj = pickle.load(f)
                
            # Check if expired
            if datetime.now().timestamp() > cache_obj['expires_at']:
                # Remove expired cache
                os.remove(cache_file)
                return False, None
                
            return True, cache_obj['data']
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return False, None
            
    def clear_cache(self, key: str = None) -> bool:
        """
        Clear cache.
        
        Args:
            key: Specific cache key to clear (None for all)
            
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            if key:
                # Clear specific key
                key_hash = hashlib.md5(key.encode()).hexdigest()
                cache_file = os.path.join(self.cache_dir, f"{key_hash}.cache")
                
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    
            else:
                # Clear all cache
                cache_files = glob.glob(os.path.join(self.cache_dir, "*.cache"))
                for file in cache_files:
                    os.remove(file)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
            
    def clear_expired_cache(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            int: Number of cleared entries
        """
        if not self.cache_enabled:
            return 0
            
        try:
            # Get all cache files
            cache_files = glob.glob(os.path.join(self.cache_dir, "*.cache"))
            cleared_count = 0
            
            for file in cache_files:
                try:
                    # Load cache object
                    with open(file, 'rb') as f:
                        cache_obj = pickle.load(f)
                        
                    # Check if expired
                    if datetime.now().timestamp() > cache_obj['expires_at']:
                        os.remove(file)
                        cleared_count += 1
                except:
                    # If any error occurs with this file, remove it
                    os.remove(file)
                    cleared_count += 1
                    
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {str(e)}")
            return 0
            
    def export_to_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """
        Export data to CSV file.
        
        Args:
            df: Data as DataFrame
            filepath: Output file path
            
        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Export to CSV
            df.to_csv(filepath)
            logger.info(f"Data exported to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return False
            
    def export_to_json(self, df: pd.DataFrame, filepath: str) -> bool:
        """
        Export data to JSON file.
        
        Args:
            df: Data as DataFrame
            filepath: Output file path
            
        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Handle datetime indices
            if isinstance(df.index, pd.DatetimeIndex):
                df_copy = df.copy()
                df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
                df_copy.to_json(filepath, orient='index', date_format='iso')
            else:
                df.to_json(filepath, orient='index', date_format='iso')
                
            logger.info(f"Data exported to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data to JSON: {str(e)}")
            return False