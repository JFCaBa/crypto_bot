"""
Historical Data Provider
=====================
Provides historical price data for backtesting and analysis using CCXT.
"""

import os
import glob
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import ccxt.async_support as ccxt_async
from loguru import logger

from cryptobot.data.database import DatabaseManager
from cryptobot.utils.helpers import timeframe_to_seconds, parse_timeframe


class HistoricalDataProvider:
    """
    Provider for historical price data from various sources including CCXT-supported exchanges.
    """
    
    def __init__(
        self,
        source: str = 'csv',
        data_dir: str = 'data',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        exchange_id: str = 'mexc',
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the historical data provider.
        
        Args:
            source: Data source ('csv', 'api', 'database')
            data_dir: Directory for CSV data
            api_key: API key for exchange
            api_secret: API secret for exchange
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase', 'kraken')
            db_manager: Database manager instance
        """
        self.source = source
        self.data_dir = data_dir
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_id = exchange_id
        self.db_manager = db_manager
        self.exchange = None  # Will be initialized when needed
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        logger.info(f"Initialized historical data provider with source: {source}, exchange: {exchange_id}")
        
    async def _initialize_exchange(self):
        """Initialize CCXT exchange instance if not already done."""
        if self.exchange is not None:
            return

        try:
            # Get the exchange class dynamically
            exchange_class = getattr(ccxt_async, self.exchange_id)
            
            # Create exchange instance with authentication if provided
            config = {
                'enableRateLimit': True,
                'timeout': 30000,
            }
            
            if self.api_key and self.api_secret:
                config['apiKey'] = self.api_key
                config['secret'] = self.api_secret

            logger.debug(f"API Key: {self.api_key}, API Secret: {self.api_secret}")
                
            self.exchange = exchange_class(config)
            logger.info(f"Initialized {self.exchange_id} exchange for historical data")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_id}: {str(e)}")
            raise

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data, preferring CSV if available.
        """
        if end_date is None:
            end_date = datetime.now()

        # Try CSV first
        csv_data = await self._get_data_from_csv(symbol, timeframe, start_date, end_date)
        if not csv_data.empty:
            logger.info(f"Loaded {len(csv_data)} rows from CSV for {symbol} {timeframe}")
            return csv_data

        # Fall back to API if CSV not available or source is explicitly 'api'
        if self.source == 'api':
            logger.info(f"No CSV data found, fetching from API for {symbol} {timeframe}")
            return await self._get_data_from_api(symbol, timeframe, start_date, end_date)
        elif self.source == 'database':
            return await self._get_data_from_database(symbol, timeframe, start_date, end_date)
        else:
            logger.error(f"Unsupported data source: {self.source}")
            return pd.DataFrame()
            
    async def _get_data_from_csv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data from CSV files.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Ensure start_date and end_date are timezone-aware UTC
            start_date = pd.Timestamp(start_date).tz_localize('UTC') if start_date.tzinfo is None else pd.Timestamp(start_date).tz_convert('UTC')
            end_date = pd.Timestamp(end_date).tz_localize('UTC') if end_date.tzinfo is None else pd.Timestamp(end_date).tz_convert('UTC')

            # Format symbol for filename
            symbol_filename = symbol.replace('/', '')
            
            # Find CSV files for this symbol and timeframe
            pattern = os.path.join(self.data_dir, f"{symbol_filename}_{timeframe}_*.csv")
            files = glob.glob(pattern)
            
            if not files:
                logger.warning(f"No CSV files found for {symbol} {timeframe}")
                return pd.DataFrame()
                
            # Read and combine all matching files
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                
                # Ensure timestamp column exists
                if 'timestamp' not in df.columns and 'time' in df.columns:
                    df['timestamp'] = df['time']
                    
                # Convert timestamp to datetime, timezone-aware UTC
                if 'timestamp' in df.columns:
                    if df['timestamp'].dtype == 'int64':
                        # Convert milliseconds to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    else:
                        # Try to parse as datetime string
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                        
                dfs.append(df)
                
            if not dfs:
                return pd.DataFrame()
                
            # Combine all dataframes
            data = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates and sort by timestamp
            data = data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Set timestamp as index
            data.set_index('timestamp', inplace=True)
            
            # Ensure index is timezone-aware UTC
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')
            
            # Filter by date range
            logger.debug(f"Filtering data for {symbol} {timeframe}: start_date={start_date}, end_date={end_date}")
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Ensure columns are properly named
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Rename columns
            data = data.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} in data for {symbol} {timeframe}")
                    return pd.DataFrame()
                    
            logger.debug(f"Loaded {len(data)} rows from CSV for {symbol} {timeframe} after filtering")
            return data
            
        except Exception as e:
            logger.error(f"Error reading CSV data for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    async def get_live_data(self, symbols: List[str], timeframe: str):
        """
        Simulate live data streams using historical data.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for data
        
        Yields:
            tuple: (timestamp, symbol_data)
        """
        for symbol in symbols:
            df = await self.get_historical_data(symbol, timeframe)
            for idx, row in df.iterrows():
                yield idx, {symbol: pd.DataFrame([row])}
                await asyncio.sleep(1)  # Simulate real-time delay
            
    async def _get_data_from_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data from exchange API using CCXT.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Initialize exchange if not already done
            await self._initialize_exchange()
            
            # Ensure start_date and end_date are timezone-aware UTC
            start_date = pd.Timestamp(start_date).tz_localize('UTC') if start_date.tzinfo is None else pd.Timestamp(start_date).tz_convert('UTC')
            end_date = pd.Timestamp(end_date).tz_localize('UTC') if end_date.tzinfo is None else pd.Timestamp(end_date).tz_convert('UTC')

            # Convert timestamps to milliseconds (CCXT standard)
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            # Check if exchange supports the timeframe
            if not self.exchange.has['fetchOHLCV']:
                logger.error(f"Exchange {self.exchange_id} does not support OHLCV data")
                return pd.DataFrame()
                
            # Check if timeframe is supported
            if timeframe not in self.exchange.timeframes:
                logger.warning(f"Timeframe {timeframe} not supported by {self.exchange_id}. " +
                              f"Available timeframes: {list(self.exchange.timeframes.keys())}")
                
                # Try to find a suitable alternative
                alternative = self._find_alternative_timeframe(timeframe)
                if alternative:
                    logger.info(f"Using alternative timeframe {alternative} instead of {timeframe}")
                    timeframe = alternative
                else:
                    logger.error(f"No suitable alternative timeframe found for {timeframe}")
                    return pd.DataFrame()
            
            # Some exchanges have limits on how much data can be fetched at once
            # Implement chunking to fetch data in smaller batches
            all_ohlcv = []
            current_since = since
            fetch_attempt = 0
            max_attempts = 5
            
            logger.info(f"Fetching OHLCV data for {symbol} {timeframe} from {start_date} to {end_date}")
            
            # Choose appropriate limit based on exchange
            # Default to 1000, but some exchanges have lower limits
            limit = 1000
            if hasattr(self.exchange, 'rateLimit'):
                # Adjust based on rate limits to avoid hitting them
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            while current_since < until and fetch_attempt < max_attempts:
                try:
                    # Fetch OHLCV data
                    ohlcv = await self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_since,
                        limit=limit
                    )
                    
                    if not ohlcv or len(ohlcv) == 0:
                        logger.warning(f"No data returned for {symbol} {timeframe} since {current_since}")
                        break
                    
                    # Add fetched data to our collection
                    all_ohlcv.extend(ohlcv)
                    
                    # Update since for next iteration
                    # Use the timestamp of the last candle + 1ms to avoid duplication
                    last_timestamp = ohlcv[-1][0] + 1
                    
                    # If the last timestamp is the same as current_since, we're not making progress
                    if last_timestamp <= current_since:
                        logger.warning(f"No progress in fetching data for {symbol} {timeframe}, breaking the loop")
                        break
                        
                    current_since = last_timestamp
                    
                    # Wait to respect rate limits
                    await asyncio.sleep(self.exchange.rateLimit / 1000)
                    
                    # Reset fetch attempt counter on successful fetch
                    fetch_attempt = 0
                    
                    # If we fetched fewer candles than the limit, we've likely reached the end
                    if len(ohlcv) < limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error fetching OHLCV data for {symbol} {timeframe}: {str(e)}")
                    fetch_attempt += 1
                    if fetch_attempt >= max_attempts:
                        logger.error(f"Max fetch attempts reached for {symbol} {timeframe}")
                        break
                    # Exponential backoff
                    await asyncio.sleep(2 ** fetch_attempt)
            
            if not all_ohlcv:
                logger.warning(f"No OHLCV data collected for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert OHLCV data to DataFrame
            # CCXT OHLCV format: [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and sort by timestamp
            df = df[~df.index.duplicated(keep='first')].sort_index()
            
            # Filter to the requested date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} using CCXT")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching CCXT data for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _find_alternative_timeframe(self, timeframe: str) -> Optional[str]:
        """
        Find an alternative timeframe if the requested one is not supported.
        
        Args:
            timeframe: Requested timeframe
            
        Returns:
            Optional[str]: Alternative timeframe or None if not found
        """
        # Common timeframe mappings
        mappings = {
            '1m': ['1m', '1min', 'M1', '1'],
            '5m': ['5m', '5min', 'M5', '5'],
            '15m': ['15m', '15min', 'M15', '15'],
            '30m': ['30m', '30min', 'M30', '30'],
            '1h': ['1h', '60m', '60min', 'H1', '60'],
            '2h': ['2h', '120m', '120min', 'H2', '120'],
            '4h': ['4h', '240m', '240min', 'H4', '240'],
            '1d': ['1d', '1day', 'D1', 'D', '1440'],
            '1w': ['1w', '1week', 'W1', 'W'],
            '1M': ['1M', '1month', 'M', 'MN']
        }
        
        # Extract all available timeframes from the exchange
        if not hasattr(self.exchange, 'timeframes') or not self.exchange.timeframes:
            return None
            
        available_timeframes = list(self.exchange.timeframes.keys())
        
        # First check for direct match
        if timeframe in available_timeframes:
            return timeframe
            
        # Find alternatives based on common mappings
        for standard, alternatives in mappings.items():
            if timeframe in alternatives:
                # Find a match from alternatives
                for alt in alternatives:
                    if alt in available_timeframes:
                        return alt
                        
                # If not found, check for standard format
                if standard in available_timeframes:
                    return standard
                    
        # If no match found in mappings, try to find closest matching timeframe
        # This is more complex and requires parsing the timeframe string
        try:
            import re
            # Extract number and unit from timeframe
            match = re.match(r'(\d+)([mhdwMy])', timeframe)
            if match:
                value, unit = match.groups()
                value = int(value)
                
                # Find all timeframes with the same unit
                same_unit = [tf for tf in available_timeframes if tf.endswith(unit)]
                if same_unit:
                    # Find the closest value
                    closest = min(same_unit, key=lambda x: abs(int(re.match(r'(\d+)', x).group(1)) - value))
                    return closest
        except:
            pass
            
        return None
            
    async def _get_data_from_database(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            if not self.db_manager:
                logger.error("Database manager not initialized")
                return pd.DataFrame()
                
            # Query database for OHLCV data
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = %s AND timeframe = %s AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            
            params = (symbol, timeframe, start_date, end_date)
            result = await self.db_manager.execute_query(query, params)
            
            if not result:
                logger.warning(f"No data found in database for {symbol} {timeframe}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching database data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
            
    async def save_data_to_csv(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        filename: str = None
    ) -> bool:
        """
        Save data to CSV file.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            data: OHLCV data
            filename: Custom filename (optional)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Format symbol for filename
            symbol_filename = symbol.replace('/', '')
            
            # Generate filename if not provided
            if not filename:
                current_date = datetime.now().strftime('%Y%m%d')
                filename = f"{symbol_filename}_{timeframe}_{current_date}.csv"
                
            # Ensure data directory exists
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                
            # Full path to file
            filepath = os.path.join(self.data_dir, filename)
            
            # Reset index to include timestamp column
            data_copy = data.copy()
            if data_copy.index.name == 'timestamp':
                data_copy = data_copy.reset_index()
                
            # Save to CSV
            data_copy.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
            
    async def download_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = None,
        save_to_csv: bool = True
    ) -> pd.DataFrame:
        """
        Download historical data from API and optionally save to CSV.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date (default: current time)
            save_to_csv: Whether to save data to CSV
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Get data from API
        data = await self._get_data_from_api(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            logger.warning(f"No data downloaded for {symbol} {timeframe}")
            return data
            
        # Save to CSV if requested
        if save_to_csv:
            await self.save_data_to_csv(symbol, timeframe, data)
            
        return data
    
    async def close(self):
        """Close the CCXT exchange connection."""
        if self.exchange:
            await self.exchange.close()
            logger.info(f"Closed connection to {self.exchange_id} exchange")
        
    async def update_historical_data(
        self,
        symbol: str,
        timeframe: str,
        days_to_update: int = 7
    ) -> pd.DataFrame:
        """
        Update historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            days_to_update: Number of days to update
            
        Returns:
            pd.DataFrame: Updated OHLCV data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_update)
            
            # Get existing data
            existing_data = await self._get_data_from_csv(symbol, timeframe, start_date, end_date)
            
            # Get new data from API
            new_data = await self._get_data_from_api(symbol, timeframe, start_date, end_date)
            
            if new_data.empty:
                logger.warning(f"No new data available for {symbol} {timeframe}")
                return existing_data
                
            # Combine existing and new data
            if existing_data.empty:
                combined_data = new_data
            else:
                combined_data = pd.concat([existing_data, new_data])
                
                # Remove duplicates and sort
                combined_data = combined_data.loc[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                
            # Save updated data
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol.replace('/', '')}_{timeframe}_{current_date}.csv"
            await self.save_data_to_csv(symbol, timeframe, combined_data, filename)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error updating historical data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()