"""
Historical Data Provider
=====================
Provides historical price data for backtesting and analysis.
"""

import os
import glob
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from loguru import logger

from cryptobot.data.database import DatabaseManager
from cryptobot.utils.helpers import timeframe_to_seconds, parse_timeframe


class HistoricalDataProvider:
    """
    Provider for historical price data from various sources.
    """
    
    def __init__(
        self,
        source: str = 'csv',
        data_dir: str = 'data',
        api_key: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the historical data provider.
        
        Args:
            source: Data source ('csv', 'api', 'database')
            data_dir: Directory for CSV data
            api_key: API key for external data sources
            db_manager: Database manager instance
        """
        self.source = source
        self.data_dir = data_dir
        self.api_key = api_key
        self.db_manager = db_manager
        
        # Ensure data directory exists
        if self.source == 'csv' and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        logger.info(f"Initialized historical data provider with source: {source}")
        
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date (default: current time)
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
            
        if self.source == 'csv':
            return await self._get_data_from_csv(symbol, timeframe, start_date, end_date)
        elif self.source == 'api':
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
        try:
            start_date = pd.Timestamp(start_date).tz_localize('UTC') if start_date.tzinfo is None else pd.Timestamp(start_date).tz_convert('UTC')
            end_date = pd.Timestamp(end_date).tz_localize('UTC') if end_date.tzinfo is None else pd.Timestamp(end_date).tz_convert('UTC')

            interval_seconds = timeframe_to_seconds(timeframe)
            time_diff = end_date - start_date
            time_diff_seconds = time_diff.total_seconds()
            candles_limit = 2000
            total_candles_needed = int(time_diff_seconds / interval_seconds) + 1
            num_requests = max(1, (total_candles_needed + candles_limit - 1) // candles_limit)
            logger.info(f"Fetching data for {symbol} {timeframe}: {total_candles_needed} candles needed, {num_requests} API requests")

            base_url = "https://min-api.cryptocompare.com/data"
            if interval_seconds < 3600:
                endpoint = "histominute"
                aggregate = int(interval_seconds / 60)
            elif interval_seconds < 86400:
                endpoint = "histohour"
                aggregate = int(interval_seconds / 3600)
            else:
                endpoint = "histoday"
                aggregate = int(interval_seconds / 86400)

            all_data = []
            current_end = end_date
            retries_per_chunk = 3
            for i in range(num_requests):
                current_start = max(start_date, current_end - timedelta(seconds=candles_limit * interval_seconds))
                api_url = (f"{base_url}/{endpoint}?fsym={symbol.split('/')[0]}&tsym={symbol.split('/')[1]}"
                        f"&limit={candles_limit}&aggregate={aggregate}&e=CCCAGG")
                if self.api_key:
                    api_url += f"&api_key={self.api_key}"
                api_url += f"&toTs={int(current_end.timestamp())}"

                logger.debug(f"Fetching API chunk {i+1}/{num_requests} for {symbol} {timeframe} from {current_start} to {current_end}")

                for attempt in range(retries_per_chunk):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url) as response:
                            if response.status != 200:
                                logger.warning(f"API error on attempt {attempt+1}/{retries_per_chunk}: {response.status} - {await response.text()}")
                                if attempt == retries_per_chunk - 1:
                                    logger.error(f"Failed to fetch API chunk {i+1} after {retries_per_chunk} attempts")
                                await asyncio.sleep(1)
                                continue
                            
                            data = await response.json()
                            
                            if 'Response' in data and data['Response'] == 'Error':
                                logger.warning(f"API error on attempt {attempt+1}/{retries_per_chunk}: {data['Message']}")
                                if attempt == retries_per_chunk - 1:
                                    logger.error(f"Failed API chunk {i+1} after {retries_per_chunk} attempts: {data['Message']}")
                                await asyncio.sleep(1)
                                continue
                                
                            if 'Data' not in data or not data['Data']:
                                logger.warning(f"No data returned from API for {symbol} {timeframe} in chunk {i+1}")
                                break
                                
                            ohlcv_data = data['Data']
                            logger.debug(f"Fetched {len(ohlcv_data)} candles in chunk {i+1} for {symbol} {timeframe}")
                            all_data.extend(ohlcv_data)
                            break
                
                current_end = current_start - timedelta(seconds=1)
                if current_end <= start_date:
                    break
            
            if not all_data:
                logger.warning(f"No data returned from API for {symbol} {timeframe} after all chunks")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df.loc[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            logger.debug(f"Processed {len(df)} rows after deduplication for {symbol} {timeframe} from {df.index.min()} to {df.index.max()}")

            column_mapping = {
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in API data for {symbol} {timeframe}")
                    return pd.DataFrame()
                    
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} from API")
            
            return df[required_columns]
            
        except Exception as e:
            logger.error(f"Error fetching API data for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
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