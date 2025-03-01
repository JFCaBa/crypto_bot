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
                    
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    if df['timestamp'].dtype == 'int64':
                        # Convert milliseconds to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    else:
                        # Try to parse as datetime string
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                dfs.append(df)
                
            if not dfs:
                return pd.DataFrame()
                
            # Combine all dataframes
            data = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates and sort by timestamp
            data = data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Set timestamp as index
            data.set_index('timestamp', inplace=True)
            
            # Filter by date range
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
                    
            return data
            
        except Exception as e:
            logger.error(f"Error reading CSV data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
            
    async def _get_data_from_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data from external API.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Use CryptoCompare API as an example
            # In a real implementation, you might use different APIs or sources
            
            # Convert timeframe to seconds
            interval_seconds = timeframe_to_seconds(timeframe)
            
            # Calculate limit based on date range and timeframe
            time_diff = end_date - start_date
            time_diff_seconds = time_diff.total_seconds()
            limit = min(2000, int(time_diff_seconds / interval_seconds) + 1)  # API limit is 2000
            
            # CryptoCompare API parameters
            base_url = "https://min-api.cryptocompare.com/data"
            
            # Determine API endpoint based on timeframe
            if interval_seconds < 3600:  # Less than 1 hour
                endpoint = "histominute"
                minute_param = int(interval_seconds / 60)
                api_url = f"{base_url}/{endpoint}?fsym={symbol.split('/')[0]}&tsym={symbol.split('/')[1]}&limit={limit}&aggregate={minute_param}&e=CCCAGG"
            elif interval_seconds < 86400:  # Less than 1 day
                endpoint = "histohour"
                hour_param = int(interval_seconds / 3600)
                api_url = f"{base_url}/{endpoint}?fsym={symbol.split('/')[0]}&tsym={symbol.split('/')[1]}&limit={limit}&aggregate={hour_param}&e=CCCAGG"
            else:  # Daily or greater
                endpoint = "histoday"
                day_param = int(interval_seconds / 86400)
                api_url = f"{base_url}/{endpoint}?fsym={symbol.split('/')[0]}&tsym={symbol.split('/')[1]}&limit={limit}&aggregate={day_param}&e=CCCAGG"
                
            # Add API key if provided
            if self.api_key:
                api_url += f"&api_key={self.api_key}"
                
            # Add start time in Unix timestamp format
            start_timestamp = int(start_date.timestamp())
            api_url += f"&toTs={int(end_date.timestamp())}"
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status != 200:
                        logger.error(f"API error: {response.status} - {await response.text()}")
                        return pd.DataFrame()
                        
                    data = await response.json()
                    
                    if 'Response' in data and data['Response'] == 'Error':
                        logger.error(f"API error: {data['Message']}")
                        return pd.DataFrame()
                        
                    # Process API response
                    if 'Data' not in data or not data['Data']:
                        logger.warning(f"No data returned from API for {symbol} {timeframe}")
                        return pd.DataFrame()
                        
                    ohlcv_data = data['Data']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv_data)
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Rename columns
                    column_mapping = {
                        'time': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volumefrom': 'volume'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Select required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_columns:
                        if col not in df.columns:
                            logger.warning(f"Missing column {col} in API data for {symbol} {timeframe}")
                            return pd.DataFrame()
                            
                    return df[required_columns]
                    
        except Exception as e:
            logger.error(f"Error fetching API data for {symbol} {timeframe}: {str(e)}")
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