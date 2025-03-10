#!/usr/bin/env python3
"""
Historical Data Downloader
=========================
Script to download historical cryptocurrency data for backtesting.
"""

import os
import asyncio
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from loguru import logger

from cryptobot.data.historical import HistoricalDataProvider
from cryptobot.utils.helpers import setup_logger, timeframe_to_seconds
from cryptobot.config.settings import load_config


async def download_data(
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime,
    end_date: Optional[datetime] = None,
    data_dir: str = 'data',
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    exchange_id: str = 'mexc'  # Add exchange_id parameter with default
):
    """
    Download historical data for specified symbols and timeframes.
    
    Args:
        symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframes: List of timeframes (e.g., ['1h', '4h', '1d'])
        start_date: Start date
        end_date: End date (default: current date)
        data_dir: Directory to save data
        api_key: API key for external data sources
        api_secret: API secret for external data sources
        exchange_id: Exchange identifier (e.g., 'binance', 'mexc')
    """
    # Set end_date to current date if not provided
    if end_date is None:
        end_date = datetime.now()
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    
    # Initialize data provider with API source for downloading
    data_provider = HistoricalDataProvider(
        source='api',
        data_dir=data_dir,
        api_key=api_key,
        api_secret=api_secret,
        exchange_id=exchange_id  # Pass the exchange_id
    )
    
    logger.info(f"Starting download for {len(symbols)} symbols and {len(timeframes)} timeframes")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Using exchange: {exchange_id} with API key: {'*****' + api_key[-4:] if api_key else 'None'}")
    
    # Download data for each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Downloading {symbol} {timeframe} data")
            
            try:
                # Download data
                data = await data_provider.download_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    save_to_csv=True
                )
                
                if data.empty:
                    logger.warning(f"No data downloaded for {symbol} {timeframe}")
                else:
                    logger.info(f"Successfully downloaded {len(data)} candles for {symbol} {timeframe}")
                    
                # Sleep to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} {timeframe}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info("Download completed")


async def batch_download(
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime,
    end_date: datetime,
    batch_days: int = 90,
    **kwargs
):
    """
    Download data in batches to avoid rate limits and memory issues.
    
    Args:
        symbols: List of trading pair symbols
        timeframes: List of timeframes
        start_date: Start date
        end_date: End date
        batch_days: Number of days per batch
        **kwargs: Additional arguments for download_data
    """
    current_start = start_date
    
    while current_start < end_date:
        # Calculate batch end date
        batch_end = current_start + timedelta(days=batch_days)
        if batch_end > end_date:
            batch_end = end_date
            
        logger.info(f"Downloading batch: {current_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
        
        # Download this batch
        await download_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=current_start,
            end_date=batch_end,
            **kwargs
        )
        
        # Move to next batch
        current_start = batch_end + timedelta(days=1)
        
        # Sleep between batches to avoid rate limits
        await asyncio.sleep(5)


async def download_strategy_data(
    config_path: str,
    strategy_id: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    timeframes: Optional[List[str]] = None,
    **kwargs
):
    """
    Download data specifically for a strategy from the configuration.
    
    Args:
        config_path: Path to configuration file
        strategy_id: Strategy identifier
        start_date: Start date
        end_date: End date
        timeframes: Override timeframes (optional)
        **kwargs: Additional arguments for download_data
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get strategy configuration
    strategy_config = config.get('strategies', {}).get(strategy_id)
    
    if not strategy_config:
        logger.error(f"Strategy {strategy_id} not found in configuration")
        return
    
    # Get exchanges from configuration where the exchange enabled is True
    exchanges = {k: v for k, v in config.get('exchanges', {}).items() if v.get('enabled', False)}
    if not exchanges:
        logger.error("No enabled exchange found in configuration")
        return
    
    # Select the first enabled exchange (or implement logic to choose a specific one)
    exchange_id = next(iter(exchanges))  # Get the first exchange key (e.g., 'mexc')
    exchange = exchanges[exchange_id]
    logger.info(f"Selected exchange: {exchange_id}, config: {exchange}")
    
    # Get symbols and timeframes from strategy configuration
    symbols = strategy_config.get('symbols', [])
    strategy_timeframes = strategy_config.get('timeframes', ['1h'])
    
    # Use provided timeframes if specified
    if timeframes:
        strategy_timeframes = timeframes
    
    if not symbols:
        logger.error("No symbols specified for strategy")
        return
    
    logger.info(f"Downloading data for strategy {strategy_id}: {symbols}, timeframes: {strategy_timeframes}")    
   
    # Access api_key and api_secret from the selected exchange
    api_key = exchange.get('api_key')
    api_secret = exchange.get('api_secret')
    logger.debug(f"Using API Key: {api_key}, API Secret: {api_secret}")

    if not api_key or not api_secret:
        logger.error(f"API key or secret not found for exchange {exchange_id}")
        return

    # Download data - explicitly pass the exchange_id
    await download_data(
        symbols=symbols,
        timeframes=strategy_timeframes,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        api_secret=api_secret,
        exchange_id=exchange_id,  # Pass the exchange_id
        **kwargs
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Download historical cryptocurrency data')
    
    parser.add_argument('--symbols', '-s', type=str, default='BTC/USDT,ETH/USDT,BNB/USDT',
                        help='Comma-separated list of symbols')
    parser.add_argument('--timeframes', '-t', type=str, default='1h,4h,1d',
                        help='Comma-separated list of timeframes')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to save data')
    parser.add_argument('--batch', action='store_true',
                        help='Download in batches to avoid rate limits')
    parser.add_argument('--batch-days', type=int, default=90,
                        help='Number of days per batch')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Download data for specific strategy from config')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logger(args.log_level)
    
    # Parse date strings to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Download data
    if args.strategy:
        # Download data for specific strategy
        await download_strategy_data(
            config_path=args.config,
            strategy_id=args.strategy,
            start_date=start_date,
            end_date=end_date,
            data_dir=args.data_dir
        )
    else:
        # Parse symbols and timeframes
        symbols = args.symbols.split(',')
        timeframes = args.timeframes.split(',')
        
        logger.info(f"Preparing to download data for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Check if we should download in batches
        if args.batch:
            await batch_download(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                batch_days=args.batch_days,
                data_dir=args.data_dir
            )
        else:
            await download_data(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                data_dir=args.data_dir
            )


if __name__ == "__main__":
    # Create event loop and run
    asyncio.run(main())