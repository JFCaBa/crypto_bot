"""
Database Manager
==============
Manages database connections and operations for the trading bot.
"""

import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

import asyncpg
from loguru import logger


class DatabaseManager:
    """
    Manages database connections and operations.
    """
    
    def __init__(
        self,
        url: str = "localhost",
        port: int = 5432,
        username: str = "postgres",
        password: str = None,
        database: str = "cryptobot",
        pool_size: int = 10
    ):
        """
        Initialize database manager.
        
        Args:
            url: Database URL
            port: Database port
            username: Database username
            password: Database password
            database: Database name
            pool_size: Connection pool size
        """
        self.url = url
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.pool_size = pool_size
        
        # Connection pool
        self.pool = None
        
        logger.info(f"Database manager initialized for {username}@{url}:{port}/{database}")
        
    async def connect(self) -> bool:
        """
        Connect to database.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.url,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database,
                min_size=1,
                max_size=self.pool_size
            )
            
            logger.info("Connected to database")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            traceback.print_exc()
            return False
            
    async def disconnect(self) -> bool:
        """
        Disconnect from database.
        
        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
                
            logger.info("Disconnected from database")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from database: {str(e)}")
            return False
            
    async def execute_query(
        self,
        query: str,
        params: Tuple = None,
        fetch: bool = True
    ) -> Union[List[Dict[str, Any]], int]:
        """
        Execute SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            list or int: Query results or row count
        """
        if not self.pool:
            logger.error("Database not connected")
            return [] if fetch else 0
            
        try:
            async with self.pool.acquire() as connection:
                if fetch:
                    result = await connection.fetch(query, *params) if params else await connection.fetch(query)
                    return [dict(row) for row in result]
                else:
                    result = await connection.execute(query, *params) if params else await connection.execute(query)
                    # Parse row count from result (e.g., "INSERT 0 5" -> 5)
                    if ' ' in result:
                        try:
                            return int(result.split(' ')[-1])
                        except ValueError:
                            return 0
                    return 0
                    
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return [] if fetch else 0
            
    async def create_tables(self) -> bool:
        """
        Create database tables if they don't exist.
        
        Returns:
            bool: True if created successfully, False otherwise
        """
        try:
            # Define table schemas
            schemas = [
                # OHLCV data
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume NUMERIC NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE (symbol, timeframe, timestamp)
                )
                """,
                
                # Trades
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id VARCHAR(100) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    amount NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    pnl NUMERIC,
                    pnl_percent NUMERIC,
                    fee NUMERIC,
                    slippage NUMERIC,
                    related_trade_id VARCHAR(100),
                    tags JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    exchange VARCHAR(20) NOT NULL
                )
                """,
                
                # Strategies
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    symbols JSONB NOT NULL,
                    timeframes JSONB NOT NULL,
                    params JSONB NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_updated TIMESTAMP DEFAULT NOW()
                )
                """,
                
                # Performance metrics
                """
                CREATE TABLE IF NOT EXISTS performance (
                    id SERIAL PRIMARY KEY,
                    strategy_id VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metrics JSONB NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
                )
                """,
                
                # Signals
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    strategy_id VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    action VARCHAR(20) NOT NULL,
                    price NUMERIC,
                    timestamp TIMESTAMP NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    params JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
                )
                """,
                
                # Account balance history
                """
                CREATE TABLE IF NOT EXISTS balance_history (
                    id SERIAL PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    total_balance NUMERIC NOT NULL,
                    available_balance NUMERIC NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    assets JSONB NOT NULL
                )
                """
            ]
            
            # Execute schema creation
            for schema in schemas:
                await self.execute_query(schema, fetch=False)
                
            # Create indices for faster queries
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv(symbol, timeframe)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
                "CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance(strategy_id)"
            ]
            
            for index in indices:
                await self.execute_query(index, fetch=False)
                
            logger.info("Database tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            traceback.print_exc()
            return False
            
    async def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]]
    ) -> int:
        """
        Save OHLCV data to database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            data: OHLCV data
            
        Returns:
            int: Number of rows inserted
        """
        if not data:
            return 0
            
        try:
            # Prepare query
            query = """
            INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """
            
            # Batch insert
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    count = 0
                    for row in data:
                        timestamp = row.get('timestamp')
                        # Convert datetime or timestamp to proper format
                        if isinstance(timestamp, int):
                            timestamp = datetime.fromtimestamp(timestamp / 1000)
                            
                        await connection.execute(
                            query,
                            symbol,
                            timeframe,
                            timestamp,
                            row.get('open'),
                            row.get('high'),
                            row.get('low'),
                            row.get('close'),
                            row.get('volume')
                        )
                        count += 1
                        
            logger.info(f"Saved {count} OHLCV rows for {symbol} {timeframe}")
            return count
            
        except Exception as e:
            logger.error(f"Error saving OHLCV data: {str(e)}")
            return 0
            
    async def save_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Save trade to database.
        
        Args:
            trade: Trade data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare query
            query = """
            INSERT INTO trades (
                id, symbol, side, amount, price, timestamp, strategy, timeframe,
                status, pnl, pnl_percent, fee, slippage, related_trade_id, tags, exchange
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ON CONFLICT (id) DO UPDATE
            SET status = EXCLUDED.status,
                pnl = EXCLUDED.pnl,
                pnl_percent = EXCLUDED.pnl_percent
            """
            
            # Execute query
            await self.execute_query(
                query,
                (
                    trade.get('id'),
                    trade.get('symbol'),
                    trade.get('side'),
                    trade.get('amount'),
                    trade.get('price'),
                    trade.get('timestamp'),
                    trade.get('strategy'),
                    trade.get('timeframe'),
                    trade.get('status', 'executed'),
                    trade.get('pnl'),
                    trade.get('pnl_percent'),
                    trade.get('fee'),
                    trade.get('slippage'),
                    trade.get('related_trade_id'),
                    json.dumps(trade.get('tags', [])),
                    trade.get('exchange', 'unknown')
                ),
                fetch=False
            )
            
            logger.info(f"Saved trade {trade.get('id')} for {trade.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade: {str(e)}")
            return False
            
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trades from database.
        
        Args:
            symbol: Trading pair symbol
            strategy: Strategy name
            start_time: Start time
            end_time: End time
            limit: Maximum number of trades to return
            
        Returns:
            list: List of trades
        """
        try:
            # Build query with conditions
            conditions = []
            params = []
            param_index = 1
            
            if symbol:
                conditions.append(f"symbol = ${param_index}")
                params.append(symbol)
                param_index += 1
                
            if strategy:
                conditions.append(f"strategy = ${param_index}")
                params.append(strategy)
                param_index += 1
                
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time)
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time)
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
                
            # Complete query
            query = f"""
            SELECT * FROM trades
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            trades = await self.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for trade in trades:
                if 'tags' in trade and trade['tags']:
                    try:
                        trade['tags'] = json.loads(trade['tags'])
                    except:
                        trade['tags'] = []
                        
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            return []
            
    async def save_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        Save strategy to database.
        
        Args:
            strategy: Strategy data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare query
            query = """
            INSERT INTO strategies (
                id, name, type, symbols, timeframes, params, is_active, last_updated
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (id) DO UPDATE
            SET name = EXCLUDED.name,
                type = EXCLUDED.type,
                symbols = EXCLUDED.symbols,
                timeframes = EXCLUDED.timeframes,
                params = EXCLUDED.params,
                is_active = EXCLUDED.is_active,
                last_updated = NOW()
            """
            
            # Execute query
            await self.execute_query(
                query,
                (
                    strategy.get('id'),
                    strategy.get('name'),
                    strategy.get('type'),
                    json.dumps(strategy.get('symbols', [])),
                    json.dumps(strategy.get('timeframes', [])),
                    json.dumps(strategy.get('params', {})),
                    strategy.get('is_active', True)
                ),
                fetch=False
            )
            
            logger.info(f"Saved strategy {strategy.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy: {str(e)}")
            return False
            
    async def get_strategies(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get strategies from database.
        
        Args:
            active_only: Whether to return only active strategies
            
        Returns:
            list: List of strategies
        """
        try:
            # Build query
            query = "SELECT * FROM strategies"
            if active_only:
                query += " WHERE is_active = TRUE"
                
            # Execute query
            strategies = await self.execute_query(query)
            
            # Parse JSON fields
            for strategy in strategies:
                if 'symbols' in strategy and strategy['symbols']:
                    try:
                        strategy['symbols'] = json.loads(strategy['symbols'])
                    except:
                        strategy['symbols'] = []
                        
                if 'timeframes' in strategy and strategy['timeframes']:
                    try:
                        strategy['timeframes'] = json.loads(strategy['timeframes'])
                    except:
                        strategy['timeframes'] = []
                        
                if 'params' in strategy and strategy['params']:
                    try:
                        strategy['params'] = json.loads(strategy['params'])
                    except:
                        strategy['params'] = {}
                        
            return strategies
            
        except Exception as e:
            logger.error(f"Error getting strategies: {str(e)}")
            return []
            
    async def save_performance_metrics(
        self,
        strategy_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Save performance metrics to database.
        
        Args:
            strategy_id: Strategy ID
            metrics: Performance metrics
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare query
            query = """
            INSERT INTO performance (strategy_id, timestamp, metrics)
            VALUES ($1, NOW(), $2)
            """
            
            # Execute query
            await self.execute_query(
                query,
                (strategy_id, json.dumps(metrics)),
                fetch=False
            )
            
            logger.info(f"Saved performance metrics for {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
            return False
            
    async def get_performance_metrics(
        self,
        strategy_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics from database.
        
        Args:
            strategy_id: Strategy ID
            start_time: Start time
            end_time: End time
            limit: Maximum number of metrics to return
            
        Returns:
            list: List of performance metrics
        """
        try:
            # Build query with conditions
            conditions = ["strategy_id = $1"]
            params = [strategy_id]
            param_index = 2
            
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time)
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time)
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            
            # Complete query
            query = f"""
            SELECT * FROM performance
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            metrics = await self.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for metric in metrics:
                if 'metrics' in metric and metric['metrics']:
                    try:
                        metric['metrics'] = json.loads(metric['metrics'])
                    except:
                        metric['metrics'] = {}
                        
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return []
            
    async def save_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Save signal to database.
        
        Args:
            signal: Signal data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare query
            query = """
            INSERT INTO signals (
                strategy_id, symbol, timeframe, action, price,
                timestamp, executed, params
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            # Execute query
            await self.execute_query(
                query,
                (
                    signal.get('strategy_id'),
                    signal.get('symbol'),
                    signal.get('timeframe'),
                    signal.get('action'),
                    signal.get('price'),
                    signal.get('timestamp', datetime.now()),
                    signal.get('executed', False),
                    json.dumps(signal.get('params', {}))
                ),
                fetch=False
            )
            
            logger.info(f"Saved signal for {signal.get('symbol')} with action {signal.get('action')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signal: {str(e)}")
            return False
            
    async def get_signals(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get signals from database.
        
        Args:
            strategy_id: Strategy ID
            symbol: Trading pair symbol
            executed: Whether the signal was executed
            start_time: Start time
            end_time: End time
            limit: Maximum number of signals to return
            
        Returns:
            list: List of signals
        """
        try:
            # Build query with conditions
            conditions = []
            params = []
            param_index = 1
            
            if strategy_id:
                conditions.append(f"strategy_id = ${param_index}")
                params.append(strategy_id)
                param_index += 1
                
            if symbol:
                conditions.append(f"symbol = ${param_index}")
                params.append(symbol)
                param_index += 1
                
            if executed is not None:
                conditions.append(f"executed = ${param_index}")
                params.append(executed)
                param_index += 1
                
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time)
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time)
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
                
            # Complete query
            query = f"""
            SELECT * FROM signals
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            signals = await self.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for signal in signals:
                if 'params' in signal and signal['params']:
                    try:
                        signal['params'] = json.loads(signal['params'])
                    except:
                        signal['params'] = {}
                        
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return []
            
    async def save_balance(
        self,
        exchange: str,
        total_balance: float,
        available_balance: float,
        assets: Dict[str, Any]
    ) -> bool:
        """
        Save account balance to database.
        
        Args:
            exchange: Exchange name
            total_balance: Total balance
            available_balance: Available balance
            assets: Asset balances
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Prepare query
            query = """
            INSERT INTO balance_history (
                exchange, total_balance, available_balance, timestamp, assets
            )
            VALUES ($1, $2, $3, NOW(), $4)
            """
            
            # Execute query
            await self.execute_query(
                query,
                (exchange, total_balance, available_balance, json.dumps(assets)),
                fetch=False
            )
            
            logger.info(f"Saved balance for {exchange}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving balance: {str(e)}")
            return False
            
    async def get_balance_history(
        self,
        exchange: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get balance history from database.
        
        Args:
            exchange: Exchange name
            start_time: Start time
            end_time: End time
            limit: Maximum number of records to return
            
        Returns:
            list: Balance history
        """
        try:
            # Build query with conditions
            conditions = []
            params = []
            param_index = 1
            
            if exchange:
                conditions.append(f"exchange = ${param_index}")
                params.append(exchange)
                param_index += 1
                
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time)
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time)
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
                
            # Complete query
            query = f"""
            SELECT * FROM balance_history
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            balance_history = await self.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for record in balance_history:
                if 'assets' in record and record['assets']:
                    try:
                        record['assets'] = json.loads(record['assets'])
                    except:
                        record['assets'] = {}
                        
            return balance_history
            
        except Exception as e:
            logger.error(f"Error getting balance history: {str(e)}")
            return []
            
    async def execute_transaction(self, queries: List[Tuple[str, Tuple]]) -> bool:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            bool: True if executed successfully, False otherwise
        """
        if not self.pool:
            logger.error("Database not connected")
            return False
            
        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    for query, params in queries:
                        await connection.execute(query, *params)
                        
            return True
            
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            return False
            
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            limit: Maximum number of records to return
            
        Returns:
            list: OHLCV data
        """
        try:
            # Build query with conditions
            conditions = ["symbol = $1", "timeframe = $2"]
            params = [symbol, timeframe]
            param_index = 3
            
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time)
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time)
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            
            # Complete query
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE {where_clause}
            ORDER BY timestamp ASC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            data = await self.execute_query(query, tuple(params))
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {str(e)}")
            return []
            
    async def run_backup(self, backup_path: str) -> bool:
        """
        Run database backup.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            import subprocess
            import os
            from datetime import datetime
            
            # Create backup directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Format backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{backup_path}_{timestamp}.sql"
            
            # Run pg_dump command
            command = [
                'pg_dump',
                '-h', self.url,
                '-p', str(self.port),
                '-U', self.username,
                '-F', 'c',  # Custom format
                '-f', backup_file,
                self.database
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            if self.password:
                env['PGPASSWORD'] = self.password
                
            # Execute command
            process = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                logger.error(f"Backup failed: {process.stderr.decode()}")
                return False
                
            logger.info(f"Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error running database backup: {str(e)}")
            return False
            
    async def run_restore(self, backup_file: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            bool: True if restore successful, False otherwise
        """
        try:
            import subprocess
            import os
            
            # Check if backup file exists
            if not os.path.exists(backup_file):
                logger.error(f"Backup file not found: {backup_file}")
                return False
                
            # Run pg_restore command
            command = [
                'pg_restore',
                '-h', self.url,
                '-p', str(self.port),
                '-U', self.username,
                '-d', self.database,
                '-c',  # Clean (drop) database objects before recreating
                backup_file
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            if self.password:
                env['PGPASSWORD'] = self.password
                
            # Execute command
            process = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                logger.error(f"Restore failed: {process.stderr.decode()}")
                return False
                
            logger.info(f"Database restored from: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database: {str(e)}")
            return False
            
    async def clean_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean old data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            int: Number of rows deleted
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old data
            queries = [
                (
                    "DELETE FROM ohlcv WHERE timestamp < $1",
                    (cutoff_date,)
                ),
                (
                    "DELETE FROM signals WHERE timestamp < $1",
                    (cutoff_date,)
                ),
                (
                    "DELETE FROM balance_history WHERE timestamp < $1",
                    (cutoff_date,)
                )
            ]
            
            # Execute delete queries
            total_deleted = 0
            for query, params in queries:
                result = await self.execute_query(query, params, fetch=False)
                
                # Try to parse row count
                try:
                    if isinstance(result, str) and ' ' in result:
                        deleted = int(result.split(' ')[-1])
                        total_deleted += deleted
                except:
                    pass
                    
            logger.info(f"Cleaned {total_deleted} rows of old data")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error cleaning old data: {str(e)}")
            return 0