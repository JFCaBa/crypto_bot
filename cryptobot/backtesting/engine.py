"""
Backtesting Engine
=================
Provides functionality for backtesting trading strategies using historical data.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.core.trade import Trade
from cryptobot.data.historical import HistoricalDataProvider
from cryptobot.strategies.base import BaseStrategy


class BacktestingEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: HistoricalDataProvider,
        initial_balance: float = 10000.0,
        maker_fee: float = 0.001,  # 0.1%
        taker_fee: float = 0.002,  # 0.2%
        slippage: float = 0.001,   # 0.1%
        enable_margin: bool = False,
        leverage: float = 1.0,
        debug: bool = False
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Strategy to backtest
            data_provider: Historical data provider
            initial_balance: Initial account balance
            maker_fee: Maker fee as a decimal
            taker_fee: Taker fee as a decimal
            slippage: Slippage as a decimal
            enable_margin: Whether to enable margin trading
            leverage: Leverage multiplier
            debug: Whether to print debug information
        """
        self.strategy = strategy
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.enable_margin = enable_margin
        self.leverage = leverage
        self.debug = debug
        
        # Performance metrics
        self.metrics = {
            'start_balance': initial_balance,
            'end_balance': initial_balance,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'annual_return': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'buy_hold_return': 0.0,
            'strategy_return': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'volatility': 0.0,
            'benchmark_volatility': 0.0
        }
        
        # Backtest results
        self.results = pd.DataFrame()
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Simulation state
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.positions = {}
        self.open_orders = []
        
        # Mock exchange for strategy execution
        self.mock_exchange = self._create_mock_exchange()
        
    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1h'
    ) -> Dict[str, Any]:
        """
        Run backtest from start_date to end_date.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Timeframe for backtest
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date} with timeframe {timeframe}")
        
        try:
            # Reset state for new backtest
            self._reset_state()
            
            # Ensure strategy is reset
            self.strategy.reset()
            
            # Get historical data for all symbols
            symbols = self.strategy.symbols
            data = {}
            
            for symbol in symbols:
                df = await self.data_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                    
                data[symbol] = df
                
            if not data:
                logger.error("No data available for any symbol")
                return self.metrics
                
            # Find common date range across all symbols
            common_dates = None
            for symbol, df in data.items():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates &= set(df.index)
                    
            common_dates = sorted(common_dates)
            
            if not common_dates:
                logger.error("No common dates found across symbols")
                return self.metrics
                
            # Initialize equity curve with initial balance
            self.equity_curve.append({
                'timestamp': common_dates[0],
                'equity': self.initial_balance,
                'drawdown': 0.0,
                'drawdown_pct': 0.0
            })
            
            # Initialize benchmark data (buy and hold)
            benchmark_initial_amounts = {}
            for symbol in symbols:
                first_price = data[symbol].loc[common_dates[0], 'close']
                benchmark_initial_amounts[symbol] = self.initial_balance / len(symbols) / first_price
                
            # Start strategy
            self.strategy.start()
            
            # Simulate trading for each date
            for i, date in enumerate(common_dates):
                # Skip first date (used for initialization)
                if i == 0:
                    continue
                    
                # Update strategy with latest data for all symbols
                for symbol in symbols:
                    if symbol in data:
                        # Get data up to current date
                        symbol_data = data[symbol].loc[:date].copy()
                        
                        # Update strategy data
                        self.strategy.update_data(symbol, timeframe, symbol_data)
                        
                # Generate signals for all symbols
                signals = []
                for symbol in symbols:
                    if symbol in data:
                        signal = self.strategy.generate_signals(symbol, timeframe)
                        if signal and signal.get('action') is not None:
                            signals.append(signal)
                            
                # Execute signals
                if signals:
                    executed_trades = await self.strategy.execute_signals(signals, self.mock_exchange)
                    self.trade_history.extend(executed_trades)
                    
                # Process any pending orders (stop loss, take profit)
                await self._process_pending_orders(date, data)
                
                # Update equity curve
                total_equity = self._calculate_total_equity(date, data)
                drawdown, drawdown_pct = self._calculate_drawdown(total_equity)
                
                self.equity_curve.append({
                    'timestamp': date,
                    'equity': total_equity,
                    'drawdown': drawdown,
                    'drawdown_pct': drawdown_pct
                })
                
                if self.debug and i % 100 == 0:
                    logger.info(f"Processed {i}/{len(common_dates)} dates, equity: {total_equity:.2f}")
                    
            # Stop strategy
            self.strategy.stop()
            
            # Calculate final metrics
            self._calculate_metrics(data, common_dates, benchmark_initial_amounts)
            
            # Convert equity curve to DataFrame
            self.results = pd.DataFrame(self.equity_curve)
            
            logger.info(f"Backtest complete - Final balance: {total_equity:.2f}, Return: {(total_equity/self.initial_balance-1)*100:.2f}%")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self.metrics
            
    def get_results(self) -> pd.DataFrame:
        """
        Get backtest results as a DataFrame.
        
        Returns:
            pd.DataFrame: Equity curve with performance metrics
        """
        return self.results
        
    def get_trade_history(self) -> List[Trade]:
        """
        Get backtest trade history.
        
        Returns:
            list: List of executed trades
        """
        return self.trade_history
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get backtest performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return self.metrics
        
    def plot_results(self, benchmark: bool = True, save_path: str = None):
        """
        Plot backtest results.
        
        Args:
            benchmark: Whether to include benchmark comparison
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.results.empty:
                logger.warning("No results to plot")
                return
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1 = axes[0]
            ax1.plot(self.results['timestamp'], self.results['equity'], label='Strategy')
            
            # Plot benchmark if requested
            if benchmark and 'benchmark_equity' in self.results.columns:
                ax1.plot(self.results['timestamp'], self.results['benchmark_equity'], label='Buy & Hold', alpha=0.7)
                
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            ax2 = axes[1]
            ax2.fill_between(self.results['timestamp'], 0, self.results['drawdown_pct'], color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.set_ylim(bottom=0)
            ax2.invert_yaxis()
            ax2.grid(True)
            
            # Format plot
            plt.tight_layout()
            
            # Save plot if requested
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
                
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available. Install it to plot results.")
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            
    def export_results(self, filepath: str) -> bool:
        """
        Export backtest results to CSV.
        
        Args:
            filepath: Path to save CSV file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            if self.results.empty:
                logger.warning("No results to export")
                return False
                
            self.results.to_csv(filepath)
            logger.info(f"Results exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False
            
    def export_trade_history(self, filepath: str) -> bool:
        """
        Export trade history to CSV.
        
        Args:
            filepath: Path to save CSV file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            if not self.trade_history:
                logger.warning("No trade history to export")
                return False
                
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame([t.to_dict() for t in self.trade_history])
            trades_df.to_csv(filepath)
            logger.info(f"Trade history exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting trade history: {str(e)}")
            return False
            
    def _reset_state(self):
        """Reset backtesting state for a new run."""
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.positions = {}
        self.open_orders = []
        self.results = pd.DataFrame()
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Reset metrics
        self.metrics = {
            'start_balance': self.initial_balance,
            'end_balance': self.initial_balance,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'annual_return': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'buy_hold_return': 0.0,
            'strategy_return': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'volatility': 0.0,
            'benchmark_volatility': 0.0
        }
        
        for symbol in self.strategy.symbols:
            self.positions[symbol] = {
                'is_active': False,
                'side': None,
                'entry_price': None,
                'amount': None,
                'entry_time': None,
                'order_id': None,
                'stop_loss': None,
                'take_profit': None
            }
            
    def _create_mock_exchange(self):
        """
        Create a mock exchange for strategy execution.
        
        Returns:
            object: Mock exchange object
        """
        class MockExchange:
            def __init__(self, engine):
                self.engine = engine
                from loguru import logger
                self.logger = logger
                
            async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
                """Mock order creation."""
                if params is None:
                    params = {}
                    
                # Get current price from latest data
                current_price = price
                
                # If price is None, try to get the current price from the latest data
                if current_price is None:
                    try:
                        # In BacktestingEngine, we already have the data loaded in the strategy object
                        # Let's use that instead of trying to access the HistoricalDataProvider directly
                        if symbol in self.engine.strategy.data:
                            for timeframe in self.engine.strategy.timeframes:
                                if timeframe in self.engine.strategy.data[symbol]:
                                    df = self.engine.strategy.data[symbol][timeframe]
                                    if not df.empty:
                                        current_price = df.iloc[-1]['close']
                                        self.logger.info(f"Using current price {current_price} for {symbol} from strategy data")
                                        break
                    except (KeyError, IndexError, AttributeError) as e:
                        self.logger.error(f"Failed to get current price for {symbol} from strategy data: {str(e)}")
                        # Let's try one more method - get it from the current_data that should be available during backtest
                        try:
                            # During backtesting, the engine should be processing data for one date at a time
                            # The current_data for the symbol should be available in the data dictionary
                            for sym, data_dict in self.engine.data.items():
                                if sym == symbol:
                                    current_price = data_dict.get('close')
                                    if current_price:
                                        self.logger.info(f"Using current price {current_price} for {symbol} from current data")
                                        break
                        except (KeyError, AttributeError) as e:
                            self.logger.error(f"Failed to get current price for {symbol} from current data: {str(e)}")
                            # Fallback to a default price to prevent failure
                            current_price = 1.0
                            self.logger.warning(f"Using fallback price {current_price} for {symbol}")
                
                # Ensure we have a valid price
                if current_price is None:
                    self.logger.error(f"No price available for {symbol}, using default price of 1.0")
                    current_price = 1.0
                    
                # Generate order ID
                import time
                from datetime import datetime
                
                order_id = f"order_{int(time.time() * 1000)}_{len(self.engine.trade_history)}"
                
                # Apply slippage to market orders
                if order_type == 'market':
                    slippage_factor = 1 + (self.engine.slippage * (1 if side == 'buy' else -1))
                    current_price = current_price * slippage_factor
                    
                # Calculate fee
                fee = self.engine.taker_fee if order_type == 'market' else self.engine.maker_fee
                fee_amount = amount * current_price * fee
                
                # Update balance
                if side == 'buy':
                    cost = amount * current_price + fee_amount
                    if cost > self.engine.current_balance:
                        # Not enough balance
                        self.logger.warning(f"Not enough balance for order: {cost:.2f} > {self.engine.current_balance:.2f}")
                        return None
                    self.engine.current_balance -= cost
                elif side == 'sell':
                    revenue = amount * current_price - fee_amount
                    self.engine.current_balance += revenue
                    
                # Update positions
                position = self.engine.positions.get(symbol, {
                    'is_active': False,
                    'side': None,
                    'entry_price': None,
                    'amount': None,
                    'entry_time': None,
                    'order_id': None,
                    'stop_loss': None,
                    'take_profit': None
                })
                
                if side == 'buy' and not position['is_active']:
                    # Open long position
                    self.engine.positions[symbol] = {
                        'is_active': True,
                        'side': 'long',
                        'entry_price': current_price,
                        'amount': amount,
                        'entry_time': datetime.now(),  # In real backtest this would be the current candle time
                        'order_id': order_id,
                        'stop_loss': params.get('stopLoss'),
                        'take_profit': params.get('takeProfit')
                    }
                elif side == 'sell' and not position['is_active']:
                    # Open short position
                    self.engine.positions[symbol] = {
                        'is_active': True,
                        'side': 'short',
                        'entry_price': current_price,
                        'amount': amount,
                        'entry_time': datetime.now(),  # In real backtest this would be the current candle time
                        'order_id': order_id,
                        'stop_loss': params.get('stopLoss'),
                        'take_profit': params.get('takeProfit')
                    }
                elif position['is_active']:
                    # Close position
                    position_side = position['side']
                    entry_price = position['entry_price']
                    position_amount = position['amount']
                    
                    # Calculate PnL
                    if position_side == 'long' and side == 'sell':
                        pnl = (current_price - entry_price) * position_amount
                        pnl_percent = (current_price / entry_price - 1) * 100
                    elif position_side == 'short' and side == 'buy':
                        pnl = (entry_price - current_price) * position_amount
                        pnl_percent = (entry_price / current_price - 1) * 100
                    else:
                        pnl = 0
                        pnl_percent = 0
                        
                    # Reset position
                    self.engine.positions[symbol] = {
                        'is_active': False,
                        'side': None,
                        'entry_price': None,
                        'amount': None,
                        'entry_time': None,
                        'order_id': None,
                        'stop_loss': None,
                        'take_profit': None
                    }
                    
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'price': current_price,
                    'amount': amount,
                    'fee': fee_amount,
                    'timestamp': datetime.now().timestamp()
                }
                
        return MockExchange(self)
        
    async def _process_pending_orders(self, date, data):
        """
        Process pending orders (stop loss, take profit).
        
        Args:
            date: Current date
            data: Historical data
        """
        for symbol, position in list(self.positions.items()):
            if not position['is_active']:
                continue
                
            # Get current price
            if symbol not in data:
                continue
                
            current_data = data[symbol].loc[date]
            current_price = current_data['close']
            
            # Check stop loss
            if position['stop_loss'] is not None:
                if (position['side'] == 'long' and current_data['low'] <= position['stop_loss']) or \
                   (position['side'] == 'short' and current_data['high'] >= position['stop_loss']):
                    # Stop loss triggered
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    order = await self.mock_exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side=side,
                        amount=position['amount'],
                        price=position['stop_loss']  # Use stop loss price
                    )
                    
                    if order:
                        # Create trade record
                        trade = Trade(
                            id=order['id'],
                            symbol=symbol,
                            side=side,
                            amount=position['amount'],
                            price=order['price'],
                            timestamp=date,
                            strategy=self.strategy.name,
                            timeframe='backtest',
                            status='executed',
                            pnl=(order['price'] - position['entry_price']) * position['amount'] if position['side'] == 'long' \
                                else (position['entry_price'] - order['price']) * position['amount'],
                            pnl_percent=(order['price'] / position['entry_price'] - 1) * 100 if position['side'] == 'long' \
                                else (position['entry_price'] / order['price'] - 1) * 100,
                            related_trade_id=position['order_id'],
                            tags=['stop_loss']
                        )
                        
                        self.trade_history.append(trade)
                        
                        if self.debug:
                            logger.info(f"Stop loss triggered for {symbol} at {position['stop_loss']}")
                            
            # Check take profit
            if position['take_profit'] is not None:
                if (position['side'] == 'long' and current_data['high'] >= position['take_profit']) or \
                   (position['side'] == 'short' and current_data['low'] <= position['take_profit']):
                    # Take profit triggered
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    order = await self.mock_exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side=side,
                        amount=position['amount'],
                        price=position['take_profit']  # Use take profit price
                    )
                    
                    if order:
                        # Create trade record
                        trade = Trade(
                            id=order['id'],
                            symbol=symbol,
                            side=side,
                            amount=position['amount'],
                            price=order['price'],
                            timestamp=date,
                            strategy=self.strategy.name,
                            timeframe='backtest',
                            status='executed',
                            pnl=(order['price'] - position['entry_price']) * position['amount'] if position['side'] == 'long' \
                                else (position['entry_price'] - order['price']) * position['amount'],
                            pnl_percent=(order['price'] / position['entry_price'] - 1) * 100 if position['side'] == 'long' \
                                else (position['entry_price'] / order['price'] - 1) * 100,
                            related_trade_id=position['order_id'],
                            tags=['take_profit']
                        )
                        
                        self.trade_history.append(trade)
                        
                        if self.debug:
                            logger.info(f"Take profit triggered for {symbol} at {position['take_profit']}")
                            
    def _calculate_total_equity(self, date, data):
        """
        Calculate total equity (balance + positions value).
        
        Args:
            date: Current date
            data: Historical data
            
        Returns:
            float: Total equity
        """
        total_equity = self.current_balance
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            if position['is_active'] and symbol in data:
                current_price = data[symbol].loc[date, 'close']
                amount = position['amount']
                entry_price = position['entry_price']
                
                if position['side'] == 'long':
                    position_value = amount * current_price
                    total_equity += position_value - (amount * entry_price)
                elif position['side'] == 'short':
                    position_value = amount * entry_price
                    total_equity += position_value - (amount * current_price)
                    
        # Update peak balance if new peak
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
            
        return total_equity
        
    def _calculate_drawdown(self, current_equity):
        """
        Calculate current drawdown.
        
        Args:
            current_equity: Current equity
            
        Returns:
            tuple: (drawdown_amount, drawdown_percent)
        """
        if self.peak_balance <= 0:
            return 0, 0
            
        drawdown = self.peak_balance - current_equity
        drawdown_percent = (drawdown / self.peak_balance) * 100
        
        return drawdown, drawdown_percent
        
    def _calculate_metrics(self, data, dates, benchmark_initial_amounts):
        """
        Calculate performance metrics for the strategy.
        
        Args:
            data: Historical data
            dates: List of dates
            benchmark_initial_amounts: Initial amounts for benchmark
        """
        # Basic metrics
        self.metrics['total_trades'] = len(self.trade_history)
        self.metrics['end_balance'] = self.equity_curve[-1]['equity']
        
        # Calculate returns
        start_value = self.metrics['start_balance']
        end_value = self.metrics['end_balance']
        total_return = (end_value / start_value) - 1
        self.metrics['strategy_return'] = total_return * 100  # Convert to percentage
        
        # Calculate benchmark return (buy and hold)
        benchmark_end_value = 0
        for symbol, amount in benchmark_initial_amounts.items():
            if symbol in data:
                first_price = data[symbol].loc[dates[0], 'close']
                last_price = data[symbol].loc[dates[-1], 'close']
                benchmark_end_value += amount * last_price
                
        benchmark_return = (benchmark_end_value / start_value) - 1
        self.metrics['buy_hold_return'] = benchmark_return * 100  # Convert to percentage
        
        # Calculate alpha and beta
        self.metrics['alpha'] = total_return - benchmark_return
        
        # Calculate annual return
        days = (dates[-1] - dates[0]).days
        if days > 0:
            self.metrics['annual_return'] = ((1 + total_return) ** (365 / days) - 1) * 100
            
        # Calculate win rate and profit factor
        # Filter out trades with None pnl values
        valid_trades = [t for t in self.trade_history if t.pnl is not None]
        winning_trades = [t for t in valid_trades if t.pnl > 0]
        losing_trades = [t for t in valid_trades if t.pnl <= 0]
        
        self.metrics['winning_trades'] = len(winning_trades)
        self.metrics['losing_trades'] = len(losing_trades)
        
        if valid_trades:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / len(valid_trades)) * 100
        else:
            self.metrics['win_rate'] = 0.0
            
        # Calculate average profit and loss
        if winning_trades:
            self.metrics['avg_profit'] = sum(t.pnl_percent for t in winning_trades if t.pnl_percent is not None) / len(winning_trades)
        
        if losing_trades:
            self.metrics['avg_loss'] = sum(t.pnl_percent for t in losing_trades if t.pnl_percent is not None) / len(losing_trades)
            
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl is not None) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl is not None)) if losing_trades else 0
        
        if gross_loss > 0:
            self.metrics['profit_factor'] = gross_profit / gross_loss
            
        # Calculate maximum drawdown
        max_dd = 0
        max_dd_pct = 0
        
        for point in self.equity_curve:
            max_dd = max(max_dd, point['drawdown'])
            max_dd_pct = max(max_dd_pct, point['drawdown_pct'])
            
        self.metrics['max_drawdown'] = max_dd
        self.metrics['max_drawdown_pct'] = max_dd_pct
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]['equity']
            curr_equity = self.equity_curve[i]['equity']
            daily_return = (curr_equity / prev_equity) - 1
            returns.append(daily_return)
            
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_rate = 0.0  # Simplified
            
            if std_return > 0:
                self.metrics['sharpe_ratio'] = (avg_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
                
        # Calculate Sortino ratio (downside risk only)
        if returns:
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    self.metrics['sortino_ratio'] = (avg_return - risk_free_rate) / downside_deviation * np.sqrt(252)
                    
        # Calculate Calmar ratio
        if self.metrics['max_drawdown_pct'] > 0:
            self.metrics['calmar_ratio'] = self.metrics['annual_return'] / self.metrics['max_drawdown_pct']
            
        # Calculate volatility
        self.metrics['volatility'] = np.std(returns) * np.sqrt(252) * 100 if returns else 0
        
        # Calculate total fees and slippage
        self.metrics['total_fees'] = sum(t.fee for t in self.trade_history if hasattr(t, 'fee') and t.fee is not None)
        self.metrics['total_slippage'] = sum(t.slippage for t in self.trade_history if hasattr(t, 'slippage') and t.slippage is not None)