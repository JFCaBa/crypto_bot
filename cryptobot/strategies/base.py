"""
Base Strategy Class
=================
Abstract base class for all trading strategies that defines the interface
that all strategy implementations must follow.
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from cryptobot.core.trade import Trade
from cryptobot.risk_management.manager import RiskManager


class BaseStrategy(abc.ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        timeframes: List[str],
        params: Dict[str, Any] = None,
        risk_manager: RiskManager = None
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            params: Strategy parameters
            risk_manager: Risk manager instance
        """
        self.name = name
        self.symbols = symbols
        self.timeframes = timeframes
        self.params = params or {}
        self.risk_manager = risk_manager
        
        # Data storage for each symbol and timeframe
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {
            symbol: {tf: pd.DataFrame() for tf in timeframes} 
            for symbol in symbols
        }
        
        # Indicators for each symbol and timeframe
        self.indicators: Dict[str, Dict[str, Dict[str, pd.Series]]] = {
            symbol: {tf: {} for tf in timeframes} 
            for symbol in symbols
        }
        
        # Track active positions
        self.positions: Dict[str, Dict[str, Any]] = {
            symbol: {'is_active': False, 'side': None, 'entry_price': None, 'amount': None, 'entry_time': None}
            for symbol in symbols
        }
        
        # Track trade history
        self.trade_history: List[Trade] = []
        
        # Strategy state
        self.is_active = False
        self.last_update_time = None
        
    def start(self) -> bool:
        """
        Start the strategy.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            self.is_active = True
            logger.info(f"Strategy {self.name} started")
            return True
        except Exception as e:
            logger.error(f"Error starting strategy {self.name}: {str(e)}")
            return False
            
    def stop(self) -> bool:
        """
        Stop the strategy.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            self.is_active = False
            logger.info(f"Strategy {self.name} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping strategy {self.name}: {str(e)}")
            return False
            
    def update_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Update historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            data: OHLCV data as pandas DataFrame
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            if symbol not in self.symbols:
                raise ValueError(f"Symbol {symbol} not in strategy symbols")
                
            if timeframe not in self.timeframes:
                raise ValueError(f"Timeframe {timeframe} not in strategy timeframes")
                
            self.data[symbol][timeframe] = data
            self.last_update_time = datetime.now()
            
            # Calculate indicators after data update
            self.calculate_indicators(symbol, timeframe)
            
            return True
        except Exception as e:
            logger.error(f"Error updating data for {symbol} {timeframe}: {str(e)}")
            return False
            
    @abc.abstractmethod
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate technical indicators for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if calculated successfully, False otherwise
        """
        pass
        
    @abc.abstractmethod
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            dict: Signal information including action, price, etc.
        """
        pass
        
    async def execute_signals(self, signals: List[Dict[str, Any]], exchange) -> List[Trade]:
        """
        Execute trading signals through an exchange.
        
        Args:
            signals: List of signals to execute
            exchange: Exchange instance for trade execution
            
        Returns:
            list: List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            if not self.is_active:
                logger.warning("Strategy is not active. Skipping signal execution.")
                break
                
            symbol = signal.get('symbol')
            action = signal.get('action')
            
            if not symbol or not action:
                logger.warning(f"Invalid signal: {signal}")
                continue
                
            try:
                # Apply risk management rules
                if self.risk_manager:
                    approved, adjusted_signal = self.risk_manager.validate_signal(signal, self.positions, self.trade_history)
                    if not approved:
                        logger.warning(f"Signal rejected by risk manager: {signal}")
                        continue
                    signal = adjusted_signal
                
                # Execute the signal based on the action
                if action == 'buy' and not self.positions[symbol]['is_active']:
                    # Open long position
                    amount = signal.get('amount')
                    price = signal.get('price')
                    params = signal.get('params', {})
                    
                    # Skip if amount or price is None or zero
                    if not amount or not price:
                        logger.warning(f"Invalid amount or price in signal: {signal}")
                        continue
                    
                    # Add stop loss and take profit if specified
                    if 'stop_loss' in signal:
                        params['stopLoss'] = signal['stop_loss']
                    if 'take_profit' in signal:
                        params['takeProfit'] = signal['take_profit']
                        
                    # Create market or limit order
                    order_type = signal.get('order_type', 'market')
                    result = await exchange.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side='buy',
                        amount=amount,
                        price=price if order_type == 'limit' else None,
                        params=params
                    )
                    
                    # Update position status
                    self.positions[symbol] = {
                        'is_active': True,
                        'side': 'long',
                        'entry_price': result.get('price') or price,
                        'amount': amount,
                        'entry_time': datetime.now(),
                        'order_id': result.get('id')
                    }
                    
                    # Create trade record
                    trade = Trade(
                        id=result.get('id'),
                        symbol=symbol,
                        side='buy',
                        amount=amount,
                        price=result.get('price') or price,
                        timestamp=datetime.now(),
                        strategy=self.name,
                        timeframe=signal.get('timeframe'),
                        status='executed'
                    )
                    
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    logger.info(f"Executed buy signal for {symbol}: {trade}")
                    
                elif action == 'sell' and not self.positions[symbol]['is_active']:
                    # Open short position (if supported)
                    amount = signal.get('amount')
                    price = signal.get('price')
                    params = signal.get('params', {})
                    
                    # Skip if amount or price is None or zero
                    if not amount or not price:
                        logger.warning(f"Invalid amount or price in signal: {signal}")
                        continue
                    
                    # Add stop loss and take profit if specified
                    if 'stop_loss' in signal:
                        params['stopLoss'] = signal['stop_loss']
                    if 'take_profit' in signal:
                        params['takeProfit'] = signal['take_profit']
                        
                    # Create market or limit order
                    order_type = signal.get('order_type', 'market')
                    result = await exchange.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side='sell',
                        amount=amount,
                        price=price if order_type == 'limit' else None,
                        params=params
                    )
                    
                    # Update position status
                    self.positions[symbol] = {
                        'is_active': True,
                        'side': 'short',
                        'entry_price': result.get('price') or price,
                        'amount': amount,
                        'entry_time': datetime.now(),
                        'order_id': result.get('id')
                    }
                    
                    # Create trade record
                    trade = Trade(
                        id=result.get('id'),
                        symbol=symbol,
                        side='sell',
                        amount=amount,
                        price=result.get('price') or price,
                        timestamp=datetime.now(),
                        strategy=self.name,
                        timeframe=signal.get('timeframe'),
                        status='executed'
                    )
                    
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    logger.info(f"Executed sell signal for {symbol}: {trade}")
                    
                elif action == 'close' and self.positions[symbol]['is_active']:
                    # Close position
                    position = self.positions[symbol]
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = position['amount']
                    params = signal.get('params', {})
                    
                    # Create market or limit order
                    order_type = signal.get('order_type', 'market')
                    price = signal.get('price')
                    
                    result = await exchange.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side=side,
                        amount=amount,
                        price=price if order_type == 'limit' else None,
                        params=params
                    )
                    
                    # Reset position status
                    self.positions[symbol] = {
                        'is_active': False,
                        'side': None,
                        'entry_price': None,
                        'amount': None,
                        'entry_time': None,
                        'order_id': None
                    }
                    
                    # Calculate profit/loss
                    exit_price = result.get('price') or price
                    if position['side'] == 'long':
                        pnl = (exit_price - position['entry_price']) * amount
                        pnl_percent = (exit_price / position['entry_price'] - 1) * 100
                    else:
                        pnl = (position['entry_price'] - exit_price) * amount
                        pnl_percent = (position['entry_price'] / exit_price - 1) * 100
                        
                    # Create trade record
                    trade = Trade(
                        id=result.get('id'),
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=exit_price,
                        timestamp=datetime.now(),
                        strategy=self.name,
                        timeframe=signal.get('timeframe'),
                        status='executed',
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        related_trade_id=position.get('order_id')
                    )
                    
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    logger.info(f"Closed position for {symbol}: {trade}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
                    
            except Exception as e:
                logger.error(f"Error executing signal for {symbol}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
        return executed_trades
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate performance statistics for the strategy.
        
        Returns:
            dict: Performance statistics
        """
        stats = {
            'name': self.name,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'total_trades': len(self.trade_history),
            'active_positions': sum(1 for pos in self.positions.values() if pos['is_active']),
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'total_pnl_percent': 0.0
        }
        
        # Calculate performance metrics from trade history
        if self.trade_history:
            # Filter closing trades with PnL information
            closing_trades = [t for t in self.trade_history if t.pnl is not None]
            
            if closing_trades:
                # Calculate win rate
                winning_trades = sum(1 for t in closing_trades if t.pnl > 0)
                stats['win_rate'] = winning_trades / len(closing_trades) * 100
                
                # Calculate average profit
                stats['avg_profit'] = sum(t.pnl_percent for t in closing_trades) / len(closing_trades)
                
                # Calculate total PnL
                stats['total_pnl'] = sum(t.pnl for t in closing_trades)
                stats['total_pnl_percent'] = sum(t.pnl_percent for t in closing_trades)
                
                # Calculate profit factor
                gross_profit = sum(t.pnl for t in closing_trades if t.pnl > 0)
                gross_loss = abs(sum(t.pnl for t in closing_trades if t.pnl < 0))
                stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Calculate maximum drawdown
                # Sort trades by timestamp
                sorted_trades = sorted(closing_trades, key=lambda t: t.timestamp)
                
                # Calculate cumulative PnL
                cumulative_pnl = [0]
                for trade in sorted_trades:
                    cumulative_pnl.append(cumulative_pnl[-1] + trade.pnl)
                    
                # Calculate drawdown
                peak = 0
                drawdown = 0
                for pnl in cumulative_pnl:
                    if pnl > peak:
                        peak = pnl
                    dd = (peak - pnl) / peak * 100 if peak > 0 else 0
                    drawdown = max(drawdown, dd)
                    
                stats['max_drawdown'] = drawdown
                
                # Calculate Sharpe ratio (simplified)
                pnl_series = [t.pnl_percent for t in sorted_trades]
                if len(pnl_series) > 1:
                    returns_mean = np.mean(pnl_series)
                    returns_std = np.std(pnl_series)
                    stats['sharpe_ratio'] = returns_mean / returns_std if returns_std > 0 else 0
                    
        return stats
        
    def reset(self) -> bool:
        """
        Reset the strategy state.
        
        Returns:
            bool: True if reset successfully, False otherwise
        """
        try:
            # Reset data
            self.data = {
                symbol: {tf: pd.DataFrame() for tf in self.timeframes} 
                for symbol in self.symbols
            }
            
            # Reset indicators
            self.indicators = {
                symbol: {tf: {} for tf in self.timeframes} 
                for symbol in self.symbols
            }
            
            # Reset positions
            self.positions = {
                symbol: {'is_active': False, 'side': None, 'entry_price': None, 'amount': None, 'entry_time': None}
                for symbol in self.symbols
            }
            
            # Reset trade history
            self.trade_history = []
            
            logger.info(f"Strategy {self.name} reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting strategy {self.name}: {str(e)}")
            return False
            
    def save_state(self, filename: str) -> bool:
        """
        Save strategy state to file.
        
        Args:
            filename: Path to save file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            import pickle
            
            # Prepare state
            state = {
                'name': self.name,
                'symbols': self.symbols,
                'timeframes': self.timeframes,
                'params': self.params,
                'positions': self.positions,
                'trade_history': self.trade_history,
                'is_active': self.is_active,
                'last_update_time': self.last_update_time
            }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Strategy state saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving strategy state: {str(e)}")
            return False
            
    def load_state(self, filename: str) -> bool:
        """
        Load strategy state from file.
        
        Args:
            filename: Path to state file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            import pickle
            
            # Load from file
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                
            # Validate state
            if state['name'] != self.name:
                logger.warning(f"Strategy name mismatch: {state['name']} != {self.name}")
                
            # Update state
            self.symbols = state['symbols']
            self.timeframes = state['timeframes']
            self.params = state['params']
            self.positions = state['positions']
            self.trade_history = state['trade_history']
            self.is_active = state['is_active']
            self.last_update_time = state['last_update_time']
            
            # Reinitialize data structures
            self.data = {
                symbol: {tf: pd.DataFrame() for tf in self.timeframes} 
                for symbol in self.symbols
            }
            
            self.indicators = {
                symbol: {tf: {} for tf in self.timeframes} 
                for symbol in self.symbols
            }
            
            logger.info(f"Strategy state loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading strategy state: {str(e)}")
            return False
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary.
        
        Returns:
            dict: Strategy information
        """
        return {
            'name': self.name,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'params': self.params,
            'is_active': self.is_active,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'positions': self.positions,
            'trade_count': len(self.trade_history),
            'performance': self.get_performance_stats()
        }