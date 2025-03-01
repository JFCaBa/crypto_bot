"""
Portfolio Management
=================
Manages the trading portfolio including positions, balances, and performance metrics.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from loguru import logger

from cryptobot.core.trade import Trade


class Portfolio:
    """
    Portfolio management class for tracking assets, positions, and performance.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        base_currency: str = 'USDT',
        risk_free_rate: float = 0.02  # 2% annual risk-free rate for performance calculations
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_balance: Initial balance in base currency
            base_currency: Base currency for portfolio valuation
            risk_free_rate: Annual risk-free rate for performance calculations
        """
        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        
        # Asset balances by currency code
        self.balances: Dict[str, float] = {base_currency: initial_balance}
        
        # Open positions
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position data
        
        # Position history
        self.closed_positions: List[Dict[str, Any]] = []
        
        # Trade history
        self.trades: List[Trade] = []
        
        # Performance metrics
        self.equity_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Portfolio state
        self.total_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.peak_equity: float = initial_balance
        
        # Last update time
        self.last_update_time: float = time.time()
        
        logger.info(f"Portfolio initialized with {initial_balance} {base_currency}")
        
    def update_balances(self, balances: Dict[str, float]) -> bool:
        """
        Update account balances.
        
        Args:
            balances: Dictionary of currency balances
                {
                    'BTC': 0.1,
                    'ETH': 2.5,
                    'USDT': 5000.0,
                    ...
                }
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Update balances
            for currency, amount in balances.items():
                self.balances[currency] = float(amount)
                
            self.last_update_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error updating balances: {str(e)}")
            return False
            
    def update_positions(
        self,
        current_prices: Dict[str, float],
        positions: Dict[str, Dict[str, Any]] = None
    ) -> bool:
        """
        Update positions with current market prices.
        
        Args:
            current_prices: Current market prices by symbol
                {
                    'BTC/USDT': 50000.0,
                    'ETH/USDT': 3000.0,
                    ...
                }
            positions: Updated position information (optional)
                {
                    'BTC/USDT': {
                        'amount': 0.1,
                        'entry_price': 45000.0,
                        'current_price': 50000.0,
                        ...
                    },
                    ...
                }
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if positions:
                # Update from external position data
                self.positions = positions
            
            # Update current prices in positions
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position['current_price'] = current_prices[symbol]
                    
                    # Calculate position PnL
                    entry_price = position.get('entry_price', 0.0)
                    amount = position.get('amount', 0.0)
                    side = position.get('side', 'long')
                    
                    if side == 'long':
                        position['unrealized_pnl'] = (current_prices[symbol] - entry_price) * amount
                        position['unrealized_pnl_percent'] = ((current_prices[symbol] / entry_price) - 1) * 100 if entry_price > 0 else 0
                    else:  # short
                        position['unrealized_pnl'] = (entry_price - current_prices[symbol]) * amount
                        position['unrealized_pnl_percent'] = ((entry_price / current_prices[symbol]) - 1) * 100 if current_prices[symbol] > 0 else 0
            
            # Calculate total unrealized PnL
            self.unrealized_pnl = sum(p.get('unrealized_pnl', 0.0) for p in self.positions.values())
            
            # Update portfolio equity
            self._update_equity(current_prices)
            
            self.last_update_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            return False
            
    def add_position(
        self,
        symbol: str,
        amount: float,
        entry_price: float,
        side: str = 'long',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            amount: Position size in base currency
            entry_price: Entry price
            side: Position side ('long' or 'short')
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Position leverage (1.0 = no leverage)
            timestamp: Position opening timestamp
            metadata: Additional position metadata
            
        Returns:
            dict: Position information
        """
        if timestamp is None:
            timestamp = time.time()
            
        if metadata is None:
            metadata = {}
            
        # Check if position already exists
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists, adding to existing position")
            existing_position = self.positions[symbol]
            
            # Calculate new average entry price
            total_value = (existing_position['entry_price'] * existing_position['amount']) + (entry_price * amount)
            total_amount = existing_position['amount'] + amount
            new_entry_price = total_value / total_amount if total_amount > 0 else entry_price
            
            # Update existing position
            existing_position['amount'] = total_amount
            existing_position['entry_price'] = new_entry_price
            existing_position['stop_loss'] = stop_loss
            existing_position['take_profit'] = take_profit
            existing_position['modified_at'] = timestamp
            
            # Update metadata
            existing_position['metadata'].update(metadata)
            
            return existing_position
        
        # Create new position
        position = {
            'symbol': symbol,
            'amount': amount,
            'entry_price': entry_price,
            'current_price': entry_price,
            'side': side,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'leverage': leverage,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_percent': 0.0,
            'created_at': timestamp,
            'modified_at': timestamp,
            'metadata': metadata
        }
        
        # Add position to portfolio
        self.positions[symbol] = position
        
        logger.info(f"Added new position for {symbol}: {amount} @ {entry_price}")
        return position
        
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: Optional[float] = None,
        partial_amount: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close a position in the portfolio.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            exit_price: Exit price
            timestamp: Position closing timestamp
            partial_amount: Amount to close (None for full position)
            metadata: Additional metadata about the close
            
        Returns:
            dict: Closed position information or None if position not found
        """
        if timestamp is None:
            timestamp = time.time()
            
        if metadata is None:
            metadata = {}
            
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
            
        position = self.positions[symbol]
        close_amount = partial_amount if partial_amount is not None else position['amount']
        
        if close_amount <= 0 or close_amount > position['amount']:
            logger.warning(f"Invalid close amount: {close_amount}")
            return None
            
        # Calculate realized PnL
        entry_price = position['entry_price']
        side = position['side']
        
        if side == 'long':
            realized_pnl = (exit_price - entry_price) * close_amount
            realized_pnl_percent = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        else:  # short
            realized_pnl = (entry_price - exit_price) * close_amount
            realized_pnl_percent = ((entry_price / exit_price) - 1) * 100 if exit_price > 0 else 0
            
        # Create a record of the closed position
        closed_position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': close_amount,
            'side': side,
            'realized_pnl': realized_pnl,
            'realized_pnl_percent': realized_pnl_percent,
            'entry_time': position['created_at'],
            'exit_time': timestamp,
            'duration': timestamp - position['created_at'],
            'metadata': {**position['metadata'], **metadata}
        }
        
        # Add to closed positions
        self.closed_positions.append(closed_position)
        
        # Update realized PnL
        self.realized_pnl += realized_pnl
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # If partial close, update position
        if close_amount < position['amount']:
            position['amount'] -= close_amount
            position['modified_at'] = timestamp
            logger.info(f"Partially closed position for {symbol}: {close_amount} @ {exit_price}, PnL: {realized_pnl}")
        else:
            # Full close, remove position
            del self.positions[symbol]
            logger.info(f"Closed position for {symbol}: {close_amount} @ {exit_price}, PnL: {realized_pnl}")
            
        return closed_position
        
    def add_trade(self, trade: Trade) -> bool:
        """
        Add a trade to the portfolio history.
        
        Args:
            trade: Trade object
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self.trades.append(trade)
            
            # If the trade has PnL information, update portfolio PnL
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                self.realized_pnl += trade.pnl
                self.total_pnl = self.realized_pnl + self.unrealized_pnl
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
            return False
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            dict: Performance metrics including returns, drawdown, ratios, etc.
        """
        if not self.equity_history:
            return self.performance_metrics
            
        try:
            # Calculate daily returns
            equity_df = pd.DataFrame(self.equity_history)
            equity_df['date'] = pd.to_datetime(equity_df['timestamp'], unit='s')
            equity_df = equity_df.set_index('date')
            
            # Resample to daily and calculate returns
            daily_equity = equity_df['equity'].resample('D').last().dropna()
            if len(daily_equity) > 1:
                daily_returns = daily_equity.pct_change().dropna()
                self.daily_returns = daily_returns.tolist()
                
                # Calculate annualized return
                total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
                if total_days > 0:
                    total_return = (daily_equity.iloc[-1] / daily_equity.iloc[0]) - 1
                    annualized_return = ((1 + total_return) ** (365.0 / total_days)) - 1
                else:
                    annualized_return = 0.0
                    
                # Calculate volatility
                if len(daily_returns) > 1:
                    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                else:
                    volatility = 0.0
                    
                # Calculate Sharpe ratio
                if volatility > 0:
                    daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
                    excess_returns = daily_returns - daily_risk_free
                    sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                    
                # Calculate Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino_ratio = (daily_returns.mean() - daily_risk_free) / downside_returns.std() * np.sqrt(252)
                else:
                    sortino_ratio = 0.0
                    
                # Calculate Calmar ratio
                if self.max_drawdown > 0:
                    calmar_ratio = annualized_return / self.max_drawdown
                else:
                    calmar_ratio = 0.0
                    
                # Calculate win rate
                profitable_trades = [t for t in self.closed_positions if t['realized_pnl'] > 0]
                win_rate = len(profitable_trades) / len(self.closed_positions) * 100 if self.closed_positions else 0.0
                
                # Calculate profit factor
                gross_profit = sum(t['realized_pnl'] for t in self.closed_positions if t['realized_pnl'] > 0)
                gross_loss = abs(sum(t['realized_pnl'] for t in self.closed_positions if t['realized_pnl'] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Update performance metrics
                self.performance_metrics = {
                    'total_return': total_return * 100,  # As percentage
                    'annualized_return': annualized_return * 100,  # As percentage
                    'volatility': volatility * 100,  # As percentage
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'max_drawdown': self.max_drawdown * 100,  # As percentage
                    'current_drawdown': self.current_drawdown * 100,  # As percentage
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': len(self.trades),
                    'closed_positions': len(self.closed_positions),
                    'open_positions': len(self.positions),
                    'realized_pnl': self.realized_pnl,
                    'unrealized_pnl': self.unrealized_pnl,
                    'total_pnl': self.total_pnl
                }
                
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return self.performance_metrics
            
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all positions.
        
        Returns:
            dict: Position summary
        """
        summary = {
            'open_positions': len(self.positions),
            'total_exposure': sum(p['amount'] * p['current_price'] for p in self.positions.values()),
            'unrealized_pnl': self.unrealized_pnl,
            'positions': []
        }
        
        for symbol, position in self.positions.items():
            summary['positions'].append({
                'symbol': symbol,
                'side': position['side'],
                'amount': position['amount'],
                'entry_price': position['entry_price'],
                'current_price': position['current_price'],
                'unrealized_pnl': position['unrealized_pnl'],
                'unrealized_pnl_percent': position['unrealized_pnl_percent']
            })
            
        return summary
        
    def get_trade_history(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trade history filtered by time and symbols.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            symbols: List of symbols to filter
            
        Returns:
            list: Filtered trade history
        """
        filtered_trades = []
        
        for trade in self.trades:
            # Apply time filter
            trade_time = trade.timestamp.timestamp() if isinstance(trade.timestamp, datetime) else trade.timestamp
            if start_time and trade_time < start_time:
                continue
            if end_time and trade_time > end_time:
                continue
                
            # Apply symbol filter
            if symbols and trade.symbol not in symbols:
                continue
                
            filtered_trades.append(trade.to_dict())
            
        return filtered_trades
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert portfolio to dictionary.
        
        Returns:
            dict: Portfolio information
        """
        return {
            'base_currency': self.base_currency,
            'initial_balance': self.initial_balance,
            'current_equity': self._calculate_total_equity(),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'balances': self.balances,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_trades': len(self.trades),
            'performance': self.performance_metrics,
            'last_update_time': self.last_update_time
        }
        
    def _update_equity(self, current_prices: Dict[str, float]) -> bool:
        """
        Update portfolio equity and record it in history.
        
        Args:
            current_prices: Current market prices by symbol
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Calculate total equity
            equity = self._calculate_total_equity()
            
            # Record in equity history
            entry = {
                'timestamp': time.time(),
                'equity': equity,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl
            }
            self.equity_history.append(entry)
            
            # Calculate drawdown
            if equity > self.peak_equity:
                self.peak_equity = equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
                
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating equity: {str(e)}")
            return False
            
    def _calculate_total_equity(self) -> float:
        """
        Calculate total portfolio equity.
        
        Returns:
            float: Total equity in base currency
        """
        # Base currency balance
        equity = self.balances.get(self.base_currency, 0.0)
        
        # Add unrealized PnL
        equity += self.unrealized_pnl
        
        # Add value of other assets (requires converting to base currency)
        # This would normally use current market prices, but for now,
        # we'll rely on position data being up-to-date
        for symbol, position in self.positions.items():
            if position.get('current_price') and position.get('amount'):
                equity += position['current_price'] * position['amount']
        
        # Add value of other currency balances not in positions
        # This would normally need exchange rates, but we'll assume
        # these are tracked elsewhere
        
        return equity
        
    def asset_allocation(self) -> Dict[str, float]:
        """
        Calculate current asset allocation as percentages.
        
        Returns:
            dict: Asset allocation percentages
        """
        total_equity = self._calculate_total_equity()
        
        if total_equity <= 0:
            return {}
            
        allocation = {}
        
        # Base currency allocation
        base_balance = self.balances.get(self.base_currency, 0.0)
        allocation[self.base_currency] = (base_balance / total_equity) * 100
        
        # Position allocations
        for symbol, position in self.positions.items():
            if position.get('current_price') and position.get('amount'):
                position_value = position['current_price'] * position['amount']
                allocation[symbol] = (position_value / total_equity) * 100
                
        # Other currency balances
        for currency, amount in self.balances.items():
            if currency != self.base_currency and amount > 0:
                # We would need to convert this to base currency value
                # For now, we'll skip these in the allocation
                pass
                
        return allocation
        
    def rebalance_targets(self, target_allocation: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate rebalancing targets to achieve desired allocation.
        
        Args:
            target_allocation: Target allocation percentages by asset
            
        Returns:
            dict: Rebalancing instructions by asset
        """
        current_allocation = self.asset_allocation()
        total_equity = self._calculate_total_equity()
        
        rebalance_instructions = {}
        
        for asset, target_pct in target_allocation.items():
            current_pct = current_allocation.get(asset, 0.0)
            
            # Calculate target value
            target_value = (target_pct / 100) * total_equity
            
            # Calculate current value
            current_value = 0.0
            
            if asset == self.base_currency:
                current_value = self.balances.get(self.base_currency, 0.0)
            elif asset in self.positions:
                position = self.positions[asset]
                current_value = position.get('current_price', 0.0) * position.get('amount', 0.0)
            else:
                # Might be another currency or asset not in a position
                current_value = self.balances.get(asset, 0.0)
                # Would need price conversion here
                
            # Calculate difference
            value_diff = target_value - current_value
            pct_diff = target_pct - current_pct
            
            rebalance_instructions[asset] = {
                'current_allocation': current_pct,
                'target_allocation': target_pct,
                'current_value': current_value,
                'target_value': target_value,
                'value_difference': value_diff,
                'percentage_difference': pct_diff
            }
            
        return rebalance_instructions
        
    def calculate_diversification_metrics(self) -> Dict[str, float]:
        """
        Calculate diversification metrics for the portfolio.
        
        Returns:
            dict: Diversification metrics
        """
        # Get allocations
        allocation = self.asset_allocation()
        
        if not allocation:
            return {
                'asset_count': 0,
                'concentration': 0.0,
                'herfindahl_index': 0.0,
                'effective_n': 0.0
            }
            
        # Number of assets
        asset_count = len(allocation)
        
        # Largest position concentration
        concentration = max(allocation.values())
        
        # Herfindahl-Hirschman Index (sum of squared percentages)
        # Lower values indicate better diversification
        hhi = sum((pct / 100) ** 2 for pct in allocation.values())
        
        # Effective N (inverse of HHI)
        # Higher values indicate better diversification
        effective_n = 1 / hhi if hhi > 0 else 0
        
        return {
            'asset_count': asset_count,
            'concentration': concentration,
            'herfindahl_index': hhi,
            'effective_n': effective_n
        }
        
    def simulate_drawdown(self, percent_change: Dict[str, float]) -> float:
        """
        Simulate portfolio drawdown based on percentage changes.
        
        Args:
            percent_change: Percentage change by asset
            
        Returns:
            float: Simulated drawdown percentage
        """
        current_equity = self._calculate_total_equity()
        
        # Calculate new equity
        new_equity = self.balances.get(self.base_currency, 0.0)
        
        # Apply changes to positions
        for symbol, position in self.positions.items():
            if symbol in percent_change and position.get('current_price') and position.get('amount'):
                change = percent_change[symbol] / 100
                new_price = position['current_price'] * (1 + change)
                new_equity += new_price * position['amount']
            elif position.get('current_price') and position.get('amount'):
                # No change for this asset
                new_equity += position['current_price'] * position['amount']
                
        # Calculate drawdown
        if current_equity <= 0:
            return 0.0
            
        drawdown = (current_equity - new_equity) / current_equity
        return drawdown * 100  # As percentage