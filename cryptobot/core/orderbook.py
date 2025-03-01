"""
Order Book Implementation
=======================
Manages and tracks the order book for trading pairs, providing methods to
analyze market depth, liquidity, and order book imbalances.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
import numpy as np
from loguru import logger


class OrderBook:
    """
    Manages an order book for a trading pair.
    
    The OrderBook class keeps track of all asks and bids for a specific trading pair,
    providing methods to analyze market depth, calculate spreads, and identify
    potential support/resistance levels.
    """
    
    def __init__(
        self,
        symbol: str,
        max_depth: int = 100,
        history_size: int = 10,
        grouping_precision: int = 4
    ):
        """
        Initialize the OrderBook.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            max_depth: Maximum depth of order book to maintain
            history_size: Number of historical snapshots to keep
            grouping_precision: Decimal precision for price grouping
        """
        self.symbol = symbol
        self.max_depth = max_depth
        self.history_size = history_size
        self.grouping_precision = grouping_precision
        
        # Current order book
        self.bids: List[List[float]] = []  # [price, amount]
        self.asks: List[List[float]] = []  # [price, amount]
        
        # Keep track of best bid/ask prices
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.spread: float = 0.0
        self.mid_price: float = 0.0
        
        # Order book statistics
        self.bid_volume: float = 0.0
        self.ask_volume: float = 0.0
        self.bid_depth_price: Dict[float, float] = {}
        self.ask_depth_price: Dict[float, float] = {}
        
        # Update tracking
        self.last_update_time: float = 0.0
        self.update_count: int = 0
        
        # Historical snapshots for analysis
        self.history = deque(maxlen=history_size)
        
        logger.debug(f"Initialized order book for {symbol}")
        
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Update the order book with new data.
        
        Args:
            data: Order book data from exchange
                {
                    'bids': [[price, amount], ...],
                    'asks': [[price, amount], ...],
                    'timestamp': timestamp,
                    ...
                }
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Extract data
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            timestamp = data.get('timestamp', time.time() * 1000)
            
            # Save current state to history
            self._save_to_history()
            
            # Update bids and asks
            self.bids = sorted(bids, key=lambda x: float(x[0]), reverse=True)[:self.max_depth]
            self.asks = sorted(asks, key=lambda x: float(x[0]))[:self.max_depth]
            
            # Update best prices
            if self.bids:
                self.best_bid = float(self.bids[0][0])
            if self.asks:
                self.best_ask = float(self.asks[0][0])
            
            # Calculate spread and mid price
            if self.best_bid > 0 and self.best_ask > 0:
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
            
            # Calculate volumes
            self.bid_volume = sum(float(bid[1]) for bid in self.bids)
            self.ask_volume = sum(float(ask[1]) for ask in self.asks)
            
            # Calculate depth at various price levels
            self._calculate_depth()
            
            # Update tracking info
            self.last_update_time = timestamp
            self.update_count += 1
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating order book for {self.symbol}: {str(e)}")
            return False
    
    def update_from_delta(self, delta: Dict[str, Any]) -> bool:
        """
        Update the order book from a delta update (WebSocket).
        
        Args:
            delta: Delta update data
                {
                    'bids': [[price, amount], ...],  # Updated bids
                    'asks': [[price, amount], ...],  # Updated asks
                    'timestamp': timestamp
                }
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Extract data
            bid_updates = delta.get('bids', [])
            ask_updates = delta.get('asks', [])
            timestamp = delta.get('timestamp', time.time() * 1000)
            
            # Save current state to history
            self._save_to_history()
            
            # Process bid updates
            for bid_update in bid_updates:
                price = float(bid_update[0])
                amount = float(bid_update[1])
                
                # Remove price level if amount is zero
                if amount == 0:
                    self.bids = [bid for bid in self.bids if float(bid[0]) != price]
                else:
                    # Update existing price level or add new one
                    updated = False
                    for i, bid in enumerate(self.bids):
                        if float(bid[0]) == price:
                            self.bids[i] = [price, amount]
                            updated = True
                            break
                    
                    if not updated:
                        self.bids.append([price, amount])
            
            # Process ask updates
            for ask_update in ask_updates:
                price = float(ask_update[0])
                amount = float(ask_update[1])
                
                # Remove price level if amount is zero
                if amount == 0:
                    self.asks = [ask for ask in self.asks if float(ask[0]) != price]
                else:
                    # Update existing price level or add new one
                    updated = False
                    for i, ask in enumerate(self.asks):
                        if float(ask[0]) == price:
                            self.asks[i] = [price, amount]
                            updated = True
                            break
                    
                    if not updated:
                        self.asks.append([price, amount])
            
            # Sort and trim
            self.bids = sorted(self.bids, key=lambda x: float(x[0]), reverse=True)[:self.max_depth]
            self.asks = sorted(self.asks, key=lambda x: float(x[0]))[:self.max_depth]
            
            # Update best prices
            if self.bids:
                self.best_bid = float(self.bids[0][0])
            if self.asks:
                self.best_ask = float(self.asks[0][0])
            
            # Calculate spread and mid price
            if self.best_bid > 0 and self.best_ask > 0:
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
            
            # Calculate volumes
            self.bid_volume = sum(float(bid[1]) for bid in self.bids)
            self.ask_volume = sum(float(ask[1]) for ask in self.asks)
            
            # Calculate depth at various price levels
            self._calculate_depth()
            
            # Update tracking info
            self.last_update_time = timestamp
            self.update_count += 1
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating order book from delta for {self.symbol}: {str(e)}")
            return False
    
    def get_price_at_volume(self, volume: float, side: str = 'buy') -> float:
        """
        Calculate the price needed to buy/sell a specific volume.
        
        Args:
            volume: Target volume to buy/sell
            side: 'buy' or 'sell'
            
        Returns:
            float: Estimated price to execute the entire volume
        """
        if volume <= 0:
            return 0.0
        
        if side.lower() == 'buy':
            # For buys, we look at the ask side
            cumulative_volume = 0.0
            for ask in self.asks:
                price = float(ask[0])
                amount = float(ask[1])
                
                if cumulative_volume + amount >= volume:
                    # Found enough volume
                    remaining = volume - cumulative_volume
                    return price
                
                cumulative_volume += amount
            
            # Not enough volume in the order book
            if self.asks:
                return float(self.asks[-1][0])
            return 0.0
            
        elif side.lower() == 'sell':
            # For sells, we look at the bid side
            cumulative_volume = 0.0
            for bid in self.bids:
                price = float(bid[0])
                amount = float(bid[1])
                
                if cumulative_volume + amount >= volume:
                    # Found enough volume
                    remaining = volume - cumulative_volume
                    return price
                
                cumulative_volume += amount
            
            # Not enough volume in the order book
            if self.bids:
                return float(self.bids[-1][0])
            return 0.0
        
        return 0.0
    
    def calculate_market_impact(self, amount: float, side: str = 'buy') -> Tuple[float, float]:
        """
        Calculate the market impact of a market order.
        
        Args:
            amount: Amount to buy/sell
            side: 'buy' or 'sell'
            
        Returns:
            tuple: (average_price, price_impact_percent)
        """
        if amount <= 0:
            return 0.0, 0.0
        
        if side.lower() == 'buy':
            # For buys, we look at the ask side
            total_cost = 0.0
            total_filled = 0.0
            
            for ask in self.asks:
                price = float(ask[0])
                available = float(ask[1])
                
                if total_filled + available >= amount:
                    # This level will complete the order
                    remaining = amount - total_filled
                    total_cost += price * remaining
                    total_filled += remaining
                    break
                else:
                    # Take all available at this level
                    total_cost += price * available
                    total_filled += available
            
            if total_filled == 0:
                return 0.0, 0.0
                
            # Calculate average price
            avg_price = total_cost / total_filled
            
            # Calculate price impact
            if self.best_ask > 0:
                price_impact = (avg_price - self.best_ask) / self.best_ask * 100
            else:
                price_impact = 0.0
                
            return avg_price, price_impact
        
        elif side.lower() == 'sell':
            # For sells, we look at the bid side
            total_revenue = 0.0
            total_filled = 0.0
            
            for bid in self.bids:
                price = float(bid[0])
                available = float(bid[1])
                
                if total_filled + available >= amount:
                    # This level will complete the order
                    remaining = amount - total_filled
                    total_revenue += price * remaining
                    total_filled += remaining
                    break
                else:
                    # Take all available at this level
                    total_revenue += price * available
                    total_filled += available
            
            if total_filled == 0:
                return 0.0, 0.0
                
            # Calculate average price
            avg_price = total_revenue / total_filled
            
            # Calculate price impact
            if self.best_bid > 0:
                price_impact = (self.best_bid - avg_price) / self.best_bid * 100
            else:
                price_impact = 0.0
                
            return avg_price, price_impact
        
        return 0.0, 0.0
    
    def get_liquidity_score(self, reference_price: Optional[float] = None) -> float:
        """
        Calculate a liquidity score (0-100) based on spread and depth.
        
        Args:
            reference_price: Reference price for calculations (default: mid price)
            
        Returns:
            float: Liquidity score (0-100)
        """
        if reference_price is None:
            reference_price = self.mid_price
            
        if reference_price <= 0:
            return 0.0
        
        # Calculate relative spread (lower is better)
        rel_spread = (self.spread / reference_price) if reference_price > 0 else 1.0
        spread_score = max(0, min(40, 40 * (1 - rel_spread * 100)))
        
        # Calculate depth score (higher is better)
        depth_1pct = self._get_depth_at_percent(0.01, reference_price)
        depth_score = min(40, depth_1pct / 100)
        
        # Calculate bid/ask balance (closer to 1.0 is better)
        if self.bid_volume > 0 and self.ask_volume > 0:
            volume_ratio = min(self.bid_volume, self.ask_volume) / max(self.bid_volume, self.ask_volume)
            balance_score = 20 * volume_ratio
        else:
            balance_score = 0
        
        # Combine scores
        liquidity_score = spread_score + depth_score + balance_score
        
        return min(100, max(0, liquidity_score))
    
    def detect_support_resistance(self, num_levels: int = 3) -> Tuple[List[float], List[float]]:
        """
        Detect potential support and resistance levels.
        
        Args:
            num_levels: Number of levels to identify
            
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        support_levels = []
        resistance_levels = []
        
        # Group bids and asks by price with the given precision
        grouped_bids = self._group_by_price(self.bids)
        grouped_asks = self._group_by_price(self.asks)
        
        # Find levels with highest volume
        bid_items = sorted(grouped_bids.items(), key=lambda x: x[1], reverse=True)
        ask_items = sorted(grouped_asks.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top levels
        support_levels = [price for price, _ in bid_items[:num_levels]]
        resistance_levels = [price for price, _ in ask_items[:num_levels]]
        
        return support_levels, resistance_levels
    
    def calculate_order_imbalance(self) -> float:
        """
        Calculate the order book imbalance metric (-1.0 to 1.0).
        
        A value close to 1.0 indicates strong buying pressure,
        while a value close to -1.0 indicates strong selling pressure.
        
        Returns:
            float: Order imbalance between -1 and 1
        """
        if self.bid_volume == 0 and self.ask_volume == 0:
            return 0.0
            
        total_volume = self.bid_volume + self.ask_volume
        
        if total_volume == 0:
            return 0.0
            
        # Calculate imbalance (-1 to 1)
        imbalance = (self.bid_volume - self.ask_volume) / total_volume
        
        return imbalance
    
    def get_vwap(self, depth_percentage: float = 0.01) -> Tuple[float, float]:
        """
        Calculate Volume-Weighted Average Price for bids and asks.
        
        Args:
            depth_percentage: Percentage of price range to consider
            
        Returns:
            tuple: (bid_vwap, ask_vwap)
        """
        if self.mid_price <= 0:
            return 0.0, 0.0
            
        depth_price = self.mid_price * depth_percentage
        
        # Calculate VWAP for bids
        bid_volume_sum = 0.0
        bid_weighted_sum = 0.0
        
        for bid in self.bids:
            price = float(bid[0])
            volume = float(bid[1])
            
            if price >= self.mid_price - depth_price:
                bid_weighted_sum += price * volume
                bid_volume_sum += volume
        
        # Calculate VWAP for asks
        ask_volume_sum = 0.0
        ask_weighted_sum = 0.0
        
        for ask in self.asks:
            price = float(ask[0])
            volume = float(ask[1])
            
            if price <= self.mid_price + depth_price:
                ask_weighted_sum += price * volume
                ask_volume_sum += volume
        
        # Calculate VWAPs
        bid_vwap = bid_weighted_sum / bid_volume_sum if bid_volume_sum > 0 else 0.0
        ask_vwap = ask_weighted_sum / ask_volume_sum if ask_volume_sum > 0 else 0.0
        
        return bid_vwap, ask_vwap
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order book to dictionary.
        
        Returns:
            dict: Order book information
        """
        return {
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'imbalance': self.calculate_order_imbalance(),
            'liquidity_score': self.get_liquidity_score(),
            'last_update_time': self.last_update_time,
            'top_bids': self.bids[:5] if self.bids else [],
            'top_asks': self.asks[:5] if self.asks else []
        }
        
    def _save_to_history(self):
        """Save current state to history."""
        if not self.bids and not self.asks:
            return
            
        snapshot = {
            'timestamp': self.last_update_time,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'imbalance': self.calculate_order_imbalance()
        }
        
        self.history.append(snapshot)
        
    def _calculate_depth(self):
        """Calculate order book depth at various price levels."""
        if not self.bids or not self.asks or self.mid_price <= 0:
            return
            
        # Calculate cumulative depths at different percentages
        for percent in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            self.bid_depth_price[percent] = self._calculate_depth_at_price(
                self.mid_price * (1 - percent), side='bid'
            )
            self.ask_depth_price[percent] = self._calculate_depth_at_price(
                self.mid_price * (1 + percent), side='ask'
            )
            
    def _calculate_depth_at_price(self, price: float, side: str) -> float:
        """
        Calculate cumulative order book depth at a specific price.
        
        Args:
            price: Price level
            side: 'bid' or 'ask'
            
        Returns:
            float: Cumulative volume at price
        """
        if side.lower() == 'bid':
            return sum(float(bid[1]) for bid in self.bids if float(bid[0]) >= price)
        else:  # ask
            return sum(float(ask[1]) for ask in self.asks if float(ask[0]) <= price)
            
    def _get_depth_at_percent(self, percent: float, reference_price: float) -> float:
        """
        Get depth at a percentage from the reference price.
        
        Args:
            percent: Percentage away from reference price
            reference_price: Reference price
            
        Returns:
            float: Combined bid and ask depth
        """
        bid_depth = self._calculate_depth_at_price(
            reference_price * (1 - percent), side='bid'
        )
        ask_depth = self._calculate_depth_at_price(
            reference_price * (1 + percent), side='ask'
        )
        
        return bid_depth + ask_depth
        
    def _group_by_price(self, orders: List[List[float]]) -> Dict[float, float]:
        """
        Group orders by price levels.
        
        Args:
            orders: List of [price, amount] entries
            
        Returns:
            dict: {grouped_price: total_volume}
        """
        grouped = {}
        
        for order in orders:
            price = float(order[0])
            amount = float(order[1])
            
            # Round price to grouping precision
            grouped_price = round(price, self.grouping_precision)
            
            if grouped_price in grouped:
                grouped[grouped_price] += amount
            else:
                grouped[grouped_price] = amount
                
        return grouped