"""
Trade Model
==========
Represents a trading transaction with associated metadata.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any


class Trade:
    """
    Represents a trading transaction with associated metadata.
    """
    
    def __init__(
        self,
        id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        timestamp: Union[datetime, str],
        strategy: str,
        timeframe: str,
        status: str = 'executed',
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        fee: Optional[float] = None,
        slippage: Optional[float] = None,
        related_trade_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize a trade object.
        
        Args:
            id: Trade ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Execution price
            timestamp: Execution timestamp
            strategy: Strategy that generated the trade
            timeframe: Timeframe used for the strategy
            status: Trade status ('pending', 'executed', 'canceled', 'failed')
            pnl: Profit and loss amount
            pnl_percent: Profit and loss percentage
            fee: Trading fee
            slippage: Price slippage
            related_trade_id: ID of related trade (e.g., for closing orders)
            tags: Additional tags for categorization
        """
        self.id = id
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.price = price
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            self.timestamp = datetime.fromisoformat(timestamp)
        else:
            self.timestamp = timestamp
            
        self.strategy = strategy
        self.timeframe = timeframe
        self.status = status
        self.pnl = pnl
        self.pnl_percent = pnl_percent
        self.fee = fee
        self.slippage = slippage
        self.related_trade_id = related_trade_id
        self.tags = tags or []
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary.
        
        Returns:
            dict: Trade information
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'amount': self.amount,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy,
            'timeframe': self.timeframe,
            'status': self.status,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'fee': self.fee,
            'slippage': self.slippage,
            'related_trade_id': self.related_trade_id,
            'tags': self.tags
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """
        Create a trade from dictionary.
        
        Args:
            data: Trade data dictionary
            
        Returns:
            Trade: Trade object
        """
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=data['side'],
            amount=data['amount'],
            price=data['price'],
            timestamp=data['timestamp'],
            strategy=data['strategy'],
            timeframe=data['timeframe'],
            status=data.get('status', 'executed'),
            pnl=data.get('pnl'),
            pnl_percent=data.get('pnl_percent'),
            fee=data.get('fee'),
            slippage=data.get('slippage'),
            related_trade_id=data.get('related_trade_id'),
            tags=data.get('tags', [])
        )
        
    def __str__(self) -> str:
        """
        Get string representation of trade.
        
        Returns:
            str: Trade information
        """
        return (
            f"Trade(id={self.id}, symbol={self.symbol}, side={self.side}, "
            f"amount={self.amount:.8f}, price={self.price:.8f})"
        )
        
    def __repr__(self) -> str:
        """
        Get string representation of trade.
        
        Returns:
            str: Trade information
        """
        return self.__str__()