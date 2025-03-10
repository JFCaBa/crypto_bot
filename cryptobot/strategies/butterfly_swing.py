"""
Simplified Butterfly Swing Strategy
==================================
A basic swing trading strategy using RSI for momentum and ATR for risk management.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from loguru import logger
from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager
import pandas_ta as ta

class ButterflySwingStrategy(BaseStrategy):
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        default_params = {
            "rsi_period": 14,
            "rsi_buy": 40,
            "rsi_sell": 60,
            "atr_period": 7,
            "atr_multiplier": 1.0,
            "rr_multiplier": 1.5,
            "trailing_stop_pct": 0.01,
            "position_size_pct": 0.02
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="ButterflySwing",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        self.stop_loss_levels: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
        self.take_profit_levels: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
        
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        try:
            df = self.data[symbol][timeframe]
            if df.empty or len(df) < self.params['rsi_period']:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return False
            df['rsi'] = ta.rsi(df['close'], length=self.params['rsi_period'])
            df['rsi'] = df['rsi'].fillna(50)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.params['atr_period'])
            df['atr'] = df['atr'].fillna(df['close'].pct_change().std() * 100)
            self.indicators[symbol][timeframe] = {'rsi': df['rsi'], 'atr': df['atr']}
            self.data[symbol][timeframe] = df
            logger.debug(f"{symbol} {timeframe}: RSI={df['rsi'].iloc[-1]:.2f}, ATR={df['atr'].iloc[-1]:.8f}")
            return True
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            return False
            
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        signal = {"action": None, "symbol": symbol, "timeframe": timeframe, "price": None, "amount": None, "stop_loss": None, "take_profit": None}
        try:
            df = self.data[symbol][timeframe]
            if df.empty or len(df) < 2 or not self.calculate_indicators(symbol, timeframe):
                logger.debug(f"{symbol} {timeframe}: No data or indicators failed")
                return signal
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            current_price = latest['close']
            position = self.positions[symbol]
            is_active = position['is_active']
            rsi = latest['rsi']
            prev_rsi = previous['rsi']
            atr = latest['atr']
            
            if not is_active:
                if prev_rsi <= self.params['rsi_buy'] and rsi > self.params['rsi_buy']:
                    signal['action'] = 'buy'
                    signal['price'] = current_price
                    stop_loss = current_price - (atr * self.params['atr_multiplier'])
                    take_profit = current_price + (atr * self.params['atr_multiplier'] * self.params['rr_multiplier'])
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    signal['amount'] = self._calculate_position_size(symbol, current_price, atr)
                    self.stop_loss_levels[symbol][timeframe] = stop_loss
                    self.take_profit_levels[symbol][timeframe] = take_profit
                    logger.info(f"{symbol} {timeframe}: BUY at {current_price:.8f}, SL={stop_loss:.8f}, TP={take_profit:.8f}")
                elif prev_rsi >= self.params['rsi_sell'] and rsi < self.params['rsi_sell']:
                    signal['action'] = 'sell'
                    signal['price'] = current_price
                    stop_loss = current_price + (atr * self.params['atr_multiplier'])
                    take_profit = current_price - (atr * self.params['atr_multiplier'] * self.params['rr_multiplier'])
                    signal['stop_loss'] = stop_loss
                    signal['take_profit'] = take_profit
                    signal['amount'] = self._calculate_position_size(symbol, current_price, atr)
                    self.stop_loss_levels[symbol][timeframe] = stop_loss
                    self.take_profit_levels[symbol][timeframe] = take_profit
                    logger.info(f"{symbol} {timeframe}: SELL at {current_price:.8f}, SL={stop_loss:.8f}, TP={take_profit:.8f}")
            else:
                logger.debug(f"{symbol} {timeframe}: Active position side={position['side']}, Price={current_price:.8f}, SL={self.stop_loss_levels[symbol][timeframe]}, TP={self.take_profit_levels[symbol][timeframe]}")
                stop_loss = self.stop_loss_levels.get(symbol, {}).get(timeframe)
                take_profit = self.take_profit_levels.get(symbol, {}).get(timeframe)
                trailing_stop = self.params['trailing_stop_pct']
                if trailing_stop > 0:
                    if position['side'] == 'long':
                        peak = df['high'].iloc[-5:].max()
                        trail_price = peak * (1 - trailing_stop)
                        if current_price <= trail_price:
                            signal['action'] = 'close'
                            logger.info(f"{symbol} {timeframe}: Trailing stop hit at {current_price:.8f}")
                    elif position['side'] == 'short':
                        trough = df['low'].iloc[-5:].min()
                        trail_price = trough * (1 + trailing_stop)
                        if current_price >= trail_price:
                            signal['action'] = 'close'
                            logger.info(f"{symbol} {timeframe}: Trailing stop hit at {current_price:.8f}")
                if position['side'] == 'long':
                    if stop_loss and current_price <= stop_loss:
                        signal['action'] = 'close'
                        logger.info(f"{symbol} {timeframe}: Stop loss hit at {current_price:.8f}")
                    elif take_profit and current_price >= take_profit:
                        signal['action'] = 'close'
                        logger.info(f"{symbol} {timeframe}: Take profit hit at {current_price:.8f}")
                elif position['side'] == 'short':
                    if stop_loss and current_price >= stop_loss:
                        signal['action'] = 'close'
                        logger.info(f"{symbol} {timeframe}: Stop loss hit at {current_price:.8f}")
                    elif take_profit and current_price <= take_profit:
                        signal['action'] = 'close'
                        logger.info(f"{symbol} {timeframe}: Take profit hit at {current_price:.8f}")
                if signal['action'] == 'close':
                    signal['price'] = current_price
                    signal['amount'] = position['amount']
                    self.stop_loss_levels[symbol][timeframe] = None
                    self.take_profit_levels[symbol][timeframe] = None
            return signal
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return signal
            
    def _calculate_position_size(self, symbol: str, price: float, atr: float) -> float:
        try:
            if price <= 0 or atr <= 0:
                logger.warning(f"Invalid price or ATR for {symbol}: price={price}, atr={atr}")
                return 1.0
            account_balance = getattr(self, 'account_size', 10000.0)
            if hasattr(self.risk_manager, 'get_available_balance'):
                account_balance = self.risk_manager.get_available_balance()
            elif hasattr(self, 'balance'):
                account_balance = self.balance
            risk_amount = account_balance * self.params['position_size_pct']
            risk_per_unit = atr * self.params['atr_multiplier']
            if risk_per_unit < 0.000001:
                logger.warning(f"Risk per unit too small: {risk_per_unit}")
                return 1.0
            position_size = risk_amount / risk_per_unit
            max_size = account_balance / price * 0.5
            if position_size > max_size:
                position_size = max_size
                logger.debug(f"Position size capped at {max_size} due to balance {account_balance}")
            elif position_size * price > account_balance:
                position_size = account_balance / price
                logger.debug(f"Position size adjusted to {position_size} to fit balance {account_balance}")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 1.0