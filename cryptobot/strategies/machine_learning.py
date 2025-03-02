"""
Machine Learning Strategy
=========================
Implementation of a machine learning-based trading strategy using XGBoost
for price movement prediction.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cryptobot.strategies.base import BaseStrategy
from cryptobot.risk_management.manager import RiskManager


class MachineLearningStrategy(BaseStrategy):
    """
    Machine Learning Strategy.
    
    This strategy uses an XGBoost model to predict price movements based on
    technical indicators (RSI, Bollinger Bands, Moving Averages) and generates
    trading signals accordingly.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        lookback_period: int = 100,
        train_interval: int = 1440,  # Train every 1440 minutes (1 day)
        risk_manager: RiskManager = None,
        params: Dict[str, Any] = None
    ):
        """
        Initialize the Machine Learning strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframes: List of timeframes to analyze
            lookback_period: Number of historical candles to use for training
            train_interval: Interval (in minutes) to retrain the model
            risk_manager: Risk manager instance
            params: Additional strategy parameters
        """
        # Initialize base strategy
        default_params = {
            'lookback_period': lookback_period,
            'train_interval': train_interval,
            'rsi_period': 14,
            'boll_period': 20,
            'boll_dev': 2,
            'ma_period': 50,
            'prediction_threshold': 0.6,  # Threshold for ML prediction confidence
            'stop_loss': 0.02,  # Stop loss (% of entry price)
            'take_profit': 0.04,  # Take profit (% of entry price)
            'risk_per_trade': 0.01,  # Risk per trade (% of account)
            'train_test_split': 0.8,  # Split ratio for training/testing
            'xgb_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,  # Reduce depth to prevent overfitting
                'learning_rate': 0.05,  # Lower learning rate for slower learning
                'n_estimators': 100,
                'random_state': 42,
                'lambda': 1.0,  # L2 regularization
                'alpha': 0.1,   # L1 regularization
                'subsample': 0.8,  # Subsample ratio of training instances
                'colsample_bytree': 0.8  # Subsample ratio of columns
            }
        }
        
        # Merge default and user params
        if params:
            default_params.update(params)
            
        super().__init__(
            name="MachineLearningStrategy",
            symbols=symbols,
            timeframes=timeframes,
            params=default_params,
            risk_manager=risk_manager
        )
        
        # Initialize ML-related attributes
        self.models: Dict[str, Dict[str, Optional[xgb.XGBClassifier]]] = {
            symbol: {tf: None for tf in timeframes} 
            for symbol in symbols
        }
        self.scalers: Dict[str, Dict[str, StandardScaler]] = {
            symbol: {tf: StandardScaler() for tf in timeframes} 
            for symbol in symbols
        }
        self.last_train_time: Dict[str, Dict[str, Optional[pd.Timestamp]]] = {
            symbol: {tf: None for tf in timeframes} 
            for symbol in symbols
        }
        
    def calculate_indicators(self, symbol: str, timeframe: str) -> bool:
        """
        Calculate technical indicators for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if calculated successfully, False otherwise
        """
        try:
            df = self.data[symbol][timeframe]
            
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return False
                
            # Ensure we have OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in data for {symbol} {timeframe}: {missing_columns}")
                return False
                
            # Extract parameters
            rsi_period = self.params['rsi_period']
            boll_period = self.params['boll_period']
            boll_dev = self.params['boll_dev']
            ma_period = self.params['ma_period']
            
            # Check if we have enough data points
            min_required = max(rsi_period, boll_period, ma_period) + 10
            if len(df) < min_required:
                logger.warning(f"Not enough data points for {symbol} {timeframe}. Need at least {min_required}, got {len(df)}.")
                return False
                
            # Calculate RSI with safely handling NaN and division by zero
            delta = df['close'].diff().fillna(0)
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['boll_mid'] = df['close'].rolling(window=boll_period).mean()
            df['boll_std'] = df['close'].rolling(window=boll_period).std()
            df['boll_upper'] = df['boll_mid'] + (boll_dev * df['boll_std'])
            df['boll_lower'] = df['boll_mid'] - (boll_dev * df['boll_std'])
            
            # Calculate Moving Average
            df['ma'] = df['close'].ewm(span=ma_period, adjust=False).mean()
            
            # Fill NaN values for all indicators
            df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI
            df['boll_mid'] = df['boll_mid'].fillna(df['close'])
            df['boll_std'] = df['boll_std'].fillna(0)
            df['boll_upper'] = df['boll_upper'].fillna(df['close'] * 1.01)  # 1% above price
            df['boll_lower'] = df['boll_lower'].fillna(df['close'] * 0.99)  # 1% below price
            df['ma'] = df['ma'].fillna(df['close'])
            
            # Store calculated indicators
            indicators = {
                'rsi': df['rsi'],
                'boll_mid': df['boll_mid'],
                'boll_upper': df['boll_upper'],
                'boll_lower': df['boll_lower'],
                'ma': df['ma']
            }
            
            # Save indicators to instance
            self.indicators[symbol][timeframe] = indicators
            
            # Update data with calculated indicators
            self.data[symbol][timeframe] = df
            
            # Train or retrain the model if needed
            self._train_model(symbol, timeframe)
            
            return True
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML model training.
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Define features
        features = [
            'rsi', 'boll_mid', 'boll_upper', 'boll_lower', 'ma',
            'close', 'volume'
        ]
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Create empty DataFrames to avoid errors
            empty_X = pd.DataFrame(columns=features)
            empty_y = pd.Series(dtype=int)
            return empty_X, empty_y
            
        # Create feature DataFrame
        X = df[features].copy()
        
        # Create target: 1 if next candle closes higher, 0 otherwise
        try:
            y = (df['close'].shift(-1) > df['close']).astype(int)
        except:
            # Handle the case where shift operation fails
            y = pd.Series(index=X.index, data=0)
            
        # Drop rows with NaN values
        X = X.dropna()
        if len(X) == 0:
            # If all rows are dropped, return empty DataFrames
            empty_X = pd.DataFrame(columns=features)
            empty_y = pd.Series(dtype=int)
            return empty_X, empty_y
            
        y = y.loc[X.index]
        
        return X, y
        
    def _train_model(self, symbol: str, timeframe: str) -> bool:
        """
        Train or retrain the XGBoost model for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if trained successfully, False otherwise
        """
        try:
            df = self.data[symbol][timeframe]
            if len(df) < self.params['lookback_period']:
                logger.warning(f"Insufficient data to train model for {symbol} {timeframe}")
                return False
                
            # Check if we need to train/retrain
            now = pd.Timestamp.now()
            last_train = self.last_train_time[symbol][timeframe]
            train_interval = pd.Timedelta(minutes=self.params['train_interval'])
            
            if last_train is not None and (now - last_train) < train_interval:
                logger.info(f"Skipping training for {symbol} {timeframe}, last trained at {last_train}")
                return True
                
            # Prepare features and target
            X, y = self._prepare_features(df.tail(self.params['lookback_period']))
            
            if len(X) < 50:  # Minimum data points for training
                logger.warning(f"Too few data points after preprocessing for {symbol} {timeframe}: {len(X)}")
                return False
                
            # Check if there's enough variation in the target
            y_counts = y.value_counts()
            if len(y_counts) < 2 or y_counts.min() < 10:
                logger.warning(f"Not enough variation in target for {symbol} {timeframe}: {y_counts.to_dict()}")
                return False
                
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=self.params['train_test_split'], shuffle=False
                )
            except Exception as e:
                logger.error(f"Error splitting data for {symbol} {timeframe}: {str(e)}")
                return False
                
            # Scale features
            try:
                scaler = self.scalers[symbol][timeframe]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                logger.error(f"Error scaling features for {symbol} {timeframe}: {str(e)}")
                return False
                
            # Train XGBoost model
            try:
                model = xgb.XGBClassifier(**self.params['xgb_params'])
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                logger.error(f"Error training model for {symbol} {timeframe}: {str(e)}")
                return False
                
            # Evaluate model
            try:
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                logger.info(f"Model for {symbol} {timeframe} - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
            except Exception as e:
                logger.error(f"Error evaluating model for {symbol} {timeframe}: {str(e)}")
                return False
                
            # Save model and update training time
            self.models[symbol][timeframe] = model
            self.last_train_time[symbol][timeframe] = now
            
            return True
        except Exception as e:
            logger.error(f"Error in model training process for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def generate_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signals for a symbol and timeframe using ML predictions.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            dict: Signal information including action, price, etc.
        """
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': None,
            'price': None,
            'amount': None,
            'params': {},
            'timestamp': pd.Timestamp.now()
        }
        
        try:
            # Ensure indicators are calculated
            if not self.calculate_indicators(symbol, timeframe):
                logger.warning(f"Failed to calculate indicators for {symbol} {timeframe}")
                return signal
                
            # Get model and scaler
            model = self.models[symbol][timeframe]
            scaler = self.scalers[symbol][timeframe]
            
            if model is None:
                logger.warning(f"No trained model for {symbol} {timeframe}")
                return signal
                
            # Get latest data point
            df = self.data[symbol][timeframe]
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                return signal
                
            # Prepare features for prediction
            latest_data = df.tail(1)
            X, _ = self._prepare_features(latest_data)
            if X.empty:
                logger.warning(f"Insufficient features for prediction for {symbol} {timeframe}")
                return signal
                
            # Scale features
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                logger.error(f"Error scaling features for prediction for {symbol} {timeframe}: {str(e)}")
                return signal
                
            # Predict probability
            try:
                pred_proba = model.predict_proba(X_scaled)[0]
                bullish_prob = pred_proba[1] if len(pred_proba) > 1 else 0.5  # Probability of price increase
            except Exception as e:
                logger.error(f"Error making prediction for {symbol} {timeframe}: {str(e)}")
                return signal
                
            # Extract parameters
            threshold = self.params['prediction_threshold']
            stop_loss = self.params['stop_loss']
            take_profit = self.params['take_profit']
            risk_per_trade = self.params['risk_per_trade']
            
            # Get current price
            current_price = latest_data['close'].iloc[-1]
            
            # Check position status
            position = self.positions[symbol]
            is_active = position['is_active']
            
            if not is_active:
                # Generate entry signals
                if bullish_prob >= threshold:
                    signal['action'] = 'buy'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(
                        symbol, current_price, risk_per_trade, stop_loss
                    )
                    if stop_loss > 0:
                        signal['stop_loss'] = current_price * (1 - stop_loss)
                    if take_profit > 0:
                        signal['take_profit'] = current_price * (1 + take_profit)
                elif bullish_prob <= (1 - threshold):
                    signal['action'] = 'sell'
                    signal['price'] = current_price
                    signal['amount'] = self._calculate_position_size(
                        symbol, current_price, risk_per_trade, stop_loss
                    )
                    if stop_loss > 0:
                        signal['stop_loss'] = current_price * (1 + stop_loss)
                    if take_profit > 0:
                        signal['take_profit'] = current_price * (1 - take_profit)
            else:
                # Generate exit signals
                if position['side'] == 'long' and bullish_prob <= (1 - threshold):
                    signal['action'] = 'close'
                    signal['price'] = current_price
                    signal['amount'] = position['amount']
                elif position['side'] == 'short' and bullish_prob >= threshold:
                    signal['action'] = 'close'
                    signal['price'] = current_price
                    signal['amount'] = position['amount']
            
            return signal
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return signal
            
    def _calculate_position_size(self, symbol: str, price: float, risk_percent: float, stop_loss_percent: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            risk_percent: Risk per trade as percentage of account
            stop_loss_percent: Stop loss percentage
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Input validation
            if not price or price <= 0:
                logger.warning(f"Invalid price for position size calculation: {price}")
                return 1.0
                
            # Default position size (1 unit) if risk parameters are invalid
            if risk_percent <= 0 or stop_loss_percent <= 0:
                logger.info(f"Using default position size due to invalid risk parameters: risk_percent={risk_percent}, stop_loss_percent={stop_loss_percent}")
                return 1.0
                
            # Get account balance from backtesting engine or use default
            account_balance = getattr(self, 'account_size', 10000.0)
            
            # Calculate max amount to risk
            risk_amount = account_balance * risk_percent
            
            # Calculate potential loss per unit based on stop loss
            loss_per_unit = price * stop_loss_percent
            
            # Safeguard against division by zero or very small numbers
            if loss_per_unit < 0.000001:
                logger.warning(f"Stop loss too small for calculation: {loss_per_unit}")
                return 1.0
                
            # Calculate position size
            position_size = risk_amount / loss_per_unit
            
            # Cap position size if it's unreasonably large
            max_size = account_balance / price * 0.5  # Max 50% of account in a single position
            if position_size > max_size:
                logger.warning(f"Position size capped from {position_size} to {max_size}")
                position_size = max_size
                
            logger.info(f"Calculated position size: {position_size} units at price {price}")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default fallback
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary.
        
        Returns:
            dict: Strategy information
        """
        base_dict = super().to_dict()
        base_dict['last_train_time'] = {
            symbol: {
                tf: str(time) if time else None
                for tf, time in tf_data.items()
            }
            for symbol, tf_data in self.last_train_time.items()
        }
        return base_dict