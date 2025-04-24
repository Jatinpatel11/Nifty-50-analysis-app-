import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for storing models and predictions
os.makedirs('models', exist_ok=True)
os.makedirs('data/predictions', exist_ok=True)
os.makedirs('plots/predictions', exist_ok=True)

class PricePredictionModel:
    def __init__(self, price_data_path='data/nifty50_with_indicators.csv', 
                 sentiment_data_path='data/sentiment/overall_sentiment.csv'):
        """
        Initialize the Price Prediction Model
        
        Parameters:
        -----------
        price_data_path : str
            Path to the price data with technical indicators
        sentiment_data_path : str
            Path to the sentiment data
        """
        print("Initializing Price Prediction Model...")
        
        # Load price data with technical indicators
        try:
            print(f"Loading price data from {price_data_path}...")
            self.price_data = pd.read_csv(price_data_path)
            self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
            self.price_data.set_index('Date', inplace=True)
            print(f"Loaded price data with {len(self.price_data)} rows and {len(self.price_data.columns)} columns")
            
            # Print column names for debugging
            print("Available columns:", self.price_data.columns.tolist())
        except Exception as e:
            print(f"Error loading price data: {e}")
            # If the file with indicators doesn't exist, try loading the original price data
            try:
                print("Trying to load original price data...")
                self.price_data = pd.read_csv('data/nifty50_data.csv')
                self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
                self.price_data.set_index('Date', inplace=True)
                print(f"Loaded original price data with {len(self.price_data)} rows")
                print("Technical indicators will be calculated during preprocessing")
            except Exception as e:
                print(f"Error loading original price data: {e}")
                self.price_data = pd.DataFrame()
        
        # Load sentiment data if available
        try:
            print(f"Loading sentiment data from {sentiment_data_path}...")
            self.sentiment_data = pd.read_csv(sentiment_data_path)
            print(f"Loaded sentiment data with {len(self.sentiment_data)} rows")
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
            self.sentiment_data = pd.DataFrame()
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialize models
        self.models = {}
    
    def preprocess_data(self, prediction_horizon=5, sequence_length=60, test_size=0.2):
        """
        Preprocess data for model training
        
        Parameters:
        -----------
        prediction_horizon : int
            Number of days to predict ahead
        sequence_length : int
            Number of previous time steps to use for LSTM model
        test_size : float
            Proportion of data to use for testing
        """
        print(f"Preprocessing data with prediction horizon of {prediction_horizon} days...")
        
        if self.price_data.empty:
            print("No price data available for preprocessing")
            return None, None, None, None
        
        # Make a copy of the price data
        data = self.price_data.copy()
        
        # Calculate additional technical indicators if needed
        print("Calculating additional technical indicators...")
        
        # Price momentum (if not already present)
        if 'Price_Momentum' not in data.columns:
            data['Price_Momentum'] = data['Close'] - data['Close'].shift(5)
        
        # Volume ratio (if not already present)
        if 'Volume_Ratio' not in data.columns:
            if 'Volume_MA_5' not in data.columns:
                data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_5']
        
        # Create target variable (future price change)
        data['Target'] = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
        
        # Add sentiment data if available
        if not self.sentiment_data.empty:
            print("Adding sentiment data...")
            # Use the latest sentiment score for all days (since we have limited sentiment data)
            latest_sentiment = self.sentiment_data['compound'].iloc[0]
            data['Sentiment'] = latest_sentiment
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Select features for prediction
        # Check which features are available and use only those
        available_features = []
        for feature in ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                       'BB_Width', 'Price_Momentum', 'Volume_Ratio']:
            if feature in data.columns:
                available_features.append(feature)
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Add sentiment if available
        if 'Sentiment' in data.columns:
            available_features.append('Sentiment')
        
        # Split data into features and target
        X = data[available_features]
        y = data['Target']
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Save the feature names for later use
        self.feature_names = available_features
        
        # Split data into training and testing sets
        # For time series data, we use the last portion as test set
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Prepare data for LSTM model
        X_lstm_train, y_lstm_train = self._create_sequences(X_train, y_train.values, sequence_length)
        X_lstm_test, y_lstm_test = self._create_sequences(X_test, y_test.values, sequence_length)
        
        print(f"LSTM training data shape: {X_lstm_train.shape}")
        print(f"LSTM testing data shape: {X_lstm_test.shape}")
        
        # Save the original data for later use
        self.data = data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_lstm_train = X_lstm_train
        self.X_lstm_test = X_lstm_test
        self.y_lstm_train = y_lstm_train
        self.y_lstm_test = y_lstm_test
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, X, y, sequence_length):
        """
        Create sequences for LSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        sequence_length : int
            Number of previous time steps to use
        
        Returns:
        --------
        X_seq : numpy.ndarray
            Sequence of features
        y_seq : numpy.ndarray
            Sequence of targets
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_linear_regression_model(self):
        """
        Build and train a linear regression model
        """
        print("Building Linear Regression model...")
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("Data not preprocessed. Run preprocess_data() first.")
            return None
        
        # Create and train the model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Linear Regression - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Linear Regression - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Linear Regression - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Store the model
        self.models['linear_regression'] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        # Save the model
        joblib.dump(model, 'models/linear_regression_model.pkl')
        print("Linear Regression model saved to models/linear_regression_model.pkl")
        
        return model
    
    def build_random_forest_model(self):
        """
        Build and train a random forest model
        """
        print("Building Random Forest model...")
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("Data not preprocessed. Run preprocess_data() first.")
            return None
        
        # Create and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Random Forest - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Random Forest - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Random Forest - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 important features:")
        print(feature_importances.head(5))
        
        # Store the model
        self.models['random_forest'] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'feature_importances': feature_importances
        }
        
        # Save the model
        joblib.dump(model, 'models/random_forest_model.pkl')
        print("Random Forest model saved to models/random_forest_model.pkl")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
        plt.title('Random Forest Feature Importances')
        plt.tight_layout()
        plt.savefig('plots/predictions/feature_importances.png')
        print("Feature importances plot saved to plots/predictions/feature_importances.png")
        
        return model
    
    def build_gradient_boosting_model(self):
        """
        Build and train a gradient boosting model
        """
        print("Building Gradient Boosting model...")
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("Data not preprocessed. Run preprocess_data() first.")
            return None
        
        # Create and train the model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Gradient Boosting - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Gradient Boosting - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Gradient Boosting - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Store the model
        self.models['gradient_boosting'] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        # Save the model
        joblib.dump(model, 'models/gradient_boosting_model.pkl')
        print("Gradient Boosting model saved to models/gradient_boosting_model.pkl")
        
        return model
    
    def build_lstm_model(self):
        """
        Build and train an LSTM model
        """
        print("Building LSTM model...")
        
        if not hasattr(self, 'X_lstm_train') or self.X_lstm_train is None:
            print("Data not preprocessed. Run preprocess_data() first.")
            return None
        
        # Define the model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.X_lstm_train.shape[1], self.X_lstm_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        history = model.fit(
            self.X_lstm_train, self.y_lstm_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_train = model.predict(self.X_lstm_train)
        y_pred_test = model.predict(self.X_lstm_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_lstm_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_lstm_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_lstm_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_lstm_test, y_pred_test)
        
        print(f"LSTM - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"LSTM - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        # Store the model
        self.models['lstm'] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'predictions': {
                'train': y_pred_train.flatten(),
                'test': y_pred_test.flatten()
            },
            'history': history.history
        }
        
        # Save the model
        model.save('models/lstm_model')
        print("LSTM model saved to models/lstm_model")
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/predictions/lstm_training_history.png')
        print("LSTM training history plot saved to plots/predictions/lstm_training_history.png")
        
        return model
    
    def compare_models(self):
        """
        Compare the performance of different models
        """
        print("Comparing model performance...")
        
        if not self.models:
            print("No models to compare. Build models first.")
            return
        
        # Create a DataFrame to store model metrics
        metrics = []
        
        for model_name, model_info in self.models.items():
            if 'metrics' in model_info:
                metric = {
                    'Model': model_name,
                    'Train RMSE': model_info['metrics'].get('train_rmse', np.nan),
                    'Test RMSE': model_info['metrics'].get('test_rmse', np.nan),
                    'Train MAE': model_info['metrics'].get('train_mae', np.nan),
                    'Test MAE': model_info['metrics'].get('test_mae', np.nan)
                }
                
                if 'train_r2' in model_info['metrics']:
                    metric['Train R²'] = model_info['metrics']['train_r2']
                    metric['Test R²'] = model_info['metrics']['test_r2']
                
                metrics.append(metric)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Save metrics to CSV
        metrics_df.to_csv('data/predictions/model_comparison.csv', index=False)
        print("Model comparison saved to data/predictions/model_comparison.csv")
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        # Plot RMSE
        plt.subplot(1, 2, 1)
        metrics_df_plot = metrics_df.set_index('Model')
        metrics_df_plot[['Train RMSE', 'Test RMSE']].plot(kind='bar', ax=plt.gca())
        plt.title('RMSE Comparison')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        metrics_df_plot[['Train MAE', 'Test MAE']].plot(kind='bar', ax=plt.gca())
        plt.title('MAE Comparison')
        plt.ylabel('MAE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/predictions/model_comparison.png')
        print("Model comparison plot saved to plots/predictions/model_comparison.png")
        
        return metrics_df
    
    def plot_predictions(self, model_name='random_forest'):
        """
        Plot actual vs predicted values for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot predictions for
        """
        print(f"Plotting predictions for {model_name} model...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Build the model first.")
            return
        
        model_info = self.models[model_name]
        
        if 'predictions' not in model_info:
            print(f"No predictions found for {model_name} model.")
            return
        
        # Get predictions
        y_train = self.y_train
        y_test = self.y_test
        y_pred_train = model_info['predictions']['train']
        y_pred_test = model_info['predictions']['test']
        
        # Plot actual vs predicted for test set
        plt.figure(figsize=(14, 7))
        
        # Convert target (percent change) to actual future prices for better visualization
        last_train_idx = len(y_train)
        test_indices = np.arange(last_train_idx, last_train_idx + len(y_test))
        
        # Get the dates for the test set
        test_dates = self.data.index[test_indices]
        
        # Plot actual vs predicted percent changes
        plt.subplot(2, 1, 1)
        plt.plot(test_dates, y_test.values, label='Actual % Change', color='blue')
        plt.plot(test_dates, y_pred_test, label='Predicted % Change', color='red', linestyle='--')
        plt.title(f'{model_name.replace("_", " ").title()} - Actual vs Predicted (% Change)')
        plt.xlabel('Date')
        plt.ylabel('Price Change (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and plot actual vs predicted prices
        plt.subplot(2, 1, 2)
        
        # Get the actual close prices for the test period
        actual_prices = self.data['Close'].iloc[test_indices].values
        
        # Calculate predicted prices based on percent changes
        predicted_prices = actual_prices * (1 + y_pred_test)
        
        # Shift to get the future prices (prediction_horizon days ahead)
        future_actual_prices = self.data['Close'].shift(-self.prediction_horizon).iloc[test_indices].values
        
        # Remove NaN values
        valid_indices = ~np.isnan(future_actual_prices)
        future_actual_prices = future_actual_prices[valid_indices]
        predicted_prices = predicted_prices[valid_indices]
        plot_dates = test_dates[valid_indices]
        
        plt.plot(plot_dates, future_actual_prices, label=f'Actual Price ({self.prediction_horizon} days ahead)', color='blue')
        plt.plot(plot_dates, predicted_prices, label=f'Predicted Price ({self.prediction_horizon} days ahead)', color='red', linestyle='--')
        plt.title(f'{model_name.replace("_", " ").title()} - Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/predictions/{model_name}_predictions.png')
        print(f"Predictions plot saved to plots/predictions/{model_name}_predictions.png")
    
    def make_future_predictions(self, days=30, model_name='random_forest'):
        """
        Make predictions for future dates
        
        Parameters:
        -----------
        days : int
            Number of days to predict into the future
        model_name : str
            Name of the model to use for predictions
        """
        print(f"Making future predictions for the next {days} days using {model_name} model...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Build the model first.")
            return None
        
        if self.data.empty:
            print("No data available for predictions.")
            return None
        
        model = self.models[model_name]['model']
        
        # Get the latest data
        latest_data = self.data.iloc[-self.sequence_length:].copy()
        
        # Initialize arrays to store predictions
        future_dates = [self.data.index[-1] + timedelta(days=i+1) for i in range(days)]
        future_prices = []
        
        # Get the last known price
        last_price = self.data['Close'].iloc[-1]
        current_price = last_price
        
        # Make predictions for each future day
        for i in range(days):
            # Prepare features for prediction
            if model_name == 'lstm':
                # For LSTM, we need a sequence of data
                X = latest_data[self.feature_names].values
                X_scaled = self.feature_scaler.transform(X)
                X_lstm = X_scaled.reshape(1, self.sequence_length, len(self.feature_names))
                
                # Make prediction
                pred_percent_change = model.predict(X_lstm)[0][0]
            else:
                # For other models, we use the latest data point
                X = latest_data[self.feature_names].iloc[-1:].values
                X_scaled = self.feature_scaler.transform(X)
                
                # Make prediction
                pred_percent_change = model.predict(X_scaled)[0]
            
            # Calculate predicted price
            pred_price = current_price * (1 + pred_percent_change)
            future_prices.append(pred_price)
            
            # Update current price for next iteration
            current_price = pred_price
            
            # Update latest data for next iteration (simplified approach)
            # In a real scenario, we would need to update all features
            new_row = latest_data.iloc[-1].copy()
            new_row['Close'] = pred_price
            new_row.name = future_dates[i]
            latest_data = pd.concat([latest_data, pd.DataFrame([new_row])])
            latest_data = latest_data.iloc[1:]  # Remove the oldest row
        
        # Create a DataFrame with the predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_prices,
            'Percent_Change': [(price / last_price - 1) * 100 for price in future_prices]
        })
        
        # Save predictions to CSV
        future_df.to_csv('data/predictions/future_predictions.csv', index=False)
        print("Future predictions saved to data/predictions/future_predictions.csv")
        
        # Plot future predictions
        plt.figure(figsize=(14, 7))
        
        # Plot historical and predicted prices
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index[-30:], self.data['Close'].iloc[-30:], label='Historical Close', color='blue')
        plt.plot(future_df['Date'], future_df['Predicted_Close'], label='Predicted Close', color='red', linestyle='--')
        plt.title(f'Nifty 50 Price Prediction - Next {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot predicted percent change
        plt.subplot(2, 1, 2)
        plt.bar(future_df['Date'], future_df['Percent_Change'], color='green', alpha=0.7)
        plt.title('Predicted Percent Change from Last Close')
        plt.xlabel('Date')
        plt.ylabel('Percent Change (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/predictions/future_predictions.png')
        print("Future predictions plot saved to plots/predictions/future_predictions.png")
        
        return future_df
    
    def run_all_models(self, prediction_horizon=5, sequence_length=60):
        """
        Run all models and compare their performance
        
        Parameters:
        -----------
        prediction_horizon : int
            Number of days to predict ahead
        sequence_length : int
            Number of previous time steps to use for LSTM model
        """
        print("Running all models...")
        
        # Preprocess data
        self.preprocess_data(prediction_horizon=prediction_horizon, sequence_length=sequence_length)
        
        # Build and train models
        self.build_linear_regression_model()
        self.build_random_forest_model()
        self.build_gradient_boosting_model()
        self.build_lstm_model()
        
        # Compare models
        metrics_df = self.compare_models()
        
        # Plot predictions for the best model
        if metrics_df is not None and not metrics_df.empty:
            # Find the model with the lowest test RMSE
            best_model = metrics_df.loc[metrics_df['Test RMSE'].idxmin(), 'Model']
            print(f"Best model based on Test RMSE: {best_model}")
            
            # Plot predictions for the best model
            self.plot_predictions(model_name=best_model)
            
            # Make future predictions using the best model
            self.make_future_predictions(days=30, model_name=best_model)
        
        print("All models completed successfully!")
        
        return metrics_df

if __name__ == "__main__":
    # Create an instance of the PricePredictionModel
    price_prediction = PricePredictionModel()
    
    # Run all models
    metrics_df = price_prediction.run_all_models(prediction_horizon=5, sequence_length=60)
