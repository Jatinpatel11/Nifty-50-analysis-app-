import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import os

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

class PriceAnalysis:
    def __init__(self, data_path='data/nifty50_data.csv'):
        """
        Initialize the Price Analysis class
        
        Parameters:
        -----------
        data_path : str
            Path to the Nifty 50 data CSV file
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Create directory for saving plots
        os.makedirs('plots', exist_ok=True)
        
        print(f"Data loaded successfully with {len(self.data)} rows")
        
    def add_technical_indicators(self):
        """
        Add technical indicators to the data
        """
        print("Adding technical indicators...")
        
        # Trend indicators
        # SMA - Simple Moving Average
        self.data['SMA_20'] = SMAIndicator(close=self.data['Close'], window=20).sma_indicator()
        self.data['SMA_50'] = SMAIndicator(close=self.data['Close'], window=50).sma_indicator()
        self.data['SMA_200'] = SMAIndicator(close=self.data['Close'], window=200).sma_indicator()
        
        # EMA - Exponential Moving Average
        self.data['EMA_20'] = EMAIndicator(close=self.data['Close'], window=20).ema_indicator()
        self.data['EMA_50'] = EMAIndicator(close=self.data['Close'], window=50).ema_indicator()
        
        # MACD - Moving Average Convergence Divergence
        macd = MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        
        # Momentum indicators
        # RSI - Relative Strength Index
        self.data['RSI'] = RSIIndicator(close=self.data['Close']).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'])
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        
        # Volatility indicators
        # Bollinger Bands
        bollinger = BollingerBands(close=self.data['Close'])
        self.data['BB_High'] = bollinger.bollinger_hband()
        self.data['BB_Mid'] = bollinger.bollinger_mavg()
        self.data['BB_Low'] = bollinger.bollinger_lband()
        self.data['BB_Width'] = bollinger.bollinger_wband()
        
        # ATR - Average True Range
        self.data['ATR'] = AverageTrueRange(high=self.data['High'], low=self.data['Low'], close=self.data['Close']).average_true_range()
        
        # Volume indicators
        # OBV - On Balance Volume
        self.data['OBV'] = OnBalanceVolumeIndicator(close=self.data['Close'], volume=self.data['Volume']).on_balance_volume()
        
        # VWAP - Volume Weighted Average Price (daily)
        self.data['VWAP'] = VolumeWeightedAveragePrice(
            high=self.data['High'], 
            low=self.data['Low'], 
            close=self.data['Close'], 
            volume=self.data['Volume']
        ).volume_weighted_average_price()
        
        # Calculate daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change() * 100
        
        # Calculate volatility (rolling standard deviation of returns)
        self.data['Volatility_20'] = self.data['Daily_Return'].rolling(window=20).std()
        
        print("Technical indicators added successfully")
        
        # Save the data with indicators
        self.data.to_csv('data/nifty50_with_indicators.csv')
        print("Data with indicators saved to data/nifty50_with_indicators.csv")
        
        return self.data
    
    def plot_price_trends(self):
        """
        Plot price trends with moving averages
        """
        print("Plotting price trends...")
        
        plt.figure(figsize=(16, 10))
        
        # Plot price and moving averages
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue', alpha=0.7)
        plt.plot(self.data.index, self.data['SMA_20'], label='SMA 20', color='red', alpha=0.7)
        plt.plot(self.data.index, self.data['SMA_50'], label='SMA 50', color='green', alpha=0.7)
        plt.plot(self.data.index, self.data['SMA_200'], label='SMA 200', color='purple', alpha=0.7)
        
        plt.title('Nifty 50 Price Trends with Moving Averages')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(self.data.index, self.data['Volume'], color='blue', alpha=0.5)
        plt.title('Trading Volume')
        plt.ylabel('Volume')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/price_trends.png')
        print("Price trends plot saved to plots/price_trends.png")
        
    def plot_bollinger_bands(self):
        """
        Plot Bollinger Bands
        """
        print("Plotting Bollinger Bands...")
        
        plt.figure(figsize=(16, 8))
        
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue', alpha=0.7)
        plt.plot(self.data.index, self.data['BB_High'], label='Upper Band', color='red', alpha=0.5)
        plt.plot(self.data.index, self.data['BB_Mid'], label='Middle Band', color='green', alpha=0.5)
        plt.plot(self.data.index, self.data['BB_Low'], label='Lower Band', color='red', alpha=0.5)
        
        plt.fill_between(self.data.index, self.data['BB_High'], self.data['BB_Low'], color='gray', alpha=0.1)
        
        plt.title('Nifty 50 with Bollinger Bands')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/bollinger_bands.png')
        print("Bollinger Bands plot saved to plots/bollinger_bands.png")
        
    def plot_momentum_indicators(self):
        """
        Plot momentum indicators (RSI and Stochastic)
        """
        print("Plotting momentum indicators...")
        
        plt.figure(figsize=(16, 12))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.title('Nifty 50 Price')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot RSI
        plt.subplot(3, 1, 2)
        plt.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        plt.title('Relative Strength Index (RSI)')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        
        # Plot Stochastic Oscillator
        plt.subplot(3, 1, 3)
        plt.plot(self.data.index, self.data['Stoch_K'], label='Stoch %K', color='blue')
        plt.plot(self.data.index, self.data['Stoch_D'], label='Stoch %D', color='red')
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        plt.title('Stochastic Oscillator')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/momentum_indicators.png')
        print("Momentum indicators plot saved to plots/momentum_indicators.png")
        
    def plot_macd(self):
        """
        Plot MACD indicator
        """
        print("Plotting MACD indicator...")
        
        plt.figure(figsize=(16, 12))
        
        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.title('Nifty 50 Price')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot MACD
        plt.subplot(2, 1, 2)
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data.index, self.data['MACD_Signal'], label='Signal Line', color='red')
        
        # Plot histogram
        plt.bar(self.data.index, self.data['MACD_Histogram'], label='Histogram', color='green', alpha=0.5)
        
        plt.title('Moving Average Convergence Divergence (MACD)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/macd.png')
        print("MACD plot saved to plots/macd.png")
        
    def plot_volatility(self):
        """
        Plot volatility indicators
        """
        print("Plotting volatility indicators...")
        
        plt.figure(figsize=(16, 12))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.title('Nifty 50 Price')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot ATR
        plt.subplot(3, 1, 2)
        plt.plot(self.data.index, self.data['ATR'], label='ATR', color='purple')
        plt.title('Average True Range (ATR)')
        plt.ylabel('ATR')
        plt.legend()
        plt.grid(True)
        
        # Plot Volatility (20-day rolling std of returns)
        plt.subplot(3, 1, 3)
        plt.plot(self.data.index, self.data['Volatility_20'], label='20-Day Volatility', color='red')
        plt.title('20-Day Rolling Volatility')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/volatility.png')
        print("Volatility plot saved to plots/volatility.png")
        
    def plot_volume_indicators(self):
        """
        Plot volume indicators
        """
        print("Plotting volume indicators...")
        
        plt.figure(figsize=(16, 12))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.title('Nifty 50 Price')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot Volume
        plt.subplot(3, 1, 2)
        plt.bar(self.data.index, self.data['Volume'], label='Volume', color='blue', alpha=0.5)
        plt.title('Trading Volume')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        
        # Plot OBV
        plt.subplot(3, 1, 3)
        plt.plot(self.data.index, self.data['OBV'], label='On Balance Volume', color='green')
        plt.title('On Balance Volume (OBV)')
        plt.ylabel('OBV')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/volume_indicators.png')
        print("Volume indicators plot saved to plots/volume_indicators.png")
        
    def identify_support_resistance(self):
        """
        Identify support and resistance levels
        """
        print("Identifying support and resistance levels...")
        
        # Use a simple method to identify support and resistance levels
        # We'll use the rolling min and max over different windows
        
        # Short-term levels (20 days)
        self.data['Support_ST'] = self.data['Low'].rolling(window=20).min()
        self.data['Resistance_ST'] = self.data['High'].rolling(window=20).max()
        
        # Medium-term levels (50 days)
        self.data['Support_MT'] = self.data['Low'].rolling(window=50).min()
        self.data['Resistance_MT'] = self.data['High'].rolling(window=50).max()
        
        # Long-term levels (100 days)
        self.data['Support_LT'] = self.data['Low'].rolling(window=100).min()
        self.data['Resistance_LT'] = self.data['High'].rolling(window=100).max()
        
        # Plot the support and resistance levels
        plt.figure(figsize=(16, 10))
        
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        
        # Plot short-term levels
        plt.plot(self.data.index, self.data['Support_ST'], label='Short-term Support', color='green', linestyle='--', alpha=0.7)
        plt.plot(self.data.index, self.data['Resistance_ST'], label='Short-term Resistance', color='red', linestyle='--', alpha=0.7)
        
        # Plot medium-term levels
        plt.plot(self.data.index, self.data['Support_MT'], label='Medium-term Support', color='green', linestyle='-', alpha=0.5)
        plt.plot(self.data.index, self.data['Resistance_MT'], label='Medium-term Resistance', color='red', linestyle='-', alpha=0.5)
        
        plt.title('Nifty 50 with Support and Resistance Levels')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/support_resistance.png')
        print("Support and resistance plot saved to plots/support_resistance.png")
        
    def analyze_market_trends(self):
        """
        Analyze market trends and generate insights
        """
        print("Analyzing market trends...")
        
        # Get the latest data point
        latest_data = self.data.iloc[-1]
        
        # Calculate the current trend based on moving averages
        current_price = latest_data['Close']
        sma_20 = latest_data['SMA_20']
        sma_50 = latest_data['SMA_50']
        sma_200 = latest_data['SMA_200']
        
        # Determine trend based on moving average relationships
        if current_price > sma_20 > sma_50 > sma_200:
            trend = "Strong Uptrend"
        elif current_price > sma_20 and current_price > sma_50 and current_price > sma_200:
            trend = "Uptrend"
        elif current_price < sma_20 < sma_50 < sma_200:
            trend = "Strong Downtrend"
        elif current_price < sma_20 and current_price < sma_50 and current_price < sma_200:
            trend = "Downtrend"
        else:
            trend = "Sideways/Neutral"
            
        # Check for golden cross (SMA 50 crosses above SMA 200)
        golden_cross = False
        if self.data['SMA_50'].iloc[-2] <= self.data['SMA_200'].iloc[-2] and self.data['SMA_50'].iloc[-1] > self.data['SMA_200'].iloc[-1]:
            golden_cross = True
            
        # Check for death cross (SMA 50 crosses below SMA 200)
        death_cross = False
        if self.data['SMA_50'].iloc[-2] >= self.data['SMA_200'].iloc[-2] and self.data['SMA_50'].iloc[-1] < self.data['SMA_200'].iloc[-1]:
            death_cross = True
            
        # Check RSI conditions
        rsi = latest_data['RSI']
        if rsi > 70:
            rsi_condition = "Overbought"
        elif rsi < 30:
            rsi_condition = "Oversold"
        else:
            rsi_condition = "Neutral"
            
        # Check MACD conditions
        macd = latest_data['MACD']
        macd_signal = latest_data['MACD_Signal']
        macd_hist = latest_data['MACD_Histogram']
        
        if macd > macd_signal and macd_hist > 0:
            macd_condition = "Bullish"
        elif macd < macd_signal and macd_hist < 0:
            macd_condition = "Bearish"
        else:
            macd_condition = "Neutral"
            
        # Check Bollinger Bands conditions
        bb_width = latest_data['BB_Width']
        bb_width_avg = self.data['BB_Width'].rolling(window=20).mean().iloc[-1]
        
        if bb_width > bb_width_avg * 1.2:
            volatility_condition = "High Volatility"
        elif bb_width < bb_width_avg * 0.8:
            volatility_condition = "Low Volatility"
        else:
            volatility_condition = "Normal Volatility"
            
        # Calculate performance metrics
        # 1-month return
        one_month_return = (current_price / self.data['Close'].iloc[-22] - 1) * 100 if len(self.data) >= 22 else None
        
        # 3-month return
        three_month_return = (current_price / self.data['Close'].iloc[-66] - 1) * 100 if len(self.data) >= 66 else None
        
        # 6-month return
        six_month_return = (current_price / self.data['Close'].iloc[-132] - 1) * 100 if len(self.data) >= 132 else None
        
        # 1-year return
        one_year_return = (current_price / self.data['Close'].iloc[-252] - 1) * 100 if len(self.data) >= 252 else None
        
        # Create a dictionary with the analysis results
        analysis_results = {
            'Current Price': current_price,
            'Current Trend': trend,
            'Golden Cross': golden_cross,
            'Death Cross': death_cross,
            'RSI Condition': rsi_condition,
            'MACD Condition': macd_condition,
            'Volatility Condition': volatility_condition,
            '1-Month Return (%)': one_month_return,
            '3-Month Return (%)': three_month_return,
            '6-Month Return (%)': six_month_return,
            '1-Year Return (%)': one_year_return
        }
        
        # Save the analysis results to a CSV file
        pd.DataFrame([analysis_results]).to_csv('data/market_trend_analysis.csv', index=False)
        print("Market trend analysis saved to data/market_trend_analysis.csv")
        
        return analysis_results
    
    def run_all_analysis(self):
        """
        Run all price analysis functions
        """
        print("Running all price analysis...")
        
        # Add technical indicators
        self.add_technical_indicators()
        
        # Generate all plots
        self.plot_price_trends()
        self.plot_bollinger_bands()
        self.plot_momentum_indicators()
        self.plot_macd()
        self.plot_volatility()
        self.plot_volume_indicators()
        self.identify_support_resistance()
        
        # Analyze market trends
        analysis_results = self.analyze_market_trends()
        
        print("Price analysis completed successfully!")
        return analysis_results

if __name__ == "__main__":
    # Create an instance of the PriceAnalysis class
    price_analysis = PriceAnalysis()
    
    # Run all analysis
    analysis_results = price_analysis.run_all_analysis()
    
    # Print the analysis results
    print("\nMarket Trend Analysis Results:")
    for key, value in analysis_results.items():
        print(f"{key}: {value}")
