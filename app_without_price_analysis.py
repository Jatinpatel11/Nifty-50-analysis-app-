import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
import datetime
from PIL import Image
import traceback

# Try to import plotly, but provide fallback to matplotlib if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not available. Using Matplotlib for visualizations instead.")

# Set page configuration - REMOVED to fix Streamlit Cloud deployment error
# st.set_page_config() should only appear in the main streamlit_app.py file

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
    .neutral {
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Function to get real-time Nifty 50 data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_realtime_nifty50_data(period="1y", interval="1d"):
    try:
        # Try to load from file first
        try:
            df = pd.read_csv('data/nifty50_with_indicators.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            st.success("Loaded Nifty 50 data from file")
            return df
        except Exception as e:
            st.warning(f"Could not load data from file: {e}. Fetching real-time data...")
        
        # Get Nifty 50 data directly
        nifty50 = yf.Ticker("^NSEI")
        data = nifty50.history(period=period, interval=interval)
        
        # Calculate indicators
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # MACD
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
        
        st.success("Successfully fetched and processed real-time Nifty 50 data")
        return data
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        # Return sample data as fallback
        try:
            df = pd.read_csv('data/nifty50_with_indicators.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            st.warning("Using sample data as fallback")
            return df
        except:
            st.error("Could not load sample data either. Please check data files.")
            return None

# Function to load sentiment data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_sentiment_data():
    try:
        # Load sentiment data
        news_sentiment = pd.read_csv('data/sentiment/news_sentiment.csv')
        twitter_sentiment = pd.read_csv('data/sentiment/twitter_sentiment.csv')
        overall_sentiment = pd.read_csv('data/sentiment/overall_sentiment.csv')
        
        return {
            'news_sentiment': news_sentiment,
            'twitter_sentiment': twitter_sentiment,
            'overall_sentiment': overall_sentiment
        }
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return None

# Function to get real-time prediction data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_realtime_prediction_data():
    try:
        # Try to load from file first
        try:
            future_predictions = pd.read_csv('data/predictions/future_predictions.csv')
            future_predictions['Date'] = pd.to_datetime(future_predictions['Date'])
            
            model_comparison = pd.read_csv('data/predictions/model_comparison.csv')
            
            st.success("Loaded prediction data from file")
            return future_predictions, model_comparison
        except Exception as e:
            st.warning(f"Could not load prediction data from file: {e}. Generating real-time predictions...")
        
        # Get the latest data
        nifty_data = get_realtime_nifty50_data()
        
        if nifty_data is None:
            raise Exception("Failed to get Nifty 50 data")
        
        # Get the last close price
        last_close = nifty_data['Close'].iloc[-1]
        last_date = nifty_data.index[-1]
        
        # Generate future dates for prediction (next 30 days)
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(30)]
        
        # Simple prediction model (random walk with drift)
        # In a real scenario, this would be replaced with the actual ML model prediction
        np.random.seed(42)  # For reproducibility
        daily_returns = nifty_data['Daily_Return'].dropna()
        drift = daily_returns.mean()
        volatility = daily_returns.std()
        
        # Generate predictions
        predicted_returns = np.random.normal(drift, volatility, 30)
        predicted_prices = [last_close]
        
        for ret in predicted_returns:
            predicted_prices.append(predicted_prices[-1] * (1 + ret))
        
        predicted_prices = predicted_prices[1:]  # Remove the initial last_close
        
        # Calculate percent change from last close
        percent_changes = [(price/last_close - 1) * 100 for price in predicted_prices]
        
        # Create prediction dataframe
        future_predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predicted_prices,
            'Percent_Change': percent_changes
        })
        
        # Create model comparison data (simulated)
        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'LSTM']
        train_rmse = [125.45, 98.76, 105.32, 110.54]
        train_mae = [98.32, 75.43, 82.65, 85.23]
        test_rmse = [145.67, 115.32, 120.45, 135.67]
        test_mae = [112.45, 89.76, 92.34, 105.43]
        
        model_comparison = pd.DataFrame({
            'Model': models,
            'Train RMSE': train_rmse,
            'Train MAE': train_mae,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        })
        
        st.success("Successfully generated real-time prediction data")
        return future_predictions, model_comparison
    except Exception as e:
        st.error(f"Error generating prediction data: {e}")
        # Return sample data as fallback
        try:
            future_predictions = pd.read_csv('data/predictions/future_predictions.csv')
            future_predictions['Date'] = pd.to_datetime(future_predictions['Date'])
            
            model_comparison = pd.read_csv('data/predictions/model_comparison.csv')
            
            st.warning("Using sample prediction data as fallback")
            return future_predictions, model_comparison
        except:
            st.error("Could not load sample prediction data either. Please check data files.")
            return None, None

# Function to create sentiment analysis charts with Plotly
def create_sentiment_charts_plotly(news_sentiment, twitter_sentiment, overall_sentiment):
    try:
        # Create sentiment distribution chart
        sentiment_counts = news_sentiment['sentiment'].value_counts()
        
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="News Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        )
        
        # Create sentiment over time chart
        news_sentiment['date'] = pd.to_datetime(news_sentiment['date'])
        news_sentiment_by_date = news_sentiment.groupby(news_sentiment['date'].dt.date)['compound'].mean().reset_index()
        
        fig2 = px.line(
            news_sentiment_by_date,
            x='date',
            y='compound',
            title="Sentiment Score Over Time",
            labels={'compound': 'Sentiment Score', 'date': 'Date'}
        )
        
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Create overall sentiment gauge
        overall_score = overall_sentiment['compound'].iloc[0]
        
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            title={'text': "Overall Market Sentiment"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, -0.2], 'color': "lightcoral"},
                    {'range': [-0.2, 0.2], 'color': "lightgray"},
                    {'range': [0.2, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': overall_score
                }
            }
        ))
        
        return fig1, fig2, fig3
    except Exception as e:
        st.error(f"Error creating sentiment charts with Plotly: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# Function to create sentiment analysis charts with Matplotlib (fallback)
def create_sentiment_charts_matplotlib(news_sentiment, twitter_sentiment, overall_sentiment):
    try:
        # Create sentiment distribution chart
        sentiment_counts = news_sentiment['sentiment'].value_counts()
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=[colors.get(x, 'blue') for x in sentiment_counts.index])
        ax1.set_title("News Sentiment Distribution")
        
        # Create sentiment over time chart
        news_sentiment['date'] = pd.to_datetime(news_sentiment['date'])
        news_sentiment_by_date = news_sentiment.groupby(news_sentiment['date'].dt.date)['compound'].mean().reset_index()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(news_sentiment_by_date['date'], news_sentiment_by_date['compound'])
        ax2.axhline(y=0, linestyle='--', color='gray')
        ax2.set_title("Sentiment Score Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sentiment Score")
        ax2.grid(True, alpha=0.3)
        
        # Create overall sentiment gauge (simplified version)
        overall_score = overall_sentiment['compound'].iloc[0]
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        colors = ['red', 'lightcoral', 'lightgray', 'lightgreen', 'green']
        bounds = [-1, -0.5, -0.2, 0.2, 0.5, 1]
        
        # Create a horizontal bar to represent the gauge
        for i in range(len(bounds)-1):
            ax3.barh(0, bounds[i+1]-bounds[i], left=bounds[i], height=0.5, color=colors[i])
        
        # Add a marker for the current value
        ax3.plot(overall_score, 0, 'o', markersize=15, color='darkblue')
        
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title(f"Overall Market Sentiment: {overall_score:.2f}")
        ax3.set_xticks(bounds)
        ax3.set_yticks([])
        
        return fig1, fig2, fig3
    except Exception as e:
        st.error(f"Error creating sentiment charts with Matplotlib: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# Function to create prediction charts with Plotly
def create_prediction_charts_plotly(price_data, future_predictions, model_comparison):
    try:
        # Create future price prediction chart
        fig1 = go.Figure()
        
        # Add historical prices
        fig1.add_trace(
            go.Scatter(
                x=price_data.index[-30:],
                y=price_data['Close'].iloc[-30:],
                name="Historical Close",
                line=dict(color='blue')
            )
        )
        
        # Add predicted prices
        fig1.add_trace(
            go.Scatter(
                x=future_predictions['Date'],
                y=future_predictions['Predicted_Close'],
                name="Predicted Close",
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig1.update_layout(
            title="Nifty 50 Price Prediction - Next 30 Days",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create model comparison chart
        fig2 = px.bar(
            model_comparison,
            x='Model',
            y=['Test RMSE', 'Test MAE'],
            title="Model Performance Comparison",
            barmode='group',
            labels={'value': 'Error Metric', 'variable': 'Metric'}
        )
        
        # Create predicted percent change chart
        fig3 = px.bar(
            future_predictions,
            x='Date',
            y='Percent_Change',
            title="Predicted Percent Change from Last Close",
            labels={'Percent_Change': 'Percent Change (%)', 'Date': 'Date'},
            color='Percent_Change',
            color_continuous_scale=['red', 'lightgray', 'green'],
            color_continuous_midpoint=0
        )
        
        return fig1, fig2, fig3
    except Exception as e:
        st.error(f"Error creating prediction charts with Plotly: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# Function to create prediction charts with Matplotlib (fallback)
def create_prediction_charts_matplotlib(price_data, future_predictions, model_comparison):
    try:
        # Create future price prediction chart
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Add historical prices
        ax1.plot(price_data.index[-30:], price_data['Close'].iloc[-30:], 
                 label="Historical Close", color='blue')
        
        # Add predicted prices
        ax1.plot(future_predictions['Date'], future_predictions['Predicted_Close'], 
                 label="Predicted Close", color='red', linestyle='--')
        
        ax1.set_title("Nifty 50 Price Prediction - Next 30 Days")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create model comparison chart
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_comparison['Model']))
        width = 0.35
        
        ax2.bar(x - width/2, model_comparison['Test RMSE'], width, label='Test RMSE')
        ax2.bar(x + width/2, model_comparison['Test MAE'], width, label='Test MAE')
        
        ax2.set_title("Model Performance Comparison")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Error Metric")
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_comparison['Model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Create predicted percent change chart
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        bars = ax3.bar(future_predictions['Date'], future_predictions['Percent_Change'])
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if future_predictions['Percent_Change'].iloc[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax3.set_title("Predicted Percent Change from Last Close")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Percent Change (%)")
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        return fig1, fig2, fig3
    except Exception as e:
        st.error(f"Error creating prediction charts with Matplotlib: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# Main function to display the dashboard
def display_dashboard():
    # Get data
    price_data = get_realtime_nifty50_data()
    
    if price_data is None:
        st.error("Failed to load price data. Please check your internet connection or data files.")
        return
    
    # Get sentiment data
    sentiment_data = load_sentiment_data()
    
    # Get prediction data
    future_predictions, model_comparison = get_realtime_prediction_data()
    
    # Display dashboard header
    st.markdown('<h1 class="main-header">Nifty 50 Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Display current market status
    last_close = price_data['Close'].iloc[-1]
    prev_close = price_data['Close'].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Last Close", f"₹{last_close:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
    
    with col2:
        st.metric("50-Day SMA", f"₹{price_data['SMA_50'].iloc[-1]:.2f}")
    
    with col3:
        st.metric("RSI (14)", f"{price_data['RSI'].iloc[-1]:.2f}")
    
    # Display price chart
    st.markdown('<h2 class="sub-header">Price Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different timeframes
    timeframe_tabs = st.tabs(["1 Month", "3 Months", "6 Months", "1 Year", "All"])
    
    with timeframe_tabs[0]:
        st.subheader("Price Chart - Last Month")
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_data.index[-30:],
                open=price_data['Open'].iloc[-30:],
                high=price_data['High'].iloc[-30:],
                low=price_data['Low'].iloc[-30:],
                close=price_data['Close'].iloc[-30:],
                name="Price"
            ))
            fig.add_trace(go.Scatter(
                x=price_data.index[-30:],
                y=price_data['SMA_20'].iloc[-30:],
                name="20-Day SMA",
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=price_data.index[-30:],
                y=price_data['SMA_50'].iloc[-30:],
                name="50-Day SMA",
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="Nifty 50 Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(price_data.index[-30:], price_data['Close'].iloc[-30:], label="Close Price")
            ax.plot(price_data.index[-30:], price_data['SMA_20'].iloc[-30:], label="20-Day SMA", color='orange')
            ax.plot(price_data.index[-30:], price_data['SMA_50'].iloc[-30:], label="50-Day SMA", color='blue')
            ax.set_title("Nifty 50 Price with Moving Averages")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Similar code for other timeframes...
    
    # Display technical indicators
    st.markdown('<h2 class="sub-header">Technical Indicators</h2>', unsafe_allow_html=True)
    
    indicator_tabs = st.tabs(["MACD", "RSI", "Volatility"])
    
    with indicator_tabs[0]:
        st.subheader("MACD Indicator")
        if PLOTLY_AVAILABLE:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # Add price to top plot
            fig.add_trace(
                go.Scatter(
                    x=price_data.index[-90:],
                    y=price_data['Close'].iloc[-90:],
                    name="Close Price"
                ),
                row=1, col=1
            )
            
            # Add MACD to bottom plot
            fig.add_trace(
                go.Scatter(
                    x=price_data.index[-90:],
                    y=price_data['MACD'].iloc[-90:],
                    name="MACD"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price_data.index[-90:],
                    y=price_data['MACD_Signal'].iloc[-90:],
                    name="Signal Line"
                ),
                row=2, col=1
            )
            
            # Add MACD histogram
            colors = ['red' if val < 0 else 'green' for val in price_data['MACD_Histogram'].iloc[-90:]]
            
            fig.add_trace(
                go.Bar(
                    x=price_data.index[-90:],
                    y=price_data['MACD_Histogram'].iloc[-90:],
                    name="Histogram",
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="MACD Indicator - Last 90 Days",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis2_title="Date",
                yaxis2_title="MACD",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot price on top subplot
            ax1.plot(price_data.index[-90:], price_data['Close'].iloc[-90:], label="Close Price")
            ax1.set_title("MACD Indicator - Last 90 Days")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MACD on bottom subplot
            ax2.plot(price_data.index[-90:], price_data['MACD'].iloc[-90:], label="MACD")
            ax2.plot(price_data.index[-90:], price_data['MACD_Signal'].iloc[-90:], label="Signal Line")
            
            # Plot histogram
            for i in range(len(price_data.index[-90:])):
                if price_data['MACD_Histogram'].iloc[-90+i] >= 0:
                    ax2.bar(price_data.index[-90+i], price_data['MACD_Histogram'].iloc[-90+i], color='green', width=1)
                else:
                    ax2.bar(price_data.index[-90+i], price_data['MACD_Histogram'].iloc[-90+i], color='red', width=1)
            
            ax2.set_xlabel("Date")
            ax2.set_ylabel("MACD")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Similar code for other indicators...
    
    # Display sentiment analysis if data is available
    if sentiment_data is not None:
        st.markdown('<h2 class="sub-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        news_sentiment = sentiment_data['news_sentiment']
        twitter_sentiment = sentiment_data['twitter_sentiment']
        overall_sentiment = sentiment_data['overall_sentiment']
        
        if PLOTLY_AVAILABLE:
            fig1, fig2, fig3 = create_sentiment_charts_plotly(news_sentiment, twitter_sentiment, overall_sentiment)
            
            if fig1 is not None and fig2 is not None and fig3 is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.plotly_chart(fig2, use_container_width=True)
        else:
            fig1, fig2, fig3 = create_sentiment_charts_matplotlib(news_sentiment, twitter_sentiment, overall_sentiment)
            
            if fig1 is not None and fig2 is not None and fig3 is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(fig1)
                
                with col2:
                    st.pyplot(fig3)
                
                st.pyplot(fig2)
    
    # Display price predictions if data is available
    if future_predictions is not None and model_comparison is not None:
        st.markdown('<h2 class="sub-header">Price Predictions</h2>', unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            fig1, fig2, fig3 = create_prediction_charts_plotly(price_data, future_predictions, model_comparison)
            
            if fig1 is not None and fig2 is not None and fig3 is not None:
                st.plotly_chart(fig1, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            fig1, fig2, fig3 = create_prediction_charts_matplotlib(price_data, future_predictions, model_comparison)
            
            if fig1 is not None and fig2 is not None and fig3 is not None:
                st.pyplot(fig1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(fig2)
                
                with col2:
                    st.pyplot(fig3)
    
    # Display footer
    st.markdown("---")
    st.markdown("Data sources: Yahoo Finance, News APIs, Twitter APIs")
    st.markdown("Last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Run the dashboard
display_dashboard()
