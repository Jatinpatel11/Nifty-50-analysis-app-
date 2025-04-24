import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib
import os
import datetime
from PIL import Image

# Try to import plotly, but provide fallback to matplotlib if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not available. Using Matplotlib for visualizations instead.")

# Set page configuration
st.set_page_config(
    page_title="Nifty 50 Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function to load data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    try:
        # Load price data
        price_data = pd.read_csv('data/nifty50_with_indicators.csv')
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data.set_index('Date', inplace=True)
        
        # Load sentiment data
        news_sentiment = pd.read_csv('data/sentiment/news_sentiment.csv')
        twitter_sentiment = pd.read_csv('data/sentiment/twitter_sentiment.csv')
        overall_sentiment = pd.read_csv('data/sentiment/overall_sentiment.csv')
        
        # Load prediction data
        future_predictions = pd.read_csv('data/predictions/future_predictions.csv')
        future_predictions['Date'] = pd.to_datetime(future_predictions['Date'])
        
        # Load model comparison
        model_comparison = pd.read_csv('data/predictions/model_comparison.csv')
        
        return {
            'price_data': price_data,
            'news_sentiment': news_sentiment,
            'twitter_sentiment': twitter_sentiment,
            'overall_sentiment': overall_sentiment,
            'future_predictions': future_predictions,
            'model_comparison': model_comparison
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to get real-time data
def get_realtime_data():
    try:
        # Get real-time Nifty 50 data
        nifty50 = yf.Ticker("^NSEI")
        current_data = nifty50.history(period="1d")
        
        # Get current price
        current_price = current_data['Close'].iloc[-1]
        
        # Get daily change
        daily_change = current_data['Close'].iloc[-1] - current_data['Open'].iloc[-1]
        daily_change_pct = (daily_change / current_data['Open'].iloc[-1]) * 100
        
        return {
            'current_price': current_price,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.warning(f"Could not fetch real-time data: {e}")
        return None

# Function to create price analysis charts with Plotly
def create_price_charts_plotly(price_data):
    # Create price chart with indicators
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Price with Moving Averages", "MACD", "RSI"))
    
    # Price with moving averages
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['SMA_20'],
            name="SMA 20",
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['SMA_50'],
            name="SMA 50",
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['MACD'],
            name="MACD",
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['MACD_Signal'],
            name="Signal",
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in price_data['MACD_Histogram']]
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['MACD_Histogram'],
            name="Histogram",
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['RSI'],
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        row=3, col=1
    )
    
    # Add overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=[70] * len(price_data),
            name="Overbought",
            line=dict(color='red', width=1, dash='dash')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=[30] * len(price_data),
            name="Oversold",
            line=dict(color='green', width=1, dash='dash')
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Nifty 50 Technical Analysis",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Function to create price analysis charts with Matplotlib (fallback)
def create_price_charts_matplotlib(price_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Price with moving averages
    ax1.plot(price_data.index, price_data['Close'], label='Close', color='black')
    ax1.plot(price_data.index, price_data['SMA_20'], label='SMA 20', color='blue')
    ax1.plot(price_data.index, price_data['SMA_50'], label='SMA 50', color='orange')
    ax1.set_title('Price with Moving Averages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MACD
    ax2.plot(price_data.index, price_data['MACD'], label='MACD', color='blue')
    ax2.plot(price_data.index, price_data['MACD_Signal'], label='Signal', color='red')
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in price_data['MACD_Histogram']]
    ax2.bar(price_data.index, price_data['MACD_Histogram'], label='Histogram', color=colors, alpha=0.5)
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3.plot(price_data.index, price_data['RSI'], label='RSI', color='purple')
    ax3.axhline(y=70, color='red', linestyle='--', label='Overbought')
    ax3.axhline(y=30, color='green', linestyle='--', label='Oversold')
    ax3.set_title('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Function to create sentiment analysis charts with Plotly
def create_sentiment_charts_plotly(news_sentiment, twitter_sentiment, overall_sentiment):
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

# Function to create sentiment analysis charts with Matplotlib (fallback)
def create_sentiment_charts_matplotlib(news_sentiment, twitter_sentiment, overall_sentiment):
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

# Function to create prediction charts with Plotly
def create_prediction_charts_plotly(price_data, future_predictions, model_comparison):
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

# Function to create prediction charts with Matplotlib (fallback)
def create_prediction_charts_matplotlib(price_data, future_predictions, model_comparison):
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
    
    colors = ['green' if x >= 0 else 'red' for x in future_predictions['Percent_Change']]
    ax3.bar(future_predictions['Date'], future_predictions['Percent_Change'], color=colors)
    
    ax3.set_title("Predicted Percent Change from Last Close")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Percent Change (%)")
    ax3.grid(True, alpha=0.3)
    
    return fig1, fig2, fig3

# Main app
def main():
    # Sidebar
    try:
        st.sidebar.image("plots/nifty50_logo.png", width=200)
    except:
        st.sidebar.title("Nifty 50 Analysis")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Price Analysis", "Sentiment Analysis", "Price Prediction"])
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Get real-time data
    realtime_data = get_realtime_data()
    
    # Header
    st.markdown("<h1 class='main-header'>Nifty 50 Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Overview page
    if page == "Overview":
        st.markdown("<h2 class='sub-header'>Market Overview</h2>", unsafe_allow_html=True)
        
        # Real-time data
        col1, col2, col3 = st.columns(3)
        
        if realtime_data:
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Current Price", f"₹{realtime_data['current_price']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Daily Change", f"₹{realtime_data['daily_change']:.2f}", 
                         f"{realtime_data['daily_change_pct']:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Last Updated", realtime_data['last_updated'])
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("<h3 class='section-header'>Market Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Price trend
            price_data = data['price_data']
            current_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            trend = "Bullish" if price_change > 0 else "Bearish"
            trend_color = "positive" if price_change > 0 else "negative"
            
            st.markdown(f"<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Price Trend</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Current trend is <span class='{trend_color}'>{trend}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Last change: <span class='{trend_color}'>{price_change_pct:.2f}%</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Sentiment
            overall_sentiment = data['overall_sentiment']
            sentiment_score = overall_sentiment['compound'].iloc[0]
            sentiment = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
            sentiment_color = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
            
            st.markdown(f"<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Market Sentiment</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Current sentiment is <span class='{sentiment_color}'>{sentiment}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Sentiment score: <span class='{sentiment_color}'>{sentiment_score:.2f}</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            # Prediction
            future_predictions = data['future_predictions']
            next_day_price = future_predictions['Predicted_Close'].iloc[0]
            price_change = next_day_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            prediction = "Up" if price_change > 0 else "Down"
            prediction_color = "positive" if price_change > 0 else "negative"
            
            st.markdown(f"<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Price Prediction</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Next day prediction: <span class='{prediction_color}'>{prediction}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Expected change: <span class='{prediction_color}'>{price_change_pct:.2f}%</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent price chart
        st.markdown("<h3 class='section-header'>Recent Price Movement</h3>", unsafe_allow_html=True)
        
        recent_data = data['price_data'].iloc[-30:]
        
        if PLOTLY_AVAILABLE:
            fig = px.line(
                recent_data,
                y='Close',
                title="Nifty 50 - Last 30 Days",
                labels={'Close': 'Close Price', 'Date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(recent_data.index, recent_data['Close'])
            ax.set_title("Nifty 50 - Last 30 Days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Key insights
        st.markdown("<h3 class='section-header'>Key Insights</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Technical Indicators</h4>", unsafe_allow_html=True)
            
            # MACD
            macd = data['price_data']['MACD'].iloc[-1]
            macd_signal = data['price_data']['MACD_Signal'].iloc[-1]
            macd_hist = data['price_data']['MACD_Histogram'].iloc[-1]
            
            macd_condition = "Bullish" if macd > macd_signal else "Bearish"
            macd_color = "positive" if macd > macd_signal else "negative"
            
            st.markdown(f"<p>MACD: <span class='{macd_color}'>{macd_condition}</span> ({macd:.2f})</p>", unsafe_allow_html=True)
            
            # RSI
            rsi = data['price_data']['RSI'].iloc[-1]
            
            if rsi > 70:
                rsi_condition = "Overbought"
                rsi_color = "negative"
            elif rsi < 30:
                rsi_condition = "Oversold"
                rsi_color = "positive"
            else:
                rsi_condition = "Neutral"
                rsi_color = "neutral"
            
            st.markdown(f"<p>RSI: <span class='{rsi_color}'>{rsi_condition}</span> ({rsi:.2f})</p>", unsafe_allow_html=True)
            
            # Moving Averages
            sma_20 = data['price_data']['SMA_20'].iloc[-1]
            sma_50 = data['price_data']['SMA_50'].iloc[-1]
            
            ma_condition = "Bullish" if sma_20 > sma_50 else "Bearish"
            ma_color = "positive" if sma_20 > sma_50 else "negative"
            
            st.markdown(f"<p>Moving Averages: <span class='{ma_color}'>{ma_condition}</span></p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Market Sentiment Drivers</h4>", unsafe_allow_html=True)
            
            # Top positive news
            news_sentiment = data['news_sentiment']
            top_positive = news_sentiment[news_sentiment['sentiment'] == 'positive'].sort_values('compound', ascending=False).head(1)
            
            if not top_positive.empty:
                st.markdown("<p><strong>Top Positive News:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p>{top_positive['title'].iloc[0]}</p>", unsafe_allow_html=True)
            
            # Top negative news
            top_negative = news_sentiment[news_sentiment['sentiment'] == 'negative'].sort_values('compound').head(1)
            
            if not top_negative.empty:
                st.markdown("<p><strong>Top Negative News:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p>{top_negative['title'].iloc[0]}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Price Analysis page
    elif page == "Price Analysis":
        st.markdown("<h2 class='sub-header'>Price Analysis</h2>", unsafe_allow_html=True)
        
        # Date range selector
        st.markdown("<h3 class='section-header'>Select Date Range</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=data['price_data'].index[-90].date(),
                min_value=data['price_data'].index[0].date(),
                max_value=data['price_data'].index[-1].date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=data['price_data'].index[-1].date(),
                min_value=data['price_data'].index[0].date(),
                max_value=data['price_data'].index[-1].date()
            )
        
        # Filter data based on date range
        filtered_data = data['price_data'].loc[start_date:end_date]
        
        # Technical analysis chart
        st.markdown("<h3 class='section-header'>Technical Analysis</h3>", unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            fig = create_price_charts_plotly(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_price_charts_matplotlib(filtered_data)
            st.pyplot(fig)
        
        # Performance metrics
        st.markdown("<h3 class='section-header'>Performance Metrics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 1-month return
            one_month_ago = filtered_data.index[-1] - pd.DateOffset(months=1)
            one_month_price = filtered_data.loc[filtered_data.index >= one_month_ago, 'Close'].iloc[0]
            current_price = filtered_data['Close'].iloc[-1]
            one_month_return = (current_price / one_month_price - 1) * 100
            
            return_color = "positive" if one_month_return > 0 else "negative"
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<h4>1-Month Return</h4>", unsafe_allow_html=True)
            st.markdown(f"<p class='{return_color}'>{one_month_return:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # 3-month return
            three_month_ago = filtered_data.index[-1] - pd.DateOffset(months=3)
            three_month_price = filtered_data.loc[filtered_data.index >= three_month_ago, 'Close'].iloc[0]
            three_month_return = (current_price / three_month_price - 1) * 100
            
            return_color = "positive" if three_month_return > 0 else "negative"
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<h4>3-Month Return</h4>", unsafe_allow_html=True)
            st.markdown(f"<p class='{return_color}'>{three_month_return:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            # 6-month return
            six_month_ago = filtered_data.index[-1] - pd.DateOffset(months=6)
            six_month_price = filtered_data.loc[filtered_data.index >= six_month_ago, 'Close'].iloc[0]
            six_month_return = (current_price / six_month_price - 1) * 100
            
            return_color = "positive" if six_month_return > 0 else "negative"
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<h4>6-Month Return</h4>", unsafe_allow_html=True)
            st.markdown(f"<p class='{return_color}'>{six_month_return:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Volatility and volume analysis
        st.markdown("<h3 class='section-header'>Volatility and Volume Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volatility chart
            volatility = filtered_data['Volatility_20']
            
            if PLOTLY_AVAILABLE:
                fig = px.line(
                    x=volatility.index,
                    y=volatility.values,
                    title="20-Day Volatility",
                    labels={'x': 'Date', 'y': 'Volatility (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(volatility.index, volatility.values)
                ax.set_title("20-Day Volatility")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volatility (%)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col2:
            # Volume chart
            volume = filtered_data['Volume']
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    x=volume.index,
                    y=volume.values,
                    title="Trading Volume",
                    labels={'x': 'Date', 'y': 'Volume'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(volume.index, volume.values)
                ax.set_title("Trading Volume")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volume")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Support and resistance levels
        st.markdown("<h3 class='section-header'>Support and Resistance Levels</h3>", unsafe_allow_html=True)
        
        # Calculate support and resistance levels (simple method)
        close_prices = filtered_data['Close'].values
        
        # Find local minima and maxima
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(close_prices) - 2):
            if close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i-2] and \
               close_prices[i] < close_prices[i+1] and close_prices[i] < close_prices[i+2]:
                support_levels.append(close_prices[i])
            
            if close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i-2] and \
               close_prices[i] > close_prices[i+1] and close_prices[i] > close_prices[i+2]:
                resistance_levels.append(close_prices[i])
        
        # Get the most recent levels
        support_levels = sorted(support_levels)[-3:] if support_levels else []
        resistance_levels = sorted(resistance_levels)[:3] if resistance_levels else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Support Levels</h4>", unsafe_allow_html=True)
            
            for i, level in enumerate(support_levels):
                st.markdown(f"<p>S{i+1}: ₹{level:.2f}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Resistance Levels</h4>", unsafe_allow_html=True)
            
            for i, level in enumerate(resistance_levels):
                st.markdown(f"<p>R{i+1}: ₹{level:.2f}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Sentiment Analysis page
    elif page == "Sentiment Analysis":
        st.markdown("<h2 class='sub-header'>Sentiment Analysis</h2>", unsafe_allow_html=True)
        
        # Overall sentiment
        st.markdown("<h3 class='section-header'>Overall Market Sentiment</h3>", unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            fig1, fig2, fig3 = create_sentiment_charts_plotly(
                data['news_sentiment'],
                data['twitter_sentiment'],
                data['overall_sentiment']
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig1, fig2, fig3 = create_sentiment_charts_matplotlib(
                data['news_sentiment'],
                data['twitter_sentiment'],
                data['overall_sentiment']
            )
            st.pyplot(fig3)
        
        # Sentiment distribution and trend
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.pyplot(fig1)
        
        with col2:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.pyplot(fig2)
        
        # News sentiment
        st.markdown("<h3 class='section-header'>News Sentiment Analysis</h3>", unsafe_allow_html=True)
        
        # Display top positive and negative news
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Top Positive News</h4>", unsafe_allow_html=True)
            
            top_positive = data['news_sentiment'][data['news_sentiment']['sentiment'] == 'positive'].sort_values('compound', ascending=False).head(5)
            
            for i, (_, row) in enumerate(top_positive.iterrows()):
                st.markdown(f"<p><strong>{i+1}. {row['title']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p><em>Score: {row['compound']:.2f}</em></p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Top Negative News</h4>", unsafe_allow_html=True)
            
            top_negative = data['news_sentiment'][data['news_sentiment']['sentiment'] == 'negative'].sort_values('compound').head(5)
            
            for i, (_, row) in enumerate(top_negative.iterrows()):
                st.markdown(f"<p><strong>{i+1}. {row['title']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p><em>Score: {row['compound']:.2f}</em></p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Twitter sentiment
        st.markdown("<h3 class='section-header'>Social Media Sentiment Analysis</h3>", unsafe_allow_html=True)
        
        # Display top tweets
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Top Positive Tweets</h4>", unsafe_allow_html=True)
            
            top_positive_tweets = data['twitter_sentiment'][data['twitter_sentiment']['sentiment'] == 'positive'].sort_values('compound', ascending=False).head(3)
            
            for i, (_, row) in enumerate(top_positive_tweets.iterrows()):
                st.markdown(f"<p><strong>{i+1}. @{row['username']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p>{row['text']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><em>Score: {row['compound']:.2f}</em></p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("<h4>Top Negative Tweets</h4>", unsafe_allow_html=True)
            
            top_negative_tweets = data['twitter_sentiment'][data['twitter_sentiment']['sentiment'] == 'negative'].sort_values('compound').head(3)
            
            for i, (_, row) in enumerate(top_negative_tweets.iterrows()):
                st.markdown(f"<p><strong>{i+1}. @{row['username']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p>{row['text']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><em>Score: {row['compound']:.2f}</em></p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Word cloud
        st.markdown("<h3 class='section-header'>Key Topics</h3>", unsafe_allow_html=True)
        
        try:
            word_cloud = Image.open("plots/sentiment/word_clouds.png")
            st.image(word_cloud, caption="Word Cloud of Key Topics", use_column_width=True)
        except:
            st.warning("Word cloud image not found.")
        
        # Top topics
        try:
            top_topics = Image.open("plots/sentiment/top_topics.png")
            st.image(top_topics, caption="Top Topics in Financial News", use_column_width=True)
        except:
            st.warning("Top topics image not found.")
    
    # Price Prediction page
    elif page == "Price Prediction":
        st.markdown("<h2 class='sub-header'>Price Prediction</h2>", unsafe_allow_html=True)
        
        # Future price prediction
        st.markdown("<h3 class='section-header'>Future Price Prediction</h3>", unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            fig1, fig2, fig3 = create_prediction_charts_plotly(
                data['price_data'],
                data['future_predictions'],
                data['model_comparison']
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fig1, fig2, fig3 = create_prediction_charts_matplotlib(
                data['price_data'],
                data['future_predictions'],
                data['model_comparison']
            )
            st.pyplot(fig1)
        
        # Predicted percent change
        st.markdown("<h3 class='section-header'>Predicted Daily Changes</h3>", unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.pyplot(fig3)
        
        # Model performance
        st.markdown("<h3 class='section-header'>Model Performance</h3>", unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.pyplot(fig2)
        
        # Prediction table
        st.markdown("<h3 class='section-header'>Detailed Predictions</h3>", unsafe_allow_html=True)
        
        # Format the prediction table
        prediction_table = data['future_predictions'].copy()
        prediction_table['Date'] = prediction_table['Date'].dt.strftime('%Y-%m-%d')
        prediction_table['Predicted_Close'] = prediction_table['Predicted_Close'].round(2)
        prediction_table['Percent_Change'] = prediction_table['Percent_Change'].round(2)
        prediction_table.columns = ['Date', 'Predicted Close (₹)', 'Percent Change (%)']
        
        st.dataframe(prediction_table, use_container_width=True)
        
        # Feature importance
        st.markdown("<h3 class='section-header'>Feature Importance</h3>", unsafe_allow_html=True)
        
        try:
            feature_importance = Image.open("plots/predictions/feature_importances.png")
            st.image(feature_importance, caption="Feature Importance in Prediction Model", use_column_width=True)
        except:
            st.warning("Feature importance image not found.")
        
        # Prediction methodology
        st.markdown("<h3 class='section-header'>Prediction Methodology</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
            <p>Our price prediction model uses a combination of machine learning algorithms to forecast Nifty 50 price movements:</p>
            <ol>
                <li><strong>Linear Regression</strong>: A simple model that establishes linear relationships between features and target.</li>
                <li><strong>Random Forest</strong>: An ensemble method that builds multiple decision trees and merges their predictions.</li>
                <li><strong>Gradient Boosting</strong>: An advanced technique that builds trees sequentially, with each tree correcting errors of previous ones.</li>
            </ol>
            <p>The models are trained on historical price data along with technical indicators and sentiment analysis. The predictions shown are from the best-performing model based on test error metrics.</p>
            <p><em>Note: These predictions are for informational purposes only and should not be considered as financial advice.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
