# Nifty 50 Analysis Dashboard

A comprehensive dashboard for analyzing Nifty 50 with sentiment analysis and price prediction capabilities.

## Features

- **Sentiment Analysis**: Analyze sentiment from financial news and social media
- **Price Prediction**: Machine learning-based forecasting of Nifty 50 prices
- **Real-time Data**: Fetch and analyze the latest market data
- **Interactive Visualizations**: Explore data through interactive charts and graphs

## Live Demo

Access the live dashboard at: [Nifty 50 Analysis Dashboard](https://nifty50-analysis.streamlit.app)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nifty50-analysis.git
cd nifty50-analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_without_price_analysis.py
```

## Data Sources

- Stock price data: Yahoo Finance API
- News data: Financial news APIs and web scraping
- Social media data: Twitter API

## Technologies Used

- Python 3.10
- Streamlit for web interface
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn, Plotly for visualization
- TensorFlow, scikit-learn for machine learning
- NLTK, TextBlob for sentiment analysis
- yfinance for real-time market data

## Project Structure

- `app_without_price_analysis.py`: Main Streamlit application
- `data/`: Directory containing data files
- `plots/`: Directory containing visualization images
- `requirements.txt`: List of Python dependencies

## Deployment

This application is deployed on Streamlit Cloud. Any push to the main branch will automatically trigger a new deployment.

## License

MIT

## Author

Your Name
