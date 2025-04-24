import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import requests
from newsapi import NewsApiClient
import time

# Create a directory for storing news data
os.makedirs('data/news', exist_ok=True)

class NewsDataCollector:
    def __init__(self, api_key=None):
        """
        Initialize the News Data Collector
        
        Parameters:
        -----------
        api_key : str
            API key for News API (optional)
        """
        print("Initializing News Data Collector...")
        
        # If API key is provided, use it; otherwise, use a default key
        # Note: In a production environment, API keys should be stored securely
        self.api_key = api_key if api_key else "YOUR_API_KEY"
        
        # Initialize the News API client
        try:
            self.newsapi = NewsApiClient(api_key=self.api_key)
            print("News API client initialized successfully")
        except Exception as e:
            print(f"Error initializing News API client: {e}")
            print("Will use Twitter API as an alternative source")
        
        # Initialize the Data API client for Twitter
        self.client = ApiClient()
        
        # Load Nifty 50 constituents
        try:
            self.constituents_data = pd.read_csv('data/nifty50_constituents.csv')
            self.symbols = self.constituents_data['Symbol'].unique()
            # Extract company names from symbols (remove .NS suffix)
            self.companies = [symbol.split('.')[0] for symbol in self.symbols]
            print(f"Loaded {len(self.companies)} Nifty 50 constituent companies")
        except Exception as e:
            print(f"Error loading Nifty 50 constituents: {e}")
            # Default list of major Indian companies if constituents data is not available
            self.companies = [
                'RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICI', 
                'HINDUNILVR', 'ITC', 'SBI', 'BHARTIARTL', 'KOTAK',
                'LT', 'AXIS', 'BAJAJ', 'ASIAN', 'MARUTI'
            ]
            print(f"Using default list of {len(self.companies)} major Indian companies")
    
    def collect_news_api_data(self, days_back=7):
        """
        Collect news data using News API
        
        Parameters:
        -----------
        days_back : int
            Number of days to look back for news
        """
        print(f"Collecting news data for the past {days_back} days using News API...")
        
        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for the API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Initialize a list to store all articles
        all_articles = []
        
        try:
            # Collect news for Nifty 50 index
            print("Collecting news for Nifty 50 index...")
            nifty_articles = self.newsapi.get_everything(
                q='Nifty 50 OR "Nifty50" OR "Indian stock market"',
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            if nifty_articles['status'] == 'ok':
                print(f"Found {len(nifty_articles['articles'])} articles for Nifty 50")
                all_articles.extend(nifty_articles['articles'])
            
            # Collect news for each constituent company
            for company in self.companies:
                print(f"Collecting news for {company}...")
                company_articles = self.newsapi.get_everything(
                    q=company + ' AND (stock OR shares OR market OR finance OR business)',
                    from_param=from_date,
                    to=to_date,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20
                )
                
                if company_articles['status'] == 'ok':
                    print(f"Found {len(company_articles['articles'])} articles for {company}")
                    all_articles.extend(company_articles['articles'])
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.2)
            
            # Create a DataFrame from the collected articles
            news_df = pd.DataFrame(all_articles)
            
            # Add a source column
            news_df['source_name'] = news_df['source'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else 'Unknown')
            
            # Drop the source column (which contains dictionaries)
            news_df = news_df.drop('source', axis=1)
            
            # Convert publishedAt to datetime
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
            
            # Sort by published date
            news_df = news_df.sort_values('publishedAt', ascending=False)
            
            # Save to CSV
            news_df.to_csv('data/news/newsapi_articles.csv', index=False)
            print(f"Saved {len(news_df)} articles to data/news/newsapi_articles.csv")
            
            return news_df
        
        except Exception as e:
            print(f"Error collecting news from News API: {e}")
            print("Will try alternative sources")
            return pd.DataFrame()
    
    def collect_twitter_data(self, count=100):
        """
        Collect Twitter data using Twitter API
        
        Parameters:
        -----------
        count : int
            Number of tweets to collect per query
        """
        print(f"Collecting Twitter data...")
        
        # Initialize a list to store all tweets
        all_tweets = []
        
        try:
            # Collect tweets for Nifty 50 index
            print("Collecting tweets for Nifty 50 index...")
            nifty_tweets = self.client.call_api('Twitter/search_twitter', query={
                'query': 'Nifty50 OR "Nifty 50" OR "Indian stock market"',
                'count': count,
                'type': 'Latest'
            })
            
            # Process and extract tweets
            if nifty_tweets and 'result' in nifty_tweets and 'timeline' in nifty_tweets['result']:
                tweets = self._extract_tweets_from_response(nifty_tweets)
                all_tweets.extend(tweets)
                print(f"Found {len(tweets)} tweets for Nifty 50")
            
            # Collect tweets for each constituent company
            for company in self.companies[:10]:  # Limit to top 10 companies to avoid rate limits
                print(f"Collecting tweets for {company}...")
                company_tweets = self.client.call_api('Twitter/search_twitter', query={
                    'query': f'{company} stock OR {company} shares OR {company} market',
                    'count': 50,
                    'type': 'Latest'
                })
                
                # Process and extract tweets
                if company_tweets and 'result' in company_tweets and 'timeline' in company_tweets['result']:
                    tweets = self._extract_tweets_from_response(company_tweets)
                    all_tweets.extend(tweets)
                    print(f"Found {len(tweets)} tweets for {company}")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(1)
            
            # Create a DataFrame from the collected tweets
            if all_tweets:
                tweets_df = pd.DataFrame(all_tweets)
                
                # Save to CSV
                tweets_df.to_csv('data/news/twitter_data.csv', index=False)
                print(f"Saved {len(tweets_df)} tweets to data/news/twitter_data.csv")
                
                return tweets_df
            else:
                print("No tweets collected")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error collecting data from Twitter API: {e}")
            return pd.DataFrame()
    
    def _extract_tweets_from_response(self, response):
        """
        Extract tweets from Twitter API response
        
        Parameters:
        -----------
        response : dict
            Twitter API response
        
        Returns:
        --------
        list
            List of extracted tweets
        """
        tweets = []
        
        try:
            # Navigate through the response structure
            if 'result' in response and 'timeline' in response['result'] and 'instructions' in response['result']['timeline']:
                instructions = response['result']['timeline']['instructions']
                
                for instruction in instructions:
                    if 'entries' in instruction:
                        entries = instruction['entries']
                        
                        for entry in entries:
                            if 'content' in entry and 'items' in entry['content']:
                                items = entry['content']['items']
                                
                                for item in items:
                                    if 'item' in item and 'itemContent' in item['item']:
                                        item_content = item['item']['itemContent']
                                        
                                        # Check if it's a tweet
                                        if 'tweet_results' in item_content:
                                            tweet_result = item_content['tweet_results']['result']
                                            
                                            # Extract tweet text
                                            if 'legacy' in tweet_result and 'full_text' in tweet_result['legacy']:
                                                tweet_text = tweet_result['legacy']['full_text']
                                                created_at = tweet_result['legacy']['created_at']
                                                
                                                # Extract user information
                                                user_info = {}
                                                if 'core' in tweet_result and 'user_results' in tweet_result['core']:
                                                    user_result = tweet_result['core']['user_results']['result']
                                                    if 'legacy' in user_result:
                                                        user_legacy = user_result['legacy']
                                                        user_info = {
                                                            'username': user_legacy.get('screen_name', ''),
                                                            'name': user_legacy.get('name', ''),
                                                            'followers_count': user_legacy.get('followers_count', 0),
                                                            'verified': user_legacy.get('verified', False)
                                                        }
                                                
                                                # Create tweet object
                                                tweet = {
                                                    'text': tweet_text,
                                                    'created_at': created_at,
                                                    'username': user_info.get('username', ''),
                                                    'name': user_info.get('name', ''),
                                                    'followers_count': user_info.get('followers_count', 0),
                                                    'verified': user_info.get('verified', False)
                                                }
                                                
                                                tweets.append(tweet)
        except Exception as e:
            print(f"Error extracting tweets from response: {e}")
        
        return tweets
    
    def collect_financial_news(self, days_back=7):
        """
        Collect financial news from multiple sources
        
        Parameters:
        -----------
        days_back : int
            Number of days to look back for news
        """
        print(f"Collecting financial news from multiple sources for the past {days_back} days...")
        
        # Collect news from News API
        news_df = self.collect_news_api_data(days_back)
        
        # If News API collection failed or returned no results, try Twitter
        if news_df.empty:
            print("No news articles collected from News API, trying Twitter...")
            tweets_df = self.collect_twitter_data()
            
            # If Twitter collection also failed, create a dummy dataset for testing
            if tweets_df.empty:
                print("No tweets collected from Twitter API, creating dummy dataset for testing...")
                self._create_dummy_news_dataset()
        else:
            # Also collect Twitter data as a supplementary source
            tweets_df = self.collect_twitter_data()
        
        print("Financial news collection completed")
    
    def _create_dummy_news_dataset(self):
        """
        Create a dummy news dataset for testing purposes
        """
        print("Creating dummy news dataset...")
        
        # Create a list of dummy news articles
        dummy_articles = []
        
        # Current date
        now = datetime.now()
        
        # Sample headlines and sentiments
        headlines = [
            {"title": "Nifty 50 reaches all-time high as tech stocks surge", "sentiment": "positive"},
            {"title": "Indian markets rally on strong economic data", "sentiment": "positive"},
            {"title": "Reliance shares jump 5% after quarterly results", "sentiment": "positive"},
            {"title": "HDFC Bank reports strong profit growth", "sentiment": "positive"},
            {"title": "TCS wins major international contract", "sentiment": "positive"},
            {"title": "Infosys raises annual guidance, shares up", "sentiment": "positive"},
            {"title": "Market volatility increases as global tensions rise", "sentiment": "negative"},
            {"title": "Nifty falls on profit booking after recent rally", "sentiment": "negative"},
            {"title": "Oil price surge impacts Indian markets negatively", "sentiment": "negative"},
            {"title": "IT sector faces headwinds due to global slowdown", "sentiment": "negative"},
            {"title": "RBI keeps rates unchanged in latest policy meeting", "sentiment": "neutral"},
            {"title": "Indian rupee stable against US dollar", "sentiment": "neutral"},
            {"title": "FII inflows continue to support Indian markets", "sentiment": "positive"},
            {"title": "Auto sector shows mixed performance in monthly sales", "sentiment": "neutral"},
            {"title": "Banking stocks lead market gains on credit growth", "sentiment": "positive"}
        ]
        
        # Generate dummy articles with dates spread over the past week
        for i, headline in enumerate(headlines):
            days_ago = i % 7  # Spread over a week
            article_date = now - timedelta(days=days_ago, hours=i%24)
            
            article = {
                'title': headline['title'],
                'description': f"This is a dummy description for the article: {headline['title']}",
                'content': f"This is dummy content for testing purposes. The article is about {headline['title'].lower()}.",
                'url': f"https://example.com/news/{i}",
                'urlToImage': f"https://example.com/images/{i}.jpg",
                'publishedAt': article_date.isoformat(),
                'source_name': ['Economic Times', 'Business Standard', 'Mint', 'Financial Express', 'CNBC-TV18'][i % 5],
                'sentiment': headline['sentiment']
            }
            
            dummy_articles.append(article)
        
        # Create a DataFrame
        dummy_df = pd.DataFrame(dummy_articles)
        
        # Save to CSV
        dummy_df.to_csv('data/news/dummy_news.csv', index=False)
        print(f"Saved {len(dummy_df)} dummy articles to data/news/dummy_news.csv")
        
        # Also create dummy Twitter data
        self._create_dummy_twitter_dataset()
    
    def _create_dummy_twitter_dataset(self):
        """
        Create a dummy Twitter dataset for testing purposes
        """
        print("Creating dummy Twitter dataset...")
        
        # Create a list of dummy tweets
        dummy_tweets = []
        
        # Current date
        now = datetime.now()
        
        # Sample tweets and sentiments
        tweets = [
            {"text": "Nifty 50 looking strong today! Bullish on Indian markets for the coming weeks. #Nifty50 #IndianStocks", "sentiment": "positive"},
            {"text": "Just bought some Reliance shares. The company's growth prospects look excellent. #Reliance #Investment", "sentiment": "positive"},
            {"text": "HDFC Bank results are impressive. Banking sector leading the market rally. #HDFC #BankingStocks", "sentiment": "positive"},
            {"text": "TCS continues to deliver solid performance. Tech sector remains attractive. #TCS #TechStocks", "sentiment": "positive"},
            {"text": "Market volatility is concerning. Might reduce equity exposure for now. #MarketVolatility #StockMarket", "sentiment": "negative"},
            {"text": "Nifty technical indicators showing overbought conditions. Correction likely. #NiftyTechnicals #MarketOutlook", "sentiment": "negative"},
            {"text": "Oil prices rising rapidly. Could impact Indian markets negatively. #OilPrices #MarketImpact", "sentiment": "negative"},
            {"text": "RBI policy was as expected. Neutral for markets in the short term. #RBIPolicy #IndianEconomy", "sentiment": "neutral"},
            {"text": "FII flows remain strong into Indian equities. Positive for market sentiment. #FII #MarketSentiment", "sentiment": "positive"},
            {"text": "Auto sales numbers are mixed. Some companies doing well, others struggling. #AutoSector #SalesData", "sentiment": "neutral"}
        ]
        
        # Generate dummy tweets with dates spread over the past week
        for i, tweet_data in enumerate(tweets):
            days_ago = i % 7  # Spread over a week
            tweet_date = now - timedelta(days=days_ago, hours=i%24)
            
            tweet = {
                'text': tweet_data['text'],
                'created_at': tweet_date.strftime('%a %b %d %H:%M:%S +0000 %Y'),
                'username': f"trader{i}",
                'name': f"Stock Trader {i}",
                'followers_count': 1000 + i * 500,
                'verified': i % 3 == 0,  # Every third user is verified
                'sentiment': tweet_data['sentiment']
            }
            
            dummy_tweets.append(tweet)
        
        # Create a DataFrame
        dummy_df = pd.DataFrame(dummy_tweets)
        
        # Save to CSV
        dummy_df.to_csv('data/news/dummy_twitter.csv', index=False)
        print(f"Saved {len(dummy_df)} dummy tweets to data/news/dummy_twitter.csv")

if __name__ == "__main__":
    # Create an instance of the NewsDataCollector
    news_collector = NewsDataCollector()
    
    # Collect financial news
    news_collector.collect_financial_news(days_back=7)
