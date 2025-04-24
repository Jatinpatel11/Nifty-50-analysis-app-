import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re
import string
from collections import Counter
from wordcloud import WordCloud

# Create directories for storing sentiment analysis results
os.makedirs('data/sentiment', exist_ok=True)
os.makedirs('plots/sentiment', exist_ok=True)

class SentimentAnalysis:
    def __init__(self, news_path='data/news/dummy_news.csv', tweets_path='data/news/dummy_twitter.csv'):
        """
        Initialize the Sentiment Analysis class
        
        Parameters:
        -----------
        news_path : str
            Path to the news data CSV file
        tweets_path : str
            Path to the Twitter data CSV file
        """
        print("Initializing Sentiment Analysis...")
        
        # Initialize sentiment lexicon
        self.sentiment_lexicon = self._create_financial_lexicon()
        print("Financial sentiment lexicon created")
        
        # Load news data
        try:
            print(f"Loading news data from {news_path}...")
            self.news_df = pd.read_csv(news_path)
            print(f"Loaded {len(self.news_df)} news articles")
        except Exception as e:
            print(f"Error loading news data: {e}")
            self.news_df = pd.DataFrame()
        
        # Load Twitter data
        try:
            print(f"Loading Twitter data from {tweets_path}...")
            self.tweets_df = pd.read_csv(tweets_path)
            print(f"Loaded {len(self.tweets_df)} tweets")
        except Exception as e:
            print(f"Error loading Twitter data: {e}")
            self.tweets_df = pd.DataFrame()
    
    def _create_financial_lexicon(self):
        """
        Create a financial sentiment lexicon
        """
        # Positive financial terms
        positive_terms = {
            'bullish': 3.0,
            'rally': 2.5,
            'surge': 2.5,
            'gain': 2.0,
            'growth': 2.0,
            'profit': 2.0,
            'outperform': 2.0,
            'upgrade': 2.0,
            'beat': 1.5,
            'recovery': 1.5,
            'upside': 1.5,
            'positive': 1.5,
            'strong': 1.5,
            'opportunity': 1.0,
            'momentum': 1.0,
            'support': 1.0,
            'high': 1.0,
            'rise': 1.5,
            'up': 1.0,
            'increase': 1.0
        }
        
        # Negative financial terms
        negative_terms = {
            'bearish': -3.0,
            'crash': -3.0,
            'plunge': -2.5,
            'slump': -2.5,
            'tumble': -2.5,
            'decline': -2.0,
            'loss': -2.0,
            'downgrade': -2.0,
            'underperform': -2.0,
            'miss': -1.5,
            'weak': -1.5,
            'negative': -1.5,
            'concern': -1.5,
            'risk': -1.5,
            'volatile': -1.0,
            'resistance': -1.0,
            'caution': -1.0,
            'fall': -1.5,
            'down': -1.0,
            'decrease': -1.0
        }
        
        # Combine into a single lexicon
        lexicon = {**positive_terms, **negative_terms}
        return lexicon
    
    def _preprocess_text(self, text):
        """
        Preprocess text for analysis
        
        Parameters:
        -----------
        text : str
            Text to preprocess
        
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions and hashtags for tweets
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        
        # Remove common stop words (simplified approach)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                     'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
                     'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 
                     'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by', 'with'}
        
        tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def _calculate_sentiment_score(self, text):
        """
        Calculate sentiment score using the financial lexicon
        
        Parameters:
        -----------
        text : str
            Text to analyze
        
        Returns:
        --------
        dict
            Dictionary with sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        
        # Tokenize text
        words = text.lower().split()
        
        # Calculate sentiment
        pos_score = 0
        neg_score = 0
        
        for word in words:
            if word in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word]
                if score > 0:
                    pos_score += score
                else:
                    neg_score += abs(score)
        
        # Normalize scores
        total = pos_score + neg_score
        
        if total == 0:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        
        pos_norm = pos_score / total if total > 0 else 0
        neg_norm = neg_score / total if total > 0 else 0
        neu_norm = 1 - (pos_norm + neg_norm)
        
        # Calculate compound score (between -1 and 1)
        compound = (pos_score - neg_score) / (pos_score + neg_score) if (pos_score + neg_score) > 0 else 0
        
        return {
            'compound': compound,
            'pos': pos_norm,
            'neg': neg_norm,
            'neu': neu_norm
        }
    
    def analyze_news_sentiment(self):
        """
        Analyze sentiment of news articles
        """
        print("Analyzing news sentiment...")
        
        if self.news_df.empty:
            print("No news data available for sentiment analysis")
            return pd.DataFrame()
        
        # Create a copy of the DataFrame to avoid modifying the original
        news_sentiment = self.news_df.copy()
        
        # Combine title, description, and content for sentiment analysis
        news_sentiment['combined_text'] = news_sentiment['title'].fillna('') + ' ' + \
                                         news_sentiment['description'].fillna('') + ' ' + \
                                         news_sentiment['content'].fillna('')
        
        # Preprocess the combined text
        news_sentiment['preprocessed_text'] = news_sentiment['combined_text'].apply(self._preprocess_text)
        
        # Apply sentiment analysis
        news_sentiment['sentiment_scores'] = news_sentiment['combined_text'].apply(
            lambda x: self._calculate_sentiment_score(x) if isinstance(x, str) else None
        )
        
        # Extract sentiment scores
        news_sentiment['compound'] = news_sentiment['sentiment_scores'].apply(lambda x: x['compound'] if x else None)
        news_sentiment['pos'] = news_sentiment['sentiment_scores'].apply(lambda x: x['pos'] if x else None)
        news_sentiment['neu'] = news_sentiment['sentiment_scores'].apply(lambda x: x['neu'] if x else None)
        news_sentiment['neg'] = news_sentiment['sentiment_scores'].apply(lambda x: x['neg'] if x else None)
        
        # Classify sentiment based on compound score
        news_sentiment['sentiment'] = news_sentiment['compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        
        # Convert publishedAt to datetime if it's not already
        if 'publishedAt' in news_sentiment.columns:
            news_sentiment['publishedAt'] = pd.to_datetime(news_sentiment['publishedAt'])
            
            # Extract date for aggregation
            news_sentiment['date'] = news_sentiment['publishedAt'].dt.date
        
        # Save sentiment analysis results
        news_sentiment.to_csv('data/sentiment/news_sentiment.csv', index=False)
        print(f"News sentiment analysis saved to data/sentiment/news_sentiment.csv")
        
        return news_sentiment
    
    def analyze_twitter_sentiment(self):
        """
        Analyze sentiment of tweets
        """
        print("Analyzing Twitter sentiment...")
        
        if self.tweets_df.empty:
            print("No Twitter data available for sentiment analysis")
            return pd.DataFrame()
        
        # Create a copy of the DataFrame to avoid modifying the original
        twitter_sentiment = self.tweets_df.copy()
        
        # Preprocess the tweet text
        twitter_sentiment['preprocessed_text'] = twitter_sentiment['text'].apply(self._preprocess_text)
        
        # Apply sentiment analysis
        twitter_sentiment['sentiment_scores'] = twitter_sentiment['text'].apply(
            lambda x: self._calculate_sentiment_score(x) if isinstance(x, str) else None
        )
        
        # Extract sentiment scores
        twitter_sentiment['compound'] = twitter_sentiment['sentiment_scores'].apply(lambda x: x['compound'] if x else None)
        twitter_sentiment['pos'] = twitter_sentiment['sentiment_scores'].apply(lambda x: x['pos'] if x else None)
        twitter_sentiment['neu'] = twitter_sentiment['sentiment_scores'].apply(lambda x: x['neu'] if x else None)
        twitter_sentiment['neg'] = twitter_sentiment['sentiment_scores'].apply(lambda x: x['neg'] if x else None)
        
        # Classify sentiment based on compound score
        twitter_sentiment['sentiment'] = twitter_sentiment['compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        
        # Convert created_at to datetime if it's not already
        if 'created_at' in twitter_sentiment.columns:
            # Twitter's date format is typically like: "Wed Apr 22 13:08:45 +0000 2020"
            try:
                twitter_sentiment['created_at'] = pd.to_datetime(twitter_sentiment['created_at'], errors='coerce')
                
                # Extract date for aggregation
                twitter_sentiment['date'] = twitter_sentiment['created_at'].dt.date
            except Exception as e:
                print(f"Error converting Twitter dates: {e}")
        
        # Save sentiment analysis results
        twitter_sentiment.to_csv('data/sentiment/twitter_sentiment.csv', index=False)
        print(f"Twitter sentiment analysis saved to data/sentiment/twitter_sentiment.csv")
        
        return twitter_sentiment
    
    def calculate_overall_sentiment(self, news_sentiment, twitter_sentiment):
        """
        Calculate overall sentiment by combining news and Twitter sentiment
        
        Parameters:
        -----------
        news_sentiment : DataFrame
            News sentiment analysis results
        twitter_sentiment : DataFrame
            Twitter sentiment analysis results
        """
        print("Calculating overall sentiment...")
        
        # Initialize an empty DataFrame for overall sentiment
        overall_sentiment = pd.DataFrame()
        
        # Check if we have both news and Twitter sentiment
        if not news_sentiment.empty and not twitter_sentiment.empty:
            # Combine sentiment scores from both sources
            # We'll weight news articles higher than tweets (0.7 vs 0.3)
            
            # Calculate average sentiment scores for news
            news_avg_compound = news_sentiment['compound'].mean()
            news_avg_pos = news_sentiment['pos'].mean()
            news_avg_neu = news_sentiment['neu'].mean()
            news_avg_neg = news_sentiment['neg'].mean()
            
            # Calculate average sentiment scores for Twitter
            twitter_avg_compound = twitter_sentiment['compound'].mean()
            twitter_avg_pos = twitter_sentiment['pos'].mean()
            twitter_avg_neu = twitter_sentiment['neu'].mean()
            twitter_avg_neg = twitter_sentiment['neg'].mean()
            
            # Calculate weighted average
            overall_compound = (news_avg_compound * 0.7) + (twitter_avg_compound * 0.3)
            overall_pos = (news_avg_pos * 0.7) + (twitter_avg_pos * 0.3)
            overall_neu = (news_avg_neu * 0.7) + (twitter_avg_neu * 0.3)
            overall_neg = (news_avg_neg * 0.7) + (twitter_avg_neg * 0.3)
            
            # Determine overall sentiment
            if overall_compound >= 0.05:
                overall_sentiment_label = 'positive'
            elif overall_compound <= -0.05:
                overall_sentiment_label = 'negative'
            else:
                overall_sentiment_label = 'neutral'
            
            # Create a DataFrame with the results
            overall_sentiment = pd.DataFrame({
                'source': ['combined'],
                'compound': [overall_compound],
                'pos': [overall_pos],
                'neu': [overall_neu],
                'neg': [overall_neg],
                'sentiment': [overall_sentiment_label],
                'news_count': [len(news_sentiment)],
                'twitter_count': [len(twitter_sentiment)]
            })
            
        elif not news_sentiment.empty:
            # Only news sentiment available
            overall_sentiment = pd.DataFrame({
                'source': ['news_only'],
                'compound': [news_sentiment['compound'].mean()],
                'pos': [news_sentiment['pos'].mean()],
                'neu': [news_sentiment['neu'].mean()],
                'neg': [news_sentiment['neg'].mean()],
                'sentiment': [news_sentiment['sentiment'].mode()[0]],
                'news_count': [len(news_sentiment)],
                'twitter_count': [0]
            })
            
        elif not twitter_sentiment.empty:
            # Only Twitter sentiment available
            overall_sentiment = pd.DataFrame({
                'source': ['twitter_only'],
                'compound': [twitter_sentiment['compound'].mean()],
                'pos': [twitter_sentiment['pos'].mean()],
                'neu': [twitter_sentiment['neu'].mean()],
                'neg': [twitter_sentiment['neg'].mean()],
                'sentiment': [twitter_sentiment['sentiment'].mode()[0]],
                'news_count': [0],
                'twitter_count': [len(twitter_sentiment)]
            })
        
        # Save overall sentiment
        if not overall_sentiment.empty:
            overall_sentiment.to_csv('data/sentiment/overall_sentiment.csv', index=False)
            print(f"Overall sentiment analysis saved to data/sentiment/overall_sentiment.csv")
        
        return overall_sentiment
    
    def plot_sentiment_distribution(self, news_sentiment, twitter_sentiment):
        """
        Plot sentiment distribution for news and Twitter
        
        Parameters:
        -----------
        news_sentiment : DataFrame
            News sentiment analysis results
        twitter_sentiment : DataFrame
            Twitter sentiment analysis results
        """
        print("Plotting sentiment distribution...")
        
        plt.figure(figsize=(14, 10))
        
        # Plot news sentiment distribution
        if not news_sentiment.empty:
            plt.subplot(2, 2, 1)
            sentiment_counts = news_sentiment['sentiment'].value_counts()
            colors = ['green', 'gray', 'red']
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
            plt.title('News Sentiment Distribution')
            
            plt.subplot(2, 2, 2)
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['green', 'gray', 'red'])
            plt.title('News Sentiment Distribution')
            plt.ylabel('Count')
            plt.xlabel('Sentiment')
        
        # Plot Twitter sentiment distribution
        if not twitter_sentiment.empty:
            plt.subplot(2, 2, 3)
            sentiment_counts = twitter_sentiment['sentiment'].value_counts()
            colors = ['green', 'gray', 'red']
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
            plt.title('Twitter Sentiment Distribution')
            
            plt.subplot(2, 2, 4)
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['green', 'gray', 'red'])
            plt.title('Twitter Sentiment Distribution')
            plt.ylabel('Count')
            plt.xlabel('Sentiment')
        
        plt.tight_layout()
        plt.savefig('plots/sentiment/sentiment_distribution.png')
        print("Sentiment distribution plot saved to plots/sentiment/sentiment_distribution.png")
    
    def plot_sentiment_over_time(self, news_sentiment, twitter_sentiment):
        """
        Plot sentiment over time for news and Twitter
        
        Parameters:
        -----------
        news_sentiment : DataFrame
            News sentiment analysis results
        twitter_sentiment : DataFrame
            Twitter sentiment analysis results
        """
        print("Plotting sentiment over time...")
        
        plt.figure(figsize=(14, 10))
        
        # Plot news sentiment over time
        if not news_sentiment.empty and 'date' in news_sentiment.columns:
            plt.subplot(2, 1, 1)
            
            # Group by date and calculate average compound score
            daily_sentiment = news_sentiment.groupby('date')['compound'].mean().reset_index()
            
            # Plot
            plt.plot(daily_sentiment['date'], daily_sentiment['compound'], marker='o', linestyle='-', color='blue')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                            where=(daily_sentiment['compound'] >= 0), color='green', alpha=0.3)
            plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                            where=(daily_sentiment['compound'] < 0), color='red', alpha=0.3)
            plt.title('News Sentiment Over Time')
            plt.ylabel('Compound Sentiment Score')
            plt.grid(True, alpha=0.3)
        
        # Plot Twitter sentiment over time
        if not twitter_sentiment.empty and 'date' in twitter_sentiment.columns:
            plt.subplot(2, 1, 2)
            
            # Group by date and calculate average compound score
            daily_sentiment = twitter_sentiment.groupby('date')['compound'].mean().reset_index()
            
            # Plot
            plt.plot(daily_sentiment['date'], daily_sentiment['compound'], marker='o', linestyle='-', color='blue')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                            where=(daily_sentiment['compound'] >= 0), color='green', alpha=0.3)
            plt.fill_between(daily_sentiment['date'], daily_sentiment['compound'], 0, 
                            where=(daily_sentiment['compound'] < 0), color='red', alpha=0.3)
            plt.title('Twitter Sentiment Over Time')
            plt.ylabel('Compound Sentiment Score')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/sentiment/sentiment_over_time.png')
        print("Sentiment over time plot saved to plots/sentiment/sentiment_over_time.png")
    
    def generate_word_clouds(self, news_sentiment, twitter_sentiment):
        """
        Generate word clouds for positive and negative sentiment
        
        Parameters:
        -----------
        news_sentiment : DataFrame
            News sentiment analysis results
        twitter_sentiment : DataFrame
            Twitter sentiment analysis results
        """
        print("Generating word clouds...")
        
        # Combine news and Twitter text
        all_text = pd.DataFrame()
        
        if not news_sentiment.empty:
            news_text = news_sentiment[['preprocessed_text', 'sentiment']].copy()
            news_text['source'] = 'news'
            all_text = pd.concat([all_text, news_text])
        
        if not twitter_sentiment.empty:
            twitter_text = twitter_sentiment[['preprocessed_text', 'sentiment']].copy()
            twitter_text['source'] = 'twitter'
            all_text = pd.concat([all_text, twitter_text])
        
        if all_text.empty:
            print("No text data available for word clouds")
            return
        
        # Generate word clouds for different sentiments
        plt.figure(figsize=(16, 12))
        
        # Positive sentiment word cloud
        plt.subplot(2, 2, 1)
        positive_text = ' '.join(all_text[all_text['sentiment'] == 'positive']['preprocessed_text'])
        if positive_text.strip():
            wordcloud_positive = WordCloud(width=800, height=400, background_color='white', 
                                        max_words=100, colormap='Greens').generate(positive_text)
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis('off')
            plt.title('Positive Sentiment Word Cloud')
        else:
            plt.text(0.5, 0.5, 'No positive sentiment text available', ha='center', va='center')
            plt.axis('off')
        
        # Negative sentiment word cloud
        plt.subplot(2, 2, 2)
        negative_text = ' '.join(all_text[all_text['sentiment'] == 'negative']['preprocessed_text'])
        if negative_text.strip():
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white', 
                                        max_words=100, colormap='Reds').generate(negative_text)
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.axis('off')
            plt.title('Negative Sentiment Word Cloud')
        else:
            plt.text(0.5, 0.5, 'No negative sentiment text available', ha='center', va='center')
            plt.axis('off')
        
        # News word cloud
        plt.subplot(2, 2, 3)
        news_text_all = ' '.join(all_text[all_text['source'] == 'news']['preprocessed_text'])
        if news_text_all.strip():
            wordcloud_news = WordCloud(width=800, height=400, background_color='white', 
                                    max_words=100, colormap='Blues').generate(news_text_all)
            plt.imshow(wordcloud_news, interpolation='bilinear')
            plt.axis('off')
            plt.title('News Word Cloud')
        else:
            plt.text(0.5, 0.5, 'No news text available', ha='center', va='center')
            plt.axis('off')
        
        # Twitter word cloud
        plt.subplot(2, 2, 4)
        twitter_text_all = ' '.join(all_text[all_text['source'] == 'twitter']['preprocessed_text'])
        if twitter_text_all.strip():
            wordcloud_twitter = WordCloud(width=800, height=400, background_color='white', 
                                        max_words=100, colormap='Purples').generate(twitter_text_all)
            plt.imshow(wordcloud_twitter, interpolation='bilinear')
            plt.axis('off')
            plt.title('Twitter Word Cloud')
        else:
            plt.text(0.5, 0.5, 'No Twitter text available', ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/sentiment/word_clouds.png')
        print("Word clouds saved to plots/sentiment/word_clouds.png")
    
    def extract_key_topics(self, news_sentiment, twitter_sentiment):
        """
        Extract key topics from news and Twitter data
        
        Parameters:
        -----------
        news_sentiment : DataFrame
            News sentiment analysis results
        twitter_sentiment : DataFrame
            Twitter sentiment analysis results
        """
        print("Extracting key topics...")
        
        # Combine news and Twitter text
        all_text = pd.DataFrame()
        
        if not news_sentiment.empty:
            news_text = news_sentiment[['preprocessed_text', 'sentiment']].copy()
            news_text['source'] = 'news'
            all_text = pd.concat([all_text, news_text])
        
        if not twitter_sentiment.empty:
            twitter_text = twitter_sentiment[['preprocessed_text', 'sentiment']].copy()
            twitter_text['source'] = 'twitter'
            all_text = pd.concat([all_text, twitter_text])
        
        if all_text.empty:
            print("No text data available for topic extraction")
            return pd.DataFrame()
        
        # Tokenize all preprocessed text
        all_tokens = []
        for text in all_text['preprocessed_text']:
            if isinstance(text, str):
                tokens = text.split()
                all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Get the top 20 most common tokens
        top_tokens = token_counts.most_common(20)
        
        # Create a DataFrame with the results
        topics_df = pd.DataFrame(top_tokens, columns=['topic', 'count'])
        
        # Save the topics
        topics_df.to_csv('data/sentiment/key_topics.csv', index=False)
        print(f"Key topics saved to data/sentiment/key_topics.csv")
        
        # Plot the top topics
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y='topic', data=topics_df.sort_values('count', ascending=False))
        plt.title('Top 20 Topics in Financial News and Tweets')
        plt.xlabel('Frequency')
        plt.ylabel('Topic')
        plt.tight_layout()
        plt.savefig('plots/sentiment/top_topics.png')
        print("Top topics plot saved to plots/sentiment/top_topics.png")
        
        return topics_df
    
    def correlate_sentiment_with_price(self, price_data_path='data/nifty50_data.csv'):
        """
        Correlate sentiment with price movements
        
        Parameters:
        -----------
        price_data_path : str
            Path to the price data CSV file
        """
        print("Correlating sentiment with price movements...")
        
        try:
            # Load price data
            price_data = pd.read_csv(price_data_path)
            price_data['Date'] = pd.to_datetime(price_data['Date'])
            price_data.set_index('Date', inplace=True)
            
            # Calculate daily returns
            price_data['Daily_Return'] = price_data['Close'].pct_change() * 100
            
            # Load sentiment data
            overall_sentiment_path = 'data/sentiment/overall_sentiment.csv'
            news_sentiment_path = 'data/sentiment/news_sentiment.csv'
            twitter_sentiment_path = 'data/sentiment/twitter_sentiment.csv'
            
            # Initialize DataFrames for correlation analysis
            correlation_data = pd.DataFrame(index=price_data.index)
            correlation_data['Close'] = price_data['Close']
            correlation_data['Daily_Return'] = price_data['Daily_Return']
            
            # Add sentiment data if available
            if os.path.exists(news_sentiment_path):
                news_sentiment = pd.read_csv(news_sentiment_path)
                if 'publishedAt' in news_sentiment.columns:
                    news_sentiment['publishedAt'] = pd.to_datetime(news_sentiment['publishedAt'])
                    news_sentiment.set_index('publishedAt', inplace=True)
                    
                    # Resample to daily frequency and calculate mean sentiment
                    daily_news_sentiment = news_sentiment['compound'].resample('D').mean()
                    
                    # Add to correlation data
                    correlation_data = correlation_data.join(daily_news_sentiment.rename('News_Sentiment'), how='left')
            
            if os.path.exists(twitter_sentiment_path):
                twitter_sentiment = pd.read_csv(twitter_sentiment_path)
                if 'created_at' in twitter_sentiment.columns:
                    twitter_sentiment['created_at'] = pd.to_datetime(twitter_sentiment['created_at'], errors='coerce')
                    twitter_sentiment.set_index('created_at', inplace=True)
                    
                    # Resample to daily frequency and calculate mean sentiment
                    daily_twitter_sentiment = twitter_sentiment['compound'].resample('D').mean()
                    
                    # Add to correlation data
                    correlation_data = correlation_data.join(daily_twitter_sentiment.rename('Twitter_Sentiment'), how='left')
            
            # Fill missing values with forward fill
            correlation_data.fillna(method='ffill', inplace=True)
            
            # Calculate correlation
            correlation_matrix = correlation_data.corr()
            
            # Save correlation matrix
            correlation_matrix.to_csv('data/sentiment/sentiment_price_correlation.csv')
            print(f"Sentiment-price correlation saved to data/sentiment/sentiment_price_correlation.csv")
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation between Sentiment and Price Movements')
            plt.tight_layout()
            plt.savefig('plots/sentiment/sentiment_price_correlation.png')
            print("Correlation heatmap saved to plots/sentiment/sentiment_price_correlation.png")
            
            # Plot sentiment and price over time
            plt.figure(figsize=(14, 10))
            
            # Plot price
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(correlation_data.index, correlation_data['Close'], color='blue')
            ax1.set_ylabel('Nifty 50 Price')
            ax1.set_title('Nifty 50 Price and Sentiment Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Plot news sentiment if available
            if 'News_Sentiment' in correlation_data.columns:
                ax2 = plt.subplot(3, 1, 2, sharex=ax1)
                ax2.plot(correlation_data.index, correlation_data['News_Sentiment'], color='green')
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax2.set_ylabel('News Sentiment')
                ax2.grid(True, alpha=0.3)
            
            # Plot Twitter sentiment if available
            if 'Twitter_Sentiment' in correlation_data.columns:
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                ax3.plot(correlation_data.index, correlation_data['Twitter_Sentiment'], color='purple')
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax3.set_ylabel('Twitter Sentiment')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/sentiment/sentiment_price_time_series.png')
            print("Sentiment and price time series plot saved to plots/sentiment/sentiment_price_time_series.png")
            
            return correlation_matrix
            
        except Exception as e:
            print(f"Error correlating sentiment with price: {e}")
            return None
    
    def run_sentiment_analysis(self):
        """
        Run all sentiment analysis functions
        """
        print("Running all sentiment analysis...")
        
        # Analyze news sentiment
        news_sentiment = self.analyze_news_sentiment()
        
        # Analyze Twitter sentiment
        twitter_sentiment = self.analyze_twitter_sentiment()
        
        # Calculate overall sentiment
        overall_sentiment = self.calculate_overall_sentiment(news_sentiment, twitter_sentiment)
        
        # Generate visualizations
        self.plot_sentiment_distribution(news_sentiment, twitter_sentiment)
        self.plot_sentiment_over_time(news_sentiment, twitter_sentiment)
        self.generate_word_clouds(news_sentiment, twitter_sentiment)
        
        # Extract key topics
        key_topics = self.extract_key_topics(news_sentiment, twitter_sentiment)
        
        # Correlate sentiment with price
        correlation_matrix = self.correlate_sentiment_with_price()
        
        print("Sentiment analysis completed successfully!")
        
        # Return the results
        return {
            'news_sentiment': news_sentiment,
            'twitter_sentiment': twitter_sentiment,
            'overall_sentiment': overall_sentiment,
            'key_topics': key_topics,
            'correlation_matrix': correlation_matrix
        }

if __name__ == "__main__":
    # Create an instance of the SentimentAnalysis class
    sentiment_analysis = SentimentAnalysis()
    
    # Run all sentiment analysis
    results = sentiment_analysis.run_sentiment_analysis()
    
    # Print overall sentiment
    if not results['overall_sentiment'].empty:
        print("\nOverall Sentiment Results:")
        print(results['overall_sentiment'])
