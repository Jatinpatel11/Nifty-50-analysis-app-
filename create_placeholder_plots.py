import matplotlib.pyplot as plt
import numpy as np

# Create a simple sentiment distribution pie chart
labels = ['Positive', 'Neutral', 'Negative']
sizes = [45, 35, 20]
colors = ['green', 'gray', 'red']
explode = (0.1, 0, 0)  # explode the 1st slice (Positive)

plt.figure(figsize=(10, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Sentiment Distribution')
plt.savefig('plots/sentiment/sentiment_distribution.png')
plt.close()

# Create a word cloud placeholder
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.5, 'Nifty 50 Word Cloud', 
         horizontalalignment='center', verticalalignment='center', 
         fontsize=30, color='blue')
plt.axis('off')
plt.savefig('plots/sentiment/word_clouds.png')
plt.close()

# Create a top topics placeholder
plt.figure(figsize=(12, 8))
topics = ['Economy', 'Banking', 'IT Sector', 'Auto', 'Pharma']
frequency = [30, 25, 20, 15, 10]

plt.barh(topics, frequency, color='skyblue')
plt.xlabel('Frequency')
plt.title('Top Topics in Financial News')
plt.tight_layout()
plt.savefig('plots/sentiment/top_topics.png')
plt.close()

# Create feature importance chart
features = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_20', 'Volatility', 'Sentiment']
importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]

plt.figure(figsize=(12, 8))
plt.barh(features, importance, color='lightgreen')
plt.xlabel('Importance')
plt.title('Feature Importance in Prediction Model')
plt.tight_layout()
plt.savefig('plots/predictions/feature_importances.png')
plt.close()

# Create price trends chart
dates = np.arange(30)
close_prices = 22000 + 500 * np.sin(dates/10) + dates * 10
sma_20 = close_prices - 200
sma_50 = close_prices - 400

plt.figure(figsize=(12, 8))
plt.plot(dates, close_prices, label='Close Price', color='blue')
plt.plot(dates, sma_20, label='SMA 20', color='orange')
plt.plot(dates, sma_50, label='SMA 50', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Nifty 50 Price Trends')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/price_trends.png')
plt.close()

print("All placeholder plot images created successfully!")
