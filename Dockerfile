FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data/sentiment data/predictions plots/sentiment plots/predictions

# Download NLTK resources
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Expose the port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app_fixed_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
