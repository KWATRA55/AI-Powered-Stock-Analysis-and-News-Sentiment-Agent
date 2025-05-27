# news_fetcher.py
# Module for fetching news articles using NewsAPI.org

import os
from newsapi import NewsApiClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY not found in environment variables. News fetching will fail.")
    newsapi = None
else:
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    except Exception as e:
        print(f"Error initializing NewsApiClient: {e}")
        newsapi = None

def get_top_headlines_for_stock(
    stock_name: str, 
    stock_ticker: str, # Added stock_ticker for more precise querying
    num_articles: int = 10, # Increased default to fetch more for filtering
    language: str = 'en', 
    sort_by: str = 'relevancy' # Changed to 'relevancy' for initial fetch
    ):
    """
    Fetches top news headlines for a given stock query.
    Args:
        stock_name (str): The company name (e.g., "Tesla, Inc.").
        stock_ticker (str): The stock ticker (e.g., "TSLA").
        num_articles (int): The number of articles to retrieve.
        language (str): The language of the articles (e.g., 'en').
        sort_by (str): How to sort the articles ('relevancy', 'popularity', 'publishedAt').
    Returns:
        list: A list of dictionaries, where each dictionary contains article details,
              or an empty list if an error occurs or no articles are found.
    """
    if not newsapi:
        print("NewsApiClient not initialized. Cannot fetch news.")
        return []
    
    # Construct a more specific query
    # Using exact phrases for company name and ticker, and common financial terms
    # NewsAPI supports boolean operators and phrase matching with quotes.
    query = f'("{stock_name}" OR "{stock_ticker}") AND (stock OR shares OR earnings OR "price target" OR analyst OR market OR investors)'
    # Alternative simpler query: query = f'"{stock_name}" AND "{stock_ticker}"'
    # Or even just: query = f'{stock_ticker} {stock_name}'
    
    print(f"NewsAPI query: {query}")

    try:
        all_articles = newsapi.get_everything(
            q=query,
            language=language,
            sort_by=sort_by, 
            page_size=num_articles 
        )

        articles_to_return = []
        if all_articles['status'] == 'ok':
            for article in all_articles['articles']:
                articles_to_return.append({
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'content': article.get('content'), # Also fetch content if available for better relevance check
                    'url': article.get('url'),
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name')
                })
            return articles_to_return
        else:
            print(f"Error from NewsAPI: {all_articles.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Error fetching news for '{query}': {e}")
        return []

if __name__ == '__main__':
    if not NEWS_API_KEY:
        print("Please set your NEWS_API_KEY in a .env file to run this example.")
    else:
        sample_stock_name = "Tesla, Inc."
        sample_stock_ticker = "TSLA"
        print(f"Fetching news articles for '{sample_stock_name} ({sample_stock_ticker})'...")
        news = get_top_headlines_for_stock(sample_stock_name, sample_stock_ticker, num_articles=5)
        
        if news:
            for i, article in enumerate(news):
                print(f"\nArticle {i+1}:")
                print(f"  Title: {article['title']}")
                print(f"  Description: {article['description']}")
                # print(f"  Content snippet: {article.get('content', '')[:100]}...") # Show snippet of content
                print(f"  Source: {article['source']}")
        else:
            print("No news articles found or an error occurred.")