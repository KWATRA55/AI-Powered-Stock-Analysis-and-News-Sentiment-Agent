# stock_data.py
# Module for fetching stock data using yfinance

import yfinance as yf
import pandas as pd

def get_stock_info(ticker_symbol: str):
    """
    Fetches basic company information for a given stock ticker.
    Args:
        ticker_symbol (str): The stock ticker (e.g., "AAPL").
    Returns:
        dict: A dictionary containing company information, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        # Selecting a subset of useful information
        # You can expand this list based on what you find relevant
        relevant_info = {
            "symbol": info.get("symbol"),
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "regularMarketPrice": info.get("regularMarketPrice"),
            "regularMarketVolume": info.get("regularMarketVolume"),
            "shortSummary": info.get("longBusinessSummary", "N/A")[:500] + "..." if info.get("longBusinessSummary") else "N/A"
        }
        return relevant_info
    except Exception as e:
        print(f"Error fetching stock info for {ticker_symbol}: {e}")
        return None

def get_historical_stock_data(ticker_symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Fetches historical stock data (OHLCV) for a given ticker.
    Args:
        ticker_symbol (str): The stock ticker (e.g., "AAPL").
        period (str): The period for which to fetch data (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
        interval (str): The data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").
    Returns:
        pandas.DataFrame: A DataFrame containing historical OHLCV data, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        hist_data = stock.history(period=period, interval=interval)
        if hist_data.empty:
            print(f"No historical data found for {ticker_symbol} with period {period} and interval {interval}.")
            return None
        return hist_data
    except Exception as e:
        print(f"Error fetching historical stock data for {ticker_symbol}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    sample_ticker = "MSFT"
    print(f"Fetching info for {sample_ticker}...")
    info = get_stock_info(sample_ticker)
    if info:
        print("\nStock Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print(f"\nFetching historical data for {sample_ticker} (last 1 month)...")
    hist_df = get_historical_stock_data(sample_ticker, period="1mo")
    if hist_df is not None:
        print("\nHistorical Data (last 5 days):")
        print(hist_df.tail())