# quantitative_analysis.py
# Module for calculating basic technical indicators

import pandas as pd
import numpy as np # Ensure numpy is imported for np.nan

def calculate_sma(data: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
    """Calculates the Simple Moving Average (SMA)."""
    if price_col not in data.columns:
        # raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
        print(f"Warning: Price column '{price_col}' not found in DataFrame for SMA. Returning empty Series.")
        return pd.Series(dtype='float64') # Return empty series to avoid crashing, let caller handle
    if len(data) < window:
        return pd.Series(index=data.index, dtype='float64')
    return data[price_col].rolling(window=window).mean()

def calculate_ema(data: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
    """Calculates the Exponential Moving Average (EMA)."""
    if price_col not in data.columns:
        # raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
        print(f"Warning: Price column '{price_col}' not found in DataFrame for EMA. Returning empty Series.")
        return pd.Series(dtype='float64')
    if len(data) < window: 
        return pd.Series(index=data.index, dtype='float64')
    return data[price_col].ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.DataFrame, window: int = 14, price_col: str = 'Close') -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    if price_col not in data.columns:
        print(f"Warning: Price column '{price_col}' not found in DataFrame for RSI. Returning empty Series.")
        return pd.Series(dtype='float64')
    if len(data) < window + 1: 
        return pd.Series(index=data.index, dtype='float64')

    delta = data[price_col].diff(1)
    gain = delta.where(delta > 0, 0.0) 
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing for RSI (com = window - 1)
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    
    # Avoid division by zero if avg_loss is 0 for some periods
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN to avoid division by zero, then fill
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle cases where avg_loss was 0:
    # If avg_loss is 0 and avg_gain is > 0, RSI is 100.
    # If avg_loss is 0 and avg_gain is 0 (no change), RSI could be considered neutral (e.g., 50 or carry previous).
    # For simplicity, if rs is inf (avg_loss was 0, avg_gain > 0), rsi becomes 100.
    # If rs is NaN (e.g. avg_loss and avg_gain were 0 or NaN), rsi remains NaN.
    rsi.loc[rs == np.inf] = 100.0
    rsi.loc[rs.isna() & avg_loss.eq(0) & avg_gain.eq(0)] = 50.0 # Or some other neutral value if both are zero

    # RSI is typically not defined for the first 'window' periods for calculation stability.
    if len(rsi) >= window:
        rsi.iloc[:window] = np.nan
    else: # If data is too short, fill all with NaN
        rsi[:] = np.nan
    return rsi

def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9, price_col: str = 'Close'):
    """
    Calculates MACD (Moving Average Convergence Divergence).
    Returns MACD line, Signal line, and MACD Histogram.
    """
    if price_col not in data.columns:
        print(f"Warning: Price column '{price_col}' not found in DataFrame for MACD. Returning empty Series for all.")
        nan_series = pd.Series(dtype='float64')
        return nan_series, nan_series, nan_series
    
    min_len_required = long_window + signal_window 
    if len(data) < min_len_required:
        nan_series = pd.Series(index=data.index, dtype='float64')
        return nan_series, nan_series, nan_series 

    short_ema = calculate_ema(data, short_window, price_col)
    long_ema = calculate_ema(data, long_window, price_col)
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def get_technical_indicators(hist_data: pd.DataFrame):
    """
    Calculates a set of technical indicators from historical data.
    Returns a dictionary containing the latest values of the calculated indicators.
    """
    if hist_data is None or hist_data.empty:
        return {
            "error": "Historical data is empty or None."
        }
    if 'Close' not in hist_data.columns: # Ensure Close column exists
        return {
            "error": "Historical data must contain a 'Close' column."
        }


    indicators = {}
    try:
        # SMAs
        sma_50 = calculate_sma(hist_data, 50)
        indicators["sma_50"] = round(sma_50.iloc[-1], 2) if not sma_50.empty and pd.notna(sma_50.iloc[-1]) else None
        sma_200 = calculate_sma(hist_data, 200)
        indicators["sma_200"] = round(sma_200.iloc[-1], 2) if not sma_200.empty and pd.notna(sma_200.iloc[-1]) else None
        
        # RSI
        rsi_14 = calculate_rsi(hist_data, 14)
        indicators["rsi_14"] = round(rsi_14.iloc[-1], 2) if not rsi_14.empty and pd.notna(rsi_14.iloc[-1]) else None

        # MACD
        macd_line, signal_line, macd_hist = calculate_macd(hist_data)
        indicators["macd_line"] = round(macd_line.iloc[-1], 2) if not macd_line.empty and pd.notna(macd_line.iloc[-1]) else None
        indicators["macd_signal"] = round(signal_line.iloc[-1], 2) if not signal_line.empty and pd.notna(signal_line.iloc[-1]) else None
        indicators["macd_histogram"] = round(macd_hist.iloc[-1], 2) if not macd_hist.empty and pd.notna(macd_hist.iloc[-1]) else None
        
        if indicators["macd_histogram"] is not None and len(macd_hist) > 1 and pd.notna(macd_hist.iloc[-1]) and pd.notna(macd_hist.iloc[-2]):
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                indicators["macd_signal_cross"] = "Bullish Crossover"
            elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
                indicators["macd_signal_cross"] = "Bearish Crossover"
            elif macd_hist.iloc[-1] > 0:
                indicators["macd_signal_cross"] = "Bullish (MACD > Signal)"
            elif macd_hist.iloc[-1] < 0:
                indicators["macd_signal_cross"] = "Bearish (MACD < Signal)"
            else:
                indicators["macd_signal_cross"] = "Neutral (On Signal Line)"
        else:
            indicators["macd_signal_cross"] = "N/A (Insufficient data or values)"

    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        # Return any indicators calculated so far along with the error
        indicators["error"] = f"Error in calculations: {str(e)}"
        return indicators # Return partially filled dict with error
    return indicators

if __name__ == '__main__':
    # Create a sample DataFrame for testing - NOW WITH APPROX 1 YEAR OF DATA
    # A typical trading year has about 252 days.
    date_rng = pd.date_range(start='2023-05-01', end='2024-05-24', freq='B') 
    print(f"Generating sample data with {len(date_rng)} trading days for standalone test.") # This will be around 250-260 days
    sample_close_prices = [150 + i*0.1 + (i%20)*0.5 - (i%15)*0.3 + (i%5)*1 for i in range(len(date_rng))] 
    sample_df = pd.DataFrame(date_rng, columns=['Date'])
    sample_df['Close'] = sample_close_prices
    sample_df.set_index('Date', inplace=True)
    
    print("\nCalculating indicators for sample data (approx 1 year):")
    indicators_result = get_technical_indicators(sample_df.copy())
    if "error" in indicators_result:
        print(f"  Error: {indicators_result['error']}")
        for key, value in indicators_result.items():
            if key != "error":
                print(f"  Partial {key}: {value}")
    else:
        for key, value in indicators_result.items():
            print(f"  {key}: {value}")

    print("\nCalculating indicators for very short sample data (10 days):")
    short_date_rng = pd.date_range(start='2024-05-10', periods=10, freq='B')
    short_close_prices = [150 + i*0.5 for i in range(10)]
    short_sample_df = pd.DataFrame({'Close': short_close_prices}, index=short_date_rng)
    
    short_indicators_result = get_technical_indicators(short_sample_df.copy())
    if "error" in short_indicators_result:
        print(f"  Error: {short_indicators_result['error']}")
        for key, value in short_indicators_result.items():
            if key != "error":
                print(f"  Partial {key}: {value}")
    else:
        for key, value in short_indicators_result.items():
            print(f"  {key}: {value}")

    print("\nCalculating indicators for data without 'Close' column:")
    no_close_df = pd.DataFrame({'Open': [1,2,3]}, index=pd.date_range(start='2024-01-01', periods=3))
    no_close_indicators = get_technical_indicators(no_close_df)
    print(no_close_indicators)

    print("\nCalculating indicators for empty data:")
    empty_df = pd.DataFrame()
    empty_indicators = get_technical_indicators(empty_df)
    print(empty_indicators)