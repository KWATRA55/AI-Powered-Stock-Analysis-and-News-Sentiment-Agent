# main.py
# FastAPI application for stock analysis agent

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time # For potential delays

# Import your modules
from stock_data import get_stock_info, get_historical_stock_data
from quantitative_analysis import get_technical_indicators
from news_fetcher import get_top_headlines_for_stock
from sentiment_analyzer import analyze_sentiment_gemini, get_news_relevance_gemini # Import new function

# --- Pydantic Models (StockAnalysisResponse might need a field for raw_news_count if desired) ---
class StockAnalysisRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol")

class NewsArticleSentiment(BaseModel):
    title: Optional[str] = Field(None, example="Stock Hits Record High")
    description: Optional[str] = Field(None, example="Detailed description of the news.")
    url: Optional[str] = Field(None, example="https://news.example.com/article")
    publishedAt: Optional[str] = Field(None, example="2024-05-25T12:00:00Z")
    source: Optional[str] = Field(None, example="Example News Source")
    relevance_score: Optional[int] = Field(None, example=5) # Add relevance score
    sentiment: Optional[str] = Field(None, example="Positive")
    justification: Optional[str] = Field(None, example="The news indicates strong performance.")


class StockAnalysisResponse(BaseModel):
    stock_info: Optional[Dict[str, Any]] = Field(None, example={"symbol": "AAPL", "longName": "Apple Inc."})
    technical_indicators: Optional[Dict[str, Any]] = Field(None, example={"sma_50": 170.50, "rsi_14": 65.0})
    news_with_sentiment: List[NewsArticleSentiment] = Field(default_factory=list)
    overall_assessment: str = Field(..., example="Positive Outlook")
    assessment_confidence: str = Field(..., example="Medium")
    assessment_drivers: List[str] = Field(default_factory=list, example=["Positive SMA Trend", "RSI Bullish Zone"])
    raw_news_fetched_count: int = Field(0, example=10)
    relevant_news_analyzed_count: int = Field(0, example=3)
    error_message: Optional[str] = Field(None, example="Could not fetch external data.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stock Analysis Agent API",
    description="Provides quantitative stock analysis and news sentiment analysis using Google Gemini, with relevance filtering.",
    version="0.2.0" # Incremented version
)

# --- CORS Configuration (remains the same) ---
origins = [ "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- determine_overall_assessment (remains the same as main_py_fastapi_v2) ---
def determine_overall_assessment(tech_indicators: Optional[dict], news_sentiments: List[NewsArticleSentiment]) -> tuple[str, str, List[str]]:
    assessment_drivers = []
    tech_score = 0.0 
    tech_signals_available = 0 

    if tech_indicators and not tech_indicators.get("error"):
        sma_50 = tech_indicators.get("sma_50")
        sma_200 = tech_indicators.get("sma_200")
        rsi = tech_indicators.get("rsi_14")
        macd_signal_cross = tech_indicators.get("macd_signal_cross")
        macd_histogram = tech_indicators.get("macd_histogram")

        if sma_50 is not None and sma_200 is not None:
            tech_signals_available += 1
            if sma_50 > sma_200:
                tech_score += 0.75
                assessment_drivers.append(f"Positive SMA Trend (50-day: {sma_50} > 200-day: {sma_200})")
            elif sma_50 < sma_200:
                tech_score -= 0.75
                assessment_drivers.append(f"Negative SMA Trend (50-day: {sma_50} < 200-day: {sma_200})")
            else:
                 assessment_drivers.append(f"Neutral SMA Trend (50-day: {sma_50} = 200-day: {sma_200})")
        
        if rsi is not None:
            tech_signals_available += 1
            if rsi < 30:
                tech_score += 0.5 
                assessment_drivers.append(f"RSI Oversold ({rsi:.2f}) - Potential Rebound")
            elif rsi > 70:
                tech_score -= 0.5 
                assessment_drivers.append(f"RSI Overbought ({rsi:.2f}) - Potential Pullback")
            elif 30 <= rsi < 45:
                 assessment_drivers.append(f"RSI Bearish Zone ({rsi:.2f})")
            elif 55 < rsi <= 70:
                 assessment_drivers.append(f"RSI Bullish Zone ({rsi:.2f})")
            else: 
                assessment_drivers.append(f"RSI Neutral ({rsi:.2f})")

        if macd_signal_cross and macd_signal_cross != "N/A (Insufficient data or values)":
            tech_signals_available += 1
            hist_val_str = f"(Hist: {macd_histogram})" if macd_histogram is not None else ""
            if "Bullish Crossover" in macd_signal_cross:
                tech_score += 1.0
                assessment_drivers.append(f"MACD Bullish Crossover {hist_val_str}")
            elif "Bearish Crossover" in macd_signal_cross:
                tech_score -= 1.0
                assessment_drivers.append(f"MACD Bearish Crossover {hist_val_str}")
            elif "Bullish (MACD > Signal)" in macd_signal_cross:
                tech_score += 0.5
                assessment_drivers.append(f"MACD Bullish Trend {hist_val_str}")
            elif "Bearish (MACD < Signal)" in macd_signal_cross:
                tech_score -= 0.5
                assessment_drivers.append(f"MACD Bearish Trend {hist_val_str}")
            else: 
                assessment_drivers.append(f"MACD Neutral {hist_val_str}")
    elif tech_indicators and tech_indicators.get("error"):
        assessment_drivers.append(f"Technical indicators error: {tech_indicators.get('error')}")
    else: 
        assessment_drivers.append("Technical indicators unavailable.")

    news_score = 0.0
    valid_news_items = 0
    if news_sentiments: # news_sentiments here are already filtered for relevance and have sentiment
        positive_news_count = sum(1 for item in news_sentiments if item.sentiment == "Positive")
        negative_news_count = sum(1 for item in news_sentiments if item.sentiment == "Negative")
        valid_news_items = len(news_sentiments) # All items in this list should be valid by now
        
        if valid_news_items > 0:
            news_score = (positive_news_count - negative_news_count) / valid_news_items
            driver_text = f"Relevant News Sentiment ({positive_news_count} Pos, {negative_news_count} Neg of {valid_news_items} analyzed)"
            if news_score > 0.33: assessment_drivers.append(f"Overall Positive {driver_text}")
            elif news_score < -0.33: assessment_drivers.append(f"Overall Negative {driver_text}")
            else: assessment_drivers.append(f"Overall Neutral/Mixed {driver_text}")
        else:
            assessment_drivers.append("No relevant news with sentiment to score.")
    # If news_sentiments list is empty, it means no relevant news was found/analyzed.
    # The caller of this function will ensure news_sentiments only contains relevant items.
    
    normalized_tech_score = 0.0
    if tech_signals_available > 0:
        normalizing_factor = 2.25 
        normalized_tech_score = tech_score / normalizing_factor
        normalized_tech_score = max(-1.0, min(1.0, normalized_tech_score)) 
    
    tech_weight = 0.6
    news_weight = 0.4
    final_score = 0.0

    if tech_signals_available == 0 and valid_news_items == 0:
        final_score = 0.0
        assessment_drivers = ["Insufficient data for assessment."] 
    elif tech_signals_available == 0: 
        final_score = news_score
        assessment_drivers.append("Assessment based solely on news sentiment.")
    elif valid_news_items == 0: 
        final_score = normalized_tech_score
        assessment_drivers.append("Assessment based solely on technical indicators.")
    else: 
        final_score = (normalized_tech_score * tech_weight) + (news_score * news_weight)

    outlook: str
    confidence: str

    if final_score > 0.6:
        outlook = "Strongly Positive Outlook"
        confidence = "High"
    elif final_score > 0.2:
        outlook = "Positive Outlook"
        confidence = "Medium"
    elif final_score >= -0.2:
        outlook = "Neutral Outlook"
        if abs(final_score) < 0.05 and tech_signals_available > 0 and valid_news_items > 0:
             confidence = "High" 
        elif abs(final_score) < 0.05:
             confidence = "Low" 
        else:
             confidence = "Medium"
    elif final_score >= -0.6:
        outlook = "Negative Outlook"
        confidence = "Medium"
    else: 
        outlook = "Strongly Negative Outlook"
        confidence = "High"

    if tech_signals_available == 0 and valid_news_items == 0:
        outlook = "Indeterminate"
        confidence = "Very Low"
    elif (tech_signals_available < 2 and valid_news_items == 0) or \
         (tech_signals_available == 0 and valid_news_items < 2) or \
         (tech_signals_available <= 1 and valid_news_items <= 1 and not (tech_signals_available == 0 and valid_news_items == 0)):
        if confidence == "High": confidence = "Medium"
        elif confidence == "Medium": confidence = "Low"
    
    if not assessment_drivers:
        assessment_drivers.append("No specific drivers identified.")
    elif outlook == "Indeterminate" and len(assessment_drivers) > 1 and assessment_drivers[0] != "Insufficient data for assessment.":
        assessment_drivers = ["Insufficient data for assessment."]

    return outlook, confidence, assessment_drivers

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Stock Analysis Agent API! Visit /docs for API documentation."}

@app.post("/analyze_stock/", response_model=StockAnalysisResponse, summary="Analyze Stock Data and News Sentiment",
          description="Fetches stock data, calculates technical indicators, retrieves news, performs relevance & sentiment analysis via Gemini, and provides an overall assessment.")
async def analyze_stock_endpoint(request: StockAnalysisRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    print(f"Analyzing ticker: {ticker}")

    MAX_RELEVANT_ARTICLES_TO_PROCESS = 3 
    MIN_RELEVANCE_SCORE_THRESHOLD = 4 

    stock_info = get_stock_info(ticker)
    company_name = stock_info.get("longName", ticker) if stock_info and stock_info.get("longName") else ticker
    
    hist_data = get_historical_stock_data(ticker, period="1y", interval="1d") 
    
    tech_indicators_result = None
    if hist_data is None or hist_data.empty:
        print(f"Warning: Could not fetch historical data for {ticker}. Technical indicators will be unavailable.")
        tech_indicators_result = {"error": f"Could not fetch historical data for {ticker}."}
    elif 'Close' not in hist_data.columns:
        print(f"Warning: Historical data for {ticker} is missing 'Close' column.")
        tech_indicators_result = {"error": "Historical data missing 'Close' column."}
    else:
        tech_indicators_result = get_technical_indicators(hist_data.copy())

    print(f"Fetching news for {company_name} ({ticker})...")
    raw_news_articles_data = get_top_headlines_for_stock(
        stock_name=company_name, 
        stock_ticker=ticker, 
        num_articles=10 
    )
    raw_news_fetched_count = len(raw_news_articles_data)
    print(f"Fetched {raw_news_fetched_count} raw articles.")

    articles_with_relevance = []
    if raw_news_articles_data:
        for article_data in raw_news_articles_data:
            title = article_data.get('title', '')
            snippet = article_data.get('description') or article_data.get('content', '') 
            if not snippet and title: snippet = title 
            
            if not title and not snippet: 
                print(f"Skipping article due to no title/snippet: {article_data.get('url')}")
                continue
            
            # Consider a small delay if making many rapid API calls
            # time.sleep(0.5) # 0.5s delay can help with strict API rate limits on free tiers
            
            relevance_info = get_news_relevance_gemini(title, snippet, ticker, company_name)
            articles_with_relevance.append({
                **article_data, 
                "relevance_score": relevance_info.get('relevance_score', 1),
                "relevance_justification": relevance_info.get('relevance_justification', 'N/A')
            })
            print(f"  Article: \"{title[:50]}...\" Relevance: {relevance_info.get('relevance_score')}")
        
        articles_with_relevance.sort(key=lambda x: x.get('relevance_score', 1), reverse=True)
    
    selected_articles_for_sentiment_input = []
    for art in articles_with_relevance:
        if art.get('relevance_score', 1) >= MIN_RELEVANCE_SCORE_THRESHOLD and \
           len(selected_articles_for_sentiment_input) < MAX_RELEVANT_ARTICLES_TO_PROCESS:
            selected_articles_for_sentiment_input.append(art)
    
    relevant_news_analyzed_count = len(selected_articles_for_sentiment_input)
    print(f"Selected {relevant_news_analyzed_count} articles for sentiment analysis (Relevance >= {MIN_RELEVANCE_SCORE_THRESHOLD}).")

    final_news_with_sentiment_list: List[NewsArticleSentiment] = []
    if selected_articles_for_sentiment_input:
        for relevant_article_data in selected_articles_for_sentiment_input:
            article_title = relevant_article_data.get('title', '')
            article_desc = relevant_article_data.get('description') or relevant_article_data.get('content', '')
            article_text_for_sentiment = article_title
            if article_desc:
                article_text_for_sentiment += ". " + article_desc
            
            sentiment_result_dict = {'sentiment': 'Neutral', 'justification': 'Not analyzed or no content.'}
            if article_text_for_sentiment.strip():
                # time.sleep(0.5) # Optional delay
                sentiment_result_dict = analyze_sentiment_gemini(
                    text_content=article_text_for_sentiment[:2000], 
                    stock_ticker=ticker
                )
            
            final_news_with_sentiment_list.append(
                NewsArticleSentiment(
                    title=article_title,
                    description=relevant_article_data.get('description'), 
                    url=relevant_article_data.get('url'),
                    publishedAt=relevant_article_data.get('publishedAt'),
                    source=relevant_article_data.get('source'),
                    relevance_score=relevant_article_data.get('relevance_score'),
                    sentiment=sentiment_result_dict.get('sentiment'),
                    justification=sentiment_result_dict.get('justification')
                )
            )
            print(f"  Sentiment for \"{article_title[:50]}...\": {sentiment_result_dict.get('sentiment')}")

    overall_outlook, confidence, drivers = determine_overall_assessment(
        tech_indicators_result, 
        final_news_with_sentiment_list 
    )

    return StockAnalysisResponse(
        stock_info=stock_info,
        technical_indicators=tech_indicators_result,
        news_with_sentiment=final_news_with_sentiment_list, 
        overall_assessment=overall_outlook,
        assessment_confidence=confidence,
        assessment_drivers=drivers,
        raw_news_fetched_count=raw_news_fetched_count,
        relevant_news_analyzed_count=relevant_news_analyzed_count
    )