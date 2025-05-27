# AI-Powered-Stock-Analysis-and-News-Sentiment-Agent
This Python-based system dives deep into market data to offer a multi-faceted outlook on publicly traded stocks

This Python-based system dives deep into market data to offer a multi-faceted outlook on publicly traded stocks. Here’s how it works:

Quantitative Engine: Calculates key technical indicators like SMAs, RSI, and MACD from historical stock data (via yfinance).
Intelligent News Analysis (with Google Gemini):
Fetches a broad set of recent news articles (via NewsAPI).
Critically, it then uses Google Gemini Pro to perform a relevance check, filtering down to only the news items directly pertinent to the specific stock being analyzed. This was a key iteration to ensure signal quality!
For these highly relevant articles, Gemini is again leveraged for nuanced sentiment analysis, determining if the news is positive, negative, or neutral for the stock.
Synthesized Outlook: The agent combines these quantitative signals and qualitative news sentiment insights to generate an overall assessment (e.g., "Positive Outlook," "Neutral," "Negative Outlook"), complete with a confidence level and the key drivers behind the assessment.
Tech Backbone: The backend is built with Python and FastAPI, serving up the analysis through a clean API interface.
