# AI-Powered Stock Analysis & News Sentiment Agent

A Python-based system that combines technical market analysis with AI-driven news sentiment evaluation to provide a comprehensive outlook on publicly traded stocks.

---

## 🚀 Features

### 📊 Quantitative Engine
- Calculates key technical indicators using **historical stock data** from `yfinance`:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)

### 📰 Intelligent News Analysis (Powered by Google Gemini)
- Fetches recent **news articles** using `NewsAPI`.
- Uses **Google Gemini Pro** for:
  - **Relevance Filtering** – Only includes articles directly related to the target stock.
  - **Sentiment Analysis** – Determines whether the news is **positive**, **negative**, or **neutral** for the stock.

### 🧠 Synthesized Outlook
- Merges **technical signals** and **news sentiment** into an overall assessment:
  - Examples: _Positive Outlook_, _Neutral_, _Negative Outlook_
- Includes:
  - Confidence score
  - Key drivers influencing the assessment

---

## 🛠️ Tech Stack
- **Backend:** Python + FastAPI
- **Data Sources:** `yfinance`, `NewsAPI`
- **AI Analysis:** Google Gemini Pro

---

## 📌 How It Works
1. Pulls **historical stock data** for technical analysis.
2. Gathers **recent news articles** for the stock.
3. Filters news to keep only **highly relevant items**.
4. Analyzes sentiment using **Google Gemini Pro**.
5. Combines quantitative + qualitative data into a final stock outlook.
