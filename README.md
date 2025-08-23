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

---

## 🖥️ Running the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/KWATRA55/AI-Powered-Stock-Analysis-and-News-Sentiment-Agent.git
cd <new-repo>
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
*(Windows PowerShell: `venv\Scripts\Activate`)*

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the project root with your API keys:
```env
NEWSAPI_KEY=your_newsapi_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### 5️⃣ Start the FastAPI Server
```bash
uvicorn main:app --reload
```
You should see something like:
```
Uvicorn running on http://127.0.0.1:8000
```

### 6️⃣ Access the API Documentation
- Swagger UI: **http://127.0.0.1:8000/docs**
- ReDoc: **http://127.0.0.1:8000/redoc**

---

⚠ **Note:**  
- On macOS, you might see a `NotOpenSSLWarning` from `urllib3` about LibreSSL — it’s harmless and can be ignored.  
- Make sure you have valid API keys for both **NewsAPI** and **Google Gemini** for the project to run fully.
