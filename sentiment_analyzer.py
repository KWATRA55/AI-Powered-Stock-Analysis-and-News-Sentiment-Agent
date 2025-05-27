# sentiment_analyzer.py
# Module for performing sentiment analysis and relevance scoring using Google Gemini API

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_model = None # Initialize as None

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Sentiment analysis and relevance scoring will fail.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using a model suitable for both tasks. gemini-1.5-flash is often good for speed/cost.
        genai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Test generation to ensure model is configured (optional, can remove)
        # genai_model.generate_content("test") 
        print("Gemini model configured successfully.")
    except Exception as e:
        print(f"Error configuring Google Gemini API: {e}")
        genai_model = None # Ensure it's None if configuration fails


def call_gemini_with_retry(prompt_text: str, max_retries: int = 2, delay: int = 5):
    """ Helper function to call Gemini API with retry logic for specific errors. """
    if not genai_model:
        return {'error': 'Gemini model not initialized.'}
    
    for attempt in range(max_retries + 1):
        try:
            response = genai_model.generate_content(prompt_text)
            # Check for specific blockages (though the SDK might raise errors for these too)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name
                print(f"Gemini API call blocked. Reason: {reason}. Prompt: '{prompt_text[:100]}...'")
                return {'error': f"Blocked by Gemini API due to {reason}."}
            
            return response.text # Return the text part of the response

        except Exception as e:
            # Specific Google API errors that might be retriable (e.g., 429, 500, 503)
            # The google-generativeai SDK might handle some of these internally or raise specific exceptions.
            # For simplicity here, we're catching general exceptions that might include these.
            # A more robust solution would inspect the type of exception or error code.
            if "429" in str(e) or "500" in str(e) or "503" in str(e) or "Resource has been exhausted" in str(e):
                print(f"Gemini API error (possibly rate limit or temporary issue): {e}. Attempt {attempt + 1} of {max_retries + 1}.")
                if attempt < max_retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    print("Max retries reached for Gemini API call.")
                    return {'error': f"Max retries reached. Last error: {str(e)}"}
            else: # Non-retriable error
                print(f"Gemini API call failed with non-retriable error: {e}")
                return {'error': f"Gemini API call failed: {str(e)}"}
    return {'error': 'Gemini call failed after retries.'} # Should be caught by the loop


def get_news_relevance_gemini(article_title: str, article_snippet: str, stock_ticker: str, stock_name: str):
    """
    Analyzes the relevance of a news article to a specific stock using Gemini.
    Args:
        article_title (str): The title of the news article.
        article_snippet (str): A snippet/description of the news article.
        stock_ticker (str): The stock ticker (e.g., "TSLA").
        stock_name (str): The name of the company (e.g., "Tesla, Inc.").
    Returns:
        dict: With 'relevance_score' (1-5, 5 is most relevant) and 'relevance_justification', 
              or error details. Defaults to low relevance on error.
    """
    if not article_title and not article_snippet:
        return {'relevance_score': 1, 'relevance_justification': 'No article content provided.'}

    text_content = f"Title: {article_title}\nSnippet: {article_snippet}"
    # Limit length to avoid overly long prompts
    max_len = 2000 
    truncated_text = text_content[:max_len] if len(text_content) > max_len else text_content

    prompt = f"""Analyze the relevance of the following news article to the stock {stock_ticker} ({stock_name}).
Is this news item DIRECTLY about {stock_name} ({stock_ticker}) or its products, financials, market performance, leadership, or major partnerships?
News Article:
---
{truncated_text}
---
Score its direct relevance to {stock_name} ({stock_ticker}) on a scale of 1 to 5, where:
1 = Not relevant at all (e.g., about a completely different company or topic).
2 = Slightly relevant (e.g., mentions the industry but not the company, or a minor, indirect link).
3 = Moderately relevant (e.g., discusses a competitor, or a broader market trend affecting the company).
4 = Relevant (e.g., directly discusses the company, its products, or market situation but may not be major news).
5 = Highly relevant (e.g., significant news directly impacting {stock_name}'s ({stock_ticker}) stock, like earnings, major announcements, legal issues, price targets by reputable analysts for THIS stock).

Provide a brief justification for your relevance score.
Return ONLY a JSON object with two keys: "relevance_score" (integer between 1 and 5) and "relevance_justification" (string).
Example for high relevance: {{"relevance_score": 5, "relevance_justification": "The article directly reports on {stock_name}'s quarterly earnings announcement."}}
Example for low relevance: {{"relevance_score": 1, "relevance_justification": "The article is about a different company in an unrelated sector."}}
"""
    
    response_text_or_error = call_gemini_with_retry(prompt)

    if isinstance(response_text_or_error, dict) and 'error' in response_text_or_error:
        print(f"Error in get_news_relevance_gemini: {response_text_or_error['error']}")
        return {'relevance_score': 1, 'relevance_justification': f"Error calling Gemini: {response_text_or_error['error']}"}

    try:
        cleaned_response_text = response_text_or_error.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        result = json.loads(cleaned_response_text.strip())
        
        if isinstance(result, dict) and "relevance_score" in result and "relevance_justification" in result:
            score = result["relevance_score"]
            if isinstance(score, int) and 1 <= score <= 5:
                return result
            else:
                print(f"Gemini returned invalid relevance_score: {score}. Defaulting relevance to 1.")
                return {'relevance_score': 1, 'relevance_justification': f"Invalid score from Gemini: {score}. Original justification: {result.get('relevance_justification')}"}
        else:
            print(f"Gemini relevance response was not the expected JSON format: {response_text_or_error}")
            return {'relevance_score': 1, 'relevance_justification': f"Invalid JSON structure for relevance: {response_text_or_error[:100]}"}
    except json.JSONDecodeError as json_e:
        print(f"Error decoding JSON from Gemini for relevance: {json_e} - Response was: {response_text_or_error}")
        return {'relevance_score': 1, 'relevance_justification': f"Could not parse relevance JSON. Raw: {response_text_or_error[:100]}"}
    except Exception as e:
        print(f"Unexpected error parsing Gemini relevance response: {e}")
        return {'relevance_score': 1, 'relevance_justification': f"Unexpected error: {str(e)}"}


def analyze_sentiment_gemini(text_content: str, stock_ticker: str = "this stock"):
    """
    Analyzes the sentiment of a given text using the Google Gemini API. (Existing function)
    """
    if not text_content or not text_content.strip():
        return {'sentiment': 'Neutral', 'justification': 'No text content provided for analysis.'}

    max_len = 2000 # Consistent length limit with relevance
    truncated_text = text_content[:max_len] if len(text_content) > max_len else text_content

    prompt = f"""Analyze the sentiment of the following news text SPECIFICALLY FOR its potential impact on the stock: "{stock_ticker}".
The news text is: "{truncated_text}"

Consider ONLY the direct implications for the stock's value or investor perception of "{stock_ticker}".
If the news is not about "{stock_ticker}" or has no clear financial implication for it, classify as Neutral.
Classify the sentiment strictly as 'Positive', 'Negative', or 'Neutral'.
Provide a brief, one-sentence justification for your classification, focusing on the key reasons for THIS stock.

Return ONLY a JSON object with two keys: "sentiment" (string) and "justification" (string).
Example for Positive: {{"sentiment": "Positive", "justification": "The report of increased earnings for {stock_ticker} is likely to boost investor confidence."}}
Example for Neutral due to irrelevance: {{"sentiment": "Neutral", "justification": "This news is about a different company and not relevant to {stock_ticker}."}}
"""

    response_text_or_error = call_gemini_with_retry(prompt)

    if isinstance(response_text_or_error, dict) and 'error' in response_text_or_error:
        print(f"Error in analyze_sentiment_gemini: {response_text_or_error['error']}")
        return {'sentiment': 'Error', 'justification': f"Error calling Gemini for sentiment: {response_text_or_error['error']}"}

    try:
        cleaned_response_text = response_text_or_error.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        result = json.loads(cleaned_response_text.strip())
        
        if isinstance(result, dict) and "sentiment" in result and "justification" in result:
            valid_sentiments = ["Positive", "Negative", "Neutral"]
            if result["sentiment"] not in valid_sentiments:
                # Attempt to normalize if slightly off (e.g. lowercase)
                if result["sentiment"].capitalize() in valid_sentiments:
                     result["sentiment"] = result["sentiment"].capitalize()
                else:
                    print(f"Gemini returned an invalid sentiment '{result['sentiment']}'. Defaulting to Neutral.")
                    result["sentiment"] = "Neutral"
                    result["justification"] += " (Original sentiment was invalid, defaulted to Neutral)"
            return result
        else:
            print(f"Gemini sentiment response was not the expected JSON format: {response_text_or_error}")
            return {'sentiment': 'Error', 'justification': f"Invalid JSON structure from Gemini for sentiment: {response_text_or_error[:100]}"}
    except json.JSONDecodeError as json_e:
        print(f"Error decoding JSON from Gemini for sentiment: {json_e} - Response was: {response_text_or_error}")
        if "positive" in response_text_or_error.lower(): sentiment = "Positive"
        elif "negative" in response_text_or_error.lower(): sentiment = "Negative"
        else: sentiment = "Neutral"
        return {'sentiment': sentiment, 'justification': f"Could not parse sentiment JSON. Raw response: {response_text_or_error[:200]}"}
    except Exception as e:
        print(f"An unexpected error occurred parsing Gemini sentiment response: {e}")
        return {'sentiment': 'Error', 'justification': f"Unexpected error parsing sentiment response: {str(e)}"}


if __name__ == '__main__':
    if not GEMINI_API_KEY or not genai_model:
        print("Please set your GEMINI_API_KEY in a .env file and ensure model is configured.")
    else:
        print("Testing Gemini Relevance Analyzer...")
        test_stock_ticker = "XYZ"
        test_stock_name = "XYZ Corp"
        
        relevant_article = {
            "title": f"{test_stock_name} announces record profits for Q1",
            "snippet": f"Shares of {test_stock_ticker} surged today after the company reported earnings that beat analyst expectations. CEO John Doe expressed optimism for the upcoming year."
        }
        irrelevant_article = {
            "title": "Global widget market sees steady growth",
            "snippet": "The worldwide market for widgets is projected to expand by 5% annually, according to a new report by MarketWatchers Inc. Key players include ABC Widgets and BestWidgets Ltd."
        }
        another_company_article = {
            "title": "ABC Corp (ticker ABC) launches new product",
            "snippet": "ABC Corp today unveiled its latest innovation, the SuperWidget 3000, which analysts believe will disrupt the industry."
        }

        test_articles = [relevant_article, irrelevant_article, another_company_article]
        for i, article in enumerate(test_articles):
            print(f"\n--- Relevance Test Article {i+1} ---")
            relevance_result = get_news_relevance_gemini(article["title"], article["snippet"], test_stock_ticker, test_stock_name)
            print(f"  Relevance Score: {relevance_result.get('relevance_score')}")
            print(f"  Relevance Justification: {relevance_result.get('relevance_justification')}")

            if relevance_result.get('relevance_score', 0) >= 4: # Only analyze sentiment if deemed relevant
                print(f"--- Sentiment Analysis for Relevant Article ---")
                sentiment_text = f"Title: {article['title']}. Description: {article['snippet']}"
                sentiment_result = analyze_sentiment_gemini(text_content=sentiment_text, stock_ticker=test_stock_ticker)
                print(f"  Sentiment: {sentiment_result.get('sentiment')}")
                print(f"  Sentiment Justification: {sentiment_result.get('justification')}")
            else:
                print(f"  Skipping sentiment analysis due to low relevance ({relevance_result.get('relevance_score')}).")

        print("\n--- Testing Sentiment Analyzer directly (as before) ---")
        sentiment_test_cases = [
            (f"{test_stock_name} reports record profits and strong future guidance.", test_stock_name),
            (f"Analyst downgrades {test_stock_ticker} due to increasing competition.", test_stock_ticker),
            ("This is a generic statement about bananas.", test_stock_ticker)
        ]
        for text, stock in sentiment_test_cases:
            print(f"\nAnalyzing sentiment for '{stock}': \"{text}\"")
            analysis = analyze_sentiment_gemini(text_content=text, stock_ticker=stock)
            print(f"  Sentiment: {analysis.get('sentiment')}")
            print(f"  Justification: {analysis.get('justification')}")