"""Prompt templates for the AlgoSignals LLM layer."""

NEWS_SENTIMENT_PROMPT = """\
You are a financial analyst assistant. Given a list of recent news headlines for the stock symbol {symbol}, \
assess the overall sentiment.

Headlines:
{headlines}

Respond with a JSON object with exactly two fields:
- "score": a float between -1.0 (very bearish) and +1.0 (very bullish)
- "summary": a single sentence (max 60 words) summarising the key themes

Respond ONLY with the JSON object. No extra text.
"""

RATIONALE_PROMPT = """\
You are a financial analyst assistant. Summarise the following signal data for {symbol} into a \
concise, plain-English investment rationale (2-4 sentences, max 120 words).

Factor scores (each -1.0 to +1.0):
- Technical:   {technical:.2f}  ({technical_note})
- News:        {news:.2f}       ({news_note})
- Events:      {events:.2f}     ({events_note})
- Financials:  {financials:.2f} ({financials_note})
- Earnings:    {earnings:.2f}   ({earnings_note})

Composite score: {composite:.2f}
Recommendation: {action}

Write a rationale that explains WHY this recommendation was produced. \
Do not give specific price targets or guarantee outcomes. \
End with: "This is not investment advice."
"""

FALLBACK_RATIONALE_TEMPLATE = (
    "{symbol} has a composite score of {composite:.2f} ({action}). "
    "Technical: {technical:.2f}. News: {news:.2f}. Events: {events:.2f}. "
    "Financials: {financials:.2f}. Earnings: {earnings:.2f}. "
    "This is not investment advice."
)
