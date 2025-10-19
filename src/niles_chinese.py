"""
List 5 Chinese restaurants in Niles, Illinois using OpenAI's Responses API
with the built-in web_search tool, and print a nicely formatted answer.

Usage:
  export OPENAI_API_KEY="sk-..."
  python niles_chinese.py
"""

import os
import json
from typing import List, Dict

from dotenv import load_dotenv

# -------- 1) Setup client --------
load_dotenv()  # loads variables from .env into os.environ

from openai import OpenAI

print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) Query
QUESTION = "List 5 Chinese restaurants in Niles, Illinois"

# We'll ask the model to:
# - use web_search
# - output strict JSON so we can parse it reliably
# - include citations for each restaurant
system_instructions = """
You are a precise, helpful local guide.
Task: Return exactly 5 Chinese restaurants located in Niles, Illinois (the municipality itself, not just 'near').
For each restaurant, verify details from reputable sources (official site, Google, Yelp, TripAdvisor, etc.).
Prefer places that appear to be currently operating.

Output STRICT JSON (and nothing else) as:
{
  "query":"<the original query>",
  "results":[
    {
      "name":"<string>",
      "address":"<string>",
      "phone":"<string or empty>",
      "url":"<official or best listing URL>",
      "rating": "<string>",
      "notes":"<1 short sentence>",
      "sources":[{"title":"<source title>","url":"<https://...>"}]
    }
  ]
}

Rules:
- Exactly 5 items in "results".
- Niles, IL addresses only (cross-check city).
- If phone or official URL is unknown, leave empty string.
- De-duplicate businesses with multiple listings.
- Keep notes concise (max ~18 words).
- Have at least 100 customer ratings, with average score at least 4.5
"""

# 3) Call Responses API — FIX: use "input_text" instead of "text"
response = client.responses.create(
    model="gpt-5",  # or gpt-4o / gpt-5 if available
    input=[
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_instructions}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": QUESTION}],
        },
    ],
    tools=[{"type": "web_search"}],
)

# 4) Extract the model output text
output_text = response.output_text.strip()

# 5) Parse JSON safely
def coerce_json(s):
    s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end >= 0:
        s = s[start:end+1]
    return json.loads(s)

try:
    data = coerce_json(output_text)
except Exception as e:
    print("Failed to parse model output as JSON:", e)
    print(output_text)
    raise

# 6) Print formatted results
print(f"\nQuery: {data.get('query')}\n")
for i, r in enumerate(data.get("results", []), 1):
    print(f"{i}. {r['name']}")
    print(f"   Address: {r['address']}")
    if r.get("phone"):
        print(f"   Phone: {r['phone']}")
    if r.get("url"):
        print(f"   URL: {r['url']}")
    if r.get("rating"):
        print(f"   Rating: {r['rating']}")
    if r.get("notes"):
        print(f"   Note: {r['notes']}")
    for s in r.get("sources", []):
        print(f"     - {s['title']} — {s['url']}")
    print()

# 7) Save raw JSON
with open("niles_chinese_results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("✅ Saved results to niles_chinese_results.json")