"""
agent.py

The AI brain of the pricing agent. Reads pre-calculated elasticity results
and uses Google Gemini to answer business questions in plain English.

Architecture decision: stats (elasticity.py) does the math, the LLM does
the communication. The LLM never sees raw data or runs calculations — it
only interprets results that have already been validated. This is the
correct production pattern: explainable, auditable, and cost-efficient
since we're not burning tokens on computation.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load API key from .env file — never hardcode secrets in source code
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Path to pre-calculated elasticity results
RESULTS_PATH = Path(__file__).resolve().parent / "data" / "elasticity_results.csv"


def load_elasticity_results() -> pd.DataFrame:
    """Load the elasticity results calculated by elasticity.py."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "elasticity_results.csv not found. Run elasticity.py first."
        )
    return pd.read_csv(RESULTS_PATH)


def build_context(df: pd.DataFrame) -> str:
    """
    Convert the elasticity dataframe into a plain text summary that
    Gemini can read and reason about. We keep this concise — the model
    doesn't need the full CSV, just the key numbers per product.
    """
    lines = ["PRODUCT PRICE ELASTICITY ANALYSIS RESULTS:", ""]
    for _, row in df.iterrows():
        sensitivity = "HIGH" if abs(row["estimated_elasticity"]) > 1.5 else \
                      "MEDIUM" if abs(row["estimated_elasticity"]) > 1.0 else "LOW"
        lines.append(
            f"- {row['product']} ({row['category']}): "
            f"elasticity={row['estimated_elasticity']}, "
            f"model_fit_R2={row['r_squared']}, "
            f"price_sensitivity={sensitivity}"
        )
    lines.append("")
    lines.append(
        "Note: More negative elasticity = more price sensitive. "
        "Electronics tend to be more price sensitive than Apparel in this dataset."
    )
    return "\n".join(lines)


def ask_agent(question: str, df: pd.DataFrame) -> str:
    """
    Takes a business question and the elasticity data, sends both to
    Gemini, and returns a plain English answer grounded in the numbers.
    """
    context = build_context(df)

    prompt = f"""
You are a pricing strategy analyst for an e-commerce business.
You have access to statistically estimated price elasticity data for 8 products.
Elasticity measures how sensitive sales volume is to price changes.
A more negative number means sales drop more sharply when price rises (or increase 
more sharply when price falls).

Here is the elasticity data:

{context}

Business question: {question}

Instructions:
- Answer directly and concisely in plain English, no jargon
- Always reference specific elasticity numbers to justify your recommendation
- If recommending a discount, estimate the expected sales volume increase
- If the question is outside the scope of this data, say so clearly
- Keep your answer under 150 words
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text


def main():
    """
    Simple command-line chat loop so we can test the agent before
    building the Streamlit UI in the next phase.
    """
    print("Loading elasticity data...")
    df = load_elasticity_results()
    print("Pricing Agent ready. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue

        print("\nAgent thinking...\n")
        answer = ask_agent(question, df)
        print(f"Agent: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
