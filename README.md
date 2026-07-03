# Causal Pricing Intelligence Agent

A production-grade AI agent that estimates **price elasticity of demand** 
using causal regression — isolating price effects from confounders like 
holidays, competitor pricing, and day-of-week patterns.

## Live Demo
[pricing-agent-dh7btdr4kba2hafptcmdru.streamlit.app](https://pricing-agent-dh7btdr4kba2hafptcmdru.streamlit.app)

## What Makes This Different
Most pricing models correlate price with sales. This agent uses **log-log 
regression controlling for confounders** to estimate the true causal effect 
of price changes — the same methodology used in academic econometrics and 
enterprise revenue management systems.

## Features
- **Elasticity Dashboard** — ranks 8 products by price sensitivity
- **Price Simulator** — real-time sales uplift calculator using elasticity formula
- **AI Chat** — Gemini-powered business recommendations grounded in statistical estimates

## Tech Stack
Python · Pandas · Scikit-learn · Statsmodels · Google Gemini · Streamlit · Plotly

## Methodology
1. Synthetic data generated with known ground-truth elasticity values
2. Log-log OLS regression per product controlling for holidays, 
   competitor price, and day-of-week
3. Estimated elasticity validated against ground truth
4. LLM layer interprets results — stats does the math, AI does the communication

## Run Locally
```bash
git clone https://github.com/Hiba-P/pricing-agent
cd pricing-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```