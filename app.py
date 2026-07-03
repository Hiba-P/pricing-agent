"""
app.py

Streamlit UI for the Causal Pricing & Inventory Decision Agent.
Three sections:
1. Elasticity Dashboard - visual chart of all 8 products
2. Price Simulator - real-time sales uplift calculator using elasticity formula
3. AI Chat - ask business questions (requires Gemini API quota)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pricing Intelligence Agent",
    page_icon="📊",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "data" / "sales_data.csv"
ELASTICITY_PATH = Path(__file__).resolve().parent / "data" / "elasticity_results.csv"


@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elasticity = pd.read_csv(ELASTICITY_PATH)
    return sales, elasticity


sales_df, elasticity_df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Causal Pricing Intelligence Agent")
st.markdown(
    "Estimates **price elasticity of demand** using causal regression — "
    "controlling for holidays, competitor pricing, and day-of-week effects — "
    "to give business-actionable pricing recommendations."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Elasticity Dashboard
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Price Elasticity by Product")
st.markdown(
    "More negative = more price sensitive. "
    "Discount high-sensitivity products for maximum volume uplift."
)

col1, col2 = st.columns([2, 1])

with col1:
    import plotly.express as px
    chart_df = elasticity_df.sort_values("estimated_elasticity")
    fig = px.bar(
        chart_df,
        x="estimated_elasticity",
        y="product",
        orientation="h",
        color="category",
        color_discrete_map={"Electronics": "#636EFA", "Apparel": "#EF553B"},
        labels={"estimated_elasticity": "Price Elasticity", "product": ""},
        height=350,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend_title="Category",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Elasticity Summary Table**")
    display_df = elasticity_df[
        ["product", "category", "estimated_elasticity", "r_squared"]
    ].sort_values("estimated_elasticity")
    display_df.columns = ["Product", "Category", "Elasticity", "R²"]
    st.dataframe(display_df, hide_index=True, height=350)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Price Simulator
# ══════════════════════════════════════════════════════════════════════════════
st.header("🎯 Price Discount Simulator")
st.markdown(
    "Select a product and discount percentage to instantly calculate "
    "the expected sales volume change based on estimated elasticity."
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    selected_product = st.selectbox(
        "Select Product",
        options=elasticity_df["product"].tolist(),
    )
    discount_pct = st.slider(
        "Discount Percentage (%)",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

# Get elasticity for selected product
product_row = elasticity_df[elasticity_df["product"] == selected_product].iloc[0]
elasticity_val = product_row["estimated_elasticity"]
base_price = sales_df[sales_df["product"] == selected_product]["price"].max()
base_units = sales_df[sales_df["product"] == selected_product]["units_sold"].mean()

# Apply elasticity formula
pct_price_change = -discount_pct / 100
pct_units_change = elasticity_val * pct_price_change
new_units = base_units * (1 + pct_units_change)
revenue_before = base_price * base_units
new_price = base_price * (1 - discount_pct / 100)
revenue_after = new_price * new_units

with sim_col2:
    st.markdown(f"**Results for {selected_product}**")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            "Expected Sales Uplift",
            f"+{pct_units_change * 100:.1f}%",
            delta=f"+{new_units - base_units:.0f} units/day",
        )
        st.metric(
            "New Daily Units",
            f"{new_units:.0f}",
            delta=f"from {base_units:.0f} baseline",
        )
    with metric_col2:
        st.metric(
            "Revenue Impact",
            f"${revenue_after:,.0f}",
            delta=f"{((revenue_after - revenue_before) / revenue_before * 100):.1f}% vs baseline",
            delta_color="normal",
        )
        st.metric(
            "Price Elasticity",
            f"{elasticity_val:.3f}",
            delta="HIGH sensitivity" if abs(elasticity_val) > 1.5 else "LOW sensitivity",
            delta_color="off",
        )

    if pct_units_change > 0.15:
        st.success(
            f"✅ **Recommend discounting.** A {discount_pct}% price cut is "
            f"expected to increase daily units by {pct_units_change*100:.1f}%, "
            f"which likely offsets the margin loss."
        )
    elif revenue_after >= revenue_before * 0.95:
        st.warning(
            f"⚠️ **Marginal benefit.** Volume uplift exists but revenue "
            f"impact is minimal. Consider a smaller discount."
        )
    else:
        st.error(
            f"❌ **Do not discount.** Low elasticity means volume gain "
            f"won't compensate for margin loss on {selected_product}."
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: AI Chat (Gemini)
# ══════════════════════════════════════════════════════════════════════════════
st.header("🤖 Ask the Pricing Agent")
st.markdown(
    "Ask any business question about pricing strategy. "
    "The agent uses your elasticity data to give grounded recommendations."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. Which products should I discount before the holiday season?",
)

if st.button("Ask Agent", type="primary"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        try:
            from agent import ask_agent
            with st.spinner("Agent thinking..."):
                answer = ask_agent(question, elasticity_df)
            st.markdown("**Agent Response:**")
            st.info(answer)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                st.error(
                    "⚠️ Gemini API quota exhausted for today. "
                    "The Price Simulator above works without API calls — "
                    "use it for instant elasticity-based recommendations. "
                    "AI chat will resume when quota resets."
                )
            else:
                st.error(f"Agent error: {error_str}")

st.divider()
st.caption(
    "Built with Python · Statsmodels · Scikit-learn · Google Gemini · Streamlit | "
    "Causal inference approach: log-log regression controlling for holidays, "
    "competitor pricing, and day-of-week effects."
)