"""
data_generator.py

Generates a synthetic, but realistically-structured, e-commerce sales dataset
for 8 products across 2 categories (electronics, apparel).

Why synthetic data, and why built this way:
Real causal analysis requires knowing the TRUE effect of price on sales so we
can later verify our model recovered something close to the truth. Real-world
data never gives you that ground truth. Here, WE control the true elasticity
for each product, deliberately inject confounders (holidays, competitor
price, day-of-week), and then the regression model (built in Phase 6) has to
correctly separate price's effect from the noise we added — exactly the
skill this project is meant to demonstrate.

Output: data/sales_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

PRODUCTS = {
    "Wireless Earbuds":   {"category": "Electronics", "base_price": 49.99,  "base_demand": 120, "true_elasticity": -2.4},
    "Smartwatch":         {"category": "Electronics", "base_price": 129.99, "base_demand": 60,  "true_elasticity": -2.1},
    "Bluetooth Speaker":  {"category": "Electronics", "base_price": 39.99,  "base_demand": 90,  "true_elasticity": -1.8},
    "Laptop Stand":       {"category": "Electronics", "base_price": 24.99,  "base_demand": 150, "true_elasticity": -1.5},
    "Cotton T-Shirt":     {"category": "Apparel",     "base_price": 14.99,  "base_demand": 200, "true_elasticity": -0.6},
    "Denim Jeans":        {"category": "Apparel",     "base_price": 44.99,  "base_demand": 80,  "true_elasticity": -0.9},
    "Running Shoes":      {"category": "Apparel",     "base_price": 69.99,  "base_demand": 70,  "true_elasticity": -1.2},
    "Winter Jacket":      {"category": "Apparel",     "base_price": 89.99,  "base_demand": 40,  "true_elasticity": -1.4},
}

N_DAYS = 365
START_DATE = pd.Timestamp("2025-07-01")

HOLIDAYS = pd.to_datetime([
    "2025-11-28",
    "2025-12-25",
    "2026-01-01",
    "2026-02-14",
])


def generate_product_data(name: str, info: dict) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    base_price = info["base_price"]
    base_demand = info["base_demand"]
    elasticity = info["true_elasticity"]

    price = np.full(N_DAYS, base_price)
    day = 0
    while day < N_DAYS:
        promo_gap = np.random.randint(18, 26)
        day += promo_gap
        if day >= N_DAYS:
            break
        promo_length = np.random.randint(3, 8)
        discount_pct = np.random.uniform(0.05, 0.20)
        end = min(day + promo_length, N_DAYS)
        price[day:end] = base_price * (1 - discount_pct)
        day = end

    competitor_price = price * np.random.uniform(0.95, 1.08, size=N_DAYS)

    day_of_week = dates.dayofweek
    weekend_boost = np.where(day_of_week >= 5, 1.15, 1.0)

    is_holiday = dates.isin(HOLIDAYS).astype(int)
    holiday_boost = np.where(is_holiday == 1, 1.8, 1.0)

    pct_price_change = (price - base_price) / base_price
    pct_demand_change = elasticity * pct_price_change
    demand = base_demand * (1 + pct_demand_change)

    demand = demand * weekend_boost * holiday_boost

    noise = np.random.normal(loc=0, scale=base_demand * 0.08, size=N_DAYS)
    units_sold = np.maximum(0, np.round(demand + noise)).astype(int)

    return pd.DataFrame({
        "date": dates,
        "product": name,
        "category": info["category"],
        "price": np.round(price, 2),
        "competitor_price": np.round(competitor_price, 2),
        "is_holiday": is_holiday,
        "day_of_week": day_of_week,
        "units_sold": units_sold,
    })


def main():
    all_data = [generate_product_data(name, info) for name, info in PRODUCTS.items()]
    full_df = pd.concat(all_data, ignore_index=True)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sales_data.csv"
    full_df.to_csv(out_path, index=False)

    print(f"Generated {len(full_df)} rows across {len(PRODUCTS)} products.")
    print(f"Saved to: {out_path}")
    print("\nPreview:")
    print(full_df.head(10))


if __name__ == "__main__":
    main()
    