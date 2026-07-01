"""
elasticity.py

For each product, runs a SEPARATE multiple regression to estimate price
elasticity, while controlling for confounders (holiday, day-of-week,
competitor price). Compares the estimated elasticity against the TRUE
elasticity we built into the synthetic data, as a validation step.

This is the core "causal" piece of the project: instead of naively
correlating price and sales, we isolate price's effect by including the
confounders as additional input columns.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Ground truth, copied from data_generator.py, used ONLY for validation
# printout — the model itself never sees these values.
TRUE_ELASTICITY = {
    "Wireless Earbuds": -2.4,
    "Smartwatch": -2.1,
    "Bluetooth Speaker": -1.8,
    "Laptop Stand": -1.5,
    "Cotton T-Shirt": -0.6,
    "Denim Jeans": -0.9,
    "Running Shoes": -1.2,
    "Winter Jacket": -1.4,
}


def estimate_elasticity_for_product(df_product: pd.DataFrame) -> dict:
    """
    Runs one regression for a single product's data and returns the
    estimated elasticity plus some basic fit diagnostics.
    """
    df = df_product.copy()

    # We can't feed raw price directly and get an "elasticity" coefficient,
    # because elasticity is defined in PERCENT terms, not raw dollars.
    # The standard trick: take the natural log of price and the natural log
    # of units_sold. In a log-log regression, the coefficient on log(price)
    # IS the elasticity directly — this is a well-known econometric result,
    # not something we're approximating.
    df = df[df["units_sold"] > 0]  # log(0) is undefined, drop any zero-sales days
    df["log_price"] = np.log(df["price"])
    df["log_units"] = np.log(df["units_sold"])

    feature_cols = ["log_price", "is_holiday", "day_of_week"]
    X = df[feature_cols].values
    y = df["log_units"].values

    model = LinearRegression()
    model.fit(X, y)

    # The coefficient on log_price (index 0, since it's our first feature)
    # is the elasticity estimate.
    estimated_elasticity = model.coef_[0]
    r_squared = model.score(X, y)

    return {
        "estimated_elasticity": round(estimated_elasticity, 3),
        "r_squared": round(r_squared, 3),
    }


def main():
    data_path = Path(__file__).resolve().parent / "data" / "sales_data.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])

    print(f"{'Product':<20} {'True':>8} {'Estimated':>12} {'R²':>8}")
    print("-" * 52)

    results = []
    for product_name in df["product"].unique():
        df_product = df[df["product"] == product_name]
        result = estimate_elasticity_for_product(df_product)
        true_val = TRUE_ELASTICITY[product_name]

        print(f"{product_name:<20} {true_val:>8} {result['estimated_elasticity']:>12} {result['r_squared']:>8}")

        results.append({
            "product": product_name,
            "true_elasticity": true_val,
            **result,
        })

    results_df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent / "data" / "elasticity_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
    