from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def make_housing_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 1200
    area = rng.integers(650, 4200, rows)
    bedrooms = rng.integers(1, 6, rows)
    bathrooms = rng.integers(1, 5, rows)
    floors = rng.integers(1, 4, rows)
    house_age = rng.integers(0, 40, rows)
    distance_to_city = rng.uniform(1, 35, rows)
    school_rating = rng.uniform(2.0, 10.0, rows)
    crime_index = rng.uniform(1.0, 10.0, rows)
    garage_spaces = rng.integers(0, 4, rows)
    neighborhood = rng.choice(
        ["Urban", "Suburban", "Premium", "Family"],
        rows,
        p=[0.25, 0.35, 0.15, 0.25],
    )
    condition = rng.choice(["Needs Work", "Standard", "Renovated"], rows, p=[0.15, 0.6, 0.25])

    price = (
        area * 210
        + bedrooms * 14000
        + bathrooms * 18000
        + floors * 9000
        - house_age * 1700
        - distance_to_city * 3200
        + school_rating * 12500
        - crime_index * 9000
        + garage_spaces * 8500
        + np.select(
            [neighborhood == "Premium", neighborhood == "Suburban", neighborhood == "Family"],
            [85000, 28000, 18000],
            default=0,
        )
        + np.select(
            [condition == "Renovated", condition == "Standard"],
            [45000, 12000],
            default=-18000,
        )
        + rng.normal(0, 25000, rows)
    )

    return pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "house_age": house_age,
            "distance_to_city_km": distance_to_city.round(2),
            "school_rating": school_rating.round(2),
            "crime_index": crime_index.round(2),
            "garage_spaces": garage_spaces,
            "neighborhood": neighborhood,
            "condition": condition,
            "price": price.round(0),
        }
    )


def make_churn_data() -> pd.DataFrame:
    rng = np.random.default_rng(43)
    rows = 1500
    tenure = rng.integers(1, 72, rows)
    monthly_charges = rng.uniform(20, 130, rows)
    support_calls = rng.integers(0, 8, rows)
    internet_service = rng.choice(["DSL", "Fiber", "None"], rows, p=[0.35, 0.5, 0.15])
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.2])
    payment_method = rng.choice(
        ["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
        rows,
        p=[0.38, 0.22, 0.25, 0.15],
    )
    tech_support = rng.choice(["Yes", "No"], rows, p=[0.42, 0.58])
    paperless_billing = rng.choice(["Yes", "No"], rows, p=[0.68, 0.32])
    senior_citizen = rng.choice([0, 1], rows, p=[0.82, 0.18])

    churn_score = (
        1.4
        - 0.025 * tenure
        + 0.022 * monthly_charges
        + 0.38 * support_calls
        + 0.65 * (contract_type == "Month-to-month")
        - 0.5 * (contract_type == "Two year")
        + 0.55 * (internet_service == "Fiber")
        + 0.4 * (payment_method == "Electronic check")
        - 0.35 * (tech_support == "Yes")
        + 0.3 * (paperless_billing == "Yes")
        + 0.25 * senior_citizen
        + rng.normal(0, 0.7, rows)
    )
    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = (rng.random(rows) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges.round(2),
            "support_calls": support_calls,
            "internet_service": internet_service,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "churn": churn,
        }
    )


def make_demand_data() -> pd.DataFrame:
    rng = np.random.default_rng(44)
    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    stores = ["Store_A", "Store_B", "Store_C"]
    categories = ["Beverages", "Snacks", "Household"]
    records = []

    for store_idx, store in enumerate(stores):
        for category_idx, category in enumerate(categories):
            base = 110 + 15 * store_idx + 12 * category_idx
            for day_idx, current_date in enumerate(dates):
                is_weekend = int(current_date.dayofweek >= 5)
                month = current_date.month
                promo = int(rng.random() < 0.18)
                holiday = int(month in [11, 12] and current_date.day in range(20, 32))
                price = 12 + category_idx * 2 + rng.normal(0, 0.7)
                seasonality = 14 * np.sin(2 * np.pi * day_idx / 7) + 9 * np.sin(2 * np.pi * day_idx / 30)
                trend = day_idx * 0.04
                demand = (
                    base
                    + seasonality
                    + trend
                    + promo * 22
                    + holiday * 35
                    - price * 4
                    + is_weekend * 18
                    + rng.normal(0, 8)
                )
                records.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "store_id": store,
                        "product_category": category,
                        "price": round(price, 2),
                        "promotion_active": promo,
                        "holiday_flag": holiday,
                        "demand": max(0, round(demand)),
                    }
                )

    return pd.DataFrame(records)


def make_fraud_data() -> pd.DataFrame:
    x, y = make_classification(
        n_samples=3000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        class_sep=1.2,
        random_state=45,
    )
    dataframe = pd.DataFrame(x, columns=[f"feature_{idx}" for idx in range(10)])
    dataframe["transaction_amount"] = np.exp(dataframe["feature_0"] + 4).round(2)
    dataframe["transaction_hour"] = np.random.default_rng(46).integers(0, 24, len(dataframe))
    dataframe["is_international"] = np.random.default_rng(47).choice([0, 1], len(dataframe), p=[0.85, 0.15])
    dataframe["fraud"] = y
    return dataframe


def make_sales_data() -> pd.DataFrame:
    rng = np.random.default_rng(48)
    rows = 1000
    advertising_spend = rng.uniform(500, 8000, rows)
    store_visits = rng.integers(80, 1200, rows)
    discount_pct = rng.uniform(0, 35, rows)
    holiday_flag = rng.choice([0, 1], rows, p=[0.85, 0.15])
    competitor_price_index = rng.uniform(0.8, 1.2, rows)
    season = rng.choice(["Winter", "Spring", "Summer", "Autumn"], rows)

    sales = (
        advertising_spend * 1.8
        + store_visits * 18
        + discount_pct * 250
        + holiday_flag * 4200
        - competitor_price_index * 3000
        + np.select(
            [season == "Summer", season == "Winter", season == "Autumn"],
            [2500, 1200, 800],
            default=0,
        )
        + rng.normal(0, 3500, rows)
    )

    return pd.DataFrame(
        {
            "advertising_spend": advertising_spend.round(2),
            "store_visits": store_visits,
            "discount_pct": discount_pct.round(2),
            "holiday_flag": holiday_flag,
            "competitor_price_index": competitor_price_index.round(3),
            "season": season,
            "sales": sales.round(2),
        }
    )


def main() -> None:
    datasets = {
        "housing/housing_prices.csv": make_housing_data(),
        "churn/customer_churn.csv": make_churn_data(),
        "demand/daily_demand.csv": make_demand_data(),
        "fraud/fraud_transactions.csv": make_fraud_data(),
        "sales/sales_data.csv": make_sales_data(),
    }

    for relative_path, dataframe in datasets.items():
        destination = DATA_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(destination, index=False)
        print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
