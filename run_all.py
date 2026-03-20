from pathlib import Path
import runpy


PROJECT_SCRIPTS = [
    "projects/01_house_price_regression/train.py",
    "projects/02_customer_churn_classification/train.py",
    "projects/03_demand_forecasting/train.py",
    "projects/04_fraud_detection/train.py",
    "projects/05_sales_prediction_app/train.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in PROJECT_SCRIPTS:
        script_path = root / script
        print(f"\nRunning {script_path.relative_to(root)}")
        runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
