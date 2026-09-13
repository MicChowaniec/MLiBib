from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OBSERVATION_END = pd.Timestamp("2011-05-31 23:59:59")
PREDICTION_END = pd.Timestamp("2011-08-29 23:59:59")
NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "items_purchased",
    "monetary_value",
    "mean_order_value",
    "unique_products",
    "average_unit_price",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["Country"]


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    counts: dict[str, int] = {"raw_rows": int(len(raw))}
    data = raw.drop_duplicates().copy()
    counts["duplicate_rows_removed"] = int(len(raw) - len(data))

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["CustomerID"] = pd.to_numeric(data["CustomerID"], errors="coerce")

    counts["rows_missing_customer_id"] = int(data["CustomerID"].isna().sum())
    invalid_date = data["InvoiceDate"].isna()
    cancelled = data["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    invalid_quantity = data["Quantity"].isna() | data["Quantity"].le(0)
    invalid_price = data["UnitPrice"].isna() | data["UnitPrice"].le(0)
    counts["rows_invalid_date"] = int(invalid_date.sum())
    counts["rows_cancelled"] = int(cancelled.sum())
    counts["rows_nonpositive_or_missing_quantity"] = int(invalid_quantity.sum())
    counts["rows_nonpositive_or_missing_price"] = int(invalid_price.sum())

    valid = ~(data["CustomerID"].isna() | invalid_date | cancelled | invalid_quantity | invalid_price)
    data = data.loc[valid].copy()
    data["CustomerID"] = data["CustomerID"].astype("int64")
    data["Country"] = data["Country"].fillna("Unknown").astype(str).str.strip()
    data["line_value"] = data["Quantity"] * data["UnitPrice"]
    counts["clean_rows"] = int(len(data))
    return data, counts


def build_customer_dataset(clean: pd.DataFrame) -> pd.DataFrame:
    observation = clean.loc[clean["InvoiceDate"] <= OBSERVATION_END].copy()
    future = clean.loc[
        clean["InvoiceDate"].gt(OBSERVATION_END)
        & clean["InvoiceDate"].le(PREDICTION_END)
    ]
    if observation.empty:
        raise ValueError("The observation window contains no valid transactions.")

    invoice_totals = (
        observation.groupby(["CustomerID", "InvoiceNo"], as_index=False)["line_value"].sum()
    )
    order_stats = invoice_totals.groupby("CustomerID").agg(
        frequency=("InvoiceNo", "nunique"),
        mean_order_value=("line_value", "mean"),
    )
    features = observation.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        items_purchased=("Quantity", "sum"),
        monetary_value=("line_value", "sum"),
        unique_products=("StockCode", "nunique"),
        average_unit_price=("UnitPrice", "mean"),
        Country=("Country", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    )
    features = features.join(order_stats)
    features["recency_days"] = (OBSERVATION_END - features["last_purchase"]).dt.days
    features["tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    repeat_customers = set(future["CustomerID"].unique())
    features["repeat_purchase"] = features.index.to_series().isin(repeat_customers).astype(int)
    return features.reset_index()[["CustomerID", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "repeat_purchase"]]


class IQRClipper:
    """Scikit-learn-compatible transformer that caps columns at fitted IQR limits."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, x, y=None):
        frame = pd.DataFrame(x)
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_ = (q3 + self.factor * iqr).to_numpy()
        return self

    def transform(self, x):
        return np.clip(np.asarray(x, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_model(random_state: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("outliers", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=random_state
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def save_plots(y_test, probabilities, predictions, target, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["No", "Yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Repeat-purchase prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(fpr, tpr, label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["No repeat purchase", "Repeat purchase"], counts.values, color=["#7a8da6", "#2878b5"])
    ax.set(ylabel="Customers", title="Target distribution")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)


def run(input_path: Path, output_dir: Path, random_state: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(input_path, sheet_name="Online Retail")
    clean, quality = clean_transactions(raw)
    dataset = build_customer_dataset(clean)
    x = dataset[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    y = dataset["repeat_purchase"]
    if y.nunique() != 2:
        raise ValueError("Both target classes are required for classification.")

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x,
        y,
        dataset["CustomerID"],
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    model = make_model(random_state)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_test, predictions)
    metrics = {
        "problem": "customer repeat purchase within 90 days",
        "observation_end": str(OBSERVATION_END.date()),
        "prediction_end": str(PREDICTION_END.date()),
        "random_state": random_state,
        "customers": int(len(dataset)),
        "train_customers": int(len(x_train)),
        "test_customers": int(len(x_test)),
        "positive_customers": int(y.sum()),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "data_quality": quality,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "CustomerID": id_test.to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "probability_repeat_purchase": probabilities,
        }
    ).sort_values("probability_repeat_purchase", ascending=False).to_csv(
        output_dir / "predictions.csv", index=False
    )

    names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    pd.DataFrame({"feature": names, "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients)}).sort_values(
        "absolute_coefficient", ascending=False
    ).to_csv(output_dir / "feature_coefficients.csv", index=False)
    dataset.to_csv(output_dir / "customer_features.csv", index=False)
    save_plots(y_test, probabilities, predictions, y, output_dir)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a customer repeat-purchase classifier.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "Online Retail 2.xlsx",
        help="Path to the Online Retail workbook.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.input, args.output_dir, args.random_state)
    print(json.dumps(result, indent=2))
