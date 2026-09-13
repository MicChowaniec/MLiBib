from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_customer_dataset, clean_transactions, make_model


def evaluate(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
    }, probability, prediction


def run(input_path: Path, output_dir: Path, random_state: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    clean, _ = clean_transactions(pd.read_excel(input_path, sheet_name="Online Retail"))
    data = build_customer_dataset(clean)
    x = data[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    y = data["repeat_purchase"]
    x_train, x_test, y_train, y_test, _, id_test = train_test_split(x, y, data["CustomerID"], test_size=.2, stratify=y, random_state=random_state)
    preprocessing = make_model(random_state).named_steps["preprocessing"]
    candidates = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    rows = []
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    grids = {
        "logistic_regression": {"model__C": [0.01, .1, 1, 10], "model__solver": ["lbfgs", "liblinear"]},
        "random_forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 4, 10], "model__max_features": ["sqrt", .7]},
        "gradient_boosting": {"model__n_estimators": [100, 250], "model__learning_rate": [.03, .1], "model__max_depth": [1, 2, 3]},
        "knn": {"model__n_neighbors": [7, 15, 25, 41], "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
    }
    search = GridSearchCV(Pipeline([("preprocessing", preprocessing), ("model", candidates[best_name])]), grids[best_name], scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    metrics, probability, prediction = evaluate(search.best_estimator_, x_test, y_test)
    pd.DataFrame({"CustomerID": id_test.to_numpy(), "actual": y_test.to_numpy(), "predicted": prediction, "probability": probability}).to_csv(output_dir / "predictions.csv", index=False)
    importance = permutation_importance(search.best_estimator_, x_test, y_test, scoring="roc_auc", n_repeats=15, random_state=random_state, n_jobs=-1)
    pd.DataFrame({"feature": x.columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    result = {"selected_model": best_name, "best_parameters": search.best_params_, "best_cv_roc_auc": search.best_score_, "test_metrics": metrics}
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fpr, tpr, _ = roc_curve(y_test, probability)
    fig, ax = plt.subplots(figsize=(6.5, 4.8)); ax.plot(fpr, tpr, label=f"{best_name} (AUC={metrics['roc_auc']:.3f})"); ax.plot([0,1],[0,1],"--",color="grey"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Optimized classifier ROC curve"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.8)); ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues", colorbar=False); ax.set_title("Optimized classifier errors"); fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=ROOT/"data"/"Online Retail 2.xlsx",help="Path to Online Retail 2.xlsx"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); p.add_argument("--random-state",type=int,default=42); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir,a.random_state),indent=2))
