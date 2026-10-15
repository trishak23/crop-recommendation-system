"""Compare classifiers for crop recommendation, evaluate and save the best.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import DATA_PATH, FEATURES, FIGURES, MODEL_PATH, REPORTS, TARGET

SEED = 42


def candidates() -> dict[str, Pipeline]:
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    return {
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=12, random_state=SEED)),
        "k-Nearest Neighbours": pipe(KNeighborsClassifier(n_neighbors=7)),
        "Gaussian Naive Bayes": pipe(GaussianNB()),
        "Logistic Regression": pipe(LogisticRegression(max_iter=3000)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    }


def train(df: pd.DataFrame, save: bool = True) -> dict:
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc = {}
    for name, model in candidates().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_acc[name] = float(scores.mean())
        print(f"{name:<22} CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_acc, key=cv_acc.get)
    model = candidates()[best_name].fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "best_model": best_name,
        "cv_accuracy": cv_acc,
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_top3_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)), 4),
        "test_macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "n_classes": int(y.nunique()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, preds))
    print(f"Top-3 accuracy: {metrics['test_top3_accuracy']}")

    if save:
        REPORTS.mkdir(exist_ok=True)
        FIGURES.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Greens", colorbar=False, xticks_rotation=45)
        ax.set_title(f"Confusion matrix - {best_name}")
        fig.tight_layout()
        fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1)
        series = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh(series.index, series.values, color="#16a34a")
        ax.set(title="Which inputs matter most? (permutation importance)", xlabel="Drop in accuracy when shuffled")
        fig.tight_layout()
        fig.savefig(FIGURES / "feature_importance.png", dpi=150)
        plt.close(fig)
        print("\nFeature importance:\n" + series.sort_values(ascending=False).round(3).to_string())
        print(f"\nModel saved to {MODEL_PATH}")
    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("data/crop_data.csv not found. Run `python -m src.generate_data` first.")
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].str.title()
    train(df)


if __name__ == "__main__":
    main()
