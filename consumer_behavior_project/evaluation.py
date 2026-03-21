import matplotlib
matplotlib.use("QtAgg")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def save_plot(filename, folder="outputs/plots", show=True):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    plt.savefig(filepath, bbox_inches="tight")
    print(f"Saved plot: {filepath}")

    if show:
        plt.show()

    plt.close("all")  


def plot_confusion_matrix(y_test, y_pred, title, class_names=None):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    if class_names is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
        )

    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(f"confusion_matrix_{title.replace(' ', '_').lower()}.png")


def get_feature_importance_dataframe(model, feature_names):
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    return feature_importance.sort_values(by="Importance", ascending=False)


def plot_feature_importance(feature_importance, top_n=10):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(top_n),
        x="Importance",
        y="Feature"
    )
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("feature_importance_random_forest.png")


def plot_model_metric_bar(results, metric):
    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame.")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results, x="Model", y=metric, hue="Model", legend=False)
    plt.title(f"Model {metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    save_plot(f"{metric.lower().replace(' ', '_')}_comparison.png")


def plot_model_comparison(results):
    metrics_to_plot = [
        col for col in ["Accuracy", "Weighted F1", "Macro F1", "Balanced Accuracy"]
        if col in results.columns
    ]

    if not metrics_to_plot:
        raise ValueError("No valid metric columns found for model comparison plotting.")

    results.set_index("Model")[metrics_to_plot].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    plt.grid(axis="y")
    save_plot("model_comparison.png")


def plot_multiclass_roc(model, X_test, y_test, title, classes=None, class_names=None):
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"The model '{title}' does not support predict_proba().")

    if classes is None:
        classes = sorted(pd.Series(y_test).unique())

    y_test_bin = label_binarize(y_test, classes=list(classes))
    y_score = model.predict_proba(X_test)
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        if class_names is not None:
            label = f"{class_names[i]} (AUC = {roc_auc:.2f})"
        else:
            label = f"Class {classes[i]} (AUC = {roc_auc:.2f})"

        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_plot(f"roc_{title.replace(' ', '_').lower()}.png")


def plot_multiclass_precision_recall(model, X_test, y_test, title, classes=None, class_names=None):
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"The model '{title}' does not support predict_proba().")

    if classes is None:
        classes = sorted(pd.Series(y_test).unique())

    y_test_bin = label_binarize(y_test, classes=list(classes))
    y_score = model.predict_proba(X_test)
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])

        if class_names is not None:
            label = f"{class_names[i]} (AP = {ap:.2f})"
        else:
            label = f"Class {classes[i]} (AP = {ap:.2f})"

        plt.plot(recall, precision, label=label)

    plt.title(f"Precision-Recall Curve - {title}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save_plot(f"pr_{title.replace(' ', '_').lower()}.png")