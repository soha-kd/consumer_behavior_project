import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.svm import SVC


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
        probability=True
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def train_logistic_regression_smote(X_train, y_train):
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_smote, y_train_smote)
    return model, X_train_smote, y_train_smote


def evaluate_model(model, X_test, y_test, zero_division=0):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=zero_division)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=zero_division)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        zero_division=zero_division,
        output_dict=True
    )

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=zero_division,
        output_dict=False
    )

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "balanced_accuracy": balanced_acc,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text
    }


def build_results_table(
    log_acc,
    log_f1_weighted,
    smote_acc,
    smote_f1_weighted,
    svm_acc,
    svm_f1_weighted,
    rf_acc,
    rf_f1_weighted,
    log_f1_macro=None,
    smote_f1_macro=None,
    svm_f1_macro=None,
    rf_f1_macro=None,
    log_bal_acc=None,
    smote_bal_acc=None,
    svm_bal_acc=None,
    rf_bal_acc=None,
):
    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Logistic Regression (SMOTE)",
            "SVM",
            "Random Forest"
        ],
        "Accuracy": [log_acc, smote_acc, svm_acc, rf_acc],
        "Weighted F1": [log_f1_weighted, smote_f1_weighted, svm_f1_weighted, rf_f1_weighted]
    })

    if all(value is not None for value in [log_f1_macro, smote_f1_macro, svm_f1_macro, rf_f1_macro]):
        results["Macro F1"] = [log_f1_macro, smote_f1_macro, svm_f1_macro, rf_f1_macro]

    if all(value is not None for value in [log_bal_acc, smote_bal_acc, svm_bal_acc, rf_bal_acc]):
        results["Balanced Accuracy"] = [log_bal_acc, smote_bal_acc, svm_bal_acc, rf_bal_acc]

    sort_column = "Macro F1" if "Macro F1" in results.columns else "Weighted F1"
    return results.sort_values(by=sort_column, ascending=False)


def print_results(name, results):
    print(f"\n=== {name} ===")
    print("Accuracy:", results["accuracy"])
    print("Weighted F1 Score:", results["f1_weighted"])
    print("Macro F1 Score:", results["f1_macro"])
    print("Balanced Accuracy:", results["balanced_accuracy"])
    print("\nClassification Report:\n")
    print(results["classification_report_text"])