import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, zero_division=zero_division, output_dict=False)

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report
    }


def build_results_table(log_acc, log_f1, smote_acc, smote_f1, svm_acc, svm_f1, rf_acc, rf_f1):
    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Logistic Regression (SMOTE)",
            "SVM",
            "Random Forest"
        ],
        "Accuracy": [log_acc, smote_acc, svm_acc, rf_acc],
        "F1 Score": [log_f1, smote_f1, svm_f1, rf_f1]
    })

    return results.sort_values(by="F1 Score", ascending=False)

def print_results(name, results):
    print(f"\n=== {name} ===")
    print("Accuracy:", results["accuracy"])
    print("F1 Score:", results["f1_score"])
    print("\nClassification Report:\n")
    print(results["report"])    

    