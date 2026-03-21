from consumer_behavior_project.preprocessing import load_data, preprocess_data
from consumer_behavior_project.model import (
    train_logistic_regression,
    train_svm,
    train_random_forest,
    train_logistic_regression_smote,
    evaluate_model,
    print_results,
    build_results_table,
)


def main():
    df = load_data("data/online vs store shopping dataset.csv")
    X_train, X_test, y_train, y_test, le = preprocess_data(df)

    # Logistic Regression
    log_model = train_logistic_regression(X_train, y_train)
    log_results = evaluate_model(log_model, X_test, y_test)
    print_results("Logistic Regression", log_results)

    # SVM
    svm_model = train_svm(X_train, y_train)
    svm_results = evaluate_model(svm_model, X_test, y_test)
    print_results("SVM", svm_results)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print_results("Random Forest", rf_results)

    # Logistic Regression with SMOTE
    log_smote_model, X_train_smote, y_train_smote = train_logistic_regression_smote(X_train, y_train)
    smote_results = evaluate_model(log_smote_model, X_test, y_test)
    print_results("Logistic Regression (SMOTE)", smote_results)

    # Comparison table
    results = build_results_table(
        log_results["accuracy"], log_results["f1_score"],
        smote_results["accuracy"], smote_results["f1_score"],
        svm_results["accuracy"], svm_results["f1_score"],
        rf_results["accuracy"], rf_results["f1_score"],
    )

    print("\n=== Model Comparison ===")
    print(results)


if __name__ == "__main__":
    main()