from pathlib import Path

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
from consumer_behavior_project.evaluation import (
    plot_confusion_matrix,
    get_feature_importance_dataframe,
    plot_feature_importance,
    plot_model_comparison,
    plot_multiclass_roc,
    plot_multiclass_precision_recall,
)


def main():
    data_path = Path("data") / "online vs store shopping dataset.csv"
    df = load_data(data_path)

    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    # Logistic Regression
    log_model = train_logistic_regression(X_train, y_train)
    log_results = evaluate_model(log_model, X_test, y_test)
    print_results("Logistic Regression", log_results)
    plot_confusion_matrix(y_test, log_results["y_pred"], "Logistic Regression")

    # SVM
    svm_model = train_svm(X_train, y_train)
    svm_results = evaluate_model(svm_model, X_test, y_test)
    print_results("SVM", svm_results)
    plot_confusion_matrix(y_test, svm_results["y_pred"], "SVM")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print_results("Random Forest", rf_results)
    plot_confusion_matrix(y_test, rf_results["y_pred"], "Random Forest")

    feature_importance = get_feature_importance_dataframe(rf_model, X_train.columns)
    plot_feature_importance(feature_importance)

    # Logistic Regression with SMOTE
    log_smote_model, _, _ = train_logistic_regression_smote(X_train, y_train)
    smote_results = evaluate_model(log_smote_model, X_test, y_test)
    print_results("Logistic Regression (SMOTE)", smote_results)
    plot_confusion_matrix(y_test, smote_results["y_pred"], "Logistic Regression (SMOTE)")

    # Comparison table
    results = build_results_table(
        log_results["accuracy"],
        log_results["f1_weighted"],
        smote_results["accuracy"],
        smote_results["f1_weighted"],
        svm_results["accuracy"],
        svm_results["f1_weighted"],
        rf_results["accuracy"],
        rf_results["f1_weighted"],
        log_results["f1_macro"],
        smote_results["f1_macro"],
        svm_results["f1_macro"],
        rf_results["f1_macro"],
        log_results["balanced_accuracy"],
        smote_results["balanced_accuracy"],
        svm_results["balanced_accuracy"],
        rf_results["balanced_accuracy"],
    )

    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))

    # Comparison plots
    plot_model_comparison(results)

    # ROC curves
    plot_multiclass_roc(log_model, X_test, y_test, "Logistic Regression")
    plot_multiclass_roc(rf_model, X_test, y_test, "Random Forest")

    # Precision-Recall curves
    plot_multiclass_precision_recall(log_model, X_test, y_test, "Logistic Regression")
    plot_multiclass_precision_recall(rf_model, X_test, y_test, "Random Forest")


if __name__ == "__main__":
    main()