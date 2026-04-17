"""
This file contains functions for training different machine learning models
(Logistic Regression, SVM, Random Forest) and handling class imbalance using SMOTE.

"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from consumer_behavior_project.utils import wrapper_for_fit

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    wrapper_for_fit(model.fit, X_train, y_train, 'Logistic Regression')
    return model


def train_svm(X_train, y_train):
    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
        probability=True
    )
    wrapper_for_fit(model.fit, X_train, y_train, 'SVM')
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    wrapper_for_fit(model.fit, X_train, y_train, desc='Random Forest')
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
    wrapper_for_fit(model.fit, X_train_smote, y_train_smote, 'SMOTE Logistic Regression')
    return model, X_train_smote, y_train_smote
