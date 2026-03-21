import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Basic validation
    if df.empty:
        raise ValueError("The input dataframe is empty.")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the dataframe."
        )

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().copy()
    else:
        df = df.copy()

    # Define features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if X.empty:
        raise ValueError("No feature columns remain after removing the target column.")

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split before feature encoding to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Identify categorical columns from training data
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    # One-hot encode train and test separately, then align test to train columns
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train_encoded.columns,
        index=X_train_encoded.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_train_encoded.columns,
        index=X_test_encoded.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder