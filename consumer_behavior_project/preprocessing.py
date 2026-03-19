import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

##Loading
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    test_size: float = 0.2,
    random_state: int = 42,
):

    # Check duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().copy()

    # Define features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder