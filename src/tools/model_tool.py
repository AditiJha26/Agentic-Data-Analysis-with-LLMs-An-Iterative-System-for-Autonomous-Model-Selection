from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def apply_preprocessing(df, target_column, preprocessing_steps):
    """
    Apply preprocessing steps suggested by the LLM.
    """
    df = df.copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])

    # Drop useless columns (basic rule)
    drop_cols = []
    for col in df.columns:
        if col.lower() in ["name", "ticket"]:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Handle missing values
    if any("missing" in step.lower() for step in preprocessing_steps):
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        categorical_cols = df.select_dtypes(include=["object"]).columns
        df[categorical_cols] = df[categorical_cols].fillna("missing")

    # Drop high-missing columns (like Cabin)
    for col in df.columns:
        if df[col].isnull().mean() > 0.5:
            df = df.drop(columns=[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    if any("encode" in step.lower() for step in preprocessing_steps):
        categorical_cols = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y


def train_models(df, target_column, preprocessing_steps):
    """
    Train models using agent-driven preprocessing.
    """
    X, y = apply_preprocessing(df, target_column, preprocessing_steps)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    results["LogisticRegression"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "f1_score": float(f1_score(y_test, y_pred_lr))
    }

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results["RandomForest"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "f1_score": float(f1_score(y_test, y_pred_rf))
    }

    return results