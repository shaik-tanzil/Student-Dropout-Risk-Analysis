import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Train a Random Forest classifier
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Evaluate the model using accuracy, precision, recall metrics
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

# Select relevant feature columns
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        'Age at enrollment', 
        'Gender', 
        'Debtor', 
        'Tuition fees up to date', 
        'Curricular units 1st sem (approved)'
    ]
    X = df[features]
    return X

# Fill missing values (categorical: 'Unknown', numerical: median)
def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X.loc[:, categorical_cols] = X[categorical_cols].fillna('Unknown')
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    return X

# One-hot encode categorical columns
def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X

# Split data into train and test sets (stratify if classification)
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y  # maintains class balance between splits
    )
    return X_train, X_test, y_train, y_test

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the dataset
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)

    # Step 1: Select relevant features for the analysis
    features = select_features(df)

    # Step 2: Handle missing values
    features = handle_missing_values(features)

    # Step 3: One-hot encode categorical columns
    features = encode_categorical_features(features)

    # Step 4: Set the target variable 'Target'
    y = df['Target']

    # Step 5: Split data into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = split_data(features, y, test_size=0.3)

    # Step 6: Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate Random Forest model
    rf_accuracy, rf_precision, rf_recall = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Model Precision:", rf_precision)
    print("Random Forest Model Recall:", rf_recall)

#Output
#Random Forest Model Accuracy: 0.6807228915662651
#Random Forest Model Precision: 0.6495406357253748
#Random Forest Model Recall: 0.6807228915662651
