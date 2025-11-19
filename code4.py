import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Train a Random Forest classifier
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Train a Gradient Boosting classifier
def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model

# Apply K-fold cross-validation and report mean accuracy
def apply_k_fold_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5) -> float:
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    return cv_scores.mean()

# Evaluate model on test set
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, cm

# Select relevant feature columns
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = ['Age at enrollment', 'Gender', 'Debtor', 'Tuition fees up to date', 'Curricular units 1st sem (approved)']
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

# Split data into train and test sets
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the dataset
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)
    
    # Step 1: Select relevant features for the analysis
    features = select_features(df)
    
    # Step 2: Handle missing values
    features = features.copy()
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    features.loc[:, categorical_cols] = features[categorical_cols].fillna('Unknown')
    
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    features.loc[:, numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].median())
    
    # Step 3: One-hot encode categorical columns
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    
    # Step 4: Set the target variable 'Target'
    y = df['Target']
    
    # Step 5: Split data into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    
    # Step 6: Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Apply K-fold cross-validation for Random Forest
    rf_cv_score = apply_k_fold_cross_validation(rf_model, X_train, y_train)
    print("Random Forest Cross-validation Accuracy:", rf_cv_score)
    
    # Evaluate Random Forest model
    rf_accuracy, rf_precision, rf_recall, rf_cm = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Model Precision:", rf_precision)
    print("Random Forest Model Recall:", rf_recall)
    print("Random Forest Confusion Matrix:\n", rf_cm)
    
    # Step 7: Train and evaluate Gradient Boosting model
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Apply K-fold cross-validation for Gradient Boosting
    gb_cv_score = apply_k_fold_cross_validation(gb_model, X_train, y_train)
    print("Gradient Boosting Cross-Validation Accuracy:", gb_cv_score)
    
    # Evaluate Gradient Boosting model
    gb_accuracy, gb_precision, gb_recall, gb_cm = evaluate_model(gb_model, X_test, y_test)
    print("\nGradient Boosting Model Accuracy:", gb_accuracy)
    print("Gradient Boosting Model Precision:", gb_precision)
    print("Gradient Boosting Model Recall:", gb_recall)
    print("Gradient Boosting Confusion Matrix:\n", gb_cm)