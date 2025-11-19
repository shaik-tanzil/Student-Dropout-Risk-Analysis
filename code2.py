import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # Select relevant feature columns from DataFrame
    features = ['Age at enrollment', 'Gender', 'Debtor', 
                'Tuition fees up to date', 'Curricular units 1st sem (approved)']
    X = df[features]
    return X

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    # Fill missing categorical values with 'Unknown', numerical with median
    X = X.copy()
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X.loc[:, categorical_cols] = X[categorical_cols].fillna('Unknown')
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    return X

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    # Convert categorical columns to numeric using one-hot encoding
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    # Train a Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    # Predict and calculate accuracy and confusion matrix
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def apply_l2_regularization(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    # Train a Logistic Regression model with L2 regularization (Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    reg_model_l2 = LogisticRegression(penalty='l2', random_state=42, max_iter=2000)
    reg_model_l2.fit(X_train_scaled, y_train)
    return reg_model_l2, scaler, X_test_scaled

def apply_l1_regularization(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    # Train a Logistic Regression model with L1 regularization (Lasso)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    reg_model_l1 = LogisticRegression(
        penalty='l1', 
        solver='saga', 
        random_state=42, 
        max_iter=2000
    )
    reg_model_l1.fit(X_train_scaled, y_train)
    return reg_model_l1, scaler, X_test_scaled

if __name__ == "__main__":
    # Load the dataset
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)
    # Select and preprocess features
    X = select_features(df)
    X = handle_missing_values(X)
    X = encode_categorical_features(X)
    y = df['Target']
    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Train and evaluate Decision Tree
    model = train_decision_tree(X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)
    print("\nDecision Tree Model Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    # L2 regularization with Logistic Regression
    reg_model_l2, scaler_l2, X_test_scaled_l2 = apply_l2_regularization(X_train, y_train, X_test, y_test)
    accuracy_l2 = reg_model_l2.score(X_test_scaled_l2, y_test)
    print("\nL2 Regularized Model Accuracy (Ridge):", accuracy_l2)
    # L1 regularization with Logistic Regression
    reg_model_l1, scaler_l1, X_test_scaled_l1 = apply_l1_regularization(X_train, y_train, X_test, y_test)
    accuracy_l1 = reg_model_l1.score(X_test_scaled_l1, y_test)
    print("\nL1 Regularized Model Accuracy (Lasso):", accuracy_l1)


#Output:

#Decision Tree Model Accuracy: 0.6641566265060241
#Confusion Matrix:
#[[297  46  98]
# [ 81  41 123]
# [ 54  44 544]]


#L2 Regularized Model Accuracy (Ridge): 0.6634036144578314
#L1 Regularized Model Accuracy (Lasso): 0.6634036144578314

