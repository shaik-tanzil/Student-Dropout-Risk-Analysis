import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def explore_data(df: pd.DataFrame):
    print("Dataset Shape:", df.shape)
    print("Column Names:")
    print(df.dtypes)
    print("Missing Values:")
    print(df.isnull().sum())

    # Corrected column names based on your dataset
    print("\nGender Distribution:")
    print(df['Gender'].value_counts())

    print("\nCourse Distribution:")
    print(df['Course'].value_counts())

    print("\nTuition fees up to Date Distribution:")
    print(df['Tuition fees up to date'].value_counts())

    print("\nDebtor Distribution:")
    print(df['Debtor'].value_counts())

def create_approval_rate(df: pd.DataFrame):
    """
    Create the 'approval_rate' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (enrolled)' columns.
    
    Args:
        df (pd.DataFrame): The DataFrame with student data.
    
    Returns:
        pd.DataFrame: DataFrame with the 'approval_rate' column.
    """
    df['approval_rate'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
    return df

def create_performance_score(df: pd.DataFrame):
    """
    Create the 'performance_score' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (evaluations)' columns.
    
    Args:
        df (pd.DataFrame): The DataFrame with student data.
    
    Returns:
        pd.DataFrame: DataFrame with the 'performance_score' column.
    """
    df['performance_score'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (evaluations)']
    return df

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """    
    Create all engineered features needed for the analysis.
    
    Args:
        df (pd.DataFrame): The DataFrame with student data.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df = create_approval_rate(df)
    df = create_performance_score(df)
    print(df[['Course', 'Gender', 'approval_rate', 'performance_score']].head())
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    file_path = 'dataset.csv'
    
    df = load_data(file_path)
    
    explore_data(df)
    
    # Create engineered features for the model
    df = create_engineered_features(df)