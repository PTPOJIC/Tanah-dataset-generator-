import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Function to load the dataset
def load_dataset(file_path):
    """
    Load dataset from the given file path.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(file_path)

# Preprocessing and splitting the dataset
def preprocess_data(df, target_column):
    """
    Preprocess the dataset for training the model.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The column name of the target variable.
        
    Returns:
        tuple: Preprocessed features, target, and the feature names.
    """
    # Separate features and target
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

# Train and evaluate models
def train_and_evaluate(X, y, preprocessor, models, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple models.
    
    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target variable.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        models (dict): Dictionary of models to train.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Evaluation metrics for each model.
    """
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    results = {}
    
    for name, model in models.items():
        # Create a pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the pipeline
        model_file = f"{name}_model.pkl"
        joblib.dump(pipeline, model_file)
        print(f"Model '{name}' saved as '{model_file}'.")
        
        # Store results
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'R2 Score': r2,
            'Model File': model_file
        }
    
    return results

# Main script
if __name__ == "__main__":
    # Path to the dataset
    dataset_path = 'soil_suitability_with_soil_voltage.csv'
    
    # Load the dataset
    print("Loading dataset...")
    df = load_dataset(dataset_path)
    
    # Target column
    target = 'Suitability_Score_Perc'
    
    # Preprocess the data
    print("Preprocessing dataset...")
    X, y, preprocessor = preprocess_data(df, target)
    
    # Define models to train
    model_dict = {
        'GradientBoosting': GradientBoostingRegressor(),
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression()
    }
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = train_and_evaluate(X, y, preprocessor, model_dict)
    
    # Display results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
