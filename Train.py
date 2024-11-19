import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict

def create_derived_features(df):
    """
    Add derived features for enhanced modeling.
    """
    df['N_to_P_Ratio'] = df['Nitrogen_Content_Perc'] / (df['Phosphorus_Content_ppm'] + 1e-6)
    df['Moisture_pH_Interaction'] = df['Soil_pH'] * df['Soil_Moisture_Perc']
    return df

def train_soil_suitability_model(dataset_path, model_output_path='soil_suitability_model.pkl'):
    """
    Train a complex AI model to predict soil suitability with probabilistic results.
    """
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Preprocessing: Convert categorical data to numerical (e.g., Drainage_Analysis, Soil_Type)
    data = pd.get_dummies(data, columns=["Drainage_Analysis", "Soil_Type"], drop_first=True)
    
    # Add derived features
    data = create_derived_features(data)
    
    # Define features and target
    X = data.drop(columns=["Suitability_Score_Perc"])
    y_regression = data["Suitability_Score_Perc"]  # Regression target
    y_classification = pd.cut(
        y_regression, bins=[-np.inf, 33, 66, np.inf], labels=["Low", "Medium", "High"]
    )  # Classification target

    # Split into training and testing sets
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

    # Create a pipeline with scaling
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # Regression model
    reg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=200, random_state=42))
    ])
    reg_model.fit(X_train, y_train_reg)

    # Classification model
    cls_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    cls_model.fit(X_train, y_train_cls)

    # Evaluate regression model
    y_pred_reg = reg_model.predict(X_test)
    reg_mse = mean_squared_error(y_test_reg, y_pred_reg)
    reg_r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"Regression Model Evaluation:\n- MSE: {reg_mse:.2f}\n- R²: {reg_r2:.2f}")

    # Evaluate classification model
    y_pred_cls = cls_model.predict(X_test)
    cls_accuracy = accuracy_score(y_test_cls, y_pred_cls)
    print(f"Classification Model Evaluation:\n- Accuracy: {cls_accuracy:.2f}")

    # Save the models
    import joblib
    joblib.dump(reg_model, model_output_path.replace('.pkl', '_reg.pkl'))
    joblib.dump(cls_model, model_output_path.replace('.pkl', '_cls.pkl'))
    print(f"Models saved to '{model_output_path.replace('.pkl', '_reg.pkl')}' and '{model_output_path.replace('.pkl', '_cls.pkl')}'.")

if __name__ == "__main__":
    # Path to the generated dataset
    dataset_path = 'soil_suitability_with_soil_voltage.csv'

    # Train the model using the dataset
    train_soil_suitability_model(dataset_path)
    
