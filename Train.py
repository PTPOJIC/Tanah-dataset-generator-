import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_soil_suitability_model(dataset_path, model_output_path='soil_suitability_model.pkl'):
    """
    Train a machine learning model to predict soil suitability based on soil, nutrient, and environmental data.

    Parameters:
    - dataset_path (str): Path to the CSV dataset.
    - model_output_path (str): Path to save the trained model (default: 'soil_suitability_model.pkl').
    """
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Preprocessing: Convert categorical data to numerical (e.g., Drainage_Analysis, Soil_Type)
    data = pd.get_dummies(data, columns=["Drainage_Analysis", "Soil_Type"], drop_first=True)

    # Define features and target
    X = data.drop(columns=["Suitability_Score_Perc"])
    y = data["Suitability_Score_Perc"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:\n- Mean Squared Error: {mse:.2f}\n- R^2 Score: {r2:.2f}")

    # Save the trained model
    import joblib
    joblib.dump(model, model_output_path)
    print(f"Model saved to '{model_output_path}'.")

if __name__ == "__main__":
    # Path to the generated dataset
    dataset_path = 'soil_suitability_with_soil_voltage.csv'

    # Train the model using the dataset
    train_soil_suitability_model(dataset_path)
