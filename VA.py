import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os


# Define constants
MODEL_PATH = "earth_fix_ai_model.pkl"
DATASET_PATH = "soil_suitability_with_soil_voltage.csv"


class EarthFixAssistant:
    """
    A virtual assistant for environmental restoration and soil improvement recommendations.
    """
    def __init__(self):
        self.model = None
        self.dataset = None
        self.load_model()
        self.load_dataset()

    def load_model(self):
        """
        Load the trained AI model from a file.
        """
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("AI model loaded successfully.")
        else:
            print("Warning: AI model not found. Please train and save the model.")

    def load_dataset(self):
        """
        Load the dataset for analysis and recommendations.
        """
        if os.path.exists(DATASET_PATH):
            self.dataset = pd.read_csv(DATASET_PATH)
            print("Dataset loaded successfully.")
        else:
            print("Warning: Dataset not found. Please generate or provide a dataset.")

    def interact(self):
        """
        Start the interactive session with the virtual assistant.
        """
        print("\nWelcome to the Earth Fix AI Virtual Assistant!")
        print("How can I assist you today?")
        print("Type 'help' for options or 'exit' to quit.\n")

        while True:
            user_input = input("Your query: ").strip().lower()

            if user_input in ["exit", "quit"]:
                print("Thank you for using Earth Fix AI. Goodbye!")
                break
            elif user_input == "help":
                self.show_help()
            elif "soil improvement" in user_input:
                self.handle_soil_improvement()
            elif "nitrogen content" in user_input:
                self.handle_nitrogen_content()
            elif "sustainable practices" in user_input:
                self.handle_sustainable_practices()
            else:
                print("Sorry, I didn't understand that. Type 'help' for options.\n")

    def show_help(self):
        """
        Display available commands to the user.
        """
        print("\nAvailable Commands:")
        print("- 'soil improvement': Get recommendations for improving soil conditions.")
        print("- 'nitrogen content': Learn how to increase nitrogen in the soil.")
        print("- 'sustainable practices': Get tips for sustainable farming practices.")
        print("- 'exit': Quit the virtual assistant.\n")

    def handle_soil_improvement(self):
        """
        Provide soil improvement recommendations based on the dataset and AI model.
        """
        print("\nAnalyzing soil improvement recommendations...")
        if self.model and self.dataset is not None:
            recommendations = self.generate_recommendations()
            print(f"Top Recommendations: {recommendations[:3]}\n")
        else:
            print("Error: Model or dataset not available. Please ensure they are loaded.\n")

    def handle_nitrogen_content(self):
        """
        Provide guidance on increasing nitrogen content in the soil.
        """
        print("\nFor increasing nitrogen content:")
        print("- Apply nitrogen-based fertilizers.")
        print("- Use nitrogen-fixing crops like legumes.")
        print("- Incorporate organic compost rich in nitrogen.\n")

    def handle_sustainable_practices(self):
        """
        Provide tips for sustainable farming practices.
        """
        print("\nSustainable Practices:")
        print("- Implement crop rotation to maintain soil health.")
        print("- Use cover crops to reduce erosion and retain nutrients.")
        print("- Incorporate organic matter like compost and mulch.")
        print("- Adopt water-efficient irrigation systems.\n")

    def generate_recommendations(self):
        """
        Generate suitability and soil improvement recommendations using the dataset and AI model.
        """
        # Process dataset for predictions
        data = pd.get_dummies(self.dataset, columns=["Drainage_Analysis", "Soil_Type"], drop_first=True)
        data = self.create_derived_features(data)

        # Predict suitability scores
        data['Predicted_Suitability'] = self.model.predict(data.drop(columns=["Suitability_Score_Perc"], errors='ignore'))
        data['Recommendations'] = data.apply(self.recommend_actions, axis=1)

        return data['Recommendations'].tolist()

    def create_derived_features(self, df):
        """
        Add derived features for enhanced modeling and recommendations.
        """
        df['N_to_P_Ratio'] = df['Nitrogen_Content_Perc'] / (df['Phosphorus_Content_ppm'] + 1e-6)
        df['Moisture_pH_Interaction'] = df['Soil_pH'] * df['Soil_Moisture_Perc']
        df['Carbon_Sequestration_Potential'] = df['Organic_Matter_Perc'] * 2.5
        return df

    def recommend_actions(self, row):
        """
        Generate soil restoration recommendations based on input data.
        """
        recommendations = []
        if row['Nitrogen_Content_Perc'] < 1.5:
            recommendations.append("Apply nitrogen-based fertilizers.")
        if row['Phosphorus_Content_ppm'] < 20:
            recommendations.append("Increase phosphorus supplementation.")
        if row['Potassium_Content_ppm'] < 200:
            recommendations.append("Add potassium-rich organic matter.")
        if row['Soil_pH'] < 5.5:
            recommendations.append("Apply lime to reduce soil acidity.")
        elif row['Soil_pH'] > 7.5:
            recommendations.append("Use sulfur to lower soil alkalinity.")
        if row['Soil_Moisture_Perc'] < 30:
            recommendations.append("Implement irrigation systems.")
        elif row.get('Drainage_Analysis_Buruk', 0) == 1:
            recommendations.append("Improve soil drainage.")
        if row['Organic_Matter_Perc'] < 4:
            recommendations.append("Incorporate organic compost to enrich soil.")
        if row['Temperature_Celsius'] > 35:
            recommendations.append("Plant heat-tolerant crops.")
        if row['Rainfall_Retention_mm_per_year'] < 500:
            recommendations.append("Adopt water retention techniques like mulching.")
        return recommendations


# Entry point for the application
if __name__ == "__main__":
    assistant = EarthFixAssistant()
    assistant.interact()
  
