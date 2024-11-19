import pandas as pd
import numpy as np
import os

def generate_nutrient_content(n_rows):
    """
    Generate nutrient content including Nitrogen, Phosphorus, Potassium, Calcium, and Magnesium.
    Returns a dictionary of generated nutrient columns.
    """
    return {
        "Nitrogen_Content_Perc": np.random.uniform(0.1, 3, n_rows),
        "Phosphorus_Content_ppm": np.random.uniform(5, 80, n_rows),
        "Potassium_Content_ppm": np.random.uniform(50, 600, n_rows),
        "Calcium_Content_ppm": np.random.uniform(200, 2000, n_rows),
        "Magnesium_Content_ppm": np.random.uniform(50, 300, n_rows)
    }

def generate_soil_properties(n_rows):
    """
    Generate soil properties including pH, moisture, organic matter, CEC, and bulk density.
    Returns a dictionary of generated soil properties.
    """
    return {
        "Soil_pH": np.random.uniform(4.5, 8.5, n_rows),
        "Soil_Moisture_Perc": np.random.uniform(10, 90, n_rows),
        "Organic_Matter_Perc": np.random.uniform(1, 10, n_rows),
        "Cation_Exchange_Capacity_meq": np.random.uniform(5, 40, n_rows),
        "Bulk_Density_g_per_cm3": np.random.uniform(1, 1.8, n_rows)
    }

def generate_environmental_factors(n_rows):
    """
    Generate environmental factors such as rainfall retention, temperature, and drainage analysis.
    Returns a dictionary of generated environmental factors.
    """
    return {
        "Rainfall_Retention_mm_per_year": np.random.uniform(200, 3000, n_rows),
        "Temperature_Celsius": np.random.uniform(10, 40, n_rows),
        "Drainage_Analysis": np.random.choice(["Baik", "Sedang", "Buruk"], n_rows)
    }

def generate_soil_voltage(soil_types):
    """
    Generate soil voltage based on soil type. Each soil type has a different range of voltages.
    Returns an array of voltage values based on soil type.
    """
    voltages = []
    for soil_type in soil_types:
        if soil_type == 'Lempung':
            voltages.append(np.random.uniform(200, 500))  # Voltase untuk tanah lempung
        elif soil_type == 'Pasir':
            voltages.append(np.random.uniform(50, 150))  # Voltase untuk tanah pasir
        elif soil_type == 'Gambut':
            voltages.append(np.random.uniform(300, 800))  # Voltase untuk tanah gambut
        elif soil_type == 'Liat':
            voltages.append(np.random.uniform(100, 300))  # Voltase untuk tanah liat
    return np.array(voltages)

def calculate_suitability(nutrient_data, soil_data, environmental_data):
    """
    Calculate the overall suitability score based on various factors using a weighted average.
    Returns an array of suitability scores.
    """
    # Calculate scores based on individual aspects (e.g., nutrient, soil, environment)
    nutrient_score = (nutrient_data["Nitrogen_Content_Perc"] + nutrient_data["Phosphorus_Content_ppm"] / 10 +
                      nutrient_data["Potassium_Content_ppm"] / 10) / 3
    soil_score = (soil_data["Soil_pH"] + soil_data["Soil_Moisture_Perc"] / 10 +
                  soil_data["Organic_Matter_Perc"] + soil_data["Cation_Exchange_Capacity_meq"] / 2) / 4
    environmental_score = (environmental_data["Rainfall_Retention_mm_per_year"] / 100 +
                           environmental_data["Temperature_Celsius"] / 5) / 2
    
    # Combine all scores with equal weights for simplicity (can be adjusted for more accuracy)
    suitability_score = (nutrient_score + soil_score + environmental_score) / 3
    
    # Scale scores to percentage
    return np.clip(suitability_score * 10, 0, 100)

def generate_farm_soil_suitability_dataset(n_rows=10000000, output_file='soil_suitability_with_soil_voltage.csv'):
    """
    Generates a complex synthetic dataset for agricultural soil suitability analysis with detailed nutrient content,
    environmental factors, soil voltage based on soil type, and a calculated suitability score.
    
    Parameters:
    n_rows (int): Number of rows to generate. Default is 10,000,000.
    output_file (str): Output file name to save the dataset. Default is 'soil_suitability_with_soil_voltage.csv'.
    
    Returns:
    A CSV file saved in the working directory with the specified name.
    """
    
    # Generate different aspects of the dataset
    nutrient_data = generate_nutrient_content(n_rows)
    soil_data = generate_soil_properties(n_rows)
    environmental_data = generate_environmental_factors(n_rows)
    
    # Defining soil types
    soil_types = np.random.choice(["Lempung", "Pasir", "Gambut", "Liat"], n_rows)
    
    # Generate soil voltage based on soil type
    voltages = generate_soil_voltage(soil_types)
    
    # Calculate the overall suitability score
    suitability_score = calculate_suitability(nutrient_data, soil_data, environmental_data)
    
    # Combine all data into one dictionary
    data = {**nutrient_data, **soil_data, **environmental_data,
            "Soil_Type": soil_types,
            "Soil_Voltage_mV": voltages,  # Voltase tanah berdasarkan jenis tanah
            "Suitability_Score_Perc": suitability_score}
    
    # Creating DataFrame
    df = pd.DataFrame(data)
    
    # Checking if the output file already exists
    if os.path.exists(output_file):
        print(f"Warning: The file '{output_file}' already exists and will be overwritten.")
    
    # Saving to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset generated and saved as '{output_file}'.")

if __name__ == "__main__":
    # Adjust the number of rows or output file name if necessary
    generate_farm_soil_suitability_dataset(n_rows=100, output_file='soil_suitability_with_soil_voltage.csv')
