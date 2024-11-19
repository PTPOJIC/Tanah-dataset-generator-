To set up and run the **Earth Fix AI Virtual Assistant**, you'll need to ensure the necessary prerequisites are in place. Here's a comprehensive list:

---

### 1. **System Requirements**
   - **Operating System**: Linux, macOS, or Windows (64-bit preferred).
   - **Python Version**: Python 3.8 or newer.
   - **RAM**: At least 8 GB (recommended for processing large datasets).
   - **Storage**: At least 10 GB free space.

---

### 2. **Required Libraries**
   - Core Libraries: 
     - `pandas`
     - `numpy`
   - AI/ML Libraries:
     - `scikit-learn`
   - Utility Libraries:
     - `joblib`

---

### 3. **Dataset**
   - **Soil Suitability Dataset**: 
     - Generate the dataset using the provided synthetic dataset generator or download a pre-existing dataset.
     - Save the file as `soil_suitability_with_soil_voltage.csv` in the same directory as the script.

---

### 4. **Installation Steps**

#### Step 1: Install Python
Ensure Python is installed on your system. You can verify this by running:
```bash
python3 --version
```

If Python is not installed, download it from the [official Python website](https://www.python.org/downloads/) or install it using your system's package manager:
```bash
# For Ubuntu/Debian:
sudo apt update && sudo apt install python3 python3-pip

# For CentOS/RHEL:
sudo yum install python3 python3-pip
```

---

#### Step 2: Install Libraries
Create a virtual environment to manage dependencies:
```bash
python3 -m venv earthfix_env
source earthfix_env/bin/activate  # On Windows: earthfix_env\Scripts\activate
```

Then install the required libraries:
```bash
pip install pandas numpy scikit-learn joblib
```

---

#### Step 3: Generate the Dataset
Use the synthetic dataset generator script provided earlier. Run it to generate the `soil_suitability_with_soil_voltage.csv` file:
```bash
python generate_dataset.py
```

If a dataset is already available, place it in the script directory.

---

#### Step 4: Train the AI Model
Run a script to train the AI model (if not already trained):
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv('soil_suitability_with_soil_voltage.csv')

# Prepare features and target
X = df.drop(columns=['Suitability_Score_Perc'], errors='ignore')
y = df['Suitability_Score_Perc']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'earth_fix_ai_model.pkl')
print("Model trained and saved as 'earth_fix_ai_model.pkl'.")
```

---

#### Step 5: Run the Assistant
Finally, run the virtual assistant script:
```bash
python earth_fix_assistant.py
```

---

### 5. **Optional Tools**
- **Jupyter Notebook**: For experimenting with dataset generation and analysis.
  ```bash
  pip install notebook
  jupyter notebook
  ```
- **Virtual Assistant Extensions**:
  - `pyttsx3` for text-to-speech functionality.
  - `SpeechRecognition` for speech-to-text capabilities.
  ```bash
  pip install pyttsx3 SpeechRecognition
  ```

---

### 6. **Verification**
After setting up, ensure everything is working:
1. The dataset (`soil_suitability_with_soil_voltage.csv`) exists in the directory.
2. The AI model (`earth_fix_ai_model.pkl`) is trained and available.
3. Running the assistant script starts the interactive session.

You’re now ready to use the **Earth Fix AI Virtual Assistant**!
