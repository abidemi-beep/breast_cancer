from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
# Model is saved in 'model/breast_cancer_model.pkl'
# Depending on where we run app.py, the path might be relative.
# Assuming app.py is in the root, and model is in model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')


try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features'] # List of feature names
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        if not model:
            error = "Model not loaded properly."
        else:
            try:
                # Extract features from form
                # Input order must match training order
                # 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
                input_features = [
                    float(request.form['radius_mean']),
                    float(request.form['texture_mean']),
                    float(request.form['perimeter_mean']),
                    float(request.form['area_mean']),
                    float(request.form['smoothness_mean'])
                ]
                
                # Reshape and scale
                features_array = np.array(input_features).reshape(1, -1)
                scaled_features = scaler.transform(features_array)
                
                # Predict
                # 0 = Benign, 1 = Malignant (based on our training mapping logic if we did that)
                # Wait, in the training script I did:
                # df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 1 else 1) 
                # Original sklearn: 0=Malignant, 1=Benign.
                # My remapping: 1 (was 0) = Malignant, 0 (was 1) = Benign.
                # So Prediction 1 -> Malignant, 0 -> Benign.
                
                pred_val = model.predict(scaled_features)[0]
                prediction = "Malignant" if pred_val == 1 else "Benign"
                
            except ValueError:
                error = "Invalid input. Please enter numeric values."
            except Exception as e:
                error = f"An error occurred: {e}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
