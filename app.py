from flask import Flask, render_template, request
import numpy as np
import pickle

# Load trained model
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Optional: Load LinearRegression if you want to compare or switch models
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Field names from form (note: HTML input name for EUR/USD should be EUR_USD)
        field_names = ['SPX', 'USO', 'SLV', 'EUR_USD']
        form_data = {field: request.form[field] for field in field_names}
        features = [float(form_data[field]) for field in field_names]
        input_array = np.array(features).reshape(1, -1)

        # Predict using RandomForest (no transform needed)
        prediction = regressor.predict(input_array)

        # You can switch to LinearRegression if you want:
        # prediction = clf.predict(input_array)

        result = f"Predicted Gold Price: {prediction[0]:.2f}"
        return render_template('index.html', prediction_text=result, form_data=form_data)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
