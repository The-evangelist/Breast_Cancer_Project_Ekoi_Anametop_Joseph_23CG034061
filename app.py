from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Model and Scaler
with open("model/breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [
        float(request.form['radius']),
        float(request.form['texture']),
        float(request.form['perimeter']),
        float(request.form['area']),
        float(request.form['compactness'])
    ]

    arr = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]

    result = "Benign" if pred == 1 else "Malignant"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
