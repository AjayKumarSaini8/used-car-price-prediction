from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model_file_path = "rf_model.pkl"
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

scaler_file_path = "scaler.pkl"
with open(scaler_file_path, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    present_price = float(request.form['present_price'])
    car_age = int(request.form['car_age'])
    fuel_type = int(request.form['fuel_type'])
    seller_type = int(request.form['seller_type'])
    transmission_manual = int(request.form['transmission_manual'])

    # Scale the input features
    input_scaled = scaler.transform([[present_price, car_age]])
    input_data = np.concatenate([input_scaled, [[fuel_type, seller_type, transmission_manual]]], axis=1)

    # Make the prediction
    predicted_price = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted car cost: {predicted_price:.2f} lakhs')

if __name__ == '__main__':
    app.run(debug=True)
