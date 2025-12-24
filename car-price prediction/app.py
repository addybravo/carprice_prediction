from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Load data for dropdowns (if needed)
data = pd.read_csv("auto_mobile_data.csv")
data.replace('?', pd.NA, inplace=True)
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get user input
    hp = float(request.form['horsepower'])
    engine_size = float(request.form['engine_size'])
    weight = float(request.form['curb_weight'])

    # 2. Predict car price
    features = np.array([[hp, engine_size, weight]])
    predicted_price = model.predict(features)[0]

    # 3. Calculate R² score (model accuracy)
    # Reload and clean data
    cols = ['horsepower', 'engine-size', 'curb-weight', 'price']
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=cols, inplace=True)

    # Prepare test data
    X = data[['horsepower', 'engine-size', 'curb-weight']]
    y = data['price']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred_test = model.predict(X_test)

    # Compute accuracy score
    r2 = r2_score(y_test, y_pred_test)

    # 4. Show result page with price and accuracy
    return render_template('result.html',
                           price=round(predicted_price, 2),
                           accuracy=round(r2, 2))

if __name__ == '__main__':
    app.run(debug=True)
    