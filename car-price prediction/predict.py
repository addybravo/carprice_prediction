import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv("auto_mobile_data.csv")

# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Convert numeric columns (modify these to match your CSV)
cols = ['horsepower', 'engine-size', 'curb-weight', 'price']
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any NaN values in selected columns
data.dropna(subset=cols, inplace=True)

# Features and target
X = data[['horsepower', 'engine-size', 'curb-weight']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
