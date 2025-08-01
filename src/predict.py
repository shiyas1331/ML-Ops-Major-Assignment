import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("artifacts/model.joblib")

# Load data
data = fetch_california_housing()
_, X_test, _, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Predict on first 5 samples
predictions = model.predict(X_test[:5])

print("Sample Predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred:.4f}")
