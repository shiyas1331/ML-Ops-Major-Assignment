import joblib
import numpy as np
import os

# Load the trained model
model = joblib.load("artifacts/model.joblib")

# Extract original parameters
coef = model.coef_
intercept = model.intercept_

# Save original parameters
joblib.dump({"coef": coef, "intercept": intercept}, "artifacts/unquant_params.joblib")

# Quantize (manually scale to 0-255 range)
coef_min, coef_max = coef.min(), coef.max()
quantized_coef = np.round((coef - coef_min) / (coef_max - coef_min) * 255).astype(np.uint8)
quantized_intercept = np.round(intercept).astype(np.uint8)  # Simple for demo

# Save quantized parameters
joblib.dump(
    {
        "coef": quantized_coef,
        "intercept": quantized_intercept,
        "scale": coef_max - coef_min,
        "zero_point": coef_min
    },
    "artifacts/quant_params.joblib"
)

# Dequantize for verification
dequantized_coef = quantized_coef.astype(np.float32) / 255 * (coef_max - coef_min) + coef_min

# Run a test inference
sample_input = np.random.rand(1, coef.shape[0])  # dummy input with same shape
output = np.dot(sample_input, dequantized_coef) + quantized_intercept
print("Sample inference output using quantized model:", output)
