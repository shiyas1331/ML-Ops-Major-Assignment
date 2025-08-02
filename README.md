# MLOps Linear Regression Pipeline

### Submitted By  
**Name**: Mohamed Shiyas  
**Roll No**: G24AI1038  
**Email**: g24ai1038@iitj.ac.in

---

## Objective

This project demonstrates a complete MLOps pipeline for a Linear Regression model trained on the California Housing dataset. It includes:

- Model training
- Unit testing
- Manual quantization
- Dockerization
- CI/CD using GitHub Actions

---

## Dataset & Model

- **Dataset**: California Housing dataset from `sklearn.datasets`
- **Model**: `LinearRegression` from `scikit-learn`

---

## Project Structure

```
.
├── src/
│   ├── train.py         # Train the model
│   ├── quantize.py      # Manual quantization
│   └── predict.py       # Inference script (used in Docker)
├── tests/
│   └── test_train.py    # Unit tests
├── artifacts/           # Contains saved models and parameters
├── Dockerfile           # Docker config
├── requirements.txt     # Python dependencies
├── .github/
│   └── workflows/ci.yml # GitHub Actions workflow
└── README.md
```

---

## CI/CD Workflow

| Job Name               | Description                                                | Depends On       |
|------------------------|------------------------------------------------------------|------------------|
| `test_suite`           | Runs `pytest` for unit testing                             | None             |
| `train_and_quantize`   | Trains and quantizes model, then uploads artifacts         | `test_suite`     |
| `build_and_test_container` | Builds Docker image and verifies inference via `predict.py` | `train_and_quantize` |

GitHub Actions automates this process on every push to the `main` branch.

---

## Model Performance

| Metric             | Original Model (`model.joblib`) | Quantized (`quant_params.joblib`) |
|--------------------|-------------------------------|----------------------------------|
| R² Score           | 0.6019                        | ~0.59                            |
| MSE                | 0.5217                        | ~0.56                            |
| File Size          | 697 KB                         | 456 KB                           |
| Parameters         | Float32 Coef + Intercept       | uint8 + scale + zero_point       |

> Quantized inference output: **[218.7218]**

---

## Sample Predictions (from Docker)

```
Sample 1: 0.7344
Sample 2: 1.7558
Sample 3: 2.6546
Sample 4: 2.8515
Sample 5: 2.6183
```

---

## Docker

Build:
```bash
docker build -t mlops-lr .
```

Run:
```bash
docker run --rm mlops-lr
```

---

## How to Run Locally

```bash
# Set up Conda environment
conda create -n mlops3 python=3.10 -y
conda activate mlops3
pip install -r requirements.txt

# Train and quantize
python src/train.py
python src/quantize.py

# Run inference
python src/predict.py

# Run tests
pytest tests/
```