# Loan-API

**FastAPI** service + **Keras** model to compute a 0–100 credit-likelihood score and human-readable recommendation.

## Structure
- `src/train_model.py` — Preprocess CSV, train NN, save artifacts
- `src/main.py`        — Load artifacts, expose `/loan/eligibility`
- `src/Loan_Data.csv`  — Sample CSVs for training
- `output/`            — Generated model & preprocessor
- `docs/`              — Architecture, model design, API reference

## Quickstart

```bash
# 1. Clone & cd
git clone <your-repo-url>
cd loan-api

# 2. Set up Conda env
conda create -n loan-api python=3.11 -y
conda activate loan-api

# 3. Install deps
pip install -r requirements.txt

# 4. Train model
python src/train_model.py --input Loan_Data.csv

# 5. Start API
uvicorn src.main:app --reload

# 6. Results
![alt text](image-1.png)

![alt text](image-2.png)
