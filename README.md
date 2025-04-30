# Zeta_Question2- Loan-API

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
git clone https://github.com/diyaverma1967/Zeta_Question2.git
cd Zeta_Question2

# 2. Set up Conda env
conda create -n Zeta_Question2 python=3.11 -y
conda activate Zeta_Question2

# 3. Install deps
pip install -r requirements.txt

# 4. Train model
python src/train_model.py --input Loan_Data.csv

# 5. Start API
uvicorn src.main:app --reload

