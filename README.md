# Evaluating EU-USA FTA Spillovers on the Top 6 Global Economies 🌍📈
**A Causal Gravity Approach via Double Machine Learning**

## 📌 Overview
This repository contains the code, data pipeline, and econometric models used to evaluate the macroeconomic spillovers of a hypothetical Free Trade Agreement (FTA) between the European Union (EU) and the United States (USA). 

By bridging traditional structural gravity models with modern machine learning, this project isolates the true cause-and-effect of deep integration policies on the world's top six economies (USA, China, Germany, Japan, India, and the UK). The methodology utilizes Double Machine Learning (DML) to control for complex, non-linear economic factors and multilateral resistance.

## 🚀 The Analytical Pipeline
The project follows a strict, four-stage causal inference pipeline:
1. **Data Compilation:** Aggregating panel data from UN COMTRADE (Trade Flows), World Bank (GDP), CEPII (Geographic Distance), and WTO RTAIS (Policy Shocks).
2. **High-Dimensional Fixed Effects (HDFE) Model:** A baseline structural gravity model applying exporter-year, importer-year, and country-pair fixed effects.
3. **XGBoost Residualization (Stage 1 DML):** Using tree-based algorithms to predict Trade and FTA variables based on background confounders, filtering out non-linear noise.
4. **Causal Forest Estimation (Stage 2 DML):** Regressing the orthogonalized residuals to calculate the isolated Average Treatment Effect (ATE).

## 📊 Key Findings
* **Massive Trade Creation:** The models isolated a robust Average Treatment Effect (ATE) of **+0.720 to +0.733 log points**.
* **Economic Impact:** This translates to an approximate **~105% increase** in associated trade volumes, effectively doubling trade capacity due to the removal of Non-Tariff Barriers (NTBs).
* **Third-Party Spillovers:** Non-member states (like India and the UK) experienced compounding marginal benefits, capturing massive downstream demand by supplying intermediate goods to a newly frictionless Western bloc.
* **Model Validation:** The causal findings were heavily stress-tested and validated using Randomized Permutation Placebo Tests (yielding an insignificant ATE of -0.032) and SHAP feature attribution.

## 🛠️ Tech Stack
This project was built using Python 3.10 and relies on state-of-the-art libraries for econometrics and causal inference:
* **Econometrics:** `pyfixest` (for HDFE matrix absorption)
* **Causal Inference:** `EconML` (Microsoft Research)
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Explainable AI:** `shap`
* **Data Manipulation:** `pandas`, `numpy`

## 📂 Repository Structure
```text
├── data/
│   ├── raw/                  # Raw datasets (UN COMTRADE, CEPII, etc.)
│   └── processed/            # Cleaned and merged panel data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_hdfe_gravity_model.ipynb
│   ├── 03_double_machine_learning.ipynb
│   └── 04_robustness_and_shap.ipynb
├── src/
│   ├── models/               # Python scripts for XGBoost and Causal Forest
│   └── utils/                # Helper functions for data scaling and metrics
├── results/
│   ├── figures/              # Event study, SHAP summary, and causal spillovers charts
│   └── NexTrade_Master_Results_Log.xlsx
├── report/
│   └── Final_LaTeX_Report.pdf
├── requirements.txt
└── README.
⚙️ Getting Started
1. Clone the repository:

Bash
git clone [https://github.com/yourusername/causal-gravity-fta.git](https://github.com/yourusername/causal-gravity-fta.git)
cd causal-gravity-fta
2. Install dependencies:

Bash
pip install -r requirements.txt
3. Run the pipeline:
Execute the Jupyter notebooks in sequential order, or run the master script to reproduce the Double Machine Learning results and robustness checks.

👨‍💻 Author
Aditya Acharya Indian Institute of Information Technology, Allahabad (IIITA) ID: IIB2023008

Email: adityaacharyasantoshkumar@gmail.com










md
