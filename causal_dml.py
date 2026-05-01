import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from econml.dml import CausalForestDML
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('final_dataset.csv')

# 1. Define our variables
Y = df['log_trade']       # Outcome
T = df['eu_usa_fta']      # Treatment (Binary)

# X represents our confounders. 
# We use the core structural gravity variables that dictate trade and policy.
X = df[['gdp_o', 'gdp_d', 'distance']]

# 2. Setup the Double Machine Learning (DML) Architecture
print("Configuring the Double ML stages...")

# model_y: Residualizes Trade (Y) using XGBoost
# model_t: Residualizes FTA (T) using XGBoost
# discrete_treatment=True tells the model that FTA is a 0 or 1 category.
dml_model = CausalForestDML(
    model_y=XGBRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42),
    model_t=XGBClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42),
    discrete_treatment=True,
    n_estimators=200,     # Number of trees in the final Causal Forest
    cv=5,                 # 5-fold cross-fitting to prevent overfitting
    random_state=42
)

# 3. Fit the model (This executes the Robinson Partialling-Out procedure)
print("Step 1: XGBoost is residualizing Y and T...")
print("Step 2: Causal Forest is calculating the ATE on the residuals...")
dml_model.fit(Y, T, X=X)

# 4. Extract the Causal Impact
# Calculate the Average Treatment Effect across all observations in X
ate = dml_model.ate(X)
ate_ci = dml_model.ate_interval(X)

print("\n==============================================================================")
print(" Corrected DML Results: XGBoost Residualization + Causal Forest")
print("==============================================================================")
print(f"Estimated ATE (Log Points): +{ate:.4f}")
print(f"95% Confidence Interval:    [+{ate_ci[0]:.4f}, +{ate_ci[1]:.4f}]")