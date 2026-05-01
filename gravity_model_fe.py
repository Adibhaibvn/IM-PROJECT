import pandas as pd
from pyfixest.estimation import feols
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('final_dataset.csv')

print("Creating fixed effect categories...")
# \pi_{it} : Exporter-Year
df['exp_year'] = df['country_o'].astype(str) + "_" + df['year'].astype(str)

# \chi_{jt} : Importer-Year
df['imp_year'] = df['country_d'].astype(str) + "_" + df['year'].astype(str)

# \mu_{ij} : Country-Pair
df['pair'] = df['country_o'].astype(str) + "_" + df['country_d'].astype(str)

print("Estimating High-Dimensional Fixed Effects Gravity Model...")
# The formula syntax: "Dependent_Var ~ Independent_Var | Fixed_Effect_1 + Fixed_Effect_2 + Fixed_Effect_3"
formula = "log_trade ~ eu_usa_fta | exp_year + imp_year + pair"

# Run the model
# pyfixest automatically uses robust clustered standard errors based on the fixed effects
model = feols(formula, data=df)

# Print the academic summary
print("\n==============================================================================")
print(" Structural Gravity Model with Exporter-Time, Importer-Time, and Pair Effects")
print("==============================================================================")
print(model.summary())