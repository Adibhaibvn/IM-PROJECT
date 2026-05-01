import pandas as pd
import numpy as np
from pyfixest.estimation import feols
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('final_dataset.csv')

# 1. Setup the Fixed Effects
df['exp_year'] = df['country_o'].astype(str) + "_" + df['year'].astype(str)
df['imp_year'] = df['country_d'].astype(str) + "_" + df['year'].astype(str)
df['pair'] = df['country_o'].astype(str) + "_" + df['country_d'].astype(str)

# The core Gravity equation
formula = "log_trade ~ eu_usa_fta | exp_year + imp_year + pair"

print("\n=======================================================")
print(" STEP 8: HETEROGENEITY ANALYSIS (Who benefits most?)")
print("=======================================================")

# A. Distance Split
median_dist = df['distance'].median()
mod_close = feols(formula, data=df[df['distance'] <= median_dist])
mod_far = feols(formula, data=df[df['distance'] > median_dist])
print(f"Below-Median Distance ATE: +{mod_close.coef().iloc[0]:.3f}")
print(f"Above-Median Distance ATE: +{mod_far.coef().iloc[0]:.3f}")

# B. GDP Split
df['total_gdp'] = df['gdp_o'] + df['gdp_d']
median_gdp = df['total_gdp'].median()
mod_high_gdp = feols(formula, data=df[df['total_gdp'] > median_gdp])
mod_low_gdp = feols(formula, data=df[df['total_gdp'] <= median_gdp])
print(f"Above-Median GDP Pairs ATE: +{mod_high_gdp.coef().iloc[0]:.3f}")
print(f"Below-Median GDP Pairs ATE: +{mod_low_gdp.coef().iloc[0]:.3f}")


print("\n=======================================================")
print(" STEP 9: ROBUSTNESS CHECKS (Stress-testing the model)")
print("=======================================================")

# A. Time Perturbation: What if we only look at data after the 2008 financial crash?
mod_post_2008 = feols(formula, data=df[df['year'] >= 2010])
print(f"Post-2008 Time Window ATE: +{mod_post_2008.coef().iloc[0]:.3f}")

# B. Drop Outliers: What if we completely remove the USA and China?
# This proves one massive country isn't artificially dragging the math up
df_no_outliers = df[(df['country_o'] != 'USA') & (df['country_d'] != 'USA') & 
                    (df['country_o'] != 'CHN') & (df['country_d'] != 'CHN')]
mod_no_outliers = feols(formula, data=df_no_outliers)
print(f"Drop USA & China ATE:      +{mod_no_outliers.coef().iloc[0]:.3f}")

# C. Placebo Test: Randomized Permutation Placebo
# We randomly shuffle the true FTA treatments across all years and country pairs.
# This destroys the real causal relationship but keeps the exact same number of "treated" instances.
np.random.seed(42) # Fixed seed ensures reproducibility 
df['random_placebo_fta'] = np.random.permutation(df['eu_usa_fta'].values)

try:
    mod_placebo = feols("log_trade ~ random_placebo_fta | exp_year + imp_year + pair", data=df)
    
    # Safely extract values using the tidy() dataframe
    res_df = mod_placebo.tidy()
    placebo_ate = res_df['Estimate'].iloc[0]
    placebo_pval = res_df['Pr(>|t|)'].iloc[0]
    
    print(f"Randomized Placebo ATE:      {placebo_ate:.3f}")
    print(f"Randomized Placebo P-value:  {placebo_pval:.3f} (Ideally > 0.05)")
    
    if placebo_pval > 0.05:
        print("SUCCESS: The P-value is > 0.05. The model correctly found NO causal effect for the randomized noise.")
    else:
        print("WARNING: The model found a significant effect.")
        
except Exception as e:
    print(f"Placebo Test encountered an error: {e}")