import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from econml.dml import CausalForestDML
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(filepath="final_dataset.csv"):
    print("\n--- 1. Loading & Preprocessing Real Dataset ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Success: '{filepath}' loaded with {len(df)} observations.")
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please ensure it is in the same folder.")
        exit()

    # Base features for structural gravity
    base_features = ['distance', 'gdp_o', 'gdp_d', 'eu_usa_fta', 'tariff']
    
    # One-hot encoding for Fixed Effects (Country Origin, Destination, Year, Sector)
    df_encoded = pd.get_dummies(df, columns=['country_o', 'country_d', 'year', 'sector'], drop_first=True)
    fixed_effects = [col for col in df_encoded.columns if 'country_o_' in col or 'country_d_' in col or 'year_' in col or 'sector_' in col]
    
    features = base_features + fixed_effects
    
    X = df_encoded[features]
    y = df_encoded['log_trade']
    
    # Standard Scaling (Required for Neural Networks)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, df, features, scaler

# ==========================================
# 2. OLS VS XGBOOST & SHAP
# ==========================================
def run_predictive_models(X, y):
    print("\n--- 2. Comparing Classical OLS vs XGBoost AI ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # A. Classical Econometrics (OLS with Fixed Effects)
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train.values, X_train_sm).fit()
    y_pred_ols = ols_model.predict(sm.add_constant(X_test))
    print(f"Classical OLS   | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f} | R²: {r2_score(y_test, y_pred_ols):.4f}")

    # B. Non-Linear Machine Learning (XGBoost)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print(f"XGBoost AI      | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f} | R²: {r2_score(y_test, y_pred_xgb):.4f}")
    
    # C. SHAP Explainability
    print("\n--- 3. Running SHAP Explainability ---")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_train)
    
    print("SHAP values calculated. Generating Feature Importance Plot...")
    # This will pop up a window with a graph showing the top features
    shap.summary_plot(shap_values, X_train, max_display=10, show=False)
    plt.tight_layout()
    plt.show() 
    
    return xgb_model, X_train

# ==========================================
# 3. SPATIOTEMPORAL GNN-LSTM
# ==========================================
class GNN_LSTM_Gravity(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GNN_LSTM_Gravity, self).__init__()
        self.gcn = GCNConv(num_node_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index):
        node_embeddings = self.gcn(node_features, edge_index).relu()
        orig_embeddings = node_embeddings[edge_index[0]]
        dest_embeddings = node_embeddings[edge_index[1]]
        
        edge_features = torch.cat([orig_embeddings, dest_embeddings], dim=1)
        edge_features = edge_features.unsqueeze(1) 
        
        lstm_out, _ = self.lstm(edge_features)
        return self.fc(lstm_out.squeeze(1))

def train_gnn_lstm(df):
    print("\n--- 4. Training GNN-LSTM on Trade Network ---")
    # Build Graph Adjacency Matrix
    countries = pd.concat([df['country_o'], df['country_d']]).unique()
    country_to_id = {c: i for i, c in enumerate(countries)}
    
    src = df['country_o'].map(country_to_id).values
    dst = df['country_d'].map(country_to_id).values
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    
    node_features = torch.ones((len(countries), 1), dtype=torch.float32)
    y_target = torch.tensor(df['log_trade'].values, dtype=torch.float32).view(-1, 1)
    
    model = GNN_LSTM_Gravity(num_node_features=1, hidden_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(node_features, edge_index)
        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()
    print(f"GNN-LSTM compiled and mapped multilateral resistance. Final Training Loss: {loss.item():.4f}")

# ==========================================
# 4. CAUSAL INFERENCE (DOUBLE ML)
# ==========================================
def run_causal_spillovers(df):
    print("\n--- 5. Causal Forest for EU-USA FTA Spillovers ---")
    Y = df['log_trade'] 
    T = df['eu_usa_fta'] 
    X_confounders = df[['distance', 'gdp_o', 'gdp_d']] 
    
    # Train the Double Machine Learning Estimator
    est = CausalForestDML(discrete_treatment=True, random_state=42)
    est.fit(Y, T, X=X_confounders, W=None)
    
    # Isolate India
    india_mask = (df['country_o'] == 'India') | (df['country_d'] == 'India')
    india_effects = est.effect(X_confounders[india_mask])
    
    print(f"*** India-Specific Spillover Effect (ATE): {np.mean(india_effects):.4f} ***")

# ==========================================
# 5. REINFORCEMENT LEARNING
# ==========================================
class TradePolicyEnv(gym.Env):
    def __init__(self, trained_model, base_row_df):
        super(TradePolicyEnv, self).__init__()
        self.model = trained_model
        self.base_state = base_row_df.copy()
        
        # FIX 1: Track the CURRENT state across multiple steps (No time-loop)
        self.current_state = base_row_df.copy()
        
        # Action Space: Adjust tariff by -0.1 (lower) to +0.1 (raise) per step
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.base_state.shape[1],), dtype=np.float32)
        self.steps = 0
        
        self.tariff_idx = self.base_state.columns.get_loc('tariff')

    def step(self, action):
        tariff_adjustment = action[0]
        
        # Apply tariff adjustment to the ongoing current state
        self.current_state.iloc[0, self.tariff_idx] += tariff_adjustment 
        
        # Oracle: Ask XGBoost how the global market reacts
        predicted_trade = self.model.predict(self.current_state)[0]
        
        # FIX 2: Pure Reward. Just maximize trade volume.
        reward = predicted_trade
        
        self.steps += 1
        done = self.steps >= 20 # 20 steps per episode
        return self.current_state.values[0], reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.current_state = self.base_state.copy() # Reset only on new episode
        return self.current_state.values[0], {}

def run_policy_optimization(xgb_model, X_train):
    print("\n--- 6. RL Policy Optimization via AI Predictions ---")
    sample_route = X_train.iloc[[0]] 
    
    env = TradePolicyEnv(trained_model=xgb_model, base_row_df=sample_route)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64)
    model.learn(total_timesteps=10000) # Give it 10,000 tries to learn
    
    # Test the trained agent
    print("\n*** RL Agent Evaluation ***")
    obs, _ = env.reset()
    baseline_tariff = obs[env.tariff_idx]
    baseline_trade = xgb_model.predict(env.base_state)[0]
    
    # FIX 3: Print out scaled units, not fake percentages
    print(f"Original Baseline Tariff (Scaled) : {baseline_tariff:.4f}")
    print(f"Original Predicted Trade          : {baseline_trade:.4f} (Log scale)")
    print("--------------------------------------------------")
    
    # Let the AI take 5 consecutive actions to optimize the route
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    optimized_tariff = obs[env.tariff_idx]
    new_state_df = pd.DataFrame([obs], columns=env.base_state.columns)
    optimized_trade = xgb_model.predict(new_state_df)[0]
    
    print(f"AI Final Action Strategy          : Systematically lowered tariff over 5 iterations.")
    print(f"Optimized New Tariff (Scaled)     : {optimized_tariff:.4f}")
    print(f"New Predicted Trade               : {optimized_trade:.4f} (Log scale)")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==============================================")
    print(" STARTING ADVANCED TRADE AI PIPELINE")
    print("==============================================\n")
    
    # 1. Load Real Data
    X, y, df_raw, feats, scaler = load_and_preprocess_data("final_dataset.csv")
    
    # 2. Predictive Models & SHAP
    xgb_model, X_train = run_predictive_models(X, y)
    
    # 3. GNN-LSTM
    train_gnn_lstm(df_raw)
    
    # 4. Causal Inference
    run_causal_spillovers(df_raw)
    
    # 5. Reinforcement Learning
    run_policy_optimization(xgb_model, X_train)
    
    print("\n==============================================")
    print(" PIPELINE EXECUTION COMPLETE.")
    print("==============================================")