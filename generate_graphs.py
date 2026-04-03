import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Set Seaborn theme for professional academic plots
sns.set_theme(style="whitegrid")

# Create a directory to save the plots so they don't clutter your folder
if not os.path.exists("presentation_plots"):
    os.makedirs("presentation_plots")

# ==========================================
# 1. LOAD DATA & XGBOOST (For SHAP)
# ==========================================
print("Generating Plot 1: SHAP Feature Importance...")
df = pd.read_csv("final_dataset.csv")
base_features = ['distance', 'gdp_o', 'gdp_d', 'eu_usa_fta', 'tariff']
df_encoded = pd.get_dummies(df, columns=['country_o', 'country_d', 'year', 'sector'], drop_first=True)
features = base_features + [col for col in df_encoded.columns if 'country_o_' in col or 'country_d_' in col or 'year_' in col or 'sector_' in col]

X = pd.DataFrame(StandardScaler().fit_transform(df_encoded[features]), columns=features)
y = df_encoded['log_trade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_train)

plt.figure(figsize=(10, 6))
plt.title('SHAP Feature Importance: AI Macroeconomic Logic', fontsize=14, fontweight='bold', pad=20)
shap.summary_plot(shap_values, X_train, max_display=10, show=False)
plt.tight_layout()
plt.savefig("presentation_plots/1_SHAP_Feature_Importance.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. GNN-LSTM (For Loss Curve)
# ==========================================
print("Generating Plot 2: GNN-LSTM Convergence Curve...")
class GNN_LSTM_Gravity(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GNN_LSTM_Gravity, self).__init__()
        self.gcn = GCNConv(num_node_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index):
        node_embeddings = self.gcn(node_features, edge_index).relu()
        edge_features = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1).unsqueeze(1) 
        lstm_out, _ = self.lstm(edge_features)
        return self.fc(lstm_out.squeeze(1))

countries = pd.concat([df['country_o'], df['country_d']]).unique()
country_to_id = {c: i for i, c in enumerate(countries)}
edge_index = torch.tensor(np.vstack((df['country_o'].map(country_to_id).values, df['country_d'].map(country_to_id).values)), dtype=torch.long)
node_features = torch.ones((len(countries), 1), dtype=torch.float32)
y_target = torch.tensor(df['log_trade'].values, dtype=torch.float32).view(-1, 1)

model = GNN_LSTM_Gravity(1, 16)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

loss_history = []
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model(node_features, edge_index), y_target)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 51), y=loss_history, color='purple', linewidth=2.5)
plt.title('GNN-LSTM Training Convergence (Multilateral Resistance)', fontsize=14, fontweight='bold')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (Loss)', fontsize=12)
plt.tight_layout()
plt.savefig("presentation_plots/2_GNN_Loss_Curve.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. CAUSAL FOREST (For Spillover Distribution)
# ==========================================
print("Generating Plot 3: Causal Spillover Distribution...")
est = CausalForestDML(discrete_treatment=True, random_state=42)
X_confounders = df[['distance', 'gdp_o', 'gdp_d']]
est.fit(df['log_trade'], df['eu_usa_fta'], X=X_confounders, W=None)

india_effects = est.effect(X_confounders[(df['country_o'] == 'India') | (df['country_d'] == 'India')])
ate = np.mean(india_effects)

plt.figure(figsize=(8, 5))
sns.histplot(india_effects, kde=True, color='teal', bins=30, line_kws={'linewidth': 2})
plt.axvline(ate, color='red', linestyle='--', linewidth=2.5, label=f'Mean ATE: +{ate:.4f}')
plt.title('Distribution of EU-USA FTA Causal Spillovers on India', fontsize=14, fontweight='bold')
plt.xlabel('Isolated Treatment Effect (Log Trade Points)', fontsize=12)
plt.ylabel('Number of Trade Routes', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("presentation_plots/3_Causal_Spillovers.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. REINFORCEMENT LEARNING (For Policy Trajectory)
# ==========================================
print("Generating Plot 4: RL Policy Optimization Trajectory...")
class TradePolicyEnv(gym.Env):
    def __init__(self, trained_model, base_row_df):
        super().__init__()
        self.model = trained_model
        self.base_state = base_row_df.copy()
        self.current_state = base_row_df.copy()
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.base_state.shape[1],), dtype=np.float32)
        self.steps = 0
        self.tariff_idx = self.base_state.columns.get_loc('tariff')

    def step(self, action):
        self.current_state.iloc[0, self.tariff_idx] += action[0]
        reward = self.model.predict(self.current_state)[0]
        self.steps += 1
        return self.current_state.values[0], reward, self.steps >= 20, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.current_state = self.base_state.copy() 
        return self.current_state.values[0], {}

env = TradePolicyEnv(xgb_model, X_train.iloc[[0]])
rl_model = PPO("MlpPolicy", env, verbose=0, n_steps=64)
rl_model.learn(total_timesteps=10000)

obs, _ = env.reset()
tariff_history = [obs[env.tariff_idx]]
trade_history = [xgb_model.predict(env.base_state)[0]]

for _ in range(5):
    action, _ = rl_model.predict(obs, deterministic=True)
    obs, _, _, _, _ = env.step(action)
    tariff_history.append(obs[env.tariff_idx])
    trade_history.append(xgb_model.predict(pd.DataFrame([obs], columns=env.base_state.columns))[0])

fig, ax1 = plt.subplots(figsize=(9, 5))
color = 'tab:red'
ax1.set_xlabel('RL Agent Optimization Steps', fontsize=12)
ax1.set_ylabel('Tariff Level (Scaled)', color=color, fontsize=12, fontweight='bold')
ax1.plot(range(6), tariff_history, color=color, marker='o', linewidth=3, markersize=8, label="Tariff Policy")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('Predicted Trade Volume (Log)', color=color, fontsize=12, fontweight='bold')
ax2.plot(range(6), trade_history, color=color, marker='s', linewidth=3, markersize=8, label="Predicted Trade")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Autonomous RL Policy Optimization: Minimizing Tariffs to Maximize Trade', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig("presentation_plots/4_RL_Trajectory.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nSUCCESS! All presentation plots have been saved to the 'presentation_plots' folder.")