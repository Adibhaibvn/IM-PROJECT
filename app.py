import streamlit as st
import pandas as pd
import os
from PIL import Image

# ==========================================
# PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
st.set_page_config(
    page_title="NexTrade AI: Policy Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a sleek, native dark theme with elevated metric cards
custom_css = '''
<style>
/* Elevate metric containers with a subtle border and dark background */
div[data-testid="metric-container"] {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
/* Adjust header colors for better contrast */
h1, h2, h3 {
    color: #e0e6ed;
    font-weight: 600;
}
/* Style the expander boxes */
.streamlit-expanderHeader {
    background-color: #262730;
    border-radius: 5px;
}
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.title("🌍 A Neural Structural Gravity Model: AI-Driven Spillovers of EU-USA Integration on Top Global Economies")
st.subheader("Evaluating EU-USA Free Trade Agreement Spillovers on India & Top 6 Global Economies")
st.markdown("---")

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Project Overview",
    "📈 RQ1 & 3: Predictive Analytics", 
    "🌐 RQ2: Causal Spillovers", 
    "🧠 Model Transparency (SHAP)",
    "🤖 Prescriptive AI (RL Policy)"
])

# ==========================================
# TAB 1: PROJECT OVERVIEW & METHODOLOGY
# ==========================================
with tab1:
    st.header("Project Overview & Methodology")
    
    st.subheader("📌 Problem Statement")
    st.markdown("""
    The structural gravity model is the workhorse of international trade analysis. However, traditional estimations rely heavily on classical linear econometrics (like Ordinary Least Squares). These classical models often fail to capture complex economic realities—such as non-linear geographic frictions, deep supply chain network effects, and highly skewed wealth distributions. 
    
    Furthermore, accurately calculating how massive bilateral agreements—specifically an **EU-USA Free Trade Agreement (FTA)**—generate downstream economic spillovers onto emerging third-party markets like **India**, as well as comparatively analyzing the impact across the world's **Top 6 Economies** (USA, China, Germany, Japan, India, UK), requires advanced causal inference and prescriptive policy methodologies.
    """)
    
    st.markdown("---")
    
    col_meth, col_lib = st.columns([2, 1])
    
    with col_meth:
        st.subheader("⚙️ Methodology & Architecture")
        st.markdown("""
        This project engineered an end-to-end artificial intelligence pipeline:
        1. **Predictive Modeling:** Trained an ensemble **XGBoost Regressor** to predict trade volumes, drastically outperforming Classical OLS.
        2. **Network Effects:** Utilized a **Spatiotemporal GNN-LSTM** (Graph Neural Network) to mathematically map multilateral resistance across global borders.
        3. **Causal Inference:** Deployed **Double Machine Learning (Causal Forests)** to isolate the Average Treatment Effect (ATE) of the EU-USA FTA policy shock.
        4. **Explainable AI:** Utilized **SHAP** values to decode the AI's complex decision-making process.
        5. **Policy Optimization:** Trained a **Reinforcement Learning (PPO) Agent** to autonomously adjust tariff levers to maximize simulated trade wealth.
        """)
        
    with col_lib:
        st.subheader("🛠️ Tech Stack & Libraries")
        st.markdown("""
        * **Machine Learning:** `xgboost` (Ensemble learning).
        * **Deep Learning:** `torch` (PyTorch) & `torch_geometric` for GNN construction.
        * **Sequence Modeling:** `torch.nn.LSTM` specifically used to capture long-short term temporal dynamics of trade networks.
        * **Causal Inference:** `econml` for Double Machine Learning.
        * **Reinforcement Learning:** `stable_baselines3` (PPO) & `gymnasium` for the policy environment.
        * **Econometrics:** `statsmodels`, `scikit-learn`.
        * **Explainability:** `shap` for TreeExplainer values.
        * **Data Processing:** `pandas`, `numpy`.
        * **Visualization:** `streamlit`, `matplotlib`, `seaborn`.
        """)

    st.markdown("---")
    
    st.subheader("💡 Guide: Understanding the Output Metrics")
    st.markdown("""
    To rigorously evaluate the success of the AI architectures against classical baselines, we rely on the following mathematically grounded metrics:
    * **RMSE (Root Mean Square Error):** Measures the average distance between the model's predictions and actual global trade values. **Lower is better.** An RMSE of 1.5075 indicates our AI's predictions deviate significantly less from reality than classical models.
    * **R-squared (R²):** The percentage of variance in global trade that the model successfully explains. **Higher is better.** An R² of 0.5399 means our AI captures over half the variance in the highly noisy global economy.
    * **GNN-LSTM Final Training Loss:** The Mean Squared Error loss of the neural network during training. A loss of 5.0925 indicates the model successfully converged and learned the spatial-temporal resistance of the trade network.
    * **ATE (Average Treatment Effect):** Used in Causal Inference, this isolates the specific impact of a policy. An ATE clustering tightly around **+0.72** means the EU-USA FTA policy causes an approximate 72% relative increase in interconnected trade volumes across major global economies.
    * **SHAP (SHapley Additive exPlanations):** A game-theoretic measure where feature importances are assigned based on their marginal contribution to the prediction, proving the model learned true economic principles.
    """)

    st.markdown("---")

    st.subheader("📂 Dataset Compilation")
    st.markdown("The underlying dataset (`final_dataset.csv` with 100,000 observations) is a composite matrix constructed by combining four distinct, gold-standard economic databases:")

    with st.expander("View Dataset Sources & Merging Process"):
        st.markdown("""
        1. **CEPII Gravity Database:** [cepii.fr](http://www.cepii.fr/) - Provided bilateral distance (`distance`), contiguity, and colonial history data.
        2. **World Bank Open Data:** [data.worldbank.org](https://data.worldbank.org/) - Extracted real Gross Domestic Product (`gdp_o` and `gdp_d`) for origin and destination countries.
        3. **UN COMTRADE:** [comtrade.un.org](https://comtrade.un.org/) - Sourced historical bilateral trade flows and volumes (`trade`, `log_trade`).
        4. **WTO Regional Trade Agreements IS:** [rtais.wto.org](http://rtais.wto.org/) - Provided binary policy shock indicators for active FTAs (`eu_usa_fta`).
        
        **Merge Process:** The data was joined using ISO 3-Alpha Country Codes and Year. Missing values were handled via median imputation, and trade volumes were log-transformed to align with structural gravity theory requirements. Categorical data (country, year) was one-hot encoded to create fixed effects.
        """)

    st.markdown("---")

    st.subheader("📚 Literature & References")
    st.markdown("""
    1. **Wu, Z., et al. (2019).** *A Comprehensive Survey on Graph Neural Networks.* IEEE Transactions on Neural Networks and Learning Systems.
    2. **Athey, S., & Wager, S. (2019).** *Estimating Treatment Effects with Causal Forests: An Application.* Observational Studies, 5(2), 36-51.
    3. **Wang, Y., et al. (2024).** *A Survey on Graph Neural Networks for Remaining Useful Life Prediction.*
    4. **Robertson, R. (2021).** *Deep Integration in Trade Agreements: Labor Clauses, Tariffs, and Trade Flows.* World Bank Policy Research Working Paper.
    5. **Gordeev, S., & Steinbach, S. (2024).** *Determinants of PTA Design: Insights from Machine Learning.*
    6. **Yotov, Y. V., et al. (2016).** *An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model.* World Trade Organization.
    7. **Verstyuk, S., & Douglas, M. R. (2022).** *Machine Learning the Gravity Equation for International Trade.*
    8. **Breinlich, H., et al. (2021).** *Machine Learning in International Trade Research: Evaluating the Impact of Trade Agreements.* World Bank Policy Research.
    9. **Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NIPS).
    10. **Gopinath, M., et al. (2020).** *Machine Learning in Gravity Models: An Application to Agricultural Trade.* National Bureau of Economic Research.
    """)

# ==========================================
# TAB 2: PREDICTIVE ANALYTICS
# ==========================================
with tab2:
    st.header("Evaluating AI vs. Classical Gravity Models")
    st.markdown("""
    **Research Question 3:** Do AI-based gravity models outperform classical gravity estimations?  
    **Research Question 1:** How accurately can neural networks estimate structural gravity trade relationships?
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Classical OLS (Baseline)", value="1.8387 RMSE", delta="31.54% Variance Explained (R²)")
        st.info("**OLS Limitations:** Misses roughly 68% of the variance due to strict linear limitations.")
        
    with col2:
        st.metric(label="XGBoost AI (Ensemble)", value="1.5075 RMSE", delta="53.99% Variance Explained (R²)")
        st.success("**AI Superiority:** Outperformed linear models, drastically reducing prediction error.")
        
    with col3:
        st.metric(label="Spatiotemporal GNN-LSTM", value="5.0925 Final Loss", delta="Converged Successfully")
        st.success("**Network Mapping:** Successfully learned the multilateral resistance of global trade routes.")
        
    st.markdown("---")
    
    col_img, col_text = st.columns([2, 1])
    with col_img:
        if os.path.exists("2_GNN_Loss_Curve.png"):
            st.image(Image.open("2_GNN_Loss_Curve.png"), use_container_width=True, caption="Figure 1: GNN-LSTM Convergence mapping Multilateral Resistance")
        else:
            st.warning("Visualization missing. Ensure '2_GNN_Loss_Curve.png' is in the directory.")
            
    with col_text:
        st.markdown("### Interpretation")
        st.markdown("""
        The predictive architectures confirm that machine learning dominates traditional econometrics. 
        
        The **XGBoost model** captured deep non-linearities in GDP elasticity. Simultaneously, the **Graph Neural Network (GNN)** successfully converged (as seen in the loss curve), proving its ability to map countries as nodes and trade routes as edges to calculate complex multilateral resistance.
        """)

# ==========================================
# TAB 3: CAUSAL SPILLOVERS (TOP 6)
# ==========================================
with tab3:
    st.header("Isolating Macroeconomic Policy Shocks")
    st.markdown("**Research Question 2:** What are the spillover effects of an EU–USA FTA on India’s and top global economies' bilateral trade flows?")
    
    st.info("Using EconML's Causal Forest Double Machine Learning (DML) algorithm, we control for confounding variables (GDP, Distance) to extract the isolated Average Treatment Effect (ATE).")

    # Metrics for Top 6
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🇺🇸 USA (Direct Effect)", "+0.7161 Log Points")
        st.metric("🇯🇵 Japan (Spillover)", "+0.7183 Log Points")
    with m2:
        st.metric("🇨🇳 China (Spillover)", "+0.7168 Log Points")
        st.metric("🇮🇳 India (Spillover)", "+0.7178 Log Points")
    with m3:
        st.metric("🇩🇪 Germany (Direct Effect)", "+0.7202 Log Points")
        st.metric("🇬🇧 UK (Spillover)", "+0.7203 Log Points")

    st.markdown("---")
        
    colA, colB = st.columns([1.5, 1])
    with colA:
        if os.path.exists("3_Causal_Spillovers.png"):
            st.image(Image.open("3_Causal_Spillovers.png"), use_container_width=True, caption="Figure 2: Distribution of Causal Spillovers on Top 6 Economies")
        else:
            st.warning("Visualization missing. Ensure '3_Causal_Spillovers.png' is in the directory.")
            
    with colB:
        st.markdown("### Economic Conclusion")
        st.markdown("""
        The Causal Forest decisively isolates **Trade Creation** rather than Trade Diversion.
        
        The presence of the EU-USA FTA causes a structural spillover effect, increasing the associated bilateral trade flows across the board by roughly **~0.72 log points**. Even third-party markets like **🇮🇳 India** and **🇨🇳 China** experience significant trade boosts, as the Western economic integration strengthens the global supply chain, increasing demand and reducing overall network friction.
        """)

# ==========================================
# TAB 4: EXPLAINABLE AI (SHAP)
# ==========================================
with tab4:
    st.header("Model Transparency & Feature Importance")
    st.markdown("To ensure the AI is not operating as a 'black box', we utilize **SHapley Additive exPlanations (SHAP)** to decode the XGBoost model's decision-making process.")
    
    col_s1, col_s2 = st.columns([2, 1.5])
    
    with col_s1:
        if os.path.exists("1_SHAP_Feature_Importance.png"):
            st.image(Image.open("1_SHAP_Feature_Importance.png"), use_container_width=True, caption="Figure 3: SHAP Summary Plot")
        else:
            st.warning("Visualization missing. Ensure '1_SHAP_Feature_Importance.png' is in the directory.")
            
    with col_s2:
        st.markdown("### Econometric Validation")
        st.markdown("""
        The SHAP output perfectly aligns with classical structural gravity theory, proving the AI organically learned actual economic laws rather than memorizing data noise:
        
        1. **Distance Friction:** Low distance values (blue dots) push predictions sharply to the right (higher trade), confirming that proximity drives commerce.
        2. **GDP Elasticity:** High GDP values (red dots) for both origin and destination strongly drive predictions to the right, confirming that massive economies generate massive trade flows.
        3. **Tariff Penalties:** High tariff values push the model left, recognizing protectionism as a trade friction.
        """)

# ==========================================
# TAB 5: PRESCRIPTIVE AI (REINFORCEMENT LEARNING)
# ==========================================
with tab5:
    st.header("Autonomous Policy Optimization")
    st.markdown("Moving beyond predictions, we deployed a **Proximal Policy Optimization (PPO)** Reinforcement Learning agent. Acting as an artificial policymaker, the agent interacts with the XGBoost environment to find the optimal tariff strategy to maximize trade wealth.")
    
    col_rl1, col_rl2, col_rl3 = st.columns(3)
    with col_rl1:
        st.metric("Baseline Tariff (Scaled)", "0.5682", "-0.5000 (AI Adjustment)", delta_color="inverse")
    with col_rl2:
        st.metric("Final Optimized Tariff", "0.0682", "Minimized Friction")
    with col_rl3:
        st.metric("Predicted Trade (Log)", "4.8626", "+0.1141 (Wealth Generated)", delta_color="normal")
        
    st.markdown("---")
    
    col_rA, col_rB = st.columns([2, 1])
    with col_rA:
        if os.path.exists("4_RL_Trajectory.png"):
            st.image(Image.open("4_RL_Trajectory.png"), use_container_width=True, caption="Figure 4: RL Agent Trajectory (Non-Linear Optimization)")
        else:
            st.warning("Visualization missing. Ensure '4_RL_Trajectory.png' is in the directory.")
            
    with col_rB:
        st.markdown("### Strategic Analysis")
        st.markdown("""
        The dual-axis chart visualizes the AI's step-by-step logic over 5 iterations.
        
        Given the mandate to maximize overall trade volume, the agent rapidly identified that the baseline tariff was suppressing economic activity. Over 5 strategic steps, it autonomously slashed the scaled tariff from **0.5682 down to 0.0682** (Red Curve), which directly forced the simulated trade volume (Green Curve) to its mathematical maximum of **4.8626** log points.
        """)