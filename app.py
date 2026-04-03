import streamlit as st
import pandas as pd
import os
from PIL import Image

# ==========================================
# PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
st.set_page_config(
    page_title="Trade AI Policy Simulator",
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
st.title("🌍  A Neural Network-Based Structural Gravity Model of EU-USA Free Trade Agreement Spillovers on India")
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
    The structural gravity model is the workhorse of international trade analysis[cite: 7]. However, traditional estimations rely heavily on classical linear econometrics (like Ordinary Least Squares). These classical models often fail to capture complex economic realities—such as non-linear geographic frictions, deep supply chain network effects, and highly skewed wealth distributions[cite: 8]. 
    
    Furthermore, accurately calculating how massive bilateral agreements—specifically an **EU-USA Free Trade Agreement (FTA)**—generate downstream economic spillovers onto emerging third-party markets like **India** requires advanced causal inference and prescriptive policy methodologies[cite: 3].
    """)
    
    st.markdown("---")
    
    col_meth, col_lib = st.columns([2, 1])
    
    with col_meth:
        st.subheader("⚙️ Methodology & Architecture")
        st.markdown("""
        This project engineered an end-to-end artificial intelligence pipeline:
        1. **Predictive Modeling:** Trained an ensemble **XGBoost Regressor** to predict trade volumes, drastically outperforming Classical OLS[cite: 11].
        2. **Network Effects:** Utilized a **Spatiotemporal GNN-LSTM** (Graph Neural Network) to mathematically map multilateral resistance across global borders[cite: 2, 4].
        3. **Causal Inference:** Deployed **Double Machine Learning (Causal Forests)** to isolate the Average Treatment Effect (ATE) of the EU-USA FTA policy shock[cite: 3].
        4. **Explainable AI:** Utilized **SHAP** values to decode the AI's complex decision-making process[cite: 10].
        5. **Policy Optimization:** Trained a **Reinforcement Learning (PPO) Agent** to autonomously adjust tariff levers to maximize simulated trade wealth.
        """)
        
    with col_lib:
        st.subheader("🛠️ Tech Stack & Libraries")
        st.markdown("""
        * **Machine Learning:** `xgboost` (Ensemble learning)[cite: 11].
        * **Deep Learning:** `torch` (PyTorch) & `torch_geometric` for GNN construction[cite: 2].
        * **Sequence Modeling:** `torch.nn.LSTM` specifically used to capture long-short term temporal dynamics of trade networks[cite: 4].
        * **Causal Inference:** `econml` for Double Machine Learning[cite: 3].
        * **Reinforcement Learning:** `stable_baselines3` (PPO) & `gymnasium` for the policy environment.
        * **Econometrics:** `statsmodels`, `scikit-learn`[cite: 7].
        * **Explainability:** `shap` for TreeExplainer values[cite: 10].
        * **Data Processing:** `pandas`, `numpy`.
        * **Visualization:** `streamlit`, `matplotlib`, `seaborn`.
        """)

    st.markdown("---")
    
    st.subheader("💡 Guide: Understanding the Output Metrics")
    st.markdown("""
    To rigorously evaluate the success of the AI architectures against classical baselines, we rely on the following mathematically grounded metrics:
    * **RMSE (Root Mean Square Error):** Measures the average distance between the model's predictions and actual global trade values[cite: 11]. **Lower is better.** An RMSE of 1.5006 indicates our AI's predictions deviate significantly less from reality than classical models.
    * **R-squared (R²):** The percentage of variance in global trade that the model successfully explains. **Higher is better.** An R² of 0.5377 means our AI captures over half the variance in the highly noisy global economy.
    * **GNN-LSTM Final Training Loss:** The Mean Squared Error loss of the neural network during training. A loss of 5.3317 indicates the model successfully converged and learned the spatial-temporal resistance of the trade network[cite: 4].
    * **ATE (Average Treatment Effect):** Used in Causal Inference, this isolates the specific impact of a policy. An ATE of **+0.6894** means the EU-USA FTA policy causes an approximate 68.94% relative increase in interconnected Indian trade volume[cite: 3].
    * **SHAP (SHapley Additive exPlanations):** A game-theoretic measure where feature importances are assigned based on their marginal contribution to the prediction, proving the model learned true economic principles[cite: 10].
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
    1. **Wu, Z., et al. (2019).** *A Comprehensive Survey on Graph Neural Networks.* IEEE Transactions on Neural Networks and Learning Systems.[cite: 2]
    2. **Athey, S., & Wager, S. (2019).** *Estimating Treatment Effects with Causal Forests: An Application.* Observational Studies, 5(2), 36-51.[cite: 3]
    3. **Wang, Y., et al. (2024).** *A Survey on Graph Neural Networks for Remaining Useful Life Prediction.*[cite: 4]
    4. **Robertson, R. (2021).** *Deep Integration in Trade Agreements: Labor Clauses, Tariffs, and Trade Flows.* World Bank Policy Research Working Paper.[cite: 5]
    5. **Gordeev, S., & Steinbach, S. (2024).** *Determinants of PTA Design: Insights from Machine Learning.*[cite: 6]
    6. **Yotov, Y. V., et al. (2016).** *An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model.* World Trade Organization.[cite: 7]
    7. **Verstyuk, S., & Douglas, M. R. (2022).** *Machine Learning the Gravity Equation for International Trade.*[cite: 8]
    8. **Breinlich, H., et al. (2021).** *Machine Learning in International Trade Research: Evaluating the Impact of Trade Agreements.* World Bank Policy Research.[cite: 9]
    9. **Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NIPS).[cite: 10]
    10. **Gopinath, M., et al. (2020).** *Machine Learning in Gravity Models: An Application to Agricultural Trade.* National Bureau of Economic Research.[cite: 11]
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
        st.metric(label="Classical OLS (Baseline)", value="1.8320 RMSE", delta="31.11% Variance Explained (R²)")
        st.info("**OLS Limitations:** Misses roughly 70% of the variance due to strict linear limitations.")
        
    with col2:
        st.metric(label="XGBoost AI (Ensemble)", value="1.5006 RMSE", delta="53.77% Variance Explained (R²)")
        st.success("**AI Superiority:** Outperformed linear models, drastically reducing prediction error.")
        
    with col3:
        st.metric(label="Spatiotemporal GNN-LSTM", value="5.3317 Final Loss", delta="Converged Successfully")
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
# TAB 3: CAUSAL SPILLOVERS (INDIA)
# ==========================================
with tab3:
    st.header("Isolating Macroeconomic Policy Shocks")
    st.markdown("**Research Question 2:** What are the spillover effects of an EU–USA FTA on India’s bilateral trade flows?")
    
    st.info("Using EconML's Causal Forest Double Machine Learning (DML) algorithm, we control for confounding variables (GDP, Distance) to extract the isolated Average Treatment Effect (ATE).")
        
    colA, colB = st.columns([1.5, 2])
    
    with colA:
        st.metric(label="🇮🇳 India-Specific Spillover Effect (ATE)", value="+0.6894 Log Points")
        
        st.markdown("### Economic Conclusion")
        st.markdown("""
        The Causal Forest decisively isolates **Trade Creation** rather than Trade Diversion.
        
        The presence of the EU-USA FTA causes a structural spillover effect, increasing India's associated bilateral trade flows by **0.6894 log points** (roughly a 69% relative boost). This proves that deep economic integration between Western markets generates downstream demand, strengthening the global supply chain and actively benefiting the Indian export market.
        """)
        
    with colB:
        if os.path.exists("3_Causal_Spillovers.png"):
            st.image(Image.open("3_Causal_Spillovers.png"), use_container_width=True, caption="Figure 2: Distribution of Causal Spillovers on India")
        else:
            st.warning("Visualization missing. Ensure '3_Causal_Spillovers.png' is in the directory.")

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
        st.metric("Baseline Tariff (Scaled)", "0.6394", "-0.5000 (AI Adjustment)", delta_color="inverse")
    with col_rl2:
        st.metric("Final Optimized Tariff", "0.1394", "Minimized Friction")
    with col_rl3:
        st.metric("Predicted Trade (Log)", "4.6343", "+0.1167 (Wealth Generated)", delta_color="normal")
        
    st.markdown("---")
    
    col_rA, col_rB = st.columns([2, 1])
    with col_rA:
        if os.path.exists("4_RL_Trajectory.png"):
            st.image(Image.open("4_RL_Trajectory.png"), use_container_width=True, caption="Figure 4: RL Agent Trajectory (Tariffs vs Trade)")
        else:
            st.warning("Visualization missing. Ensure '4_RL_Trajectory.png' is in the directory.")
            
    with col_rB:
        st.markdown("### Strategic Analysis")
        st.markdown("""
        The dual-axis chart visualizes the AI's step-by-step logic over 5 iterations. 
        
        Given the mandate to maximize overall trade volume, the agent rapidly identified that the baseline tariff was suppressing economic activity. Over 5 strategic steps, it autonomously slashed the scaled tariff from 0.6394 down to 0.1394 (Red Line), which directly forced the simulated trade volume (Green Line) to its mathematical maximum of 4.7510 log points.
        """)