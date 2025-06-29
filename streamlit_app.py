
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
warnings.filterwarnings('ignore')

# MLflow and experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.warning("MLflow not installed. Some features may be limited.")

# PyCaret imports
try:
    from pycaret.classification import setup as cls_setup, compare_models as cls_compare, create_model as cls_create
    from pycaret.classification import tune_model as cls_tune, finalize_model as cls_finalize, predict_model as cls_predict
    from pycaret.classification import pull as cls_pull, plot_model as cls_plot, evaluate_model as cls_evaluate
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create
    from pycaret.regression import tune_model as reg_tune, finalize_model as reg_finalize, predict_model as reg_predict
    from pycaret.regression import pull as reg_pull, plot_model as reg_plot, evaluate_model as reg_evaluate
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("PyCaret not installed. AutoML features will be limited.")

# Data profiling
#try:
#    from ydata_profiling import ProfileReport
#    from streamlit_pandas_profiling import st_profile_report
#    PROFILING_AVAILABLE = True
#except ImportError:
#    PROFILING_AVAILABLE = False

# PyTorch for deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ================== UPLOADING THE DATA ==================

df = pd.read_csv("ocd_patient_dataset.csv")

# ================== CUSTOM CSS & STYLING ==================
st.set_page_config(
    page_title="OCD Diagnosing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

st.markdown("""
<style>
    /* Main styling */
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Arial', sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content { 
        background: linear-gradient(180deg, #2C3E50, #3498DB);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning {
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 0;">OCD Diagnosing</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
        Test different factors on their predicibility of OCD using ML Models
    </p>
</div>
""", unsafe_allow_html=True)

# ================== AUTHENTICATION ==================
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.sidebar:
            st.header("🔒 Authentication")
            password = st.text_input("Enter Password", type="password", key="auth_password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔑 Login", key="login_btn"):
                    if password == "diagnosis testing":
                        st.session_state.authenticated = True
                        st.success("✅ Access Granted!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect Password")
            with col2:
                if st.button("👤 Demo Mode", key="demo_btn"):
                    st.session_state.authenticated = True
                    st.session_state.demo_mode = True
                    st.info("📊 Demo Mode Activated")
                    st.rerun()
        
        st.info("🔐 Please authenticate to access the application")
        st.stop()

check_authentication()

# ================== SESSION STATE INITIALIZATION ==================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'pycaret_setup_done' not in st.session_state:
    st.session_state.pycaret_setup_done = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'dl_models' not in st.session_state:
    st.session_state.dl_models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}

# ================== SIDEBAR NAVIGATION ==================
#PAGES
st.sidebar.title("🧭 Navigation")
pages = [
    "🏠 Home",
    "📊 Data Viz",
    "🤖 Logistical Regression",
    "🌳 Decision Tree",
    "Model Comparison"
]
#"📋 MLflow Tracking",

selected_page = st.sidebar.selectbox("Select Page", pages, key="page_selector")


# ================== PAGE CONTENT ==================

if selected_page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## OCD Diagnosis Deep Dive
        
        About the data
            There is an ongoing issue of misdiagnosis among mental illnesses, like OCD. Machine Learning has the ability to make diagnosing easier. 
            This app aims to use factors such as OCD Diagnosis Date, Duration of Symptoms in months, Previous Diagnoses, Family History of OCD, 
            Obsession Type, and Compulsion Type, to see if we accurately predict the obession and/or compulsion type. 

        """)
        
    st.table(df.head())
            
        
#DATA VIZ
elif selected_page == "📊 Data Viz":
    filtds = df.drop(columns=["Patient ID"])

    col_x = st.selectbox("Select X-axis variable (group by)", filtds.columns)
    col_y = st.selectbox("Select Y-axis variable (numeric)", filtds.columns)

    tab1, tab2, tab3, tab4 = st.tabs(["Box plot", "Bar Chart 📊","Line Chart 📈","Correlation Heatmap 🔥",])

    with tab1:
        st.subheader("Box plot")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col_x, y=col_y, ax=ax)
        ax.set_title(f"{col_y} by {col_x}")
        st.pyplot(fig)

    with tab2:
        st.subheader("Bar Chart")
        st.bar_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)

    with tab3:
        st.subheader("Line Chart")
        st.line_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)

    with tab4:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        ct = pd.crosstab(df[col_x], df[col_y])
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
        plt.xlabel(col_y)
        plt.ylabel(col_x)
        plt.title(f"Heatmap of {col_x} vs {col_y}")

#LOG REG
elif selected_page == "🤖 Logistical Regression":
    st.header("Running a Logistical Regression on our data...")
    
    target_variable = st.selectbox(
                "Select which variable you would like to predict:",
                ["Y-BOCS Score (Obsessions)", "Y-BOCS Score (Compulsions)", "Depression Diagnosis", "Anxiety Diagnosis"]
            )
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try: 
                df_sampled = df.sample(n=500, random_state=42)
                X = df_sampled.drop(columns=[target_variable])
                X = X.select_dtypes(include=["number"])
                y = df_sampled[target_variable]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression()
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)

                st.write("### Accuracy:", accuracy_score(y_test, y_pred))
                st.write("### Classification Report:")
                st.text(classification_report(y_test, y_pred))

                st.subheader("📊 SHAP Summary Plot for Logistic Regression")
                fig2 = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(bbox_inches='tight')
                plt.clf()
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

elif selected_page == "🌳 Decision Tree":
    st.header("Predictions via decision tree...")

    target_variable = st.selectbox(
                "Select which variable you would like to predict:",
                ["Y-BOCS Score (Obsessions)", "Y-BOCS Score (Compulsions)", "Depression Diagnosis", "Anxiety Diagnosis"]
            )
    
    X = df.drop(columns=[target_variable])
    X = X.select_dtypes(include=["number"])
    X = X.fillna(X.mean())
    y = df[target_variable]

    # split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)  # You can adjust depth
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    st.write("### 🌳 Decision Tree Performance")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    explainer = shap.Explainer(dt_model, X_test)
    shap_values = explainer(X_test)

    # Summary plot (global feature importance)
    st.subheader("📊 SHAP Summary Plot")
    fig1 = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()


elif selected_page == "Model Comparison":
    st.header("Decision Tree vs Logistic Regression")

    target_variable = st.selectbox(
        "🎯 Select the target variable to predict:",
        ["Y-BOCS Score (Obsessions)", "Y-BOCS Score (Compulsions)", "Depression Diagnosis", "Anxiety Diagnosis"])

    df_sampled = df.sample(n=500, random_state=42)
    X = df_sampled.drop(columns=[target_variable])
    X = X.select_dtypes(include=["number"])
    X = X.fillna(X.mean())    
    y = df_sampled[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    dtree = DecisionTreeClassifier(max_depth=5, random_state=42)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("OCD")

    col1, col2 = st.columns(2)
    with col1:
        with mlflow.start_run(run_name="Decision Tree"):
            dtree.fit(X_train, y_train)
            y_pred_tree = dtree.predict(X_test)
            y_proba_tree = dtree.predict_proba(X_test)[:, 1]

            st.markdown("### 🌿 Decision Tree")
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred_tree))
            st.text(classification_report(y_test, y_pred_tree))

            cm_tree = confusion_matrix(y_test, y_pred_tree)
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens', ax=ax1)
            ax1.set_title("Decision Tree Confusion Matrix")
            st.pyplot(fig1)
            plt.close(fig1)

            st.session_state.trained_models = st.session_state.get("trained_models", {})
            st.session_state.trained_models["Decision Tree"] = {
                "model": dtree,
                "features": X.columns.tolist(),
                "target": target_variable,
                "predictions": y_pred_tree,
                "y_test": y_test,
                "problem_type": "Classification"
            }


    with col2:
        with mlflow.start_run(run_name="Logistic Regression"):
            logreg.fit(X_train_scaled, y_train)
            y_pred_log = logreg.predict(X_test_scaled)
            y_proba_log = logreg.predict_proba(X_test_scaled)[:, 1]
        
            st.markdown("### 📈 Logistic Regression")
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred_log))
            st.text(classification_report(y_test, y_pred_log))

            cm_log = confusion_matrix(y_test, y_pred_log)
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title("Logistic Regression Confusion Matrix")
            st.pyplot(fig2)
            plt.close(fig2)

            st.session_state.trained_models = st.session_state.get("trained_models", {})
            st.session_state.trained_models["Logistic Regression"] = {
                "model": logreg,
                "features": X.columns.tolist(),
                "target": target_variable,
                "predictions": y_pred_log,
                "y_test": y_test,
                "problem_type": "Classification"
            }



# elif selected_page == "📋 MLflow Tracking":
#     st.header("📋 MLflow Experiment Tracking")

#     # --- MLflow config section ---
#     st.subheader("⚙️ MLflow Configuration")
#     tracking_uri = st.text_input("🔗 Tracking URI", value="http://localhost:5000")
#     experiment_name = st.text_input("🧪 Experiment Name", value="my_local_experiment")

#     if st.button("🔧 Set MLflow Configuration"):
#         try:
#             mlflow.set_tracking_uri(tracking_uri)
#             mlflow.set_experiment(experiment_name)
#             st.success("✅ MLflow configured successfully!")
#         except Exception as e:
#             st.error(f"❌ Failed to set MLflow config: {str(e)}")

#     # --- Log trained model ---
#     st.subheader("📤 Log Trained Model to MLflow")

#     if st.session_state.get("trained_models"):
#         model_name = st.selectbox("Select a model to log:", list(st.session_state.trained_models.keys()))
#         if st.button("📥 Log This Model"):
#             model_data = st.session_state.trained_models[model_name]
#             try:
#                 with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
#                     # Log model
#                     mlflow.sklearn.log_model(model_data["model"], "model")

#                     # Log params
#                     mlflow.log_param("model_type", model_name)
#                     mlflow.log_param("target", model_data["target"])
#                     mlflow.log_param("features", len(model_data["features"]))

#                     # Log metrics
#                     y_test = model_data["y_test"]
#                     y_pred = model_data["predictions"]
#                     if model_data["problem_type"] == "Classification":
#                         acc = accuracy_score(y_test, y_pred)
#                         mlflow.log_metric("accuracy", acc)
#                     else:
#                         mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
#                         mlflow.log_metric("r2", r2_score(y_test, y_pred))
#                         mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))

#                     st.success("✅ Model logged to MLflow!")
#             except Exception as e:
#                 st.error(f"❌ Error logging model: {str(e)}")
#     else:
#         st.info("No models found. Train some models first!")

#     # --- View past runs ---
#     st.subheader("📈 Recent Experiment Runs")

#     if st.button("🔄 Refresh Runs"):
#         try:
#             runs_df = mlflow.search_runs(order_by=["start_time desc"])
#             if not runs_df.empty:
#                 st.dataframe(
#                     runs_df[[
#                         'run_id',
#                         'status',
#                         'start_time',
#                         'params.model_type',
#                         'params.target',
#                         'metrics.accuracy',  # This will show NaN for regression
#                         'metrics.mse',
#                         'metrics.r2'
#                     ]],
#                     use_container_width=True
#                 )
#             else:
#                 st.info("📊 No runs found.")
#         except Exception as e:
#             st.error(f"❌ Error fetching runs: {str(e)}")
    
    

# ================== SIDEBAR ! ==================

# Help section
st.sidebar.markdown("---")
st.sidebar.subheader("Where to go...")
st.sidebar.markdown("""
1. 🏠 Home
2. 📊 Data Viz
3. 🤖 Logistical Regression
4. 🌳 Decision Tree
5. Model Comparison

""")

#6. 📋 MLflow Tracking