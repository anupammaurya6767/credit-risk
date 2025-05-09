import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Credit Card Approval Prediction", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Credit Card Approval Prediction Dashboard</h1>", unsafe_allow_html=True)

# Initialize session state for storing models
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Function to load dataset
@st.cache_data
def load_credit_data():
    """Load the Credit Card Approval dataset"""
    np.random.seed(42)
    n_samples = 1000
    approval_rate = 0.6
    y = np.random.choice([0, 1], size=n_samples, p=[1 - approval_rate, approval_rate])
    approved_idx = np.where(y == 1)[0]
    denied_idx = np.where(y == 0)[0]
    
    # Generate numerical features
    income_total = np.zeros(n_samples)
    income_total[approved_idx] = np.random.normal(100000, 30000, len(approved_idx))
    income_total[denied_idx] = np.random.normal(50000, 20000, len(denied_idx))
    income_total = np.clip(income_total, 10000, None)
    
    employment_years = np.zeros(n_samples)
    employment_years[approved_idx] = np.random.normal(10, 5, len(approved_idx))
    employment_years[denied_idx] = np.random.normal(3, 2, len(denied_idx))
    employment_years = np.clip(employment_years, 0, None)
    
    age_years = np.zeros(n_samples)
    age_years[approved_idx] = np.random.normal(40, 10, len(approved_idx))
    age_years[denied_idx] = np.random.normal(30, 10, len(denied_idx))
    age_years = np.clip(age_years, 18, None)
    
    # Generate categorical features
    own_car = np.zeros(n_samples, dtype=object)
    own_car[approved_idx] = np.random.choice(['Yes', 'No'], len(approved_idx), p=[0.7, 0.3])
    own_car[denied_idx] = np.random.choice(['Yes', 'No'], len(denied_idx), p=[0.3, 0.7])
    
    own_realty = np.zeros(n_samples, dtype=object)
    own_realty[approved_idx] = np.random.choice(['Yes', 'No'], len(approved_idx), p=[0.8, 0.2])
    own_realty[denied_idx] = np.random.choice(['Yes', 'No'], len(denied_idx), p=[0.4, 0.6])
    
    # Create DataFrame
    df = pd.DataFrame({
        'income_total': income_total,
        'employment_years': employment_years,
        'age_years': age_years,
        'own_car': own_car,
        'own_realty': own_realty,
        'credit_risk': y
    })
    return df

# Function to preprocess data
def preprocess_data(df, target_column='credit_risk'):
    """Preprocess the data for modeling"""
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to train models
def train_models(X_train, y_train, preprocessor, dt_max_depth, rf_n_estimators, rf_max_depth):
    """Train decision tree, random forest, and meta learner models"""
    models = {}
    
    dt_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42))
    ])
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42
        ))
    ])
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    dt_preds = dt_model.predict_proba(X_train)[:, 1]
    rf_preds = rf_model.predict_proba(X_train)[:, 1]
    
    meta_features = np.column_stack((dt_preds, rf_preds))
    meta_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    meta_model.fit(meta_features, y_train)
    models['Meta Learner'] = meta_model
    
    return models

# Function to evaluate models
def evaluate_model(model, X_test, y_test, preprocessor=None, is_meta=False):
    """Evaluate the model and return metrics"""
    if is_meta:
        dt_preds = st.session_state.models['Decision Tree'].predict_proba(X_test)[:, 1]
        rf_preds = st.session_state.models['Random Forest'].predict_proba(X_test)[:, 1]
        meta_features = np.column_stack((dt_preds, rf_preds))
        y_pred = model.predict(meta_features)
        y_pred_proba = model.predict_proba(meta_features)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, report, conf_matrix, fpr, tpr, roc_auc

# Save models function
def save_models(models, preprocessor):
    """Save all models and preprocessor to disk"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(models['Decision Tree'], 'models/decision_tree_model.pkl')
    joblib.dump(models['Random Forest'], 'models/random_forest_model.pkl')
    joblib.dump(models['Meta Learner'], 'models/meta_learner_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    st.success("All models saved successfully!")

# Load models function
def load_saved_models():
    """Load saved models from disk"""
    try:
        models = {}
        models['Decision Tree'] = joblib.load('models/decision_tree_model.pkl')
        models['Random Forest'] = joblib.load('models/random_forest_model.pkl')
        models['Meta Learner'] = joblib.load('models/meta_learner_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        st.success("Models loaded successfully!")
        return models, preprocessor
    except:
        st.error("No saved models found. Please train models first.")
        return None, None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "Model Evaluation", "Prediction"])

# Load dataset
if st.session_state.dataset is None:
    st.session_state.dataset = load_credit_data()

# Data Overview Page
if page == "Data Overview":
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
    
    df = st.session_state.dataset
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{df.shape[0]}</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Number of Applications</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        approval_rate = df['credit_risk'].mean() * 100
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{approval_rate:.2f}%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Approval Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df.head(100))
    
    st.markdown("<h3 class='sub-header'>Data Summary</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        st.dataframe(df.select_dtypes(exclude=['object']).describe())
    
    with col2:
        st.subheader("Categorical Features")
        cat_summary = {col: df[col].value_counts().shape[0] for col in df.select_dtypes(include=['object']).columns}
        st.dataframe(pd.DataFrame(cat_summary.items(), columns=['Feature', 'Unique Values']))
    
    st.markdown("<h3 class='sub-header'>Data Visualizations</h3>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    if 'credit_risk' in numeric_cols:
        numeric_cols.remove('credit_risk')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Numeric Features")
        if numeric_cols:
            selected_num_col = st.selectbox("Select a numeric feature", numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=selected_num_col, hue='credit_risk', kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_num_col} by Approval Status")
            st.pyplot(fig)
    
    with col2:
        st.subheader("Distribution of Categorical Features")
        if categorical_cols:
            selected_cat_col = st.selectbox("Select a categorical feature", categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = df.groupby([selected_cat_col, 'credit_risk']).size().unstack(fill_value=0)
            counts.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Distribution of {selected_cat_col} by Approval Status")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Count")
            ax.legend(["Denied", "Approved"])
            st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(exclude=['object'])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric features to create a correlation matrix.")

# Model Training Page
elif page == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    df = st.session_state.dataset
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{df.shape[0]}</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Number of Applications</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        approval_rate = df['credit_risk'].mean() * 100
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{approval_rate:.2f}%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Approval Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        features_count = df.shape[1] - 1
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{features_count}</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Number of Features</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        dt_max_depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=20, value=5)
    
    with col2:
        rf_n_estimators = st.slider("Random Forest Number of Trees", min_value=50, max_value=300, value=100, step=10)
        rf_max_depth = st.slider("Random Forest Max Depth", min_value=2, max_value=20, value=10)
    
    if st.button("Train Models"):
        with st.spinner("Training models, please wait..."):
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
            
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            
            st.session_state.models = train_models(
                X_train, y_train, preprocessor, dt_max_depth, rf_n_estimators, rf_max_depth
            )
            st.session_state.trained = True
            
            st.success("Models trained successfully!")
    
    if st.session_state.trained:
        if st.button("Save Models"):
            save_models(st.session_state.models, st.session_state.preprocessor)
    else:
        st.info("Train models first before saving.")
    
    st.subheader("Load Pre-trained Models")
    if st.button("Load Saved Models"):
        models, preprocessor = load_saved_models()
        if models is not None:
            st.session_state.models = models
            st.session_state.preprocessor = preprocessor
            st.session_state.trained = True

# Model Evaluation Page
elif page == "Model Evaluation":
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("No trained models found. Please go to the Model Training page to train models first.")
    else:
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select a model to evaluate", model_names)
        
        model = st.session_state.models[selected_model]
        
        if selected_model == "Meta Learner":
            accuracy, report, conf_matrix, fpr, tpr, roc_auc = evaluate_model(
                model, st.session_state.X_test, st.session_state.y_test, is_meta=True
            )
        else:
            accuracy, report, conf_matrix, fpr, tpr, roc_auc = evaluate_model(
                model, st.session_state.X_test, st.session_state.y_test
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{accuracy*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Accuracy</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            precision = report['1']['precision']
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{precision*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Precision (Approved)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            recall = report['1']['recall']
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{recall*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Recall (Approved)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Denied', 'Approved'], 
                        yticklabels=['Denied', 'Approved'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)
        
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
        if 'macro avg' in report_df.index and 'weighted avg' in report_df.index:
            report_df = report_df.drop(['macro avg', 'weighted avg'])
        
        for col in report_df.columns:
            if col != 'support':
                report_df[col] = report_df[col].apply(lambda x: f"{x*100:.2f}%")
        
        report_df.index = report_df.index.map({'0': 'Denied', '1': 'Approved'})
        
        st.dataframe(report_df)
        
        st.subheader("Model Comparison")
        
        model_metrics = {}
        for name in model_names:
            current_model = st.session_state.models[name]
            if name == "Meta Learner":
                acc, _, _, _, _, auc_score = evaluate_model(
                    current_model, st.session_state.X_test, st.session_state.y_test, is_meta=True
                )
            else:
                acc, _, _, _, _, auc_score = evaluate_model(
                    current_model, st.session_state.X_test, st.session_state.y_test
                )
            model_metrics[name] = [acc, auc_score]
        
        comparison_df = pd.DataFrame(model_metrics, index=['Accuracy', 'AUC']).transpose()
        comparison_df['Accuracy'] = comparison_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
        comparison_df['AUC'] = comparison_df['AUC'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(comparison_df)
        
        if selected_model == "Random Forest":
            st.subheader("Feature Importance")
            try:
                rf_classifier = model.named_steps['classifier']
                preprocessor = model.named_steps['preprocessor']
                
                feature_names = []
                for name, _, cols in preprocessor.transformers_:
                    if name == 'cat':
                        cat_transformer = preprocessor.named_transformers_['cat']
                        cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out(cols)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(cols)
                
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_classifier.feature_importances_
                })
                
                feature_importance = feature_importance.sort_values('Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                ax.set_title('Top 15 Feature Importance')
                st.pyplot(fig)
            except:
                st.error("Could not extract feature importance.")

# Prediction Page
elif page == "Prediction":
    st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("No trained models found. Please go to the Model Training page to train models first.")
    else:
        selected_model = "Meta Learner"
        st.markdown("<h3 class='sub-header'>Enter Application Details</h3>", unsafe_allow_html=True)
        
        df = st.session_state.dataset
        input_data = {}
        
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
        if 'credit_risk' in numerical_features:
            numerical_features.remove('credit_risk')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numerical Features")
            for feature in numerical_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_data[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
        
        with col2:
            st.subheader("Categorical Features")
            for feature in categorical_features:
                options = df[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(
                    f"{feature.replace('_', ' ').title()}", 
                    options, 
                    index=0
                )
        
        input_df = pd.DataFrame([input_data])
        
        if st.button("Predict Approval Status"):
            with st.spinner("Generating prediction..."):
                model = st.session_state.models[selected_model]
                
                if selected_model == "Meta Learner":
                    dt_model = st.session_state.models['Decision Tree']
                    rf_model = st.session_state.models['Random Forest']
                    dt_pred = dt_model.predict_proba(input_df)[:, 1]
                    rf_pred = rf_model.predict_proba(input_df)[:, 1]
                    meta_features = np.column_stack((dt_pred, rf_pred))
                    prediction = model.predict(meta_features)[0]
                    prob = model.predict_proba(meta_features)[0][1]
                else:
                    prediction = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1]
                
                st.subheader("Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("✅ APPROVED")
                    else:
                        st.error("❌ DENIED")
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<p class='metric-value'>{prob*100:.2f}%</p>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Approval Probability</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("Approval Probability")
                
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(0, 100, height=0.5, color='lightgray', alpha=0.3)
                ax.barh(0, prob*100, height=0.5, color='green' if prediction == 1 else 'red')
                ax.axvline(x=50, color='black', linestyle='--', alpha=0.7)
                ax.set_yticks([])
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.text(25, -0.5, 'Low Probability', ha='center', va='top')
                ax.text(75, -0.5, 'High Probability', ha='center', va='top')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                st.pyplot(fig)
                
                if selected_model == "Random Forest":
                    st.subheader("Factors Influencing the Decision")
                    try:
                        rf_classifier = model.named_steps['classifier']
                        preprocessor = model.named_steps['preprocessor']
                        
                        transformed_input = preprocessor.transform(input_df)
                        
                        feature_names = []
                        for name, _, cols in preprocessor.transformers_:
                            if name == 'cat':
                                cat_transformer = preprocessor.named_transformers_['cat']
                                cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out(cols)
                                feature_names.extend(cat_features)
                            else:
                                feature_names.extend(cols)
                        
                        feature_contributions = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': rf_classifier.feature_importances_
                        })
                        
                        feature_contributions = feature_contributions.sort_values('Importance', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_contributions, ax=ax)
                        ax.set_title('Top Features Influencing This Decision')
                        st.pyplot(fig)
                    except:
                        st.info("Feature contributions could not be calculated.")
                
                st.subheader("General Credit Approval Guidelines")
                st.markdown("""
                Credit card applications are typically evaluated based on:
                
                - **Income Level**: Higher and stable income increases approval chances.
                - **Employment Status**: Stable employment history is preferred.
                - **Age**: Certain age groups may have different approval rates.
                - **Car Ownership**: Owning a car can be a positive factor.
                - **Realty Ownership**: Owning real estate can be a positive factor.
                
                The model uses these patterns to predict approval likelihood.
                """)

# Main function
def main():
    pass

if __name__ == "__main__":
    main()
