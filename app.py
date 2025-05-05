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
import pickle
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

# Function to load sample dataset
@st.cache_data
def load_credit_data():
    """Load the German Credit dataset"""
    try:
        # URL for the German Credit dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        
        # Column names (these are meaningful names, not A1, A2, etc.)
        column_names = [
            'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'present_residence', 'property', 'age',
            'other_installment_plans', 'housing', 'number_credits',
            'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk'
        ]
        
        # Load the data
        df = pd.read_csv(url, sep=' ', header=None, names=column_names)
        
        # In this dataset, credit_risk is 1 for good credit and 2 for bad credit
        # Let's transform it to 1 for approved (good credit) and 0 for denied (bad credit)
        df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return a dummy dataset in case of error
        dummy_data = pd.DataFrame(np.random.rand(100, 20), columns=column_names[:-1])
        dummy_data['credit_risk'] = np.random.randint(0, 2, size=100)
        return dummy_data

# Function to load South German Credit dataset (alternative with more features)
@st.cache_data
def load_south_german_credit():
    """Load the South German Credit dataset"""
    try:
        url = "https://raw.githubusercontent.com/FrancescoBontempo/credit-risk-analysis/main/data/SouthGermanCredit.csv"
        df = pd.read_csv(url)
        
        # Rename the target column to maintain consistency with our code
        df = df.rename(columns={'Class': 'credit_risk'})
        
        # In this dataset, credit_risk is 0 for good credit and 1 for bad credit
        # Let's transform it to 1 for approved (good credit) and 0 for denied (bad credit)
        df['credit_risk'] = df['credit_risk'].map({0: 1, 1: 0})
        
        return df
    except Exception as e:
        st.error(f"Error loading South German Credit dataset: {e}")
        # Return the other dataset as a fallback
        return load_credit_data()

# Function to preprocess data
def preprocess_data(df, target_column='credit_risk'):
    """Preprocess the data for modeling"""
    # Define target
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Split features into categorical and numerical
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to train models
def train_models(X_train, y_train, preprocessor):
    """Train decision tree, random forest, and meta learner models"""
    models = {}
    
    # Create and fit decision tree pipeline
    dt_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    
    # Create and fit random forest pipeline
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Meta learner (averaging predictions)
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
        # For meta model, we need to get base model predictions first
        dt_preds = st.session_state.models['Decision Tree'].predict_proba(X_test)[:, 1]
        rf_preds = st.session_state.models['Random Forest'].predict_proba(X_test)[:, 1]
        meta_features = np.column_stack((dt_preds, rf_preds))
        y_pred = model.predict(meta_features)
        y_pred_proba = model.predict_proba(meta_features)[:, 1]
    else:
        # For base models
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, report, conf_matrix, fpr, tpr, roc_auc

# Save models function
def save_models(models, preprocessor):
    """Save all models and preprocessor to disk"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save base models
    joblib.dump(models['Decision Tree'], 'models/decision_tree_model.pkl')
    joblib.dump(models['Random Forest'], 'models/random_forest_model.pkl')
    
    # Save meta model
    joblib.dump(models['Meta Learner'], 'models/meta_learner_model.pkl')
    
    # Save preprocessor
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

categorical_mappings = {
    'status': {
        'A11': '< 0 DM',
        'A12': '0 <= ... < 200 DM',
        'A13': '>= 200 DM',
        'A14': 'no checking account'
    },
    'credit_history': {
        'A30': 'no credits taken/all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/other credits existing (not at this bank)'
    },
    'purpose': {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    },
    'savings': {
        'A61': '< 100 DM',
        'A62': '100 <= ... < 500 DM',
        'A63': '500 <= ... < 1000 DM',
        'A64': '>= 1000 DM',
        'A65': 'unknown/no savings account'
    },
    'employment_duration': {
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1 <= ... < 4 years',
        'A74': '4 <= ... < 7 years',
        'A75': '>= 7 years'
    },
    'personal_status_sex': {
        'A91': 'male: divorced/separated',
        'A92': 'female: divorced/separated/married',
        'A93': 'male: single',
        'A94': 'male: married/widowed',
        'A95': 'female: single'
    },
    'other_debtors': {
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'
    },
    'property': {
        'A121': 'real estate',
        'A122': 'building society savings agreement/life insurance',
        'A123': 'car or other',
        'A124': 'unknown/no property'
    },
    'other_installment_plans': {
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none'
    },
    'housing': {
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free'
    },
    'job': {
        'A171': 'unemployed/unskilled - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee/official',
        'A174': 'management/self-employed/highly qualified employee/officer'
    },
    'telephone': {
        'A191': 'none',
        'A192': 'yes, registered under the customer\'s name'
    },
    'foreign_worker': {
        'A201': 'yes',
        'A202': 'no'
    }
}

# Data Overview Page
if page == "Data Overview":
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
    
    df = st.session_state.dataset
    
    # Display dataset info
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
    
    # Display raw data with toggle
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df.head(100))
    
    # Data summary
    st.markdown("<h3 class='sub-header'>Data Summary</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        st.dataframe(df.select_dtypes(exclude=['object']).describe())
    
    with col2:
        st.subheader("Categorical Features")
        cat_summary = {}
        for col in df.select_dtypes(include=['object']).columns:
            cat_summary[col] = df[col].value_counts().shape[0]
        st.dataframe(pd.DataFrame(cat_summary.items(), columns=['Feature', 'Unique Values']))
    
    # Visualizations
    st.markdown("<h3 class='sub-header'>Data Visualizations</h3>", unsafe_allow_html=True)
    
    # Choose a column to visualize
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
            
            # Create count plot
            counts = df.groupby([selected_cat_col, 'credit_risk']).size().unstack(fill_value=0)
            counts.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Distribution of {selected_cat_col} by Approval Status")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Count")
            ax.legend(["Denied", "Approved"])
            st.pyplot(fig)
    
    # Correlation matrix for numeric features
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(exclude=['object'])
    if numeric_df.shape[1] > 1:  # Only plot if there are at least 2 numeric columns
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric features to create a correlation matrix.")

# Model Training Page
elif page == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    # Show dataset dimensions and approval rate
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
    
    # Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        dt_max_depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=20, value=5)
    
    with col2:
        rf_n_estimators = st.slider("Random Forest Number of Trees", min_value=50, max_value=300, value=100, step=10)
        rf_max_depth = st.slider("Random Forest Max Depth", min_value=2, max_value=20, value=10)
    
    # Train models button
    if st.button("Train Models"):
        with st.spinner("Training models, please wait..."):
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
            
            # Store test data for later evaluation
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            
            # Train base models
            dt_model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42))
            ])
            dt_model.fit(X_train, y_train)
            
            rf_model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=rf_n_estimators, 
                    max_depth=rf_max_depth, 
                    random_state=42
                ))
            ])
            rf_model.fit(X_train, y_train)
            
            # Store models
            st.session_state.models['Decision Tree'] = dt_model
            st.session_state.models['Random Forest'] = rf_model
            
            # Meta learner 
            dt_preds = dt_model.predict_proba(X_train)[:, 1]
            rf_preds = rf_model.predict_proba(X_train)[:, 1]
            
            meta_features = np.column_stack((dt_preds, rf_preds))
            meta_model = DecisionTreeClassifier(max_depth=3, random_state=42)
            meta_model.fit(meta_features, y_train)
            
            st.session_state.models['Meta Learner'] = meta_model
            st.session_state.trained = True
            
            st.success("Models trained successfully!")
    
    # Save models button (only enabled if models are trained)
    if st.session_state.trained:
        if st.button("Save Models"):
            save_models(st.session_state.models, st.session_state.preprocessor)
    else:
        st.info("Train models first before saving.")
    
    # Option to load pre-trained models
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
        # Select model to evaluate
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select a model to evaluate", model_names)
        
        # Get the selected model
        model = st.session_state.models[selected_model]
        
        # Evaluate the model
        if selected_model == "Meta Learner":
            accuracy, report, conf_matrix, fpr, tpr, roc_auc = evaluate_model(
                model, 
                st.session_state.X_test, 
                st.session_state.y_test, 
                is_meta=True
            )
        else:
            accuracy, report, conf_matrix, fpr, tpr, roc_auc = evaluate_model(
                model, 
                st.session_state.X_test, 
                st.session_state.y_test
            )
        
        # Display metrics
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
            st.markdown("<p class='metric-label'>Precision (Approval)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            recall = report['1']['recall']
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{recall*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Recall (Approval)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization of results
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
        
        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        # Filter out the unnecessary rows and columns
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
        if 'macro avg' in report_df.index and 'weighted avg' in report_df.index:
            report_df = report_df.drop(['macro avg', 'weighted avg'])
        
        # Format the report
        for col in report_df.columns:
            if col != 'support':
                report_df[col] = report_df[col].apply(lambda x: f"{x*100:.2f}%")
        
        # Map class labels
        report_df.index = report_df.index.map({'0': 'Denied', '1': 'Approved'})
        
        st.dataframe(report_df)
        
        # Compare models if there are multiple
        st.subheader("Model Comparison")
        
        model_metrics = {}
        for name in model_names:
            current_model = st.session_state.models[name]
            
            if name == "Meta Learner":
                acc, _, _, _, _, auc_score = evaluate_model(
                    current_model, 
                    st.session_state.X_test, 
                    st.session_state.y_test, 
                    is_meta=True
                )
            else:
                acc, _, _, _, _, auc_score = evaluate_model(
                    current_model, 
                    st.session_state.X_test, 
                    st.session_state.y_test
                )
            
            model_metrics[name] = [acc, auc_score]
        
        comparison_df = pd.DataFrame(model_metrics, index=['Accuracy', 'AUC']).transpose()
        comparison_df['Accuracy'] = comparison_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
        comparison_df['AUC'] = comparison_df['AUC'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(comparison_df)
        
        # If Random Forest model, show feature importance
        if selected_model == "Random Forest":
            st.subheader("Feature Importance")
            
            # Extract the random forest classifier from the pipeline
            rf_classifier = model.named_steps['classifier']
            
            # Get feature names after preprocessing (if possible)
            try:
                preprocessor = model.named_steps['preprocessor']
                cat_cols = preprocessor.transformers_[1][2]  # Categorical columns
                num_cols = preprocessor.transformers_[0][2]  # Numerical columns
                
                # Get one-hot encoded feature names
                cat_feature_names = []
                for cat_col in cat_cols:
                    unique_values = st.session_state.dataset[cat_col].dropna().unique()
                    for val in unique_values:
                        cat_feature_names.append(f"{cat_col}_{val}")
                
                # Combine with numerical feature names
                feature_names = list(num_cols) + cat_feature_names
                
                # If feature names don't match, use generic names
                if len(feature_names) != len(rf_classifier.feature_importances_):
                    feature_names = [f"Feature {i}" for i in range(len(rf_classifier.feature_importances_))]
                
                # Create a dataframe for feature importance
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_classifier.feature_importances_
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('Importance', ascending=False).head(15)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                ax.set_title('Top 15 Feature Importance')
                st.pyplot(fig)
                
            except:
                st.error("Could not extract feature importance. This could be due to the preprocessing pipeline structure.")

# Prediction Page
elif page == "Prediction":
    st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("No trained models found. Please go to the Model Training page to train models first.")
    else:
        # Select model for prediction
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select a model for prediction", model_names)
        
        st.markdown("<h3 class='sub-header'>Enter Application Details</h3>", unsafe_allow_html=True)
        
        # Get a sample from the dataset for reference
        df = st.session_state.dataset
        
        # Create input fields based on dataset columns (excluding target variable)
        input_data = {}
        
        # Group features by type for better organization
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
        
        if 'credit_risk' in numerical_features:
            numerical_features.remove('credit_risk')
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        # Add numerical feature inputs
        with col1:
            st.subheader("Numerical Features")
            for feature in numerical_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_data[feature] = st.slider(
                    f"{feature}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
        
        # Add categorical feature inputs with descriptions
        with col2:
            st.subheader("Categorical Features")
            for feature in categorical_features:
                if feature in categorical_mappings:
                    # Get the mapping for this feature
                    mapping = categorical_mappings[feature]
                    # Create a list of descriptions
                    descriptions = list(mapping.values())
                    # Set default to the first description
                    default_description = descriptions[0]
                    # Create select box with descriptions
                    selected_description = st.selectbox(f"{feature}", descriptions, index=0)
                    # Map the selected description back to the code
                    code = [k for k, v in mapping.items() if v == selected_description][0]
                    input_data[feature] = code
                else:
                    # Fallback if no mapping exists
                    options = df[feature].dropna().unique().tolist()
                    default_option = options[0] if options else ""
                    input_data[feature] = st.selectbox(f"{feature}", options, index=0)
        
        # Create a dataframe from input
        input_df = pd.DataFrame([input_data])
        
        # Make prediction button
        if st.button("Predict Approval Status"):
            with st.spinner("Generating prediction..."):
                # Get the selected model
                model = st.session_state.models[selected_model]
                
                # Make prediction
                if selected_model == "Meta Learner":
                    # Get base model predictions
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
                
                # Display result
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
                
                # Visualization of probability
                st.subheader("Approval Probability")
                
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Create a simple gauge chart
                ax.barh(0, 100, height=0.5, color='lightgray', alpha=0.3)
                ax.barh(0, prob*100, height=0.5, color='green' if prediction == 1 else 'red')
                
                # Add threshold line at 50%
                ax.axvline(x=50, color='black', linestyle='--', alpha=0.7)
                
                # Remove y-axis ticks and labels
                ax.set_yticks([])
                
                # Set x-axis limits and ticks
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                
                # Add labels for low/high probability regions
                ax.text(25, -0.5, 'Low Probability', ha='center', va='top')
                ax.text(75, -0.5, 'High Probability', ha='center', va='top')
                
                # Remove box around plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                st.pyplot(fig)
                
                # Explain the factors influencing the decision
                st.subheader("Factors Influencing the Decision")
                
                # For random forest, we can show feature importance for this prediction
                if selected_model == "Random Forest":
                    try:
                        # Extract the random forest classifier from the pipeline
                        rf_classifier = model.named_steps['classifier']
                        preprocessor = model.named_steps['preprocessor']
                        
                        # Transform the input data
                        transformed_input = preprocessor.transform(input_df)
                        
                        # Calculate feature contributions
                        feature_contributions = []
                        
                        # Get feature names (simplistic approach)
                        feature_names = []
                        for name, _, cols in preprocessor.transformers_:
                            if name == 'cat':
                                # For categorical features, get one-hot encoded names
                                for col in cols:
                                    unique_values = df[col].dropna().unique()
                                    for val in unique_values:
                                        feature_names.append(f"{col}_{val}")
                            else:
                                # For numerical features, just use the column names
                                feature_names.extend(cols)
                        
                        # If feature names don't match, use generic names
                        if len(feature_names) != transformed_input.shape[1]:
                            feature_names = [f"Feature {i}" for i in range(transformed_input.shape[1])]
                        
                        # Loop through trees to find feature contributions
                        for tree in rf_classifier.estimators_:
                            for feature_idx in range(transformed_input.shape[1]):
                                feature_contribution = 0
                                
                                # This is a simplified approach - in reality, we would need
                                # to traverse the tree to get accurate contributions
                                if feature_idx in tree.feature_importances_.nonzero()[0]:
                                    feature_contribution = tree.feature_importances_[feature_idx]
                                
                                if len(feature_contributions) <= feature_idx:
                                    feature_contributions.append(feature_contribution)
                                else:
                                    feature_contributions[feature_idx] += feature_contribution
                        
                        # Normalize contributions
                        feature_contributions = [fc / len(rf_classifier.estimators_) for fc in feature_contributions]
                        
                        # Create a dataframe of feature contributions
                        contrib_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Contribution': feature_contributions
                        })
                        
                        # Sort and display top contributors
                        contrib_df = contrib_df.sort_values('Contribution', ascending=False).head(10)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Contribution', y='Feature', data=contrib_df, ax=ax)
                        ax.set_title('Top Features Influencing This Decision')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Could not determine feature contributions: {str(e)}")
                        st.info("Feature contributions are only available for the Random Forest model.")
                else:
                    st.info("Feature contributions are only available for the Random Forest model. Please select Random Forest to see detailed explanations.")
                
                # Show general rules for credit approval
                st.subheader("General Credit Approval Guidelines")
                st.markdown("""
                Credit card applications are typically evaluated based on the following factors:
                
                - **Credit History**: Length and quality of credit history
                - **Income**: Stable source and sufficient level of income
                - **Debt-to-Income Ratio**: Lower ratios are preferred
                - **Employment Status**: Stable employment history
                - **Age**: Must meet minimum age requirements
                - **Residence**: Stability of residence
                
                The model has been trained on historical approval data and uses patterns in the data to make predictions.
                """)

# Function to run the app
def main():
    pass  # All the app logic is already in the main body

if __name__ == "__main__":
    main()
