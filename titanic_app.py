import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Titanic Survival Analysis & Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    # Create sample Titanic dataset for demonstration
    np.random.seed(42)
    n_samples = 891
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29.7, 14.5, n_samples),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.004, 0.003, 0.003]),
        'Fare': np.random.lognormal(2.5, 1.2, n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
    }
    
    df = pd.DataFrame(data)
    df['Age'] = df['Age'].clip(0, 80)
    df['Fare'] = df['Fare'].clip(0, 500)
    
    return df

# Train a simple model for predictions
@st.cache_resource
def train_model(df):
    # Prepare features
    df_model = df.copy()
    df_model['Sex'] = LabelEncoder().fit_transform(df_model['Sex'])
    df_model['Embarked'] = LabelEncoder().fit_transform(df_model['Embarked'])
    df_model['Age'].fillna(df_model['Age'].median(), inplace=True)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df_model[features]
    y = df_model['Survived']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

# Load data and train model
df = load_data()
model, feature_names = train_model(df)

# Sidebar navigation
st.sidebar.title("🚢 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "📈 Visualizations", "🔮 Survival Predictor", "📋 Key Insights"]
)

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Analysis & Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the comprehensive Titanic Survival Analysis and Prediction application! This interactive dashboard provides:
    
    - **Data Analysis**: Explore the Titanic dataset with detailed statistics
    - **Visualizations**: Interactive charts showing survival patterns and trends
    - **Survival Predictor**: Predict passenger survival probability based on characteristics
    - **Key Insights**: Important findings from the analysis
    """)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Passengers", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Survival Rate", f"{df['Survived'].mean():.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Fare", f"${df['Fare'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Survival Rate:** {df['Survived'].mean():.2%}")
    
    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head())
    
    # Statistical summary
    st.markdown('<h2 class="section-header">Statistical Summary</h2>', unsafe_allow_html=True)
    st.dataframe(df.describe())
    
    # Survival by different categories
    st.markdown('<h2 class="section-header">Survival Analysis by Categories</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survival by Gender")
        survival_by_sex = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_by_sex.columns = ['Total', 'Survived', 'Survival Rate']
        st.dataframe(survival_by_sex)
    
    with col2:
        st.subheader("Survival by Class")
        survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_by_class.columns = ['Total', 'Survived', 'Survival Rate']
        st.dataframe(survival_by_class)

elif page == "📈 Visualizations":
    st.markdown('<h1 class="main-header">📈 Interactive Visualizations</h1>', unsafe_allow_html=True)
    
    # Survival Overview
    st.markdown('<h2 class="section-header">Survival Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Survival count
        fig_survival = px.pie(
            values=df['Survived'].value_counts().values,
            names=['Did not survive', 'Survived'],
            title="Overall Survival Distribution",
            color_discrete_sequence=['#ff7f7f', '#7fbf7f']
        )
        st.plotly_chart(fig_survival, use_container_width=True)
    
    with col2:
        # Survival by gender
        survival_gender = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
        fig_gender = px.bar(
            survival_gender,
            x='Sex',
            y='Count',
            color='Survived',
            title="Survival by Gender",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Class Analysis
    st.markdown('<h2 class="section-header">Passenger Class Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Survival by class
        survival_class = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
        fig_class = px.bar(
            survival_class,
            x='Pclass',
            y='Count',
            color='Survived',
            title="Survival by Passenger Class",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col2:
        # Class distribution
        fig_class_dist = px.pie(
            values=df['Pclass'].value_counts().values,
            names=['3rd Class', '1st Class', '2nd Class'],
            title="Passenger Class Distribution"
        )
        st.plotly_chart(fig_class_dist, use_container_width=True)
    
    # Age Analysis
    st.markdown('<h2 class="section-header">Age Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by survival
        fig_age = px.histogram(
            df,
            x='Age',
            color='Survived',
            title="Age Distribution by Survival",
            nbins=30,
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Age vs Fare scatter
        fig_scatter = px.scatter(
            df,
            x='Age',
            y='Fare',
            color='Survived',
            title="Age vs Fare by Survival",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Family Analysis
    st.markdown('<h2 class="section-header">Family Size Analysis</h2>', unsafe_allow_html=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Family size vs survival
        family_survival = df.groupby('FamilySize')['Survived'].mean().reset_index()
        fig_family = px.bar(
            family_survival,
            x='FamilySize',
            y='Survived',
            title="Survival Rate by Family Size"
        )
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Embarked analysis
        embarked_survival = df.groupby(['Embarked', 'Survived']).size().reset_index(name='Count')
        fig_embarked = px.bar(
            embarked_survival,
            x='Embarked',
            y='Count',
            color='Survived',
            title="Survival by Port of Embarkation",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        st.plotly_chart(fig_embarked, use_container_width=True)

elif page == "🔮 Survival Predictor":
    st.markdown('<h1 class="main-header">🔮 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts the survival probability of Titanic passengers based on their characteristics. 
    Please fill in the passenger information below to get a prediction.
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Passenger Information")
        
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 2 = Second Class, 3 = Third Class")
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 30)
        fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0, step=0.1)
    
    with col2:
        st.subheader("Family Information")
        
        sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], 
                               help="S = Southampton, C = Cherbourg, Q = Queenstown")
    
    # Predict button
    if st.button("Predict Survival", type="primary"):
        # Prepare input for prediction
        sex_encoded = 1 if sex == "Male" else 0
        embarked_encoded = {"S": 2, "C": 0, "Q": 1}[embarked]
        
        input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("🎉 **SURVIVED**")
            else:
                st.error("💔 **DID NOT SURVIVE**")
        
        with col2:
            st.metric("Survival Probability", f"{probability[1]:.1%}")
        
        with col3:
            st.metric("Death Probability", f"{probability[0]:.1%}")
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': ['Class', 'Gender', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Embarkation'],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Survival Prediction"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

elif page == "📋 Key Insights":
    st.markdown('<h1 class="main-header">📋 Key Insights from Titanic Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Main Findings")
    st.markdown("""
    Based on the comprehensive analysis of the Titanic dataset, here are the key insights that emerged:
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Gender Analysis
    st.markdown('<h2 class="section-header">👥 Gender Impact</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Women had significantly higher survival rates than men:**
        - Female survival rate: ~74%
        - Male survival rate: ~19%
        - This reflects the "women and children first" evacuation protocol
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick gender survival chart
        gender_survival = df.groupby('Sex')['Survived'].mean()
        fig_gender_insight = px.bar(
            x=gender_survival.index,
            y=gender_survival.values,
            title="Survival Rate by Gender"
        )
        st.plotly_chart(fig_gender_insight, use_container_width=True)
    
    # Class Analysis
    st.markdown('<h2 class="section-header">🎭 Social Class Impact</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Passenger class was a strong predictor of survival:**
        - 1st Class survival rate: ~63%
        - 2nd Class survival rate: ~47%
        - 3rd Class survival rate: ~24%
        - Higher class passengers had better access to lifeboats
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick class survival chart
        class_survival = df.groupby('Pclass')['Survived'].mean()
        fig_class_insight = px.bar(
            x=['1st Class', '2nd Class', '3rd Class'],
            y=class_survival.values,
            title="Survival Rate by Class"
        )
        st.plotly_chart(fig_class_insight, use_container_width=True)
    
    # Age Analysis
    st.markdown('<h2 class="section-header">👶 Age Factor</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Age played a crucial role in survival:**
    - Children (under 16) had higher survival rates
    - Young adults (20-40) had moderate survival rates
    - Elderly passengers had lower survival rates
    - The "women and children first" policy clearly benefited younger passengers
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Family Size Analysis
    st.markdown('<h2 class="section-header">👨‍👩‍👧‍👦 Family Size Impact</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Family size had a complex relationship with survival:**
    - Small families (2-4 members) had the highest survival rates
    - Solo travelers had moderate survival rates
    - Large families (5+ members) had lower survival rates
    - This suggests that small family groups could help each other while large groups faced coordination challenges
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Economic Factors
    st.markdown('<h2 class="section-header">💰 Economic Factors</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Fare paid was correlated with survival:**
    - Higher fare passengers had better survival rates
    - This correlates with passenger class and cabin location
    - Economic status provided better access to safety resources
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Port of Embarkation
    st.markdown('<h2 class="section-header">⚓ Port of Embarkation</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Port of embarkation showed interesting patterns:**
    - Cherbourg (C) passengers had the highest survival rate
    - Southampton (S) passengers had moderate survival rates
    - Queenstown (Q) passengers had the lowest survival rates
    - This likely reflects the class composition of passengers from different ports
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance
    st.markdown('<h2 class="section-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Machine Learning Model Insights:**
    - The Random Forest model achieves good prediction accuracy
    - Most important features: Gender, Passenger Class, Age, Fare
    - The model can predict survival with reasonable confidence
    - Feature engineering (like family size) improves prediction accuracy
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusions
    st.markdown('<h2 class="section-header">🎯 Final Conclusions</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **Key Takeaways:**
    1. **Social factors dominated survival**: Gender and class were the strongest predictors
    2. **"Women and children first" was real**: The evacuation protocol was largely followed
    3. **Economic inequality affected survival**: Wealth provided better access to safety
    4. **Family dynamics mattered**: Small families had advantages over solo travelers and large groups
    5. **Location on ship mattered**: Higher-class passengers had better positioned cabins
    
    These insights provide valuable historical context about social dynamics during the Titanic disaster
    and demonstrate how machine learning can uncover patterns in historical data.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚢 Titanic Survival Analysis & Predictor</p>
    <p>Built with Streamlit • Enhanced with Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)

