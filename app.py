import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .survived {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .not-survived {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def preprocess(df2):
    """
    Preprocessing function to transform input data
    """
    # Family size category
    df2['FamilySize'] = df2['SibSp'] + df2['Parch'] + 1
    df2['FamilySizeCategory'] = df2['FamilySize'].apply(
        lambda x: 0 if x == 1 else 1 if x <= 4 else 2
    )

    # Age category (assuming no missing values)
    df2['AgeCategory'] = df2['Age'].apply(
        lambda x: 0 if x < 18 else 1 if x < 60 else 2
    )

    # Encode Sex
    if df2['Sex'].dtype == object:
        df2['Sex'] = df2['Sex'].str.strip().str.lower().map({'male': 0, 'female': 1})

    # One-hot encode Embarked (assuming no missing values)
    df2 = pd.get_dummies(df2, columns=['Embarked'], prefix='Embarked')
    # Ensure all expected embarked columns exist
    for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
        if col not in df2:
            df2[col] = 0

    # Drop unused columns
    drop_cols = ['Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'FamilySize']
    if 'Title' in df2.columns:
        drop_cols.append('Title')
    df2.drop(columns=[c for c in drop_cols if c in df2.columns], inplace=True)

    # Return columns in consistent order
    expected_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySizeCategory', 
                    'AgeCategory', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    
    # Add any missing columns with 0 (safety check)
    for col in expected_cols:
        if col not in df2.columns:
            df2[col] = 0
            
    return df2[expected_cols]

@st.cache_resource
def load_model():
    """
    Load the trained model
    """
    try:
        model_path = "best_rf_model.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"Could not load the original model ({str(e)}). Using a simple rule-based predictor for demonstration.")
        return None

def simple_predictor(pclass, sex, age, fare, sibsp, parch, embarked):
    """
    Simple rule-based predictor as fallback
    """
    # Basic survival probability based on historical patterns
    survival_score = 0.5  # Base probability
    
    # Gender (most important factor)
    if sex.lower() == 'female':
        survival_score += 0.3
    else:
        survival_score -= 0.2
    
    # Class
    if pclass == 1:
        survival_score += 0.2
    elif pclass == 3:
        survival_score -= 0.15
    
    # Age
    if age < 16:  # Children
        survival_score += 0.1
    elif age > 60:  # Elderly
        survival_score -= 0.1
    
    # Family size
    family_size = sibsp + parch + 1
    if family_size == 1:  # Alone
        survival_score -= 0.05
    elif family_size > 4:  # Large family
        survival_score -= 0.1
    
    # Fare (proxy for wealth)
    if fare > 50:
        survival_score += 0.1
    elif fare < 10:
        survival_score -= 0.05
    
    # Ensure probability is between 0 and 1
    survival_score = max(0, min(1, survival_score))
    
    # Return prediction and probabilities
    prediction = 1 if survival_score > 0.5 else 0
    probabilities = [1 - survival_score, survival_score]
    
    return prediction, probabilities

def main():
    # Header
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts the survival probability of Titanic passengers based on their characteristics.
    Please fill in the passenger information below to get a prediction.
    """)
    
    # Load model
    model = load_model()
    
    # Continue even if model is None (will use fallback predictor)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Passenger Information")
        
        # Passenger Class
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            help="1 = First Class, 2 = Second Class, 3 = Third Class"
        )
        
        # Sex
        sex = st.selectbox(
            "Gender",
            options=["Male", "Female"]
        )
        
        # Age
        age = st.slider(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            help="Age in years"
        )
        
        # Fare
        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=1000.0,
            value=32.0,
            step=0.1,
            help="Ticket fare in pounds"
        )
    
    with col2:
        st.subheader("Family Information")
        
        # Number of siblings/spouses
        sibsp = st.number_input(
            "Number of Siblings/Spouses aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
        
        # Number of parents/children
        parch = st.number_input(
            "Number of Parents/Children aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
        
        # Port of Embarkation
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["C", "Q", "S"],
            index=2,
            help="C = Cherbourg, Q = Queenstown, S = Southampton"
        )
    
    # Prediction button
    if st.button("Predict Survival", type="primary"):
        # Create input dataframe with raw data for the model's pipeline
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Embarked': [embarked]
        })
        
        # The model pipeline will handle preprocessing, so we don't need to preprocess manually
        
        # Make prediction
        try:
            if model is not None:
                # Try using the loaded model
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                except Exception as model_error:
                    # If model fails, fall back to simple predictor
                    st.warning(f"Model prediction failed ({str(model_error)}). Using fallback predictor.")
                    prediction, prob_array = simple_predictor(pclass, sex, age, fare, sibsp, parch, embarked)
                    probability = prob_array
            else:
                # Use fallback predictor
                prediction, prob_array = simple_predictor(pclass, sex, age, fare, sibsp, parch, embarked)
                probability = prob_array
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box survived">'
                    f'<h3>✅ SURVIVED</h3>'
                    f'<p>Survival Probability: {probability[1]:.2%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box not-survived">'
                    f'<h3>❌ DID NOT SURVIVE</h3>'
                    f'<p>Survival Probability: {probability[1]:.2%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Show probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Survival Probability", f"{probability[1]:.2%}")
            with col2:
                st.metric("Death Probability", f"{probability[0]:.2%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Additional information
    with st.expander("About this Model"):
        st.markdown("""
        This model was trained on the famous Titanic dataset to predict passenger survival.
        
        **Features used:**
        - Passenger Class (Pclass)
        - Gender (Sex)
        - Age
        - Fare
        - Family Size (derived from SibSp and Parch)
        - Port of Embarkation
        
        **Model Performance:**
        The model uses a Random Forest classifier and has been optimized for accuracy.
        """)

if __name__ == "__main__":
    main()

