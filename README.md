# Titanic Survival Prediction: A Machine Learning Web Application

## Description

This project delivers a comprehensive solution for predicting Titanic passenger survival, leveraging machine learning and a user-friendly web interface. It encompasses the entire machine learning lifecycle, from data preprocessing and model development to interactive deployment. The core of the system is a Random Forest Classifier, trained on the historical Titanic dataset to identify key factors influencing survival. The model is then integrated into a Streamlit web application, providing an intuitive platform for users to input passenger details and receive real-time survival predictions.

One of the key design considerations for this application was robustness. Recognizing potential challenges with model compatibility across different environments or future updates, a resilient fallback prediction mechanism has been implemented. In instances where the primary machine learning model encounters an issue (e.g., version incompatibility, corrupted file), the application seamlessly transitions to a rule-based predictor. This ensures continuous functionality and a consistent user experience, even under unforeseen circumstances, highlighting a practical approach to deploying machine learning solutions in dynamic environments.

The project serves as a practical demonstration of building and deploying an end-to-end machine learning pipeline, emphasizing best practices in data science, software engineering, and user interface design. It is ideal for those interested in understanding how predictive models can be operationalized and made accessible through interactive web applications.



## Features

*   **Interactive Web Application:** A user-friendly interface built with Streamlit allows for easy input of passenger details and instant survival predictions.
*   **Machine Learning Model:** Utilizes a Random Forest Classifier for accurate and robust survival predictions based on historical data.
*   **Comprehensive Data Preprocessing:** Includes a custom preprocessing pipeline to handle various data types and engineer relevant features (e.g., `FamilySize`, `AgeCategory`, one-hot encoding for `Embarked`).
*   **Robust Fallback Mechanism:** Features a rule-based predictor that automatically activates if the primary machine learning model encounters loading or prediction errors, ensuring continuous service availability.
*   **Clear Prediction Visualization:** Presents survival predictions and probabilities in an easily understandable format, with visual cues for survival status.
*   **Modular Codebase:** Organized and well-commented Python code, promoting readability, maintainability, and extensibility.
*   **Dependency Management:** `requirements.txt` file for straightforward installation of all necessary libraries.
*   **Customizable Input Fields:** The Streamlit interface is designed to expose only the relevant input features for prediction, simplifying the user experience.



## Technologies Used

This project is developed using the following key technologies and libraries:

*   **Python 3.x:** The primary programming language for the entire project.
*   **Streamlit:** For building the interactive web application interface.
*   **Scikit-learn:** For machine learning model development, including the Random Forest Classifier and `FunctionTransformer` for preprocessing.
*   **Pandas:** For data manipulation and analysis, crucial for handling the Titanic dataset and preprocessing steps.
*   **Joblib:** For efficient serialization and deserialization of the trained machine learning model.
*   **NumPy:** For numerical operations, often used implicitly by Pandas and Scikit-learn.
*   **Git & GitHub:** For version control and collaborative development.



## Setup and Installation

To get this project up and running on your local machine, follow these steps:

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub username and repository name.

### Create a Virtual Environment (Recommended)

It's highly recommended to create a virtual environment to manage project dependencies. This prevents conflicts with other Python projects.

```bash
python -m venv venv
```

### Activate the Virtual Environment

*   **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### Install Dependencies

Once your virtual environment is active, install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Place the Model File

Ensure your trained model file (`best_rf_model.pkl`) is placed in the root directory of the project, alongside `app.py`. If you trained your model separately, you would save it to this location. If you are using the provided `app.py`, it expects this file to be present. If the model file is not found or is incompatible, the application will automatically switch to a rule-based fallback predictor.



## Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if you haven't already):
    *   **On Windows:** `.\venv\Scripts\activate`
    *   **On macOS/Linux:** `source venv/bin/activate`

2.  **Run the Streamlit app** from the project root directory:
    ```bash
    streamlit run app.py
    ```

    This command will open a new tab in your default web browser, displaying the Titanic Survival Predictor application. If it doesn't open automatically, a local URL (e.g., `http://localhost:8501`) will be provided in your terminal, which you can copy and paste into your browser.

### Using the Application

Once the application is running, you can:

*   **Input Passenger Details:** Use the provided input fields (Passenger Class, Gender, Age, Fare, Number of Siblings/Spouses aboard, Number of Parents/Children aboard, Port of Embarkation) to enter the characteristics of a hypothetical or real Titanic passenger.
*   **Get Prediction:** Click the "Predict Survival" button to see the model's prediction (Survived or Did Not Survive) and the associated survival probabilities.
*   **Observe Fallback:** If the primary machine learning model encounters an issue, a warning message will appear, and the application will seamlessly use the built-in rule-based fallback predictor to provide a prediction.

### Access the application here
https://titanic-predition-model.streamlit.app/


## Project Structure

The repository is structured as follows:

```
titanic_streamlit_app/
├── titanic_app.py
├── best_rf_model.pkl
├── requirements.txt
└── README.md
```

*   `app.py`: The main Streamlit application script, containing the UI, preprocessing logic, and prediction handling.
*   `best_rf_model.pkl`: The serialized machine learning model (Random Forest Classifier pipeline).
*   `requirements.txt`: Lists all Python dependencies required to run the project.
*   `README.md`: This file, providing an overview and instructions for the project.



## Model Details

### Machine Learning Model

The primary prediction engine is a **Random Forest Classifier**, chosen for its robustness, ability to handle various data types, and strong predictive performance in classification tasks. The model was trained on a comprehensive dataset derived from the original Titanic passenger manifest, encompassing a range of features such as passenger class, age, gender, fare, and embarkation point.

### Preprocessing Pipeline

The model incorporates a custom preprocessing pipeline designed to transform raw input data into a format suitable for the Random Forest Classifier. This pipeline includes:

*   **Feature Engineering:**
    *   `FamilySize`: Calculated as `SibSp` (number of siblings/spouses aboard) + `Parch` (number of parents/children aboard) + 1 (for the passenger themselves). This feature captures the size of the family unit.
    *   `FamilySizeCategory`: Categorizes `FamilySize` into discrete groups (e.g., alone, small family, large family) to capture non-linear relationships.
    *   `AgeCategory`: Categorizes `Age` into age groups (e.g., child, adult, senior) to account for age-related survival patterns.
*   **Categorical Feature Encoding:**
    *   `Sex`: Encoded using a binary mapping (e.g., Male=0, Female=1).
    *   `Embarked`: One-hot encoded to convert categorical port information into numerical format (`Embarked_C`, `Embarked_Q`, `Embarked_S`).
*   **Feature Selection/Dropping:** Irrelevant or redundant columns (e.g., `Name`, `Ticket`, `Cabin`) are dropped to reduce noise and improve model efficiency.

This preprocessing is encapsulated within the saved model pipeline, meaning that raw input data provided to the deployed application is automatically transformed before being fed into the Random Forest model.

### Fallback Predictor

To ensure the application remains functional even if the primary machine learning model encounters compatibility issues or fails to load, a simple rule-based fallback predictor is integrated. This predictor makes survival predictions based on general historical patterns and common sense rules (e.g., women and children are more likely to survive, higher class passengers have better survival rates). While not as accurate as the trained machine learning model, it provides a basic level of functionality and a smooth user experience under adverse conditions.



## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes relevant tests where applicable.



## Contact

For any questions or inquiries, please open an issue in this repository or contact the author:

*   **Author:** Wayne Otieno
*   **GitHub:** https://github.com/Wayneotc 


