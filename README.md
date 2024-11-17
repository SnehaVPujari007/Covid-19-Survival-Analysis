# Covid-19 Survival Prediction

This project aims to predict the survival of Covid-19 patients based on various health and demographic factors. The goal is to use machine learning to analyze features such as age, gender, comorbidities, and treatment given to determine the likelihood of patient survival.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)

8. [Visualizations](#visualizations)

10. [License](#license)

## Project Overview

The Covid-19 pandemic has posed significant challenges to healthcare systems worldwide. Early prediction of patient survival can assist in decision-making and resource allocation. This project builds a machine learning model to predict whether a Covid-19 patient will survive based on several features. 

### Objective:
- **Target Variable**: Survival (1 = Survived, 0 = Not Survived)
- **Predictors**: Age, Gender, Comorbidities (e.g., diabetes, hypertension), Severity of symptoms, Treatment methods, and Lifestyle factors (e.g., smoking).

## Dataset

The dataset used in this project contains information about Covid-19 patients. You can download it from [Kaggle's Covid-19 Dataset](https://www.kaggle.com/datasets) or any relevant sources. 

The dataset should contain columns such as:
- **Age**: Age of the patient
- **Gender**: Gender of the patient
- **Health Conditions**: Any pre-existing conditions like diabetes, hypertension, etc.
- **Severity of Symptoms**: Mild, Moderate, Severe
- **Treatment Given**: Type of treatment like ICU care, ventilator usage, etc.
- **Survival**: Target variable indicating whether the patient survived (1) or not (0)

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `Pandas` for data manipulation
  - `NumPy` for numerical operations
  - `Scikit-learn` for machine learning algorithms
  - `Matplotlib` and `Seaborn` for data visualization
  - `Power BI` for building interactive dashboards

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SnehaVPujari/covid-survival-prediction.git

``
## Data Preprocessing

The dataset undergoes the following preprocessing steps to prepare the data for machine learning model training:

### 1. **Missing Values Handling**
   - Missing values in the dataset are handled by:
     - Filling numerical features with the **mean** value.
     - Removing rows or columns with excessive missing data if necessary.

### 2. **Categorical Encoding**
   - Categorical features, such as **gender** and **treatment types**, are encoded using:
     - **One-Hot Encoding** for nominal variables (e.g., gender, treatment types).
     - **Label Encoding** for ordinal variables if applicable.

### 3. **Feature Scaling**
   - Numerical features, such as **age** and **severity**, are standardized using **StandardScaler**. This ensures that the data is normalized for algorithms that are sensitive to the scale of the data (e.g., Logistic Regression, KNN).

### 4. **Feature Selection**
   - Correlations between features are analyzed to identify significant predictors for survival.
   - Features that do not contribute meaningfully to the model's performance are either removed or combined.
   - **Feature Importance** is determined using models like Random Forest or by using statistical tests to identify key variables affecting survival.

---

## Model Building

After data preprocessing, the dataset is ready for building and training machine learning models. The following steps outline the model building process:

### 1. **Data Split**
   - The dataset is split into **training** and **testing** sets (typically an 80-20 or 70-30 split). This ensures that the model can be trained on one set of data and evaluated on another to avoid overfitting.

### 2. **Machine Learning Models**
   The following machine learning models are used to predict the survival of Covid-19 patients:
   - **Logistic Regression**: A simple and effective linear model for binary classification.
   - **Decision Trees**: A tree-based model that helps capture non-linear relationships in the data.
   - **Random Forest**: An ensemble method that builds multiple decision trees to improve performance and reduce overfitting.
   - **XGBoost**: An advanced gradient boosting algorithm known for its efficiency and accuracy in classification tasks.
   - **Support Vector Machine (SVM)**: A powerful algorithm used for classification, especially with complex decision boundaries.

### 3. **Model Training**
   - The chosen models are trained using the **training data** and evaluated on the **testing data**.
   - We tune hyperparameters and perform cross-validation to ensure robust model performance.
   - Each model is evaluated using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC** to determine its effectiveness.

## Visualizations

Visualizations play an important role in understanding the dataset and interpreting the results of the model. In this project, the following visualizations are created to gain insights into the data and the model's performance:

### 1. **Feature Distributions**
   - Visualizations are created to show the distribution of key features in the dataset. For example:
     - Age distribution of Covid-19 patients.
     - The distribution of **comorbidities**, **severity levels**, and **treatment methods**.
   - These visualizations help in understanding the dataset and the potential relationship between features and survival.

### 2. **Survival Rates**
   - The survival rates are visualized based on different features such as:
     - **Age groups**: Comparing survival rates across various age groups.
     - **Health conditions**: Visualizing how comorbidities like hypertension, diabetes, etc., affect survival.
     - **Treatment types**: Comparing survival rates based on different treatments (e.g., ICU care, ventilators).
   - These visualizations are important to identify trends and potential factors that significantly affect patient survival.

### 3. **Confusion Matrices**
   - **Confusion matrices** are plotted to evaluate the performance of the machine learning models.
   - They show the number of **True Positives (TP)**, **True Negatives (TN)**, **False Positives (FP)**, and **False Negatives (FN)** for each model, which helps in understanding how well the model is predicting survival vs. death.

### 4. **Interactive Dashboard (Power BI)**
   - An interactive dashboard is created using **Power BI** to visualize survival predictions and compare survival rates across various factors.
   - The dashboard includes:
     - Interactive filters for age, gender, health conditions, etc.
     - Visualizations like bar charts, pie charts, and heatmaps to analyze relationships between features and survival.
     - A live prediction feature (optional), where users can input patient data to predict survival.



## License

This project has MIT License.



 
